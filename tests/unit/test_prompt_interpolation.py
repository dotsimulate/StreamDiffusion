"""Unit tests for the prompt AND seed interpolation modes in
stream_parameter_updater.py:

  - ``_multi_slerp``      – N-way iterative SLERP for embeddings (ported reference multi_slerp)
  - ``_cosine_weighted_blend`` / ``_cosine_adjusted_weights`` – cosine-similarity
    reweighting, shared by the prompt and seed paths
  - ``_apply_prompt_blending`` dispatch for "cosine_weighted" and N>2 "slerp" paths
  - ``_apply_seed_blending`` dispatch for all three methods, incl. the seed-specific
    ``_multi_slerp_noise`` (N-way slerp for noise) and ``_linear_blend_noise`` (shared by
    "average" and "cosine_weighted")
  - ``_last_prompt_interpolation_method`` / ``_last_seed_interpolation_method`` are
    sticky: recorded on every explicit call, and preserved (not clobbered) across
    method-omitted calls to ``update_stream_params`` and the six index-level methods

All tests run on CPU with float32 so no GPU is required.
"""

import types

import torch

from streamdiffusion.stream_parameter_updater import StreamParameterUpdater

# ---------------------------------------------------------------------------
# Minimal fake stream that satisfies the fields accessed during __init__ and
# _apply_prompt_blending without touching the real pipeline.
# ---------------------------------------------------------------------------


def _fake_stream():
    """Return a minimal namespace that looks like a StreamDiffusion instance."""
    stream = types.SimpleNamespace()
    stream.device = torch.device("cpu")
    stream.dtype = torch.float32
    stream.batch_size = 1
    stream.cfg_type = "none"
    stream.guidance_scale = 1.0
    stream.prompt_embeds = None
    stream.negative_prompt_embeds = None
    # Attributes accessed by OrchestratorUser.attach_orchestrator
    stream._preprocessing_orchestrator = None
    stream.embedding_hooks = []
    # Accessed unconditionally near the end of update_stream_params, regardless of
    # which optional params are passed.
    stream.kvo_cache = None
    stream.use_feature_injection = False
    # Seed-blending attributes: latent shape for _cache_seed_noise/add_seed/
    # update_seed_at_index, and the init_noise bookkeeping _apply_seed_blending touches.
    stream.latent_height = 8
    stream.latent_width = 8
    stream.generator = None
    stream.current_seed = 0
    stream.init_noise = None
    stream._init_noise_rotated = None
    return stream


def _make_updater() -> StreamParameterUpdater:
    """Construct a StreamParameterUpdater with a fake stream, bypassing __init__ side-effects."""
    stream = _fake_stream()

    # Patch OrchestratorUser.attach_orchestrator to be a no-op so we don't need
    # a real PreprocessingOrchestrator.
    orig_attach = StreamParameterUpdater.attach_orchestrator

    def _noop_attach(self, s):
        self._preprocessing_orchestrator = None

    StreamParameterUpdater.attach_orchestrator = _noop_attach
    try:
        updater = StreamParameterUpdater(stream)
    finally:
        StreamParameterUpdater.attach_orchestrator = orig_attach

    updater._embedding_orchestrator = None
    return updater


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_embed(shape=(1, 4, 8), seed=0) -> torch.Tensor:
    """Reproducible random embedding on CPU/float32."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _rand_noise(shape=(1, 4, 8, 8), seed=0) -> torch.Tensor:
    """Reproducible random seed-noise tensor on CPU/float32 (matches the
    (batch, 4, latent_h, latent_w) shape _cache_seed_noise generates)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _orthonormal_pair(dim=256):
    """Two exactly-orthogonal, exactly-unit-norm flat vectors, reshaped to the
    default seed-noise shape (1, 4, 8, 8) (dim=256). Used where the expected cosine
    similarity needs to be analytically known rather than statistical."""
    assert dim % 2 == 0
    half = dim // 2
    e1 = torch.zeros(dim)
    e1[:half] = 1.0 / (half**0.5)
    e2 = torch.zeros(dim)
    e2[half:] = 1.0 / (half**0.5)
    return e1.reshape(1, 4, 8, 8), e2.reshape(1, 4, 8, 8)


# ---------------------------------------------------------------------------
# _multi_slerp tests
# ---------------------------------------------------------------------------


class TestMultiSlerp:
    def setup_method(self):
        self.upd = _make_updater()

    def test_single_embedding_returns_scaled(self):
        e = _rand_embed(seed=1)
        result = self.upd._multi_slerp([e], [1.0])
        assert result.shape == e.shape
        # scale_factor = max(1, 1.0) = 1 → output identical to input
        assert torch.allclose(result, e)

    def test_single_embedding_weight_gt1_scales(self):
        e = _rand_embed(seed=2)
        result = self.upd._multi_slerp([e], [2.5])
        assert torch.allclose(result, e * 2.5, atol=1e-5)

    def test_two_way_matches_direct_slerp(self):
        """With two embeddings, multi_slerp result must equal _slerp(e1, e2, t)."""
        e1 = _rand_embed(seed=3)
        e2 = _rand_embed(seed=4)
        w1, w2 = 0.7, 0.3
        result_multi = self.upd._multi_slerp([e1, e2], [w1, w2])
        # _multi_slerp normalises first: scaled_w = [0.7, 0.3]; sorted desc → [0.7, 0.3]
        # t = 0.3 / (0.7 + 0.3) = 0.3
        t_expected = w2 / (w1 + w2)
        result_direct = self.upd._slerp(e1, e2, t_expected)
        # scale_factor = max(1, 1.0) = 1 → no additional scaling
        assert torch.allclose(result_multi, result_direct, atol=1e-5)

    def test_three_way_preserves_shape(self):
        es = [_rand_embed(seed=i) for i in range(3)]
        result = self.upd._multi_slerp(es, [0.5, 0.3, 0.2])
        assert result.shape == es[0].shape

    def test_zero_weight_entry_skipped(self):
        """A zero-weight prompt should have no effect."""
        e1 = _rand_embed(seed=5)
        e2 = _rand_embed(seed=6)
        e_zero = _rand_embed(seed=99)
        # With zero weight the third embedding should be entirely ignored
        result_with = self.upd._multi_slerp([e1, e2, e_zero], [0.6, 0.4, 0.0])
        result_without = self.upd._multi_slerp([e1, e2], [0.6, 0.4])
        assert torch.allclose(result_with, result_without, atol=1e-5)

    def test_weights_sum_gt1_scales_magnitude(self):
        """When sum(weights) > 1 the result magnitude is scaled accordingly."""
        e = _rand_embed(seed=7)
        # Single embedding, weight 3.0 → output = e * 3.0
        result = self.upd._multi_slerp([e], [3.0])
        assert torch.allclose(result, e * 3.0, atol=1e-5)

    def test_dtype_preserved(self):
        e1 = _rand_embed(seed=8)
        e2 = _rand_embed(seed=9)
        result = self.upd._multi_slerp([e1, e2], [0.5, 0.5])
        assert result.dtype == e1.dtype


# ---------------------------------------------------------------------------
# _cosine_weighted_blend tests
# ---------------------------------------------------------------------------


class TestCosineWeightedBlend:
    def setup_method(self):
        self.upd = _make_updater()

    def test_single_embedding_passthrough(self):
        e = _rand_embed(seed=10)
        result = self.upd._cosine_weighted_blend([e], [1.0])
        assert torch.allclose(result, e, atol=1e-5)

    def test_identical_direction_matches_multi_slerp(self):
        """When all embeddings point in the same direction, cos-sims are all 1 → same as multi_slerp."""
        base = _rand_embed(seed=11)
        # Scale copies of the same embedding by small factors (same direction)
        e1 = base * 1.0
        e2 = base * 0.5
        weights = [0.6, 0.4]
        result_cw = self.upd._cosine_weighted_blend([e1, e2], weights)
        result_ms = self.upd._multi_slerp([e1, e2], weights)
        assert torch.allclose(result_cw, result_ms, atol=1e-4)

    def test_outlier_de_emphasised(self):
        """An embedding pointing in the opposite direction to both others should be
        de-weighted, pulling the output AWAY from it compared to plain multi_slerp."""
        # Two aligned embeddings and one in the opposite direction
        e_main = _rand_embed(seed=12)
        e_aligned = _rand_embed(seed=12) * 0.9  # almost identical direction
        e_outlier = -e_main.clone()  # exact opposite
        weights = [0.4, 0.4, 0.2]

        cw_result = self.upd._cosine_weighted_blend([e_main, e_aligned, e_outlier], weights)
        ms_result = self.upd._multi_slerp([e_main, e_aligned, e_outlier], weights)

        # cosine_weighted should differ from plain multi_slerp when there's an outlier
        assert not torch.allclose(cw_result, ms_result, atol=1e-4), (
            "cosine_weighted_blend should differ from multi_slerp when an outlier is present"
        )

    def test_shape_and_dtype_preserved(self):
        es = [_rand_embed(seed=i) for i in range(3)]
        result = self.upd._cosine_weighted_blend(es, [0.5, 0.3, 0.2])
        assert result.shape == es[0].shape
        assert result.dtype == es[0].dtype


# ---------------------------------------------------------------------------
# _apply_prompt_blending dispatch + _last_prompt_interpolation_method
# ---------------------------------------------------------------------------


class TestApplyPromptBlendingDispatch:
    """Patch the actual blend helpers to just record that they were called, and verify
    the dispatch logic chooses the right one."""

    def setup_method(self):
        self.upd = _make_updater()
        # Pre-populate a two-embedding cache so _apply_prompt_blending has data.
        e1 = _rand_embed(seed=20)
        e2 = _rand_embed(seed=21)
        e3 = _rand_embed(seed=22)
        self.upd._prompt_cache = {
            "cat": {"embed": e1},
            "dog": {"embed": e2},
            "bird": {"embed": e3},
        }
        self.upd._current_prompt_list = [("cat", 0.5), ("dog", 0.3), ("bird", 0.2)]
        self.upd._current_negative_prompt = ""

    def test_slerp_n_gt_2_calls_multi_slerp(self):
        called = []
        orig = self.upd._multi_slerp

        def spy(*args, **kwargs):
            called.append("multi_slerp")
            return orig(*args, **kwargs)

        self.upd._multi_slerp = spy
        self.upd._apply_prompt_blending("slerp")
        assert "multi_slerp" in called, "slerp with N>2 should delegate to _multi_slerp"

    def test_cosine_weighted_calls_cosine_weighted_blend(self):
        called = []
        orig = self.upd._cosine_weighted_blend

        def spy(*args, **kwargs):
            called.append("cosine_weighted_blend")
            return orig(*args, **kwargs)

        self.upd._cosine_weighted_blend = spy
        self.upd._apply_prompt_blending("cosine_weighted")
        assert "cosine_weighted_blend" in called, "cosine_weighted method should delegate to _cosine_weighted_blend"

    def test_last_method_recorded_slerp(self):
        self.upd._apply_prompt_blending("slerp")
        assert self.upd._last_prompt_interpolation_method == "slerp"

    def test_last_method_recorded_cosine_weighted(self):
        self.upd._apply_prompt_blending("cosine_weighted")
        assert self.upd._last_prompt_interpolation_method == "cosine_weighted"

    def test_last_method_recorded_average(self):
        self.upd._apply_prompt_blending("average")
        assert self.upd._last_prompt_interpolation_method == "average"

    def test_slerp_2_way_uses_slerp_not_multi_slerp(self):
        """With exactly 2 embeddings, 'slerp' must NOT call _multi_slerp."""
        self.upd._current_prompt_list = [("cat", 0.6), ("dog", 0.4)]
        multi_called = []
        slerp_called = []
        orig_multi = self.upd._multi_slerp
        orig_slerp = self.upd._slerp

        def spy_multi(*a, **kw):
            multi_called.append(True)
            return orig_multi(*a, **kw)

        def spy_slerp(*a, **kw):
            slerp_called.append(True)
            return orig_slerp(*a, **kw)

        self.upd._multi_slerp = spy_multi
        self.upd._slerp = spy_slerp
        self.upd._apply_prompt_blending("slerp")
        assert not multi_called, "2-way slerp should use _slerp directly, not _multi_slerp"
        assert slerp_called, "2-way slerp should call _slerp"

    def test_last_prompt_interpolation_method_default(self):
        """Attribute must exist from __init__ with default 'slerp'."""
        fresh = _make_updater()
        assert hasattr(fresh, "_last_prompt_interpolation_method")
        assert fresh._last_prompt_interpolation_method == "slerp"

    def test_unknown_method_falls_back_to_average(self):
        """An unrecognised method string must produce the same output as 'average'."""
        # Capture the average result first on a fresh updater sharing the same embeds.
        upd_average = _make_updater()
        upd_average._prompt_cache = dict(self.upd._prompt_cache)
        upd_average._current_prompt_list = list(self.upd._current_prompt_list)
        upd_average._current_negative_prompt = ""
        upd_average._apply_prompt_blending("average")
        average_embed = upd_average.stream.prompt_embeds.clone()

        # Now run the typo'd string on our main updater.
        self.upd._apply_prompt_blending("cosine_weignted")
        unknown_embed = self.upd.stream.prompt_embeds

        assert torch.allclose(unknown_embed, average_embed, atol=1e-5), (
            "Unknown method should fall back to average interpolation"
        )

    def test_unknown_method_warns_once(self, caplog):
        """Exactly one warning per unique unknown string; 'average' never warns."""
        import logging

        with caplog.at_level(logging.WARNING, logger="streamdiffusion.stream_parameter_updater"):
            # Two calls with the same bad string → only one warning record.
            self.upd._apply_prompt_blending("cosine_weignted")
            self.upd._apply_prompt_blending("cosine_weignted")

        unknown_warnings = [r for r in caplog.records if "cosine_weignted" in r.message]
        assert len(unknown_warnings) == 1, (
            f"Expected exactly 1 warning for repeated unknown method, got {len(unknown_warnings)}"
        )

        # An 'average' call must never produce a warning.
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="streamdiffusion.stream_parameter_updater"):
            self.upd._apply_prompt_blending("average")
        assert not caplog.records, "No warning expected for the 'average' method"


# ---------------------------------------------------------------------------
# Regression: all-zero (degenerate-sum) weights used to reach an unguarded
# weights / weights.sum() in _normalize_weights, producing a 0/0 -> NaN
# embedding that then latches permanently into cross-frame pipeline buffers
# (x_t_latent_buffer, stock_noise) with no recovery -- the "Normpweights off +
# both prompts at 0 -> permanently black" bug. Repro: TD's own all-zero guard
# (StreamDiffusionExt.py Promptblock) only fires when its Normpweights toggle
# is on, so toggling it off forwards literal 0.0 weights straight to the
# backend, which had no guard of its own.
# ---------------------------------------------------------------------------


class TestNormalizeWeightsDegenerateSum:
    def setup_method(self):
        self.upd = _make_updater()

    def test_all_zero_normalize_true_returns_uniform(self):
        out = self.upd._normalize_weights([0.0, 0.0], normalize=True)
        assert torch.isfinite(out).all(), "degenerate sum must not divide by zero into NaN"
        assert torch.allclose(out, torch.tensor([0.5, 0.5]))

    def test_all_zero_normalize_false_returns_uniform(self):
        """The guard applies regardless of `normalize` -- an all-zero weight list has
        no useful interpretation either way, and normalize=False would otherwise let
        a literal-zeros embedding (average) disagree with slerp/cosine's fallback."""
        out = self.upd._normalize_weights([0.0, 0.0, 0.0], normalize=False)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3]))

    def test_near_zero_sum_also_treated_as_degenerate(self):
        out = self.upd._normalize_weights([1e-10, -1e-10], normalize=True)
        assert torch.isfinite(out).all()

    def test_nonzero_weights_unaffected_normalize_true(self):
        out = self.upd._normalize_weights([0.7, 0.3], normalize=True)
        assert torch.allclose(out, torch.tensor([0.7, 0.3]), atol=1e-6)

    def test_nonzero_weights_unaffected_normalize_false(self):
        out = self.upd._normalize_weights([0.7, 0.3], normalize=False)
        assert torch.allclose(out, torch.tensor([0.7, 0.3]), atol=1e-6)

    def test_warns_once(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="streamdiffusion.stream_parameter_updater"):
            self.upd._normalize_weights([0.0, 0.0], normalize=True)
            self.upd._normalize_weights([0.0, 0.0], normalize=True)
        degenerate_warnings = [
            r for r in caplog.records if "degenerate" not in r.message and "all weights are" in r.message
        ]
        assert len(degenerate_warnings) == 1, "expected exactly one warning across repeated degenerate calls"


class TestApplyPromptBlendingZeroWeights:
    """Same regression, exercised through the real blend entry point
    (_apply_prompt_blending) for all three interpolation methods and both
    normalize settings -- this is where the NaN was actually born and then
    unconditionally assigned to stream.prompt_embeds (:831)."""

    def setup_method(self):
        self.upd = _make_updater()
        e1 = _rand_embed(seed=200)
        e2 = _rand_embed(seed=201)
        self.upd._prompt_cache = {"cat": {"embed": e1}, "dog": {"embed": e2}}
        self.upd._current_negative_prompt = ""

    def _assert_finite_for(self, method: str, normalize: bool):
        self.upd._current_prompt_list = [("cat", 0.0), ("dog", 0.0)]
        self.upd.normalize_prompt_weights = normalize
        self.upd._apply_prompt_blending(method)
        embeds = self.upd.stream.prompt_embeds
        assert embeds is not None
        assert torch.isfinite(embeds).all(), (
            f"method={method} normalize={normalize}: all-zero prompt weights produced a non-finite embedding"
        )

    def test_slerp_normalize_true(self):
        self._assert_finite_for("slerp", True)

    def test_slerp_normalize_false(self):
        self._assert_finite_for("slerp", False)

    def test_average_normalize_true(self):
        self._assert_finite_for("average", True)

    def test_average_normalize_false(self):
        self._assert_finite_for("average", False)

    def test_cosine_weighted_normalize_true(self):
        self._assert_finite_for("cosine_weighted", True)

    def test_cosine_weighted_normalize_false(self):
        self._assert_finite_for("cosine_weighted", False)


class TestApplySeedBlendingZeroWeights:
    """Seed path shares _normalize_weights (:1022, :1050) -- a zero-sum seed_list
    would poison init_noise, the pipeline's only clean recovery source for guard 1
    (see plan Part 2)."""

    def setup_method(self):
        self.upd = _make_updater()
        n1 = _rand_noise(seed=210)
        n2 = _rand_noise(seed=211)
        self.upd._seed_cache = {0: {"noise": n1, "seed": 210}, 1: {"noise": n2, "seed": 211}}

    def _assert_finite_for(self, method: str, normalize: bool):
        self.upd._current_seed_list = [(210, 0.0), (211, 0.0)]
        self.upd.normalize_seed_weights = normalize
        self.upd._apply_seed_blending(method)
        noise = self.upd.stream.init_noise
        assert noise is not None
        assert torch.isfinite(noise).all(), (
            f"method={method} normalize={normalize}: all-zero seed weights produced non-finite init_noise"
        )

    def test_slerp_normalize_true(self):
        self._assert_finite_for("slerp", True)

    def test_average_normalize_true(self):
        self._assert_finite_for("average", True)

    def test_cosine_weighted_normalize_true(self):
        self._assert_finite_for("cosine_weighted", True)

    def test_average_normalize_false(self):
        self._assert_finite_for("average", False)


# ---------------------------------------------------------------------------
# _cosine_adjusted_weights — shared reweighting helper (used by both the prompt
# path via _cosine_weighted_blend and the seed path directly)
# ---------------------------------------------------------------------------


class TestCosineAdjustedWeights:
    """Test the shared reweighting math directly with exactly-orthogonal,
    exactly-unit-norm vectors so the expected cosine similarities (and therefore the
    expected adjusted weights) are analytically known rather than statistical."""

    def setup_method(self):
        self.upd = _make_updater()

    def test_equal_weights_orthogonal_vectors_stay_equal(self):
        e1, e2 = _orthonormal_pair()
        adjusted = self.upd._cosine_adjusted_weights([e1, e2], [0.5, 0.5])
        assert abs(adjusted[0] - adjusted[1]) < 1e-4
        assert abs(sum(adjusted) - 1.0) < 1e-4, "total weight mass must be preserved"

    def test_unequal_weights_orthogonal_vectors_sharpen_toward_dominant(self):
        """For orthogonal (uncorrelated) directions, cosine_weighted sharpens the
        weight distribution toward the already-dominant entry (adj ~ w_i^2,
        renormalised): its *share* of the total should increase relative to the raw
        input weights."""
        e1, e2 = _orthonormal_pair()
        raw = [0.7, 0.3]
        adjusted = self.upd._cosine_adjusted_weights([e1, e2], raw)
        assert abs(sum(adjusted) - sum(raw)) < 1e-3, "total weight mass must be preserved"
        assert (adjusted[0] / sum(adjusted)) > (raw[0] / sum(raw)), (
            "the dominant entry's share should increase, not just its absolute weight"
        )

    def test_single_tensor_weight_preserved(self):
        e1, _ = _orthonormal_pair()
        adjusted = self.upd._cosine_adjusted_weights([e1], [2.0])
        assert abs(adjusted[0] - 2.0) < 1e-4


# ---------------------------------------------------------------------------
# _multi_slerp_noise — N-way spherical fold for seed noise (no magnitude rescale,
# unlike _multi_slerp for embeddings)
# ---------------------------------------------------------------------------


class TestMultiSlerpNoise:
    def setup_method(self):
        self.upd = _make_updater()

    def test_single_tensor_passthrough(self):
        n = _rand_noise(seed=70)
        result = self.upd._multi_slerp_noise([n], [1.0])
        assert torch.allclose(result, n)

    def test_two_way_matches_slerp_noise(self):
        """With exactly two tensors, _multi_slerp_noise must reduce to _slerp_noise
        at the same fold ratio (mirrors _multi_slerp's 2-way parity test)."""
        n1 = _rand_noise(seed=71)
        n2 = _rand_noise(seed=72)
        w1, w2 = 0.7, 0.3
        result_multi = self.upd._multi_slerp_noise([n1, n2], [w1, w2])
        t_expected = w2 / (w1 + w2)
        result_direct = self.upd._slerp_noise(n1, n2, t_expected)
        assert torch.allclose(result_multi, result_direct, atol=1e-5)

    def test_three_way_preserves_shape_and_norm(self):
        """Independent Gaussian noise tensors are near-orthogonal in high dimension,
        so each pairwise fold should be close to norm-preserving -- unlike
        _multi_slerp for embeddings, there is no max(1, sum(weights)) rescale here."""
        g = torch.Generator()
        g.manual_seed(80)
        shape = (1, 4, 64, 64)  # large enough for near-orthogonality to actually hold
        ns = [torch.randn(*shape, generator=g) for _ in range(3)]
        weights = [0.5, 0.3, 0.2]
        result = self.upd._multi_slerp_noise(ns, weights)
        assert result.shape == shape
        avg_norm = sum(n.norm().item() for n in ns) / len(ns)
        assert abs(result.norm().item() - avg_norm) / avg_norm < 0.1

    def test_zero_weight_entry_skipped(self):
        n1 = _rand_noise(seed=81)
        n2 = _rand_noise(seed=82)
        n_zero = _rand_noise(seed=999)
        result_with = self.upd._multi_slerp_noise([n1, n2, n_zero], [0.6, 0.4, 0.0])
        result_without = self.upd._multi_slerp_noise([n1, n2], [0.6, 0.4])
        assert torch.allclose(result_with, result_without, atol=1e-5)


# ---------------------------------------------------------------------------
# Cold-guard regression: _slerp_noise's sin_theta divisor goes to zero not only
# at theta~=0 (parallel, already handled) but also at theta~=pi (antiparallel).
# dot_product is explicitly clamped to exactly -1.0 (:1176), so theta==pi is
# reachable -- e.g. a seed noise tensor reused with a sign flip upstream -- and
# the pre-fix code divided by that zero into a NaN that would poison init_noise,
# the pipeline's only clean recovery source for guard 1.
# ---------------------------------------------------------------------------


class TestSlerpNoiseAntiparallel:
    def setup_method(self):
        self.upd = _make_updater()

    def test_exactly_antiparallel_vectors_stay_finite(self):
        n1 = _rand_noise(seed=95)
        n2 = -n1  # theta == pi exactly
        result = self.upd._slerp_noise(n1, n2, 0.5)
        assert torch.isfinite(result).all(), "antiparallel noise must not divide-by-zero into NaN"

    def test_exactly_antiparallel_vectors_match_linear_fallback(self):
        """At theta==pi the fix takes the same linear-fallback branch as theta==0;
        pin the expected formula so this doesn't silently regress to something
        merely finite-but-wrong."""
        n1 = _rand_noise(seed=96)
        n2 = -n1
        t = 0.3
        result = self.upd._slerp_noise(n1, n2, t)
        expected = (1 - t) * n1 + t * n2
        assert torch.allclose(result, expected, atol=1e-5)

    def test_nearly_antiparallel_vectors_stay_finite(self):
        """theta very close to (but not exactly) pi -- sin(theta) is a tiny nonzero
        denominator pre-fix, which blows up the result rather than NaN-ing it
        outright; must still land in the linear-fallback branch."""
        n1 = _rand_noise(seed=97)
        # Perturb slightly so dot_product clamps to just inside -1.0, not exactly it.
        n2 = -n1 + 1e-7 * _rand_noise(seed=98)
        result = self.upd._slerp_noise(n1, n2, 0.5)
        assert torch.isfinite(result).all()
        assert result.norm().item() < 1e6, "near-antiparallel slerp should not blow up"


# ---------------------------------------------------------------------------
# _apply_seed_blending dispatch + all three seed methods + _last_seed_interpolation_method
# ---------------------------------------------------------------------------


class TestApplySeedBlending:
    """_apply_seed_blending, _slerp_noise, and the cosine_weighted/N-way-slerp seed
    helpers had zero test coverage before this change -- this is new ground."""

    def setup_method(self):
        self.upd = _make_updater()
        n1 = _rand_noise(seed=30)
        n2 = _rand_noise(seed=31)
        n3 = _rand_noise(seed=32)
        self.upd._seed_cache = {
            0: {"noise": n1, "seed": 30},
            1: {"noise": n2, "seed": 31},
            2: {"noise": n3, "seed": 32},
        }
        self.upd._current_seed_list = [(30, 0.5), (31, 0.3), (32, 0.2)]

    def test_slerp_2_way_uses_slerp_noise_not_multi(self):
        """With exactly 2 seeds, 'slerp' must NOT call _multi_slerp_noise."""
        self.upd._current_seed_list = [(30, 0.6), (31, 0.4)]
        multi_called = []
        slerp_called = []
        orig_multi = self.upd._multi_slerp_noise
        orig_slerp = self.upd._slerp_noise

        def spy_multi(*a, **kw):
            multi_called.append(True)
            return orig_multi(*a, **kw)

        def spy_slerp(*a, **kw):
            slerp_called.append(True)
            return orig_slerp(*a, **kw)

        self.upd._multi_slerp_noise = spy_multi
        self.upd._slerp_noise = spy_slerp
        self.upd._apply_seed_blending("slerp")
        assert not multi_called, "2-way slerp should use _slerp_noise directly, not _multi_slerp_noise"
        assert slerp_called, "2-way slerp should call _slerp_noise"

    def test_slerp_n_gt_2_calls_multi_slerp_noise(self):
        """With 3+ seeds, 'slerp' must delegate to _multi_slerp_noise (previously this
        silently fell through to average -- Defect B)."""
        called = []
        orig = self.upd._multi_slerp_noise

        def spy(*a, **kw):
            called.append(True)
            return orig(*a, **kw)

        self.upd._multi_slerp_noise = spy
        self.upd._apply_seed_blending("slerp")
        assert called, "slerp with N>2 seeds should delegate to _multi_slerp_noise"

    def test_cosine_weighted_calls_linear_blend_noise(self):
        """cosine_weighted must exist for seeds at all (previously absent -- Defect B)
        and must fold through _linear_blend_noise, not _multi_slerp_noise."""
        called = []
        orig = self.upd._linear_blend_noise

        def spy(*a, **kw):
            called.append(True)
            return orig(*a, **kw)

        self.upd._linear_blend_noise = spy
        self.upd._apply_seed_blending("cosine_weighted")
        assert called, "cosine_weighted should delegate to _linear_blend_noise"

    def test_average_calls_linear_blend_noise(self):
        called = []
        orig = self.upd._linear_blend_noise

        def spy(*a, **kw):
            called.append(True)
            return orig(*a, **kw)

        self.upd._linear_blend_noise = spy
        self.upd._apply_seed_blending("average")
        assert called, "average should delegate to _linear_blend_noise"

    def test_last_seed_interpolation_method_recorded(self):
        self.upd._apply_seed_blending("cosine_weighted")
        assert self.upd._last_seed_interpolation_method == "cosine_weighted"

    def test_last_seed_interpolation_method_default(self):
        """Attribute must exist from __init__ with default 'average'."""
        fresh = _make_updater()
        assert hasattr(fresh, "_last_seed_interpolation_method")
        assert fresh._last_seed_interpolation_method == "average"

    def test_cosine_weighted_equal_weights_matches_average_for_orthogonal_noise(self):
        """With exactly-orthogonal noise tensors and equal input weights, the cosine
        reweighting is a no-op (see TestCosineAdjustedWeights), so cosine_weighted and
        average must produce an identical seed blend."""
        n1, n2 = _orthonormal_pair()
        upd_cw = _make_updater()
        upd_avg = _make_updater()
        for u in (upd_cw, upd_avg):
            u._seed_cache = {0: {"noise": n1.clone(), "seed": 50}, 1: {"noise": n2.clone(), "seed": 51}}
            u._current_seed_list = [(50, 0.5), (51, 0.5)]
        upd_cw._apply_seed_blending("cosine_weighted")
        upd_avg._apply_seed_blending("average")
        assert torch.allclose(upd_cw.stream.init_noise, upd_avg.stream.init_noise, atol=1e-4)

    def test_cosine_weighted_unequal_weights_shifts_toward_dominant_seed(self):
        """With unequal weights, cosine_weighted should push further toward the
        dominant seed's direction than plain average blending does."""
        n1, n2 = _orthonormal_pair()
        upd_cw = _make_updater()
        upd_avg = _make_updater()
        for u in (upd_cw, upd_avg):
            u._seed_cache = {0: {"noise": n1.clone(), "seed": 50}, 1: {"noise": n2.clone(), "seed": 51}}
            u._current_seed_list = [(50, 0.7), (51, 0.3)]
        upd_cw._apply_seed_blending("cosine_weighted")
        upd_avg._apply_seed_blending("average")

        # Projection onto the dominant seed's direction (n1 is a unit vector).
        proj_cw = (upd_cw.stream.init_noise.flatten() * n1.flatten()).sum().item()
        proj_avg = (upd_avg.stream.init_noise.flatten() * n1.flatten()).sum().item()
        assert proj_cw > proj_avg, "cosine_weighted should shift further toward the dominant seed than average"

    def test_average_and_cosine_weighted_preserve_unit_variance(self):
        """The whole point of the 1/sqrt(sum(w_i^2)) restoration in _linear_blend_noise:
        blended noise should stay close to unit variance, not shrink toward the
        weighted-average's under-dispersion."""
        g = torch.Generator()
        g.manual_seed(40)
        shape = (1, 4, 64, 64)
        n1 = torch.randn(*shape, generator=g)
        n2 = torch.randn(*shape, generator=g)
        n3 = torch.randn(*shape, generator=g)
        for method in ("average", "cosine_weighted"):
            upd = _make_updater()
            upd._seed_cache = {
                0: {"noise": n1.clone(), "seed": 60},
                1: {"noise": n2.clone(), "seed": 61},
                2: {"noise": n3.clone(), "seed": 62},
            }
            upd._current_seed_list = [(60, 0.5), (61, 0.3), (62, 0.2)]
            upd._apply_seed_blending(method)
            std = upd.stream.init_noise.std().item()
            assert abs(std - 1.0) < 0.15, f"{method}: blended std={std:.3f}, expected ~1.0"

    def test_unknown_method_falls_back_to_average(self):
        """An unrecognised method string must produce the same output as 'average'
        (the seed path previously had no such fallback documented/tested at all)."""
        n1 = _rand_noise(seed=90)
        n2 = _rand_noise(seed=91)

        upd_average = _make_updater()
        upd_average._seed_cache = {0: {"noise": n1.clone(), "seed": 90}, 1: {"noise": n2.clone(), "seed": 91}}
        upd_average._current_seed_list = [(90, 0.6), (91, 0.4)]
        upd_average._apply_seed_blending("average")
        average_noise = upd_average.stream.init_noise.clone()

        upd = _make_updater()
        upd._seed_cache = {0: {"noise": n1.clone(), "seed": 90}, 1: {"noise": n2.clone(), "seed": 91}}
        upd._current_seed_list = [(90, 0.6), (91, 0.4)]
        upd._apply_seed_blending("cosine_weignted")

        assert torch.allclose(upd.stream.init_noise, average_noise, atol=1e-5), (
            "Unknown method should fall back to average interpolation"
        )

    def test_unknown_method_warns_once(self, caplog):
        """Exactly one warning per unique unknown string; 'average' never warns.
        Shares the same warn-once set as the prompt path."""
        import logging

        with caplog.at_level(logging.WARNING, logger="streamdiffusion.stream_parameter_updater"):
            self.upd._apply_seed_blending("cosine_weignted")
            self.upd._apply_seed_blending("cosine_weignted")

        unknown_warnings = [r for r in caplog.records if "cosine_weignted" in r.message]
        assert len(unknown_warnings) == 1, (
            f"Expected exactly 1 warning for repeated unknown method, got {len(unknown_warnings)}"
        )

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="streamdiffusion.stream_parameter_updater"):
            self.upd._apply_seed_blending("average")
        assert not caplog.records, "No warning expected for the 'average' method"


# ---------------------------------------------------------------------------
# Stickiness regression tests (Defect A): a method-only update must take effect
# immediately, and a subsequent list-only update must not revert to a signature
# default. Exercised through the real update_stream_params entrypoint plus a
# representative index-level method on each side (add_prompt / update_seed_at_index).
# ---------------------------------------------------------------------------


class TestStickyInterpolationMethods:
    def setup_method(self):
        self.upd = _make_updater()
        e1 = _rand_embed(seed=100)
        e2 = _rand_embed(seed=101)
        self.upd._prompt_cache = {"cat": {"embed": e1}, "dog": {"embed": e2}}
        self.upd._current_prompt_list = [("cat", 0.5), ("dog", 0.5)]
        self.upd._current_negative_prompt = ""

        n1 = _rand_noise(seed=110)
        n2 = _rand_noise(seed=111)
        self.upd._seed_cache = {0: {"noise": n1, "seed": 110}, 1: {"noise": n2, "seed": 111}}
        self.upd._current_seed_list = [(110, 0.5), (111, 0.5)]

    def test_method_only_prompt_update_reblends_immediately(self):
        """Consequence 1: pre-fix, a method-only change did nothing at all."""
        assert self.upd.stream.prompt_embeds is None  # nothing blended yet
        self.upd.update_stream_params(prompt_interpolation_method="cosine_weighted")
        assert self.upd._last_prompt_interpolation_method == "cosine_weighted"
        assert self.upd.stream.prompt_embeds is not None

    def test_method_only_seed_update_reblends_immediately(self):
        assert self.upd.stream.init_noise is None
        self.upd.update_stream_params(seed_interpolation_method="cosine_weighted")
        assert self.upd._last_seed_interpolation_method == "cosine_weighted"
        assert self.upd.stream.init_noise is not None

    def test_list_only_update_preserves_sticky_prompt_method(self):
        """Consequence 2: a prompt-weight drag must not silently revert the method."""
        self.upd.update_stream_params(prompt_interpolation_method="average")
        assert self.upd._last_prompt_interpolation_method == "average"
        # Weight-drag: new prompt_list, no method specified.
        self.upd.update_stream_params(prompt_list=[("cat", 0.6), ("dog", 0.4)])
        assert self.upd._last_prompt_interpolation_method == "average"

    def test_bare_seed_list_update_preserves_sticky_seed_method(self):
        """Consequence 3: the per-frame seed randomizer sends seed_list alone; it must
        not re-default the method to 'average' every frame."""
        self.upd.update_stream_params(seed_interpolation_method="slerp")
        assert self.upd._last_seed_interpolation_method == "slerp"
        self.upd.update_stream_params(seed_list=[(110, 0.5), (111, 0.5)])
        assert self.upd._last_seed_interpolation_method == "slerp"

    def test_add_prompt_preserves_sticky_method(self):
        """Consequence 4: add_prompt's own signature default must not clobber the
        sticky value when the caller omits the method."""
        self.upd.update_stream_params(prompt_interpolation_method="cosine_weighted")

        def fake_encode_prompt(prompt, **kwargs):
            return (_rand_embed(seed=abs(hash(prompt)) % 1000),)

        self.upd.stream.pipe = types.SimpleNamespace(encode_prompt=fake_encode_prompt)
        self.upd.add_prompt("bird", weight=0.2)
        assert self.upd._last_prompt_interpolation_method == "cosine_weighted"

    def test_update_seed_at_index_preserves_sticky_method(self):
        """Same as above, for the seed-side index methods."""
        self.upd.update_stream_params(seed_interpolation_method="slerp")
        self.upd.update_seed_at_index(0, new_seed=999)
        assert self.upd._last_seed_interpolation_method == "slerp"

    def test_method_only_update_before_any_prompt_or_seed_is_a_noop(self):
        """A method-only change before any prompt/seed list exists must be recorded but
        not crash -- both _apply_* early-return on an empty list/cache."""
        fresh = _make_updater()
        fresh.update_stream_params(prompt_interpolation_method="slerp", seed_interpolation_method="cosine_weighted")
        assert fresh._last_prompt_interpolation_method == "slerp"
        assert fresh._last_seed_interpolation_method == "cosine_weighted"
        assert fresh.stream.prompt_embeds is None
        assert fresh.stream.init_noise is None


# ---------------------------------------------------------------------------
# Part 3 regression: toggling Normpweights/normalize_seed_weights alone (no
# prompt/seed list edit) used to have zero effect until the next list update,
# because update_stream_params only re-blended on prompt_list/seed_list or an
# *_interpolation_method change. This mirrors TestStickyInterpolationMethods'
# method-only re-blend tests, but for the normalize-flag-only path.
# ---------------------------------------------------------------------------


class TestNormalizeFlagOnlyReblendsImmediately:
    def setup_method(self):
        self.upd = _make_updater()
        e1 = _rand_embed(seed=300)
        e2 = _rand_embed(seed=301)
        self.upd._prompt_cache = {"cat": {"embed": e1}, "dog": {"embed": e2}}
        self.upd._current_prompt_list = [("cat", 0.5), ("dog", 0.5)]
        self.upd._current_negative_prompt = ""

        n1 = _rand_noise(seed=310)
        n2 = _rand_noise(seed=311)
        self.upd._seed_cache = {0: {"noise": n1, "seed": 310}, 1: {"noise": n2, "seed": 311}}
        self.upd._current_seed_list = [(310, 0.5), (311, 0.5)]

    def test_normalize_prompt_weights_only_reblends_immediately(self):
        assert self.upd.stream.prompt_embeds is None  # nothing blended yet
        self.upd.update_stream_params(normalize_prompt_weights=False)
        assert self.upd.normalize_prompt_weights is False
        assert self.upd.stream.prompt_embeds is not None, (
            "a normalize_prompt_weights-only change must re-blend the cached prompts "
            "immediately, not wait for the next prompt_list update"
        )

    def test_normalize_seed_weights_only_reblends_immediately(self):
        assert self.upd.stream.init_noise is None
        self.upd.update_stream_params(normalize_seed_weights=False)
        assert self.upd.normalize_seed_weights is False
        assert self.upd.stream.init_noise is not None, (
            "a normalize_seed_weights-only change must re-blend the cached seed noise "
            "immediately, not wait for the next seed_list update"
        )

    def test_normalize_prompt_weights_flag_actually_changes_output(self):
        """Not just 'it re-blended' -- confirm the flag is wired through to the
        blend math (Totalpweights-equivalent: un-normalized weights amplify)."""
        self.upd._current_prompt_list = [("cat", 2.0), ("dog", 2.0)]
        self.upd.update_stream_params(normalize_prompt_weights=True)
        normalized_embeds = self.upd.stream.prompt_embeds.clone()

        self.upd.update_stream_params(normalize_prompt_weights=False)
        unnormalized_embeds = self.upd.stream.prompt_embeds.clone()

        assert not torch.allclose(normalized_embeds, unnormalized_embeds), (
            "toggling normalize_prompt_weights with an identical prompt_list produced "
            "identical output -- the flag isn't reaching the blend"
        )
