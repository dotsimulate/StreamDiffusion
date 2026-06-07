"""Unit tests for the new prompt interpolation modes added in
stream_parameter_updater.py:

  - ``_multi_slerp``      – N-way iterative SLERP (port of reference multi_slerp)
  - ``_cosine_weighted_blend`` – genuine cosine-similarity weighting before N-way SLERP
  - ``_apply_prompt_blending`` dispatch for "cosine_weighted" and N>2 "slerp" paths
  - ``_last_prompt_interpolation_method`` attribute is recorded and carries across calls

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
    return stream


def _make_updater() -> StreamParameterUpdater:
    """Construct a StreamParameterUpdater with a fake stream, bypassing __init__ side-effects."""
    stream = _fake_stream()

    # Patch OrchestratorUser.attach_orchestrator to be a no-op so we don't need
    # a real PreprocessingOrchestrator.
    orig_attach = StreamParameterUpdater.attach_orchestrator

    def _noop_attach(self, s):  # noqa: ANN001
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
            0: {"embed": e1, "text": "cat"},
            1: {"embed": e2, "text": "dog"},
            2: {"embed": e3, "text": "bird"},
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

    def test_last_method_recorded_linear(self):
        self.upd._apply_prompt_blending("linear")
        assert self.upd._last_prompt_interpolation_method == "linear"

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
