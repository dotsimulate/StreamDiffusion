"""
Regression tests for `_resolve_fp8_ipadapter_calibration_tokens`
(fp8-round-9.1, wrapper.py).

fp8-round-9 gated the real-token calibration path on adapter flavour
(`is_faceid and not is_plus`), which was measured against a FaceID config but
left every other adapter type -- including the shipped `type: regular`
config -- silently zero-padded. A rebuild after the fix produced a
byte-identical `fp8_uncalibrated_scales: 420` census, proving the fix was
dead code for `type: regular`.

fp8-round-9.1 extracts the token resolution into a pure, importable function
that dispatches on `image_proj_model.proj`'s actual shape instead of adapter
flavour:

- `proj` is an `nn.Sequential` (FaceIDProjectionModel) -> `proj[0].in_features`
  (id_embeddings_dim), pinning the path fp8-round-9 already measured.
- `proj` is a bare `nn.Linear` (ImageProjModel, regular adapters) ->
  `proj.in_features` (clip_embeddings_dim) -- previously unreachable.
- Anything else (Plus / Resampler / FaceID-Plus-v2) -> skip, zero-pad
  fallback, unchanged from fp8-round-9.

The producer had zero test coverage before this -- the consumer
(`_reconcile_calib_to_onnx_dims`, tests/quality/test_fp8_calib_tile.py) was
well-tested, but nothing pinned the resolver that decides what token to feed
it. This is the seam fp8-round-9.1 makes testable.
"""

import numpy as np
import torch

from streamdiffusion.wrapper import _resolve_fp8_ipadapter_calibration_tokens

# ---------------------------------------------------------------------------
# Minimal stand-ins for the two vendored diffusers_ipadapter projection
# models (ip_adapter/projection_models.py::FaceIDProjectionModel,
# ip_adapter/ip_adapter.py::ImageProjModel) -- same proj-structure contract,
# without depending on the vendored package's exact __init__ signature.
# ---------------------------------------------------------------------------


class _FakeFaceIDProjectionModel(torch.nn.Module):
    """`proj` is an nn.Sequential -- mirrors FaceIDProjectionModel."""

    def __init__(self, id_embeddings_dim=512, cross_attention_dim=2048, num_tokens=4):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        self._num_tokens = num_tokens
        self._cross_attention_dim = cross_attention_dim

    def forward(self, id_embeds):
        out = self.proj(id_embeds).reshape(-1, self._num_tokens, self._cross_attention_dim)
        return self.norm(out)


class _FakeImageProjModel(torch.nn.Module):
    """`proj` is a bare nn.Linear -- mirrors ImageProjModel (regular adapters)."""

    def __init__(self, clip_embeddings_dim=1024, cross_attention_dim=2048, clip_extra_context_tokens=4):
        super().__init__()
        self.proj = torch.nn.Linear(clip_embeddings_dim, clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        self._clip_extra_context_tokens = clip_extra_context_tokens
        self._cross_attention_dim = cross_attention_dim

    def forward(self, image_embeds):
        out = self.proj(image_embeds).reshape(-1, self._clip_extra_context_tokens, self._cross_attention_dim)
        return self.norm(out)


class _FakeUnrecognizedProjModel(torch.nn.Module):
    """`proj` is neither Sequential nor Linear -- e.g. IP-Adapter Plus's
    Resampler. Call signature unmeasured -- must stay skipped."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv1d(4, 4, 1)


class _FakeIPA:
    def __init__(
        self,
        image_proj_model,
        device="cpu",
        dtype=torch.float32,
        image_embeds=None,
        num_tokens=4,
        raise_on_get_image_embeds=None,
        fail_images=None,
    ):
        self.image_proj_model = image_proj_model
        self.device = torch.device(device)
        self.dtype = dtype
        # fp8-round-9.1: stand-ins for the real ip_adapter.py contract this
        # resolver's "image" branch depends on -- get_image_embeds(images=...)
        # returns (image_prompt_embeds, negative_embeds) and has the
        # set_tokens(N) side effect the resolver must restore afterward.
        self._image_embeds = image_embeds
        self.num_tokens = num_tokens
        self._raise_on_get_image_embeds = raise_on_get_image_embeds
        # fp8-round-9.1 §10: maps an image identifier -> the exception to
        # raise whenever that identifier appears in a get_image_embeds()
        # call's `images` list. Lets tests pin
        # _encode_fp8_calibration_images' batch-fails/per-image-retry
        # behaviour precisely (one bad image among good ones), unlike
        # raise_on_get_image_embeds's unconditional "every call raises".
        self._fail_images = fail_images or {}
        self.get_image_embeds_calls = []
        self.set_tokens_calls = []

    def get_image_embeds(self, images):
        self.get_image_embeds_calls.append(list(images))
        if self._raise_on_get_image_embeds is not None:
            raise self._raise_on_get_image_embeds
        for image in images:
            if image in self._fail_images:
                raise self._fail_images[image]
        embeds = self._image_embeds
        if embeds is None:
            embeds = torch.full((len(images), 4, 2048), 7.0)
        # Mirrors ip_adapter.py's real side effect: num_tokens scales with
        # how many images were just encoded, not the per-image invariant.
        self.set_tokens(embeds.shape[0] * self.num_tokens)
        return embeds, torch.zeros_like(embeds)

    def set_tokens(self, num_tokens):
        self.set_tokens_calls.append(num_tokens)


class TestResolveFp8IpadapterCalibrationTokens:
    def test_regular_adapter_linear_proj_yields_non_zero_surrogate(self):
        """The previously-unreachable path: `type: regular` config, bare
        nn.Linear proj. Must produce a non-degenerate (1, 4, 2048) surrogate,
        not the TypeError that `proj[0]` would raise on a non-indexable
        Linear, and not a silent zero-pad skip."""
        ipa = _FakeIPA(_FakeImageProjModel(clip_embeddings_dim=1024, cross_attention_dim=2048))

        tokens, method = _resolve_fp8_ipadapter_calibration_tokens(ipa, cached_embeddings=None)

        assert method == "surrogate"
        assert tokens is not None
        assert tokens.shape == (1, 4, 2048)
        # LayerNorm's bias on a zeros input is non-degenerate -- this is the
        # whole point of the surrogate (a real, non-zero signal for FP8
        # calibration to see), not an all-zero region indistinguishable from
        # the old zero-pad fallback.
        assert np.abs(tokens).max() > 0.0

    def test_faceid_sequential_proj_still_works(self):
        """Pins the path fp8-round-9 already measured and shipped."""
        ipa = _FakeIPA(_FakeFaceIDProjectionModel(id_embeddings_dim=512, cross_attention_dim=2048))

        tokens, method = _resolve_fp8_ipadapter_calibration_tokens(ipa, cached_embeddings=None)

        assert method == "surrogate"
        assert tokens is not None
        assert tokens.shape == (1, 4, 2048)
        assert np.abs(tokens).max() > 0.0

    def test_unrecognized_proj_shape_skips_to_zero_pad(self):
        """Plus/Resampler-shaped adapters are unmeasured -- must not attempt
        a blind zeros() call against an unknown forward signature."""
        ipa = _FakeIPA(_FakeUnrecognizedProjModel())

        tokens, method = _resolve_fp8_ipadapter_calibration_tokens(ipa, cached_embeddings=None)

        assert tokens is None
        assert method == "skipped"

    def test_ipa_none_skips_to_zero_pad(self):
        tokens, method = _resolve_fp8_ipadapter_calibration_tokens(None, cached_embeddings=None)

        assert tokens is None
        assert method == "skipped"

    def test_cached_embeddings_preferred_over_surrogate(self):
        """A real cached style embedding must win even when the surrogate
        path is also available and well-formed."""
        ipa = _FakeIPA(_FakeImageProjModel())
        cached_tensor = torch.full((1, 4, 2048), 3.5)
        cached_embeddings = (cached_tensor, torch.zeros(1))

        tokens, method = _resolve_fp8_ipadapter_calibration_tokens(ipa, cached_embeddings=cached_embeddings)

        assert method == "cached"
        assert tokens is not None
        np.testing.assert_array_equal(tokens, cached_tensor.numpy())

    def test_missing_image_proj_model_skips_to_zero_pad(self):
        """An ipa object with no image_proj_model attribute at all (e.g. a
        stub/mock in another test) must degrade gracefully, not raise
        AttributeError."""
        ipa = _FakeIPA(image_proj_model=None)

        tokens, method = _resolve_fp8_ipadapter_calibration_tokens(ipa, cached_embeddings=None)

        assert tokens is None
        assert method == "skipped"

    def test_style_images_produce_image_method_tokens(self):
        """fp8-round-9.1's new highest-priority source: real
        `fp8_calibration_style_image` file(s), encoded through
        `ipa.get_image_embeds` -- the same call the runtime
        IPAdapterEmbeddingPreprocessor uses. Must report method="image" and
        return exactly what get_image_embeds produced, not a surrogate."""
        embeds = torch.full((2, 4, 2048), 3.0)
        ipa = _FakeIPA(_FakeImageProjModel(), image_embeds=embeds)
        style_images = ["fake_pil_image_0", "fake_pil_image_1"]

        tokens, method = _resolve_fp8_ipadapter_calibration_tokens(
            ipa, cached_embeddings=None, style_images=style_images
        )

        assert method == "image"
        assert tokens is not None
        np.testing.assert_array_equal(tokens, embeds.numpy())
        assert ipa.get_image_embeds_calls == [style_images]

    def test_style_images_preferred_over_cached_and_surrogate(self):
        """An operator who deliberately configures calibration images
        expects them used, not silently superseded by an incidentally
        cached runtime embedding (or the zeros-surrogate)."""
        embeds = torch.full((1, 4, 2048), 9.0)
        ipa = _FakeIPA(_FakeImageProjModel(), image_embeds=embeds)
        cached_tensor = torch.full((1, 4, 2048), 3.5)
        cached_embeddings = (cached_tensor, torch.zeros(1))

        tokens, method = _resolve_fp8_ipadapter_calibration_tokens(
            ipa, cached_embeddings=cached_embeddings, style_images=["fake_pil_image"]
        )

        assert method == "image"
        np.testing.assert_array_equal(tokens, embeds.numpy())

    def test_style_images_restores_num_tokens_after_multi_image_encode(self):
        """Defect found during fp8-round-9.1 implementation: get_image_embeds's
        set_tokens(N) side effect mutates live IPAttnProcessor.num_tokens on
        the SAME ipa object used at runtime. With N=2 calibration images this
        would leave num_tokens=8 stuck on live state meant for single-frame
        (num_tokens=4) runtime calls. The resolver must restore
        ipa.set_tokens(ipa.num_tokens) in a finally after the encode."""
        embeds = torch.full((2, 4, 2048), 1.0)
        ipa = _FakeIPA(_FakeImageProjModel(), image_embeds=embeds, num_tokens=4)

        _resolve_fp8_ipadapter_calibration_tokens(
            ipa, cached_embeddings=None, style_images=["fake_pil_image_0", "fake_pil_image_1"]
        )

        # get_image_embeds's own side effect sets it to N * num_tokens = 8,
        # then the resolver's finally restores it to the per-image
        # invariant (4) -- both calls must be observed, in that order.
        assert ipa.set_tokens_calls == [8, 4]

    def test_style_images_single_image_failure_falls_through_without_retry(self):
        """fp8-round-9.1 §10: a single-image "batch" that fails is not
        retried (nothing smaller to retry with) -- falls straight through to
        the surrogate, and the exception does not propagate. Restoration
        (the `finally`) still runs regardless. Supersedes the pre-§10
        contract where this raised RuntimeError."""
        ipa = _FakeIPA(
            _FakeImageProjModel(),
            num_tokens=4,
            raise_on_get_image_embeds=RuntimeError("encode failed"),
        )

        tokens, method = _resolve_fp8_ipadapter_calibration_tokens(
            ipa, cached_embeddings=None, style_images=["fake_pil_image"]
        )

        assert method == "surrogate"
        assert tokens is not None
        assert ipa.set_tokens_calls == [4]
        assert ipa.get_image_embeds_calls == [["fake_pil_image"]]

    def test_style_images_partial_encode_failure_keeps_survivors(self):
        """§10: a batch encode that fails because one image among several is
        bad (e.g. FaceID rejecting a face-less image) must not cost the
        other images' contribution -- retries individually and keeps
        whatever survives."""
        ipa = _FakeIPA(
            _FakeImageProjModel(),
            num_tokens=4,
            fail_images={"bad": ValueError("No face detected in image 1")},
        )
        style_images = ["good_0", "bad", "good_1"]

        tokens, method = _resolve_fp8_ipadapter_calibration_tokens(
            ipa, cached_embeddings=None, style_images=style_images
        )

        assert method == "image"
        assert tokens is not None
        assert tokens.shape[0] == 2
        # Batch attempt (all 3, raises) then per-image retries (3 singles,
        # "bad" raises too) -- restore call closes it out at the per-image
        # invariant (4) regardless of how many images were actually kept.
        assert ipa.get_image_embeds_calls == [style_images, ["good_0"], ["bad"], ["good_1"]]
        assert ipa.set_tokens_calls[-1] == 4

    def test_style_images_all_encode_failures_fall_through_to_surrogate(self):
        """§10: every configured image rejected -- no cached embedding
        available -- must fall through to the surrogate rather than raising
        or returning a degenerate all-zero result."""
        ipa = _FakeIPA(
            _FakeImageProjModel(),
            num_tokens=4,
            fail_images={
                "bad_0": ValueError("No face detected in image 0"),
                "bad_1": ValueError("No face detected in image 1"),
            },
        )

        tokens, method = _resolve_fp8_ipadapter_calibration_tokens(
            ipa, cached_embeddings=None, style_images=["bad_0", "bad_1"]
        )

        assert method == "surrogate"
        assert tokens is not None
        assert np.abs(tokens).max() > 0.0

    def test_style_images_all_encode_failures_fall_through_to_cached(self):
        """§10: every configured image rejected, but a real cached embedding
        is available -- cached must win over the surrogate, matching the
        resolver's normal (non-image) priority order."""
        ipa = _FakeIPA(
            _FakeImageProjModel(),
            num_tokens=4,
            fail_images={"bad": ValueError("No face detected in image 0")},
        )
        cached_tensor = torch.full((1, 4, 2048), 3.5)
        cached_embeddings = (cached_tensor, torch.zeros(1))

        tokens, method = _resolve_fp8_ipadapter_calibration_tokens(
            ipa, cached_embeddings=cached_embeddings, style_images=["bad"]
        )

        assert method == "cached"
        np.testing.assert_array_equal(tokens, cached_tensor.numpy())

    def test_empty_style_images_list_falls_through_to_surrogate(self):
        """An empty list (e.g. `_load_fp8_calibration_style_images` found no
        loadable files) must behave like "not configured", not attempt to
        call get_image_embeds([]) -- falls through to the surrogate path."""
        ipa = _FakeIPA(_FakeImageProjModel())

        tokens, method = _resolve_fp8_ipadapter_calibration_tokens(ipa, cached_embeddings=None, style_images=[])

        assert method == "surrogate"
        assert ipa.get_image_embeds_calls == []
