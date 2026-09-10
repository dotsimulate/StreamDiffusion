"""
Regression tests for the Custom VAE crash (v0.4.0 Discord report).

Setting the operator's `Customvae` param to a full VAE (e.g.
`stabilityai/sd-vae-ft-mse`) crashed the server at load with
`NotImplementedError: Cannot copy out of meta tensor`. Root cause: `wrapper.py`
loaded *every* `vae_id` through `AutoencoderTiny.from_pretrained(...)`, regardless
of the checkpoint's actual architecture. With zero key overlap between an
AutoencoderKL checkpoint and the AutoencoderTiny skeleton, diffusers'
`low_cpu_mem_usage=True` default left every parameter on the meta device, and the
`.to(device=...)` call raised.

These tests exercise `streamdiffusion.wrapper`'s module-level VAE-resolution
helpers directly (`_resolve_vae`, `_resolve_vae_class`, `_load_custom_vae`,
`_validate_vae_config`, `_harden_vae_attention`, `_should_skip_trt_vae`) - the test
seam Step 1 of the fix plan created, since the dispatch decision previously lived
~500 lines inside the monolithic `_load_model` with no way to test it in
isolation. CPU-only, network-free: `AutoencoderKL.load_config`/`from_pretrained`
are patched in every test, so no HF download or GPU is required.

Follows the object.__new__-free, module-level `patch.object(wrapper_module.*,
"from_pretrained"/"load_config", staticmethod(...))` pattern established in
test_load_model_network_error_reporting.py - these helpers take no wrapper
instance at all.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
import torch

from streamdiffusion import wrapper as wrapper_module
from streamdiffusion.wrapper import (
    VaeResolutionError,
    _fork_returns_kvo_tuple,
    _harden_vae_attention,
    _load_custom_vae,
    _resolve_vae,
    _resolve_vae_class,
    _should_skip_trt_vae,
    _validate_vae_config,
)


class _FakeKL(torch.nn.Module):
    """Stand-in for a loaded AutoencoderKL: has attention (attn_processors /
    set_attn_processor), matching what _harden_vae_attention probes for."""

    def __init__(self, meta: bool = False, has_attn: bool = True):
        super().__init__()
        if meta:
            self.weight = torch.nn.Parameter(torch.empty(2, 2, device="meta"))
        self._has_attn = has_attn
        self.attn_processor_calls = []

    @property
    def attn_processors(self):
        return {"mid_block.attentions.0.processor": object()} if self._has_attn else {}

    def set_attn_processor(self, processor):
        self.attn_processor_calls.append(processor)


class _FakeTiny(torch.nn.Module):
    """Stand-in for a loaded AutoencoderTiny: genuinely lacks attn_processors /
    set_attn_processor, matching the real class - an unguarded
    vae.set_attn_processor(...) call on this must raise AttributeError."""

    def __init__(self, meta: bool = False):
        super().__init__()
        if meta:
            self.weight = torch.nn.Parameter(torch.empty(2, 2, device="meta"))


_KL_CONFIG = {
    "_class_name": "AutoencoderKL",
    "latent_channels": 4,
    "block_out_channels": [128, 256, 512, 512],
    "scaling_factor": 0.18215,
}
_TINY_CONFIG = {"_class_name": "AutoencoderTiny", "latent_channels": 4, "scaling_factor": 1.0}


def _fake_load_config(responses):
    """Build a fake `AutoencoderKL.load_config` that returns/raises per subfolder,
    keyed by the `subfolder` kwarg (None for the repo root)."""

    def _fake(vae_id, subfolder=None, **kwargs):
        response = responses[subfolder]
        if isinstance(response, Exception):
            raise response
        return response

    return _fake


class TestVaeIdRoutesToDetectedClass:
    """The bug that shipped: vae_id must route to the class its own config.json
    declares, never unconditionally to AutoencoderTiny."""

    def test_full_vae_routes_to_autoencoder_kl_not_tiny(self):
        kl_calls = []
        tiny_calls = []

        def fake_kl_from_pretrained(vae_id, **kwargs):
            kl_calls.append((vae_id, kwargs))
            return _FakeKL()

        def fake_tiny_from_pretrained(vae_id, **kwargs):
            tiny_calls.append((vae_id, kwargs))
            raise AssertionError("AutoencoderTiny.from_pretrained must not be called for a full VAE")

        with (
            patch.object(
                wrapper_module.AutoencoderKL,
                "load_config",
                staticmethod(_fake_load_config({None: _KL_CONFIG})),
            ),
            patch.object(wrapper_module.AutoencoderKL, "from_pretrained", staticmethod(fake_kl_from_pretrained)),
            patch.object(wrapper_module.AutoencoderTiny, "from_pretrained", staticmethod(fake_tiny_from_pretrained)),
            patch.object(wrapper_module, "_fork_returns_kvo_tuple", lambda: False),
        ):
            vae, class_name, source = _resolve_vae(
                "stabilityai/sd-vae-ft-mse",
                use_tiny_vae=True,
                is_sdxl=False,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        assert kl_calls == [("stabilityai/sd-vae-ft-mse", {})]
        assert tiny_calls == []
        # class_name comes from type(vae).__name__ on whatever from_pretrained
        # actually returned (the fake), not a hardcoded "AutoencoderKL" literal -
        # this also guards against _resolve_vae switching to reporting the
        # *resolved* class instead of the *returned instance's* class.
        assert class_name == _FakeKL.__name__
        assert "stabilityai/sd-vae-ft-mse" in source
        assert "config.json" in source

    def test_tiny_vae_id_routes_to_autoencoder_tiny(self):
        """Today's behaviour, preserved: a vae_id whose config is AutoencoderTiny
        still routes to AutoencoderTiny."""
        kl_calls = []
        tiny_calls = []

        def fake_kl_from_pretrained(vae_id, **kwargs):
            kl_calls.append((vae_id, kwargs))
            raise AssertionError("AutoencoderKL.from_pretrained must not be called for a tiny VAE")

        def fake_tiny_from_pretrained(vae_id, **kwargs):
            tiny_calls.append((vae_id, kwargs))
            return _FakeTiny()

        with (
            patch.object(
                wrapper_module.AutoencoderKL,
                "load_config",
                staticmethod(_fake_load_config({None: _TINY_CONFIG})),
            ),
            patch.object(wrapper_module.AutoencoderKL, "from_pretrained", staticmethod(fake_kl_from_pretrained)),
            patch.object(wrapper_module.AutoencoderTiny, "from_pretrained", staticmethod(fake_tiny_from_pretrained)),
        ):
            vae, class_name, source = _resolve_vae(
                "community/some-taesd-variant",
                use_tiny_vae=True,
                is_sdxl=False,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        assert tiny_calls == [("community/some-taesd-variant", {})]
        assert kl_calls == []
        assert class_name == _FakeTiny.__name__


class TestDefaultVaeWhenIdIsNone:
    def test_no_vae_id_uses_taesd(self):
        calls = []

        def fake_from_pretrained(model_id, **kwargs):
            calls.append(model_id)
            return _FakeTiny()

        with patch.object(wrapper_module.AutoencoderTiny, "from_pretrained", staticmethod(fake_from_pretrained)):
            vae, class_name, source = _resolve_vae(
                None,
                use_tiny_vae=True,
                is_sdxl=False,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        assert calls == ["madebyollin/taesd"]
        # Unlike the vae_id-given branch (type(vae).__name__), the default path
        # hardcodes the "AutoencoderTiny" literal since it's statically known.
        assert class_name == "AutoencoderTiny"
        assert "madebyollin/taesd" in source
        assert "default" in source

    def test_no_vae_id_and_sdxl_uses_taesdxl(self):
        calls = []

        def fake_from_pretrained(model_id, **kwargs):
            calls.append(model_id)
            return _FakeTiny()

        with patch.object(wrapper_module.AutoencoderTiny, "from_pretrained", staticmethod(fake_from_pretrained)):
            _resolve_vae(
                None,
                use_tiny_vae=True,
                is_sdxl=True,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        assert calls == ["madebyollin/taesdxl"]

    def test_default_path_moves_to_device_and_dtype_unconditionally(self):
        """F2 guard: _resolve_vae must always move the VAE to device/dtype itself -
        there is no `acceleration` parameter here to gate it on, by construction."""
        to_calls = []

        class _TrackedFakeTiny(_FakeTiny):
            def to(self, *args, **kwargs):
                to_calls.append((args, kwargs))
                return self

        with patch.object(
            wrapper_module.AutoencoderTiny, "from_pretrained", staticmethod(lambda *a, **k: _TrackedFakeTiny())
        ):
            _resolve_vae(
                None,
                use_tiny_vae=True,
                is_sdxl=False,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        assert to_calls == [((), {"device": torch.device("cpu"), "dtype": torch.float32})]


class TestModelRepoSubfolderRetry:
    """vae_id pointing at a base-model repo (e.g. stabilityai/sd-turbo), which has
    no root config.json, must retry with subfolder='vae' before failing."""

    def test_root_failure_falls_back_to_vae_subfolder(self):
        def fake_kl_from_pretrained(vae_id, **kwargs):
            assert kwargs == {"subfolder": "vae"}
            return _FakeKL()

        with (
            patch.object(
                wrapper_module.AutoencoderKL,
                "load_config",
                staticmethod(_fake_load_config({None: OSError("no config.json at repo root"), "vae": _KL_CONFIG})),
            ),
            patch.object(wrapper_module.AutoencoderKL, "from_pretrained", staticmethod(fake_kl_from_pretrained)),
            patch.object(wrapper_module, "_fork_returns_kvo_tuple", lambda: False),
        ):
            vae, class_name, _source = _resolve_vae(
                "stabilityai/sd-turbo",
                use_tiny_vae=True,
                is_sdxl=False,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        assert class_name == _FakeKL.__name__

    def test_network_error_at_root_is_not_retried(self):
        """A network failure is not 'try the vae/ subfolder' - retrying would just
        repeat the same failure, so it must surface immediately as the cause."""
        import requests

        timeout_error = requests.exceptions.ReadTimeout("Read timed out")
        subfolder_calls = []

        def fake_load_config(vae_id, subfolder=None, **kwargs):
            subfolder_calls.append(subfolder)
            raise timeout_error

        with patch.object(wrapper_module.AutoencoderKL, "load_config", staticmethod(fake_load_config)):
            with pytest.raises(VaeResolutionError) as exc_info:
                _resolve_vae_class("some/repo")

        assert subfolder_calls == [None]
        assert exc_info.value.__cause__ is timeout_error


class TestMetaTensorTripwire:
    """The direct fix for the reported crash: a class/checkpoint mismatch leaves
    parameters on the meta device, and must be caught before .to(), not after."""

    def test_meta_parameters_raise_before_to(self):
        to_calls = []

        class _MetaKL(_FakeKL):
            def to(self, *args, **kwargs):
                to_calls.append((args, kwargs))
                raise AssertionError(".to() must never be reached when parameters are on the meta device")

        with (
            patch.object(
                wrapper_module.AutoencoderKL,
                "load_config",
                staticmethod(_fake_load_config({None: _KL_CONFIG})),
            ),
            patch.object(
                wrapper_module.AutoencoderKL, "from_pretrained", staticmethod(lambda *a, **k: _MetaKL(meta=True))
            ),
        ):
            with pytest.raises(VaeResolutionError) as exc_info:
                _load_custom_vae("mismatched/checkpoint", device=torch.device("cpu"), dtype=torch.float32)

        assert to_calls == []
        message = str(exc_info.value)
        assert "meta device" in message
        assert "AutoencoderKL" in message
        assert "mismatched/checkpoint" in message


class TestTrtSkippedForFullVae:
    """Step 4: a full AutoencoderKL never reaches the TensorRT VAE engine build.
    `_should_skip_trt_vae` is the single decision every one of wrapper.py's four
    TRT-VAE gate sites (missing_engines, the two compile_and_load_engine calls,
    the AutoencoderKLEngine assignment) reads - tested directly here since actually
    exercising those sites needs TensorRT/onnx/polygraphy, unavailable in this venv
    (see test_engine_path_length.py's module docstring for the same constraint).
    """

    def test_full_vae_under_tensorrt_is_skipped(self):
        assert _should_skip_trt_vae("AutoencoderKL", "tensorrt") is True

    def test_tiny_vae_under_tensorrt_is_not_skipped(self):
        assert _should_skip_trt_vae("AutoencoderTiny", "tensorrt") is False

    def test_full_vae_under_non_tensorrt_reports_not_skipped(self):
        """No TRT engine attempt exists at all outside acceleration=='tensorrt', so
        there is nothing to skip - the flag only means something inside that branch."""
        assert _should_skip_trt_vae("AutoencoderKL", "xformers") is False
        assert _should_skip_trt_vae("AutoencoderKL", "none") is False

    def test_unvalidated_env_override_does_not_enable_full_vae_trt(self, monkeypatch):
        monkeypatch.setenv("SDTD_FULL_VAE_TRT", "1")
        assert _should_skip_trt_vae("AutoencoderKL", "tensorrt") is True

    def test_env_override_requires_exact_value(self, monkeypatch):
        monkeypatch.setenv("SDTD_FULL_VAE_TRT", "true")
        assert _should_skip_trt_vae("AutoencoderKL", "tensorrt") is True


class TestUnreadableOrAbsentClassName:
    def test_config_unreadable_at_both_locations_raises_with_cause(self):
        root_error = OSError("stabilityai/sd-vae-ft-mse-nope is not a local folder and is not a valid repo id")
        kl_calls = []
        tiny_calls = []

        def fake_load_config(vae_id, subfolder=None, **kwargs):
            raise root_error

        with (
            patch.object(wrapper_module.AutoencoderKL, "load_config", staticmethod(fake_load_config)),
            patch.object(
                wrapper_module.AutoencoderKL,
                "from_pretrained",
                staticmethod(lambda *a, **k: kl_calls.append(a) or _FakeKL()),
            ),
            patch.object(
                wrapper_module.AutoencoderTiny,
                "from_pretrained",
                staticmethod(lambda *a, **k: tiny_calls.append(a) or _FakeTiny()),
            ),
        ):
            with pytest.raises(VaeResolutionError) as exc_info:
                _resolve_vae(
                    "not/a-real-vae",
                    use_tiny_vae=True,
                    is_sdxl=False,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )

        assert kl_calls == [], "no silent fallback: from_pretrained must not run on an unreadable config"
        assert tiny_calls == [], "no silent TAESD fallback for an unreadable config"
        assert exc_info.value.__cause__ is root_error
        assert "not/a-real-vae" in str(exc_info.value)

    def test_readable_config_without_class_name_raises_without_guessing(self):
        with patch.object(
            wrapper_module.AutoencoderKL,
            "load_config",
            staticmethod(_fake_load_config({None: {"latent_channels": 4}})),  # no _class_name key
        ):
            with pytest.raises(VaeResolutionError) as exc_info:
                _resolve_vae_class("weird/repo")

        assert "_class_name" in str(exc_info.value)


class TestUnsupportedClassName:
    """6a: a valid, readable config naming an architecture this build does not
    implement must name that architecture in the error, not crash generically."""

    def test_unsupported_architecture_names_the_class(self):
        with patch.object(
            wrapper_module.AutoencoderKL,
            "load_config",
            staticmethod(_fake_load_config({None: {"_class_name": "AutoencoderKLTemporalDecoder"}})),
        ):
            with pytest.raises(VaeResolutionError) as exc_info:
                _resolve_vae_class("stabilityai/stable-video-diffusion-img2vid")

        assert "AutoencoderKLTemporalDecoder" in str(exc_info.value)


class TestLatentChannelValidation:
    """6b: a 16-channel VAE (SD3/Flux-family) must be rejected up front - this
    build's TensorRT VAE models and pipeline hardcode 4 latent channels."""

    def test_sixteen_channel_vae_is_rejected(self):
        cfg = {**_KL_CONFIG, "latent_channels": 16}
        with pytest.raises(VaeResolutionError) as exc_info:
            _validate_vae_config("black-forest-labs/FLUX.1-dev", wrapper_module.AutoencoderKL, cfg, torch.float32)

        message = str(exc_info.value)
        assert "latent_channels" in message
        assert "16" in message


class TestSpatialScaleValidation:
    def test_non_8x_block_out_channels_is_rejected(self):
        cfg = {**_KL_CONFIG, "block_out_channels": [128, 256, 512]}  # implies 4x, not 8x
        with pytest.raises(VaeResolutionError) as exc_info:
            _validate_vae_config("some/4x-vae", wrapper_module.AutoencoderKL, cfg, torch.float32)

        assert "4x" in str(exc_info.value) or "8x" in str(exc_info.value)

    def test_tiny_vae_is_not_subject_to_the_spatial_scale_check(self):
        """AutoencoderTiny configs have no block_out_channels - must not raise."""
        cfg = {"_class_name": "AutoencoderTiny", "latent_channels": 4}
        _validate_vae_config("madebyollin/taesd", wrapper_module.AutoencoderTiny, cfg, torch.float32)


class TestForceUpcastFp16Warning:
    def test_force_upcast_with_fp16_warns_but_does_not_raise(self, caplog):
        cfg = {**_KL_CONFIG, "force_upcast": True}
        with caplog.at_level(logging.WARNING, logger="streamdiffusion.wrapper"):
            _validate_vae_config("stabilityai/sd-vae-ft-mse", wrapper_module.AutoencoderKL, cfg, torch.float16)

        assert any("force_upcast" in record.getMessage() for record in caplog.records)

    def test_force_upcast_with_fp32_does_not_warn(self, caplog):
        cfg = {**_KL_CONFIG, "force_upcast": True}
        with caplog.at_level(logging.WARNING, logger="streamdiffusion.wrapper"):
            _validate_vae_config("stabilityai/sd-vae-ft-mse", wrapper_module.AutoencoderKL, cfg, torch.float32)

        assert not any("force_upcast" in record.getMessage() for record in caplog.records)


class TestHardenVaeAttentionGating:
    """6c: the Step 3 fork workaround must self-disarm when the installed
    diffusers no longer matches the kvo_cache-returning fork, and must never be
    applied to AutoencoderTiny (which lacks set_attn_processor entirely)."""

    def test_probe_false_is_a_true_no_op(self):
        """With the probe disarmed, nothing about the passed object is even
        inspected - an object with neither attn_processors nor set_attn_processor
        must not raise."""
        with patch.object(wrapper_module, "_fork_returns_kvo_tuple", lambda: False):
            _harden_vae_attention(object())  # must not raise

            kl = _FakeKL()
            _harden_vae_attention(kl)
            assert kl.attn_processor_calls == []

    def test_probe_true_swaps_processor_on_full_kl(self):
        with patch.object(wrapper_module, "_fork_returns_kvo_tuple", lambda: True):
            kl = _FakeKL()
            _harden_vae_attention(kl)
            assert len(kl.attn_processor_calls) == 1
            assert isinstance(kl.attn_processor_calls[0], wrapper_module.AttnProcessor)

    def test_probe_true_is_still_a_no_op_for_tiny_vae(self):
        """AutoencoderTiny has no set_attn_processor - an unguarded call would be
        an AttributeError. Must not raise even when the probe fires."""
        with patch.object(wrapper_module, "_fork_returns_kvo_tuple", lambda: True):
            tiny = _FakeTiny()
            _harden_vae_attention(tiny)  # must not raise AttributeError

    def test_probe_true_skips_a_kl_with_no_attention_blocks(self):
        with patch.object(wrapper_module, "_fork_returns_kvo_tuple", lambda: True):
            kl = _FakeKL(has_attn=False)
            _harden_vae_attention(kl)
            assert kl.attn_processor_calls == []

    def test_probe_reflects_the_installed_diffusers_fork(self):
        """Sanity check on the real, unpatched probe in this venv - just asserts it
        runs and returns a bool, without asserting which fork is installed (that
        would make the test depend on environment state rather than behaviour)."""
        assert isinstance(_fork_returns_kvo_tuple(), bool)
