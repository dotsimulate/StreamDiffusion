"""
Regression tests for the ControlNet loader's Xet-download-failure handling.

A laptop failed at startup with:

    OSError: Can't load the model for 'thibaud/controlnet-sd21-scribble-diffusers'.
    ... make sure '...' is the correct path to a directory containing a file
    named diffusion_pytorch_model.bin

That message is wrong. The real cause was a HuggingFace Xet transfer-layer
failure (`hf_xet`'s Rust download layer raising `OSError: I/O error: I/O
error: error decoding response body` after 33 minutes at 0 bytes transferred).
`diffusers`' `_get_model_file` catches any `EnvironmentError` (== `OSError` in
Python 3) and rewrites it into the "can't find the model" message above,
discarding the real cause into `__cause__`.

This is the same bug class already fixed for the base model load path (see
`test_load_model_network_error_reporting.py`), now fixed for
`ControlNetModule._load_pytorch_controlnet_model`: detect a Xet transfer
failure, retry once with Xet disabled (plain HTTPS) and `force_download=True`,
and if it still fails, report the real cause with an actionable hint instead
of the misleading message.

Follows the `patch.object(module.Symbol, "method", staticmethod(fake))`
pattern established in test_load_model_network_error_reporting.py. Unlike
that file, `ControlNetModule.__init__(device, dtype)` is trivial, so no
`object.__new__` shell is needed, and `ControlNetModel` is imported at module
level in controlnet_module.py, so the patch target is
`controlnet_module.ControlNetModel`.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from huggingface_hub import constants as hf_constants

from streamdiffusion.modules import controlnet_module as controlnet_module_mod
from streamdiffusion.modules.controlnet_module import ControlNetModule
from streamdiffusion.utils.hf_download import is_xet_download_error


def _make_xet_error() -> OSError:
    """Build a synthetic exception chain mirroring the real one from the log:
    an inner OSError raised from a function literally named `xet_get` (so it
    carries a genuine traceback frame matching the detection signature),
    wrapped by an outer OSError with the misleading diffusers-style message.
    """

    def download_files():
        raise OSError("I/O error: I/O error: error decoding response body")

    def xet_get():
        download_files()

    try:
        xet_get()
    except OSError as inner:
        try:
            raise OSError(
                "Can't load the model for 'thibaud/controlnet-sd21-scribble-diffusers'. ... make "
                "sure '...' is the correct path to a directory containing a file named "
                "diffusion_pytorch_model.bin"
            ) from inner
        except OSError as outer:
            return outer
    raise AssertionError("unreachable")


class _FakeControlNet:
    """Stand-in for a loaded ControlNetModel."""

    model_id: str | None = None

    def to(self, *args, **kwargs):
        return self


class TestXetDownloadErrorDetection:
    def test_detects_xet_error_chain(self):
        assert is_xet_download_error(_make_xet_error()) is True

    def test_does_not_flag_ordinary_oserror(self):
        assert is_xet_download_error(OSError("permission denied")) is False

    def test_does_not_flag_genuine_missing_file(self):
        err = OSError(
            "Can't load the model for 'some/typo-repo'. ... make sure 'some/typo-repo' is the "
            "correct path to a directory containing a file named diffusion_pytorch_model.bin"
        )
        assert is_xet_download_error(err) is False


class TestXetRetrySucceeds:
    def test_retries_with_xet_disabled_and_force_download(self):
        calls = []

        def fake_from_pretrained(model_id, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise _make_xet_error()
            return _FakeControlNet()

        module = ControlNetModule(device="cpu", dtype=torch.float32)

        with patch.object(
            controlnet_module_mod.ControlNetModel,
            "from_pretrained",
            staticmethod(fake_from_pretrained),
        ):
            result = module._load_pytorch_controlnet_model("thibaud/controlnet-sd21-scribble-diffusers")

        assert isinstance(result, _FakeControlNet)
        assert len(calls) == 2, "expected exactly one retry after the Xet failure"
        assert calls[0].get("force_download") is not True
        assert calls[1].get("force_download") is True
        assert result.model_id == "thibaud/controlnet-sd21-scribble-diffusers"


class TestXetDisableStateRestored:
    def test_restored_after_successful_retry(self):
        original = hf_constants.HF_HUB_DISABLE_XET
        seen_during_retry = {}
        calls = []

        def fake_from_pretrained(model_id, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise _make_xet_error()
            seen_during_retry["value"] = hf_constants.HF_HUB_DISABLE_XET
            return _FakeControlNet()

        module = ControlNetModule(device="cpu", dtype=torch.float32)

        try:
            with patch.object(
                controlnet_module_mod.ControlNetModel,
                "from_pretrained",
                staticmethod(fake_from_pretrained),
            ):
                module._load_pytorch_controlnet_model("thibaud/controlnet-sd21-scribble-diffusers")
        finally:
            assert original == hf_constants.HF_HUB_DISABLE_XET

        assert seen_during_retry["value"] is True

    def test_restored_after_failed_retry(self):
        original = hf_constants.HF_HUB_DISABLE_XET

        def fake_from_pretrained(model_id, **kwargs):
            raise _make_xet_error()

        module = ControlNetModule(device="cpu", dtype=torch.float32)

        try:
            with patch.object(
                controlnet_module_mod.ControlNetModel,
                "from_pretrained",
                staticmethod(fake_from_pretrained),
            ):
                with pytest.raises(RuntimeError):
                    module._load_pytorch_controlnet_model("thibaud/controlnet-sd21-scribble-diffusers")
        finally:
            assert original == hf_constants.HF_HUB_DISABLE_XET


class TestPermanentFailureMessage:
    def test_message_reports_real_cause_with_hint(self):
        xet_error = _make_xet_error()

        def fake_from_pretrained(model_id, **kwargs):
            raise xet_error

        module = ControlNetModule(device="cpu", dtype=torch.float32)

        with patch.object(
            controlnet_module_mod.ControlNetModel,
            "from_pretrained",
            staticmethod(fake_from_pretrained),
        ):
            with pytest.raises(Exception) as exc_info:
                module._load_pytorch_controlnet_model("thibaud/controlnet-sd21-scribble-diffusers")

        message = str(exc_info.value).lower()
        assert "huggingface download failure" in message
        # Unlike the original bug report (bare diffusers text, no explanation at all),
        # the final message must make clear a Xet-disabled retry was attempted.
        assert "retrying without xet" in message
        # The real cause must be preserved in the exception chain, not discarded.
        cause_chain = []
        cur = exc_info.value
        while cur is not None:
            cause_chain.append(cur)
            cur = cur.__cause__
        assert xet_error in cause_chain


class TestOfflineModeSkipsRetry:
    def test_offline_mode_does_not_retry(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        calls = []

        def fake_from_pretrained(model_id, **kwargs):
            calls.append(kwargs)
            raise _make_xet_error()

        module = ControlNetModule(device="cpu", dtype=torch.float32)

        with patch.object(
            controlnet_module_mod.ControlNetModel,
            "from_pretrained",
            staticmethod(fake_from_pretrained),
        ):
            # Offline mode's permanent-failure path is a bare `raise` (preserves the original
            # exception unchanged) rather than the RuntimeError wrapping used elsewhere - it never
            # reaches the "retry failed" or "non-offline network error" branches that wrap.
            with pytest.raises(OSError):
                module._load_pytorch_controlnet_model("thibaud/controlnet-sd21-scribble-diffusers")

        assert len(calls) == 1, "offline mode must not retry a Xet failure"
