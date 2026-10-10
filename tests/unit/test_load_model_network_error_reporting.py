"""
Regression tests for the `_load_model` error-masking fix.

A clean install's first run failed after a 56-minute HuggingFace download
timeout with a misleading `RuntimeError: ... Invalid pretrained_model_name_or_path
provided`. The real cause (a network timeout downloading a bare HF repo id) was
overwritten by two structurally-guaranteed-to-fail `from_single_file` attempts
that ran afterward, whose own failure became the reported error.

Covers three independent pieces of that fix in `StreamDiffusionWrapper._load_model`:

1. A bare HF repo id (no local path, no `.safetensors` suffix) builds a
   single-entry `loading_methods` list — `from_single_file` requires an actual
   file/URL and can never succeed on a bare repo id, so it must not be tried.
2. When the sole attempt fails with a network-class error, that error (not a
   masked "Invalid pretrained_model_name_or_path") is what `_load_model` raises,
   with the network exception preserved as `__cause__`.
3. A bare repo id that merely contains the substring "xl" routes the inner
   SDXL pipeline-mismatch retry through `StableDiffusionXLPipeline.from_pretrained`,
   not `.from_single_file`, for the same structural reason as (1).

Follows the object.__new__ shell + patch.object(wrapper_module.*, ...) pattern
established in test_wrapper_exception_hygiene.py.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest
import requests
import torch

from streamdiffusion import wrapper as wrapper_module
from streamdiffusion.wrapper import StreamDiffusionWrapper


def _make_wrapper_shell() -> StreamDiffusionWrapper:
    """Construct a minimal StreamDiffusionWrapper without model loading."""
    w = object.__new__(StreamDiffusionWrapper)
    w.device = torch.device("cpu")
    w.dtype = torch.float32
    w.cleanup_gpu_memory = lambda: None
    return w


class _FakeWrongTypePipe:
    """Stand-in for a successfully-loaded but non-SDXL pipeline object."""

    def to(self, *args, **kwargs):
        return self


class TestBareRepoIdLoadingMethods:
    def test_from_single_file_variants_are_never_attempted(self):
        sd_single_file_calls = []
        xl_single_file_calls = []

        def fake_auto_from_pretrained(path, *args, **kwargs):
            raise RuntimeError("AutoPipeline boom")

        def fake_sd_from_single_file(path, *args, **kwargs):
            sd_single_file_calls.append(path)
            raise RuntimeError("should never be reached")

        def fake_xl_from_single_file(path, *args, **kwargs):
            xl_single_file_calls.append(path)
            raise RuntimeError("should never be reached")

        w = _make_wrapper_shell()

        with (
            patch.object(
                wrapper_module.AutoPipelineForText2Image,
                "from_pretrained",
                staticmethod(fake_auto_from_pretrained),
            ),
            patch.object(
                wrapper_module.StableDiffusionPipeline,
                "from_single_file",
                staticmethod(fake_sd_from_single_file),
            ),
            patch.object(
                wrapper_module.StableDiffusionXLPipeline,
                "from_single_file",
                staticmethod(fake_xl_from_single_file),
            ),
        ):
            with pytest.raises(RuntimeError):
                w._load_model("stabilityai/sd-turbo", t_index_list=[0])

        assert sd_single_file_calls == [], "SD from_single_file must not be attempted for a bare repo id"
        assert xl_single_file_calls == [], "SDXL from_single_file must not be attempted for a bare repo id"


class TestNetworkErrorNotMasked:
    def test_read_timeout_is_reported_not_masked(self):
        """The regression test for the reported bug: a HF download timeout must
        surface as itself, not as 'Invalid pretrained_model_name_or_path'."""
        timeout_error = requests.exceptions.ReadTimeout("Read timed out. (read timeout=10)")

        def fake_auto_from_pretrained(path, *args, **kwargs):
            raise timeout_error

        w = _make_wrapper_shell()

        with patch.object(
            wrapper_module.AutoPipelineForText2Image,
            "from_pretrained",
            staticmethod(fake_auto_from_pretrained),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                w._load_model("stabilityai/sd-turbo", t_index_list=[0])

        message = str(exc_info.value).lower()
        assert "invalid `pretrained_model_name_or_path`" not in message
        assert "read timed out" in message
        assert "huggingface download failure" in message
        assert exc_info.value.__cause__ is timeout_error


class TestXlNamedBareRepoIdRetry:
    def test_inner_retry_uses_from_pretrained_not_from_single_file(self):
        """A bare repo id containing 'xl' (matching the SDXL-indicator substring
        check) is not a local file - the inner SDXL pipeline-mismatch retry must
        use from_pretrained, not from_single_file."""
        xl_from_pretrained_calls = []
        xl_from_single_file_calls = []

        def fake_auto_from_pretrained(path, *args, **kwargs):
            return _FakeWrongTypePipe()

        def fake_xl_from_pretrained(path, *args, **kwargs):
            xl_from_pretrained_calls.append(path)
            raise RuntimeError("retry boom")

        def fake_xl_from_single_file(path, *args, **kwargs):
            xl_from_single_file_calls.append(path)
            raise RuntimeError("should never be reached")

        w = _make_wrapper_shell()

        with (
            patch.object(
                wrapper_module.AutoPipelineForText2Image,
                "from_pretrained",
                staticmethod(fake_auto_from_pretrained),
            ),
            patch.object(
                wrapper_module.StableDiffusionXLPipeline,
                "from_pretrained",
                staticmethod(fake_xl_from_pretrained),
            ),
            patch.object(
                wrapper_module.StableDiffusionXLPipeline,
                "from_single_file",
                staticmethod(fake_xl_from_single_file),
            ),
        ):
            with pytest.raises(RuntimeError):
                w._load_model("some-xl-model", t_index_list=[0])

        assert xl_from_pretrained_calls == ["some-xl-model"]
        assert xl_from_single_file_calls == [], "from_single_file must not be attempted on a bare repo id retry"


def _make_fp16_wrapper_shell() -> StreamDiffusionWrapper:
    w = _make_wrapper_shell()
    w.dtype = torch.float16
    return w


class _OkPipe:
    def to(self, *args, **kwargs):
        return self


class TestLowMemoryFp16Variant:
    """Field crash (v0.4.0, German Windows): a bare from_pretrained(id) picked the fp32 UNet
    (10 GB), which the OS charges in full against RAM + paging file -> OSError 1455 -> blank
    MemoryError. The fp16 variant halves the commit."""

    def test_fp16_load_requests_fp16_variant_and_dtype(self):
        calls = []

        def fake_auto(path, *args, **kwargs):
            calls.append(kwargs)
            return _OkPipe()

        w = _make_fp16_wrapper_shell()
        with patch.object(wrapper_module.AutoPipelineForText2Image, "from_pretrained", staticmethod(fake_auto)):
            with contextlib.suppress(AttributeError):  # _OkPipe has no .unet; we only assert the loader call
                w._load_model("stabilityai/sd-turbo", t_index_list=[0])

        assert calls[0]["variant"] == "fp16"
        assert calls[0]["torch_dtype"] == torch.float16
        assert calls[0]["use_safetensors"] is True

    def test_fp32_load_does_not_request_variant(self):
        calls = []

        def fake_auto(path, *args, **kwargs):
            calls.append(kwargs)
            return _OkPipe()

        w = _make_wrapper_shell()  # float32
        with patch.object(wrapper_module.AutoPipelineForText2Image, "from_pretrained", staticmethod(fake_auto)):
            with contextlib.suppress(AttributeError):  # _OkPipe has no .unet; we only assert the loader call
                w._load_model("stabilityai/sd-turbo", t_index_list=[0])

        assert "variant" not in calls[0]

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("You are trying to load the model files of the `variant=fp16`, but no such modeling files"),
            OSError("Error no file named diffusion_pytorch_model.fp16.safetensors found in directory x"),
        ],
        ids=["value_error_variant_fp16", "os_error_missing_safetensors"],
    )
    def test_missing_variant_retries_without_variant(self, error):
        calls = []

        def fake_auto(path, *args, **kwargs):
            calls.append(kwargs)
            if "variant" in kwargs:
                raise error
            return _OkPipe()

        w = _make_fp16_wrapper_shell()
        with patch.object(wrapper_module.AutoPipelineForText2Image, "from_pretrained", staticmethod(fake_auto)):
            with contextlib.suppress(AttributeError):  # _OkPipe has no .unet; we only assert the loader call
                w._load_model("stabilityai/sd-turbo", t_index_list=[0])

        assert len(calls) == 2 and "variant" not in calls[1]

    def test_pagefile_error_is_not_retried_and_gets_hint(self):
        calls = []
        err = OSError("The paging file is too small for this operation to complete. (os error 1455)")

        def fake_auto(path, *args, **kwargs):
            calls.append(kwargs)
            raise err

        w = _make_fp16_wrapper_shell()
        with patch.object(wrapper_module.AutoPipelineForText2Image, "from_pretrained", staticmethod(fake_auto)):
            with pytest.raises(RuntimeError) as exc_info:
                w._load_model("stabilityai/sd-turbo", t_index_list=[0])

        assert len(calls) == 1
        assert "paging file" in str(exc_info.value).lower()
        assert "system managed" in str(exc_info.value).lower()
        assert exc_info.value.__cause__ is err

    def test_bare_memory_error_message_is_not_blank(self):
        def fake_auto(path, *args, **kwargs):
            raise MemoryError()

        w = _make_wrapper_shell()
        with patch.object(wrapper_module.AutoPipelineForText2Image, "from_pretrained", staticmethod(fake_auto)):
            with pytest.raises(RuntimeError) as exc_info:
                w._load_model("stabilityai/sd-turbo", t_index_list=[0])

        assert "MemoryError" in str(exc_info.value)
        assert "system managed" in str(exc_info.value).lower()

    def test_pagefile_error_nested_in_cause_chain_is_detected(self):
        from streamdiffusion.utils.hf_download import is_pagefile_error

        try:
            try:
                raise OSError("... (os error 1455)")
            except OSError as inner:
                raise OSError("Can't load the model") from inner
        except OSError as outer:
            assert is_pagefile_error(outer)
        assert not is_pagefile_error(RuntimeError("unrelated"))
