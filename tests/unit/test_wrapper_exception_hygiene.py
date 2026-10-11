"""
Regression tests for wrapper.py exception-hygiene fixes (Phase 2 of the
perf/best-practices remediation, docs/perf_bestpractices_audit_2026-07-10.md).

Covers two independent fixes:

1. `_is_oom_error` — a shared OOM-detection helper that recognizes both the
   typed `torch.cuda.OutOfMemoryError` and the string-matching heuristic used
   by third-party code (e.g. TensorRT) that surfaces OOM as a generic
   RuntimeError. Pure function, CPU-only.

2. `_load_model`'s SDXL pipeline-type-mismatch handling — previously, when an
   SDXL model was detected but loaded with the wrong pipeline type and the
   explicit `StableDiffusionXLPipeline` retry also failed, the code silently
   logged a warning and continued with the mismatched pipeline. The fix
   raises RuntimeError on retry failure instead, refusing to proceed with a
   known-wrong pipeline type. That raise is caught by the enclosing
   `for method in loading_methods` loop's own except/continue, so it falls
   through to the next loading method rather than crashing outright — this
   test drives all three loading methods to fail and asserts the final error
   carries evidence that the fail-fast fired (rather than the old silent
   continuation).

Both tests are deliberately CPU-only and model-free (no real diffusers model
is loaded), following the object.__new__ shell pattern used elsewhere in
tests/unit/ (see test_safety_checker.py, test_normal_bae_fallback.py).
"""

from unittest.mock import patch

import pytest
import torch

from streamdiffusion import wrapper as wrapper_module
from streamdiffusion.wrapper import StreamDiffusionWrapper, _is_oom_error

# ---------------------------------------------------------------------------
# _is_oom_error
# ---------------------------------------------------------------------------


class TestIsOomError:
    def test_typed_cuda_oom_is_detected(self):
        exc = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert _is_oom_error(exc) is True

    @pytest.mark.parametrize(
        "message",
        [
            "CUDA out of memory. Tried to allocate 20.00 MiB",
            "torch.OutOfMemoryError: OutOfMemory",
            "generic oom while building engine",
            "CUDA error: an illegal memory access was encountered",
        ],
        ids=["cuda_oom_substring", "torch_oom_substring", "generic_oom", "illegal_memory_access"],
    )
    def test_string_heuristic_matches_known_oom_substrings(self, message):
        assert _is_oom_error(RuntimeError(message)) is True

    def test_unrelated_error_is_not_oom(self):
        assert _is_oom_error(RuntimeError("shape mismatch: expected [1, 4, 64, 64]")) is False
        assert _is_oom_error(ValueError("invalid literal for int()")) is False


# ---------------------------------------------------------------------------
# SDXL pipeline-mismatch fail-fast
# ---------------------------------------------------------------------------


class _FakeWrongTypePipe:
    """Stand-in for a successfully-loaded but non-SDXL pipeline object."""

    def to(self, *args, **kwargs):
        return self


def _make_wrapper_shell() -> StreamDiffusionWrapper:
    """Construct a minimal StreamDiffusionWrapper without model loading."""
    w = object.__new__(StreamDiffusionWrapper)
    w.device = torch.device("cpu")
    w.dtype = torch.float32
    w.cleanup_gpu_memory = lambda: None
    return w


class TestSdxlPipelineMismatchFailFast:
    def test_retry_failure_raises_instead_of_continuing_silently(self, caplog):
        """
        Path contains 'sdxl' + .safetensors -> loading_methods tries
        StableDiffusionXLPipeline.from_single_file first. Its first call
        succeeds but returns a non-SDXL-typed pipe, triggering the mismatch
        branch; the retry (second call to the same mocked method) fails.
        The other two loading methods are also made to fail so _load_model
        exhausts all methods and raises its final RuntimeError.

        Two things must hold:
        - the mismatch branch's failure is logged as a "pipeline retry ...
          also failed" warning (proving the fail-fast fired instead of the
          old silent "continue with the originally loaded pipeline"), and
        - _load_model does not crash trying to use the discarded
          wrong-typed pipe as if it were a valid load (regression guard for
          the stale `pipe` reference the fail-fast must clear).
        """
        xl_call_count = [0]

        def fake_xl_from_single_file(path, *args, **kwargs):
            xl_call_count[0] += 1
            if xl_call_count[0] == 1:
                return _FakeWrongTypePipe()
            raise RuntimeError("SDXL pipeline retry boom")

        def fake_auto_from_pretrained(path, *args, **kwargs):
            raise RuntimeError("AutoPipeline boom")

        def fake_sd_from_single_file(path, *args, **kwargs):
            raise RuntimeError("SD pipeline boom")

        w = _make_wrapper_shell()

        with (
            patch.object(
                wrapper_module.StableDiffusionXLPipeline,
                "from_single_file",
                staticmethod(fake_xl_from_single_file),
            ),
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
            caplog.at_level("WARNING", logger="streamdiffusion.wrapper"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                w._load_model("fake_sdxl_model.safetensors", t_index_list=[0])

        # The XL method must have been called twice: once for the initial
        # attempt (wrong-typed success) and once for the mismatch retry.
        assert xl_call_count[0] == 2, "expected exactly one retry attempt after the type mismatch"

        # The final error is "all methods exhausted" (the last method's error) -
        # that's expected once every loading method has failed. What matters is
        # that the mismatch retry's own failure was surfaced, not swallowed.
        assert "all loading methods failed" in str(exc_info.value).lower()
        assert any(
            "pipeline retry" in record.message.lower() and "also failed" in record.message.lower()
            for record in caplog.records
        ), "expected a logged warning proving the SDXL retry fail-fast fired, not a silent continue"
