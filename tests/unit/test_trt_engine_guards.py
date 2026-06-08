"""
Tests for TensorRTEngine dynamic-shape and dtype guards (Findings C, D).

These tests use MagicMock for all TRT internals and verify only the pure-Python
guard logic added to allocate_buffers() and infer().

The entire module is skipped when TensorRT is not installed.

Run with: pytest tests/unit/test_trt_engine_guards.py -v
"""

from collections import OrderedDict
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Import guard — skip all tests if TRT is unavailable
# ---------------------------------------------------------------------------

try:
    from streamdiffusion.preprocessing.processors.trt_base import (
        TENSORRT_AVAILABLE,
        TensorRTEngine,
    )

    if TENSORRT_AVAILABLE:
        import tensorrt as trt
except ImportError:
    TENSORRT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TENSORRT_AVAILABLE,
    reason="TensorRT not installed — skipping engine guard tests",
)


# ---------------------------------------------------------------------------
# Fake-engine factory
# ---------------------------------------------------------------------------


def _make_fake_engine(
    input_shape=(1, 3, 512, 512),
    output_shape=(1, 1, 512, 512),
    dtype=None,
):
    """
    Build a TensorRTEngine with fully-mocked TRT internals.

    Pre-populates tensors as if ``allocate_buffers`` already ran at
    ``input_shape`` / ``output_shape`` with ``dtype`` (default float16).
    """
    import torch

    dtype = dtype or torch.float16

    eng = TensorRTEngine.__new__(TensorRTEngine)
    eng.engine_path = "/fake/test.engine"
    eng._cuda_stream = None

    eng.engine = MagicMock()
    eng.context = MagicMock()
    eng.context.execute_async_v3.return_value = True

    names = ["input", "output"]
    modes = [trt.TensorIOMode.INPUT, trt.TensorIOMode.OUTPUT]
    current_shapes = {
        "input": input_shape,
        "output": output_shape,
    }

    eng.engine.num_io_tensors = 2
    eng.engine.get_tensor_name.side_effect = lambda idx: names[idx]
    eng.engine.get_tensor_mode.side_effect = lambda n: modes[names.index(n)]
    eng.context.get_tensor_shape.side_effect = lambda n: current_shapes[n]

    eng.tensors = OrderedDict(
        input=torch.zeros(*input_shape, dtype=dtype),
        output=torch.zeros(*output_shape, dtype=dtype),
    )

    return eng, current_shapes


# ---------------------------------------------------------------------------
# infer() dtype guard (Finding D)
# ---------------------------------------------------------------------------


class TestInferDtypeGuard:
    def test_dtype_mismatch_raises_valueerror_with_context(self):
        """float32 input into a float16 engine must raise ValueError naming the tensor."""
        import torch

        eng, _ = _make_fake_engine()
        feed = {"input": torch.zeros(1, 3, 512, 512, dtype=torch.float32)}

        with pytest.raises(ValueError, match="dtype mismatch") as exc_info:
            eng.infer(feed)

        msg = str(exc_info.value)
        assert "input" in msg, "error should name the mismatched tensor"
        assert "engine_path" in msg or "engine:" in msg.lower() or "/fake/" in msg, (
            "error should include engine path for diagnosability"
        )

    def test_correct_dtype_and_shape_succeeds(self):
        """Matching dtype + shape: infer should succeed and return output."""
        import torch

        eng, _ = _make_fake_engine()
        feed = {"input": torch.zeros(1, 3, 512, 512, dtype=torch.float16)}
        result = eng.infer(feed)
        assert "output" in result

    def test_dtype_mismatch_float16_input_into_float32_engine(self):
        """float16 input into a float32 engine must also raise ValueError."""
        import torch

        eng, _ = _make_fake_engine(dtype=torch.float32)
        feed = {"input": torch.zeros(1, 3, 512, 512, dtype=torch.float16)}

        with pytest.raises(ValueError, match="dtype mismatch"):
            eng.infer(feed)


# ---------------------------------------------------------------------------
# infer() shape reconciliation — alignment guarantee (Finding C)
# ---------------------------------------------------------------------------


class TestInferShapeReconciliation:
    def test_shape_change_reallocates_output_to_new_resolution(self):
        """
        Feeding a 384×384 input after allocating at 512×512 must reallocate
        the output buffer to match the new resolution (the alignment guarantee).
        """
        import torch

        eng, current_shapes = _make_fake_engine(
            input_shape=(1, 3, 512, 512),
            output_shape=(1, 1, 512, 512),
        )

        new_input_shape = (1, 3, 384, 384)
        new_output_shape = (1, 1, 384, 384)

        # After set_input_shape(384), context.get_tensor_shape should return new sizes
        def dynamic_shape(name):
            if name == "input":
                return new_input_shape
            return new_output_shape

        eng.context.get_tensor_shape.side_effect = dynamic_shape

        feed = {"input": torch.zeros(*new_input_shape, dtype=torch.float16)}
        result = eng.infer(feed)

        assert tuple(result["output"].shape) == new_output_shape, (
            f"output shape {tuple(result['output'].shape)} != expected {new_output_shape} "
            "— dynamic-shape reconciliation is broken"
        )

    def test_same_shape_does_not_trigger_realloc(self):
        """Feeding the same shape as allocated must not call set_input_shape."""
        import torch

        eng, _ = _make_fake_engine()
        feed = {"input": torch.zeros(1, 3, 512, 512, dtype=torch.float16)}
        eng.infer(feed)

        eng.context.set_input_shape.assert_not_called()


# ---------------------------------------------------------------------------
# allocate_buffers() dynamic-shape guard (Finding C)
# ---------------------------------------------------------------------------


class TestAllocateBuffersGuard:
    def _make_engine_skeleton(self):
        """Bare TensorRTEngine without pre-allocated tensors."""
        eng = TensorRTEngine.__new__(TensorRTEngine)
        eng.engine_path = "/fake/dynamic.engine"
        eng.tensors = OrderedDict()
        eng.engine = MagicMock()
        eng.context = MagicMock()
        return eng

    def test_dynamic_dims_without_input_shape_raises_runtime_error(self):
        """
        allocate_buffers on a dynamic engine without input_shape must raise
        RuntimeError that names the problematic tensor and says 'dynamic'.
        """
        eng = self._make_engine_skeleton()

        eng.engine.num_io_tensors = 1
        eng.engine.get_tensor_name.side_effect = lambda idx: "input"
        eng.engine.get_tensor_mode.side_effect = lambda n: trt.TensorIOMode.INPUT
        # Simulate dynamic engine: shape has -1 dims
        eng.context.get_tensor_shape.return_value = (-1, 3, -1, -1)

        with pytest.raises(RuntimeError, match="dynamic") as exc_info:
            eng.allocate_buffers()

        msg = str(exc_info.value)
        assert "input" in msg, "error should name the problematic tensor"

    def test_static_dims_without_input_shape_succeeds(self):
        """
        A fully-static engine (no -1 dims) must work without input_shape.
        Verifies we didn't break the existing static-engine path.
        """
        import numpy as np

        eng = self._make_engine_skeleton()
        static_shape = (1, 3, 512, 512)

        eng.engine.num_io_tensors = 1
        eng.engine.get_tensor_name.side_effect = lambda idx: "input"
        eng.engine.get_tensor_mode.side_effect = lambda n: trt.TensorIOMode.INPUT
        eng.engine.get_tensor_dtype.side_effect = lambda n: trt.DataType.FLOAT

        # Return static (concrete) shape
        eng.context.get_tensor_shape.return_value = static_shape

        # Patch trt.nptype to return a known numpy dtype
        import streamdiffusion.preprocessing.processors.trt_base as trt_base_mod

        original_nptype = trt_base_mod.trt.nptype
        try:
            trt_base_mod.trt.nptype = lambda _: np.float32
            eng.allocate_buffers()
        finally:
            trt_base_mod.trt.nptype = original_nptype

        assert "input" in eng.tensors
        assert tuple(eng.tensors["input"].shape) == static_shape
