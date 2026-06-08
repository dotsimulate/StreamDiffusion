"""
Regression tests for Finding A: NormalBae torch-direct fallback initialization.

Before the fix, NormalBaeTensorrtPreprocessor.__new__ returned
``object.__new__(_NormalBaeTorchGPU)``.  Because _NormalBaeTorchGPU is not a
subclass of NormalBaeTensorrtPreprocessor, CPython's type.__call__ skipped
__init__ entirely.  The resulting object had no _detector, params, or device —
AttributeError on every frame.

After the fix, __new__ calls ``_NormalBaeTorchGPU(**kwargs)`` directly so
__init__ runs correctly.

Requires: controlnet_aux installed (skipped otherwise).
CPU-only: patches NormalBaeDetector.from_pretrained to avoid model downloads.
Run with: pytest tests/unit/test_normal_bae_fallback.py -v
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Skip guard — controlnet_aux must be importable
# ---------------------------------------------------------------------------

try:
    from controlnet_aux import NormalBaeDetector as _NDA  # noqa: F401 — import probe only

    _CONTROLNET_AUX_OK = True
except ImportError:
    _CONTROLNET_AUX_OK = False

pytestmark = pytest.mark.skipif(
    not _CONTROLNET_AUX_OK,
    reason="controlnet_aux not installed — skipping NormalBae fallback tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_detector():
    """Return a NormalBaeDetector stub that survives _load_model()."""
    stub = MagicMock()
    stub.model = MagicMock()
    stub.model.to.return_value = stub.model
    stub.model.eval.return_value = stub.model
    stub.norm = MagicMock()
    stub.norm.to.return_value = stub.norm
    return stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormalBaeFallbackFullyInitialized(unittest.TestCase):
    """Verify the fallback path produces a correctly-initialized _NormalBaeTorchGPU."""

    def setUp(self):
        # Reset the module-level probe cache before each test.
        import streamdiffusion.preprocessing.processors.normal_bae_tensorrt as mod

        mod._TRT_STRATEGY_AVAILABLE = None

    def test_fallback_has_device_params_and_detector(self):
        """
        Finding A regression: fallback object must have device, params, _detector.
        Before the fix this test raised AttributeError on the first frame.
        """
        import streamdiffusion.preprocessing.processors.normal_bae_tensorrt as mod
        from streamdiffusion.preprocessing.processors.normal_bae_tensorrt import (
            NormalBaeTensorrtPreprocessor,
            _NormalBaeTorchGPU,
        )

        stub_det = _make_stub_detector()

        with (
            patch.object(mod, "_probe_normal_bae_onnx_export", return_value=False),
            patch.object(mod, "TENSORRT_AVAILABLE", False),
            patch.object(mod, "NormalBaeDetector") as MockNDA,
        ):
            MockNDA.from_pretrained.return_value = stub_det
            obj = NormalBaeTensorrtPreprocessor(device="cpu")

        self.assertIsInstance(
            obj,
            _NormalBaeTorchGPU,
            "fallback must be a _NormalBaeTorchGPU instance",
        )
        self.assertTrue(hasattr(obj, "device"), "fallback must have 'device' attribute")
        self.assertTrue(hasattr(obj, "params"), "fallback must have 'params' attribute")
        self.assertTrue(hasattr(obj, "_detector"), "fallback must have '_detector' attribute")
        self.assertIsNotNone(obj._detector, "_detector must not be None after __init__")

    def test_fallback_device_is_passed_through(self):
        """Constructor kwargs (device, detect_resolution) must flow to the fallback object."""
        import streamdiffusion.preprocessing.processors.normal_bae_tensorrt as mod
        from streamdiffusion.preprocessing.processors.normal_bae_tensorrt import (
            NormalBaeTensorrtPreprocessor,
        )

        stub_det = _make_stub_detector()

        with (
            patch.object(mod, "_probe_normal_bae_onnx_export", return_value=False),
            patch.object(mod, "TENSORRT_AVAILABLE", False),
            patch.object(mod, "NormalBaeDetector") as MockNDA,
        ):
            MockNDA.from_pretrained.return_value = stub_det
            obj = NormalBaeTensorrtPreprocessor(device="cpu", detect_resolution=384)

        # params is populated by BasePreprocessor.__init__ from **kwargs
        self.assertEqual(
            obj.params.get("detect_resolution"),
            384,
            "detect_resolution kwarg must appear in fallback's params",
        )


class TestNormalBaeUninitializedGuard(unittest.TestCase):
    """Verify that the defensive guard in _process_tensor_core raises clearly."""

    def setUp(self):
        import streamdiffusion.preprocessing.processors.normal_bae_tensorrt as mod

        mod._TRT_STRATEGY_AVAILABLE = None

    def test_none_detector_raises_runtime_error_not_attribute_error(self):
        """
        Simulating the pre-fix uninitialized state: _detector=None must raise
        RuntimeError with a clear message instead of a bare AttributeError.
        """
        import torch

        import streamdiffusion.preprocessing.processors.normal_bae_tensorrt as mod
        from streamdiffusion.preprocessing.processors.normal_bae_tensorrt import (
            _NormalBaeTorchGPU,
        )

        stub_det = _make_stub_detector()

        with patch.object(mod, "NormalBaeDetector") as MockNDA:
            MockNDA.from_pretrained.return_value = stub_det
            obj = _NormalBaeTorchGPU(device="cpu")

        # Explicitly unset _detector to replicate pre-fix broken state
        obj._detector = None

        with self.assertRaises(RuntimeError) as ctx:
            obj._process_tensor_core(torch.zeros(3, 64, 64))

        err = str(ctx.exception)
        self.assertIn("_load_model", err, "error message should mention _load_model")


if __name__ == "__main__":
    unittest.main()
