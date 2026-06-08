"""
HED TensorRT preprocessor — GPU-native edge detection via TRT engine.

The HED network (ControlNetHED_Apache2) is wrapped in HEDExportWrapper so
that ONNX export sees a single output tensor (full-resolution edge map) rather
than the native 5-output multi-scale tuple.  The wrapper input/output contract:

    input  : float32 (B, 3, H, W) in [0, 1]   ← same as validate_tensor_input output
    output : float32 (B, 1, H, W) in [0, 1]   ← sigmoid edge map at full resolution
"""

import logging
from pathlib import Path

import torch

from .trt_base import SelfBuildingTRTPreprocessor


logger = logging.getLogger(__name__)

try:
    from controlnet_aux import HEDdetector

    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False


# ---------------------------------------------------------------------------
# ONNX export wrapper — returns only the full-resolution output
# ---------------------------------------------------------------------------


class HEDExportWrapper(torch.nn.Module):
    """
    Thin wrapper around ControlNetHED_Apache2 for ONNX export.

    The native forward returns a 5-element tuple of tensors at decreasing
    resolutions.  ONNX export requires a single output of consistent shape.
    This wrapper returns only output[0] (the full-resolution sigmoid map).
    """

    def __init__(self, netNetwork: torch.nn.Module):
        super().__init__()
        self.netNetwork = netNetwork

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [0, 1]
        outputs = self.netNetwork(x)
        # outputs is a list/tuple of 5 tensors at (H, H/2, H/4, H/8, H/16);
        # take the first = full-resolution edge map  shape (B, 1, H, W)
        return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class HEDTensorrtPreprocessor(SelfBuildingTRTPreprocessor):
    """
    HED edge detection via a self-built TensorRT engine.

    GPU-native: no CPU / PIL round-trip on the tensor path.
    The engine is built on first use and cached in engines/preprocessors/hed.engine
    (or the path supplied via preprocessor_params.engine_path in the YAML config).
    """

    engine_filename = "hed.engine"
    onnx_filename = "hed.onnx"
    default_detect_resolution = 512

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "HED Edge Detection (TensorRT)",
            "description": (
                "GPU-native HED (Holistically-Nested Edge Detection) via TensorRT. "
                "Self-builds its engine from the controlnet_aux model on first run. "
                "No CPU/PIL round-trips — satisfies the GPU-residency constraint."
            ),
            "parameters": {},
            "use_cases": [
                "HED ControlNet conditioning",
                "Structured edge maps (real-time)",
            ],
        }

    def __init__(self, **kwargs):
        if not CONTROLNET_AUX_AVAILABLE:
            raise ImportError(
                "controlnet_aux is required for HEDTensorrtPreprocessor. Install with: pip install controlnet_aux"
            )
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def _export_onnx(self, onnx_path: Path) -> None:
        """Load HEDdetector, wrap it, and export to ONNX."""
        logger.info("HEDTensorrtPreprocessor: loading HEDdetector for ONNX export…")
        detector = HEDdetector.from_pretrained("lllyasviel/Annotators")

        if not hasattr(detector, "netNetwork"):
            raise RuntimeError(
                "HEDTensorrtPreprocessor: HEDdetector has no 'netNetwork' attribute. "
                "controlnet_aux version may be incompatible."
            )

        wrapper = HEDExportWrapper(detector.netNetwork).to(self.device).eval()
        res = self.default_detect_resolution
        dummy = torch.zeros(1, 3, res, res, device=self.device)

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy,
                str(onnx_path),
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch", 2: "height", 3: "width"},
                },
            )

        logger.info(f"HEDTensorrtPreprocessor: ONNX exported → {onnx_path}")
        # Free GPU memory used by the export model
        del wrapper, detector
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Post-process TRT output → CHW GPU tensor
    # ------------------------------------------------------------------

    def _postprocess(self, engine_outputs: dict) -> torch.Tensor:
        """
        Convert TRT output to a 3-channel [0, 1] edge map GPU tensor (CHW).

        Input  : engine_outputs["output"]  shape (B, 1, H, W)  or (B, H, W)
        Output : (3, H, W) in [0, 1]
        """
        out = engine_outputs["output"].float()

        # Collapse batch + channel dims if present
        if out.dim() == 4:
            out = out.squeeze(1)  # (B, H, W) — B should be 1
        if out.dim() == 3:
            out = out.squeeze(0)  # (H, W)

        # Normalize to [0, 1]  (edge map may already be in this range)
        v_min, v_max = out.min(), out.max()
        if v_max > v_min:
            out = (out - v_min) / (v_max - v_min)
        out = out.clamp(0.0, 1.0)

        # Expand to 3-channel RGB  →  (3, H, W)
        return out.unsqueeze(0).repeat(3, 1, 1)
