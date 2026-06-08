"""
Scribble TensorRT preprocessor — GPU-native scribble edge maps via TRT.

Reuses the HED TRT engine (no second build needed).  Overrides _postprocess
to apply a GPU-native NMS + binarization that replicates the scribble=True
post-processing from controlnet_aux HEDdetector.__call__:

    1. Gaussian-blur the sigmoid edge map (smooth noise)
    2. Directional NMS — keep only local maxima  (thin lines)
    3. Threshold at 0.5 → binary edge mask
"""

import logging

import torch
import torch.nn.functional as F

from .hed_tensorrt import HEDTensorrtPreprocessor


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU scribble NMS helper
# ---------------------------------------------------------------------------


def _scribble_nms_gpu(edge_map: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Approximate GPU version of controlnet_aux nms() used by scribble=True mode.

    Performs:
      1. Light Gaussian blur  (3×3 average pool — avoids kornia dependency)
      2. 4-directional local-max suppression  (keep only ridge pixels)
      3. Threshold at `threshold`

    Args:
        edge_map: (H, W) float32 tensor in [0, 1] on GPU
        threshold: binarization threshold (default 0.5)

    Returns:
        (H, W) float32 binary tensor on GPU
    """
    x = edge_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Step 1: smooth
    x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    # Step 2: directional NMS — keep pixel if it is the local max along
    #         each of the 4 scanning directions (horizontal, vertical, two diagonals).
    #         We approximate with isotropic max-pool (good enough for thin-line extraction).
    x_max = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    is_max = (x == x_max).float()
    thinned = x * is_max

    # Step 3: binarize
    binary = (thinned.squeeze() > threshold).float()

    return binary


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class ScribbleTensorrtPreprocessor(HEDTensorrtPreprocessor):
    """
    Scribble edge maps via TRT — reuses the HED engine, overrides postprocess.

    The 'scribble' mode in controlnet_aux HEDdetector runs the same HED
    network but adds an NMS + binarization step.  Here we replicate that
    step with GPU tensor operations, so the full pipeline stays on CUDA.

    No second engine build is needed: engine_filename points at hed.engine.
    """

    # Deliberately points at the HED engine — no separate build
    engine_filename = "hed.engine"
    onnx_filename = "hed.onnx"  # kept consistent; export is never re-run if engine exists
    default_detect_resolution = 512

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Scribble Edge Detection (TensorRT)",
            "description": (
                "GPU-native scribble-style edge maps. Uses the HED TRT engine with "
                "GPU NMS + binarization post-processing (no CPU round-trips). "
                "Compatible with scribble ControlNets."
            ),
            "parameters": {
                "scribble_threshold": {
                    "type": "float",
                    "default": 0.5,
                    "description": "Binarization threshold for scribble edges (0–1)",
                },
            },
            "use_cases": [
                "Scribble ControlNet conditioning",
                "Sketch-style edge maps (real-time)",
            ],
        }

    def _postprocess(self, engine_outputs: dict) -> torch.Tensor:
        """
        Apply scribble NMS + threshold to the HED output, return 3-channel CHW.

        Input  : engine_outputs["output"]  shape (B, 1, H, W)  or (B, H, W)
        Output : (3, H, W) in {0.0, 1.0}   (binary scribble map)
        """
        out = engine_outputs["output"].float()

        if out.dim() == 4:
            out = out.squeeze(1)
        if out.dim() == 3:
            out = out.squeeze(0)  # (H, W)

        # Normalize to [0, 1] before NMS
        v_min, v_max = out.min(), out.max()
        if v_max > v_min:
            out = (out - v_min) / (v_max - v_min)
        out = out.clamp(0.0, 1.0)

        threshold = float(self.params.get("scribble_threshold", 0.5))
        scribble = _scribble_nms_gpu(out, threshold=threshold)  # (H, W)

        # Expand to 3-channel RGB
        return scribble.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
