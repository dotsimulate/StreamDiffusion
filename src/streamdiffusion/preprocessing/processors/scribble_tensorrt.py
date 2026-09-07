"""
Scribble TensorRT preprocessor — GPU-native scribble edge maps via TRT.

Reuses the HED TRT engine (no second build needed).  Overrides _postprocess
with a GPU port of the ``scribble=True`` branch of
``controlnet_aux.HEDdetector.__call__``:

    1. nms(edge, 127, 3.0):  Gaussian blur (sigma 3) -> keep pixels that are a
       local maximum along any of the 4 scan lines (h / v / two diagonals)
       -> threshold at 127/255
    2. Gaussian blur (sigma 3) of the binary map, re-threshold at 4/255
       (thickens the thin ridges into scribble strokes)
"""

import logging
import math

import torch
import torch.nn.functional as F

from .category_params import EDGE_SMOOTHNESS_PARAM, apply_edge_smoothness
from .hed_tensorrt import HEDTensorrtPreprocessor
from .trt_base import _first_output

logger = logging.getLogger(__name__)

#: controlnet_aux ``nms(x, t=127, s=3.0)`` threshold expressed on a [0, 1] map.
DEFAULT_SCRIBBLE_THRESHOLD = 127.0 / 255.0
#: Gaussian sigma used by both the NMS pre-blur and the stroke-thickening blur.
SCRIBBLE_SIGMA = 3.0
#: ``detected_map[detected_map > 4] = 255`` on a [0, 1] map.
THICKEN_THRESHOLD = 4.0 / 255.0


# ---------------------------------------------------------------------------
# GPU scribble NMS helpers
# ---------------------------------------------------------------------------


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (1, 1, H, W) tensor (cv2.GaussianBlur((0, 0), sigma) equivalent)."""
    radius = max(1, int(math.ceil(4.0 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(t * t) / (2.0 * sigma * sigma))
    k = k / k.sum()
    h, w = x.shape[-2:]
    # cv2 default border is BORDER_REFLECT_101 (= torch "reflect"); it needs pad < dim.
    mode = "reflect" if radius < min(h, w) else "replicate"
    x = F.pad(x, (radius, radius, 0, 0), mode=mode)
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, radius, radius), mode=mode)
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x


def _directional_nms(x: torch.Tensor) -> torch.Tensor:
    """Keep pixels of a (1, 1, H, W) map that are a local max along any of the 4 scan lines.

    Mirrors the ``cv2.dilate(x, kernel=f) == x`` loop in controlnet_aux ``nms``
    with the four 3-pixel line kernels (horizontal, vertical, both diagonals).
    Replicate padding leaves border pixels unaffected, like cv2's default
    morphology border.
    """
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    c = xp[..., 1:-1, 1:-1]

    def _dil(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.maximum(a, c), b)

    horiz = _dil(xp[..., 1:-1, :-2], xp[..., 1:-1, 2:])
    vert = _dil(xp[..., :-2, 1:-1], xp[..., 2:, 1:-1])
    diag1 = _dil(xp[..., :-2, :-2], xp[..., 2:, 2:])
    diag2 = _dil(xp[..., :-2, 2:], xp[..., 2:, :-2])
    keep = (horiz == c) | (vert == c) | (diag1 == c) | (diag2 == c)
    return torch.where(keep, c, torch.zeros_like(c))


def _scribble_nms_gpu(
    edge_map: torch.Tensor,
    threshold: float = DEFAULT_SCRIBBLE_THRESHOLD,
    sigma: float = SCRIBBLE_SIGMA,
    thicken: bool = True,
) -> torch.Tensor:
    """
    GPU port of the controlnet_aux scribble post-process.

    Args:
        edge_map:  (H, W) float tensor in [0, 1] (HED sigmoid probabilities)
        threshold: ridge threshold on the blurred map (controlnet_aux: 127/255)
        sigma:     Gaussian sigma for the pre-blur and the thickening blur
        thicken:   apply the blur + 4/255 re-threshold that widens ridges into strokes

    Returns:
        (H, W) float tensor in {0.0, 1.0}
    """
    h, w = edge_map.shape[-2:]
    x = edge_map.reshape(1, 1, h, w).float()

    x = _gaussian_blur(x, sigma)
    ridges = _directional_nms(x)
    binary = (ridges > threshold).to(x.dtype)

    if thicken:
        binary = (_gaussian_blur(binary, sigma) > THICKEN_THRESHOLD).to(x.dtype)

    return binary.reshape(h, w)


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
            "display_name": "Scribble Strokes (HED + NMS, TensorRT)",
            "description": (
                "GPU-native scribble strokes for scribble ControlNets. Runs the same "
                "hed.engine as the HED preprocessor and applies the controlnet_aux "
                "scribble=True post-process on the GPU: NMS thinning, binarisation at "
                "scribble_threshold, then stroke thickening (no CPU round-trips)."
            ),
            "parameters": {
                "scribble_threshold": {
                    "type": "float",
                    "default": round(DEFAULT_SCRIBBLE_THRESHOLD, 3),
                    "range": [0.0, 1.0],
                    "description": (
                        "Ridge threshold applied after the Gaussian-blurred directional NMS, on the "
                        "HED edge probability in [0, 1] (controlnet_aux uses 127/255). Lower keeps more edges."
                    ),
                },
                **EDGE_SMOOTHNESS_PARAM,
            },
            "use_cases": [
                "Scribble ControlNet conditioning",
                "Sketch-style edge maps (real-time)",
            ],
        }

    def _postprocess(self, engine_outputs: dict) -> torch.Tensor:
        """
        Apply scribble NMS + threshold to the HED output, return 3-channel CHW.

        Input  : engine_outputs["edge_map"]  shape (B, 1, H, W)  or (B, H, W)
        Output : (3, H, W) in {0.0, 1.0}   (binary scribble map)
        """
        out = _first_output(engine_outputs).float()

        if out.dim() == 4:
            out = out.squeeze(1)
        if out.dim() == 3:
            out = out.squeeze(0)  # (H, W)

        # Engine output is a sigmoid probability map; keep absolute values so the
        # threshold matches controlnet_aux (no min/max normalisation).
        out = out.clamp(0.0, 1.0)

        # Optional smoothness pre-blur (category-standard edge param) applied before
        # NMS so that increasing smoothness suppresses fine texture while preserving
        # the structural ridges that NMS retains.
        smoothness = float(self.params.get("smoothness", 0.0))
        if smoothness > 0.0:
            out = apply_edge_smoothness(out, smoothness)  # (H, W) in, (H, W) out

        threshold = float(self.params.get("scribble_threshold", DEFAULT_SCRIBBLE_THRESHOLD))
        scribble = _scribble_nms_gpu(out, threshold=threshold)  # (H, W)

        # Expand to 3-channel RGB
        return scribble.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
