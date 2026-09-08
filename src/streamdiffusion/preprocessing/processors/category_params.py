"""
Category-level parameter contracts for ControlNet preprocessors.

Canonical metadata fragments (merged into get_preprocessor_metadata) and GPU / NumPy
helpers that implement each category's standard post-processing step.  Preprocessors
import what they need — no inheritance change required.

Category contracts (keyed to the production xinsir SDXL CN set):
  EDGE_SMOOTHNESS_PARAM   edge-based  (canny, scribble_tensorrt, …)
  DEPTH_GRADE_PARAMS      depth-based (depth_tensorrt, …)
  POSE_DRAW_PARAMS        bodypose    (pose_tensorrt, …)
  SEGMENTATION_PARAMS     segmentation (future; mediapipe_segmentation already matches)

GPU helpers: apply_edge_smoothness, apply_depth_grade
NumPy helper: apply_depth_grade_numpy
"""

from __future__ import annotations

import math
import threading

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Canonical metadata fragments
# ---------------------------------------------------------------------------

EDGE_SMOOTHNESS_PARAM: dict = {
    "smoothness": {
        "type": "float",
        "default": 0.0,
        "range": [0.0, 1.0],
        "description": (
            "Optional pre-blur applied before edge extraction.  "
            "0 = off (sharpest edges); 1 = heaviest smoothing (σ≈2, ~13×13 kernel)."
        ),
    },
}

DEPTH_GRADE_PARAMS: dict = {
    "gamma": {
        "type": "float",
        "default": 1.0,
        "range": [0.1, 3.0],
        "description": (
            "Gamma applied to the [0,1] depth map after auto-normalization.  "
            ">1 compresses the far field (deepens contrast near the camera); "
            "<1 lifts it (stretches shadow detail)."
        ),
    },
    "black_level": {
        "type": "float",
        "default": 0.0,
        "range": [0.0, 1.0],
        "description": ("Normalization floor — depth values at or below this level map to 0 (far field)."),
    },
    "white_level": {
        "type": "float",
        "default": 1.0,
        "range": [0.0, 1.0],
        "description": ("Normalization ceiling — depth values at or above this level map to 1 (near field)."),
    },
    "invert": {
        "type": "bool",
        "default": False,
        "description": "Swap near/far (1 − depth) before grading.",
    },
}

POSE_DRAW_PARAMS: dict = {
    "keypoint_threshold": {
        "type": "float",
        "default": 0.5,
        "range": [0.0, 1.0],
        "description": "Confidence cutoff for drawing skeleton joints and keypoints.",
    },
    "joint_thickness": {
        "type": "int",
        "default": 4,
        "range": [1, 30],
        "description": "Half-width of the filled skeleton limb ellipses (pixels). Default matches "
        "the OpenPose rendering convention xinsir/controlnet-openpose-sdxl-1.0 was trained on.",
    },
    "keypoint_radius": {
        "type": "int",
        "default": 4,
        "range": [1, 30],
        "description": "Radius of keypoint dots (pixels). Default matches the OpenPose rendering "
        "convention xinsir/controlnet-openpose-sdxl-1.0 was trained on.",
    },
}

# Passthrough — zero parameters, intentionally.
# The input image is forwarded unchanged to the ControlNet; no pre-processing is applied.
# Use this when the source is already a conditioning map (depth pass, scribble, skeleton, …).
# Listed here so the empty contract is explicit rather than accidentally omitted.
PASSTHROUGH_PARAMS: dict = {}

# Segmentation — not in the production set today; defined here so future seg CNs align.
# mediapipe_segmentation already implements this exact set of parameters.
SEGMENTATION_PARAMS: dict = {
    "threshold": {
        "type": "float",
        "default": 0.5,
        "range": [0.0, 1.0],
        "description": "Mask binarization threshold.",
    },
    "blur_radius": {
        "type": "int",
        "default": 0,
        "range": [0, 20],
        "description": "Edge blur radius on the segmentation mask (pixels).",
    },
    "invert_mask": {
        "type": "bool",
        "default": False,
        "description": "Invert foreground/background.",
    },
}


# ---------------------------------------------------------------------------
# GPU helpers (operate on torch.Tensor, no CPU round-trip)
# ---------------------------------------------------------------------------

# torch.backends.cudnn.flags(...) snapshots/restores process-global state, not thread-local
# state. apply_edge_smoothness is called concurrently from ThreadPoolExecutor worker threads
# (preprocessing_orchestrator.py's parallel ControlNet dispatch), so unsynchronized enter/exit
# can leave the global cudnn.benchmark flag stuck in the wrong state. Serialize with a lock.
_EDGE_SMOOTHNESS_CUDNN_LOCK = threading.Lock()


def apply_edge_smoothness(t: torch.Tensor, strength: float) -> torch.Tensor:
    """Apply an adaptive separable Gaussian pre-blur to a grayscale / edge-map tensor.

    Designed to be inserted *before* the native Gaussian/Sobel block so that increasing
    `strength` progressively suppresses high-frequency texture, yielding sparser / softer
    edge maps.  strength=0 is a fast no-op (early return, no allocation).

    Args:
        t:        Input tensor.  Accepts (H, W), (C, H, W), or (1, C, H, W).
        strength: Blur intensity in [0, 1].  Maps to σ ∈ [0, 2] (3σ gives the kernel radius).
                  At strength=1: σ=2, radius=6, k_size=13.  cuDNN benchmarking is disabled
                  (under a lock) so a changing kernel size never re-autotunes.

    Returns:
        Blurred tensor with the same shape and dtype as *t*.
    """
    if strength <= 0.0:
        return t

    orig_shape = t.shape
    orig_dtype = t.dtype

    # Promote to float32 (1, C, H, W) for conv2d
    x = t.float()
    if x.dim() == 2:  # (H, W)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:  # (C, H, W)
        x = x.unsqueeze(0)
    # else already (1, C, H, W) — or (B, C, H, W); we treat as single image

    sigma = float(strength) * 2.0
    radius = max(1, int(math.ceil(3.0 * sigma)))
    k_size = 2 * radius + 1

    coords = torch.arange(k_size, dtype=torch.float32, device=t.device) - radius
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    c = x.shape[1]
    # Separable 1-D horizontal / vertical Gaussian convolutions
    k_h = kernel_1d.view(1, 1, k_size, 1).expand(c, 1, k_size, 1).contiguous()
    k_w = kernel_1d.view(1, 1, 1, k_size).expand(c, 1, 1, k_size).contiguous()

    # The kernel size follows the knob, so with the pipeline's global cudnn.benchmark=True
    # every new strength would be a new conv shape and trigger a ~750 ms cuDNN autotune
    # (a visible freeze while dragging the Smoothing knob). Use heuristic algo selection here,
    # guarded by a lock since the flag context manager mutates process-global state and this
    # function runs from concurrent ThreadPoolExecutor worker threads.
    with _EDGE_SMOOTHNESS_CUDNN_LOCK, torch.backends.cudnn.flags(enabled=True, benchmark=False):
        x = F.conv2d(x, k_h, padding=(radius, 0), groups=c)
        x = F.conv2d(x, k_w, padding=(0, radius), groups=c)

    # Restore original shape
    if len(orig_shape) == 2:
        x = x.squeeze(0).squeeze(0)
    elif len(orig_shape) == 3:
        x = x.squeeze(0)

    return x.to(dtype=orig_dtype)


def apply_depth_grade(
    depth: torch.Tensor,
    gamma: float = 1.0,
    black_level: float = 0.0,
    white_level: float = 1.0,
    invert: bool = False,
) -> torch.Tensor:
    """Apply normalization + gamma grade to a depth map in [0, 1].

    Operation order:
      1. Optional invert (1 − d) — swap near/far.
      2. Level remap: (d − black_level) / (white_level − black_level), clamped to [0, 1].
      3. Gamma: d^gamma (1.0 = identity).

    Args:
        depth:       Depth tensor in [0, 1], any shape.
        gamma:       Gamma exponent (1.0 = identity).
        black_level: New zero point — depth values at/below this map to 0.
        white_level: New full-scale point — depth values at/above this map to 1.
        invert:      Swap near/far before grading.

    Returns:
        Graded depth tensor with the same shape and dtype as *depth*.
    """
    orig_dtype = depth.dtype
    d = depth.float()

    if invert:
        d = 1.0 - d

    span = max(float(white_level) - float(black_level), 1e-6)
    d = ((d - float(black_level)) / span).clamp(0.0, 1.0)

    if abs(float(gamma) - 1.0) > 1e-6:
        d = d.pow(float(gamma))

    return d.clamp(0.0, 1.0).to(dtype=orig_dtype)


# ---------------------------------------------------------------------------
# NumPy helper (CPU / _process_core paths)
# ---------------------------------------------------------------------------


def apply_depth_grade_numpy(
    depth: np.ndarray,
    gamma: float = 1.0,
    black_level: float = 0.0,
    white_level: float = 1.0,
    invert: bool = False,
) -> np.ndarray:
    """NumPy equivalent of apply_depth_grade.  Expects *depth* in [0, 1] float."""
    d = depth.astype(np.float32)

    if invert:
        d = 1.0 - d

    span = max(float(white_level) - float(black_level), 1e-6)
    d = ((d - float(black_level)) / span).clip(0.0, 1.0)

    if abs(float(gamma) - 1.0) > 1e-6:
        d = np.power(d, float(gamma))

    return d.clip(0.0, 1.0)
