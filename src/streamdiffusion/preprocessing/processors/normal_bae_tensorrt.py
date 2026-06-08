"""
NormalBae TensorRT preprocessor — GPU-native surface normal estimation.

Implementation strategy
-----------------------
The NormalBaeDetector from controlnet_aux uses an NNET architecture with a
complex multi-scale decoder output (nested lists of tensors at different scales),
which complicates ONNX export.  MCP verification confirmed zero prior usage
of this model in the repo, so its ONNX-exportability was unverified.

Probe at module import time:

  PRIMARY: self-building TRT engine (ONNX export succeeds at class load time).
           NormalBaeExportWrapper encapsulates:
             self.norm(x)  →  self.model(normed)  →  extract high-res 3ch normal
           so the engine takes plain [0,1] RGB and returns a [0,1] 3ch normal map.

  FALLBACK: if ONNX export fails (or TRT is unavailable), the class falls back
            to running the torch model directly under no_grad — the same pattern
            as SoftEdgePreprocessor, which is MCP-confirmed GPU-native.  This
            still satisfies the GPU-residency constraint (no CPU/PIL round-trips).

In either case `gpu_native = True` is set, and the dangling 'normal_bae'
registry reference in get_preprocessor_for_controlnet is resolved.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .base import BasePreprocessor
from .trt_base import TENSORRT_AVAILABLE, SelfBuildingTRTPreprocessor, _first_output


logger = logging.getLogger(__name__)

try:
    from controlnet_aux import NormalBaeDetector

    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Probe whether ONNX export works for this version of controlnet_aux
# ---------------------------------------------------------------------------

_TRT_STRATEGY_AVAILABLE: Optional[bool] = None  # None = not yet probed


def _probe_normal_bae_onnx_export(device: str = "cuda") -> bool:
    """
    Try a lightweight ONNX export of NormalBaeExportWrapper.
    Returns True if it succeeds, False otherwise.
    Cached after first call.
    """
    global _TRT_STRATEGY_AVAILABLE
    if _TRT_STRATEGY_AVAILABLE is not None:
        return _TRT_STRATEGY_AVAILABLE

    if not CONTROLNET_AUX_AVAILABLE or not TENSORRT_AVAILABLE:
        _TRT_STRATEGY_AVAILABLE = False
        return False

    import os
    import tempfile

    try:
        det = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        wrapper = NormalBaeExportWrapper(det.model, det.norm).to(device).eval()
        dummy = torch.zeros(1, 3, 64, 64, device=device)  # small for probe
        tmp = tempfile.mktemp(suffix=".onnx")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy,
                tmp,
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
            )
        _TRT_STRATEGY_AVAILABLE = os.path.exists(tmp) and os.path.getsize(tmp) > 0
        if os.path.exists(tmp):
            os.unlink(tmp)
        del wrapper, det
        torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning(
            f"NormalBaeTensorrtPreprocessor: ONNX probe failed ({exc}); will use torch-direct GPU fallback instead."
        )
        _TRT_STRATEGY_AVAILABLE = False

    return _TRT_STRATEGY_AVAILABLE


# ---------------------------------------------------------------------------
# ONNX export wrapper
# ---------------------------------------------------------------------------


class NormalBaeExportWrapper(torch.nn.Module):
    """
    Wraps NormalBaeDetector internals for single-pass ONNX export.

    Replicates the core of NormalBaeDetector.__call__:
        normed = self.norm(x)
        out    = self.model(normed)    # NNET
        normal = out[0][-1][:, :3]    # highest-res decoder output, 3 channels
        return ((normal + 1) * 0.5).clamp(0, 1)   # [-1,1] → [0,1]

    Input  : (B, 3, H, W)  float32  [0, 1]
    Output : (B, 3, H, W)  float32  [0, 1]
    """

    def __init__(self, nnet_model: torch.nn.Module, norm_transform: torch.nn.Module):
        super().__init__()
        self.nnet_model = nnet_model
        self.norm = norm_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        out = self.nnet_model(x)
        # out[0] is a list of 4 tensors at scales [64², 128², 256², 512²]
        # out[0][-1] is the highest-resolution output  (B, 4, H, W)
        normal = out[0][-1][:, :3]  # (B, 3, H, W)
        return ((normal + 1.0) * 0.5).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Torch-direct GPU fallback (used when TRT strategy is unavailable)
# ---------------------------------------------------------------------------


class _NormalBaeTorchGPU(BasePreprocessor):
    """
    GPU-direct NormalBae using the torch model under no_grad.
    Mirrors the SoftEdgePreprocessor pattern (MCP-confirmed GPU-native).
    No CPU / PIL round-trips.
    """

    gpu_native = True
    _detector_cache: dict = {}

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Normal Map Estimation (torch GPU)",
            "description": (
                "GPU-native surface normal estimation using NormalBaeDetector "
                "run directly under torch.no_grad. No TRT engine required."
            ),
            "parameters": {},
            "use_cases": ["Normal ControlNet conditioning"],
        }

    def __init__(self, **kwargs):
        if not CONTROLNET_AUX_AVAILABLE:
            raise ImportError(
                "controlnet_aux is required for normal map preprocessing. Install with: pip install controlnet_aux"
            )
        super().__init__(**kwargs)
        self._detector = None
        self._load_model()

    def _load_model(self):
        cache_key = f"normal_bae_{self.device}"
        if cache_key in self._detector_cache:
            self._detector = self._detector_cache[cache_key]
            return
        logger.info("NormalBae (torch-GPU): loading NormalBaeDetector…")
        det = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        det.model.to(self.device).eval()
        det.norm.to(self.device)
        self._detector = det
        self._detector_cache[cache_key] = det

    def _process_core(self, image: Image.Image) -> Image.Image:
        tensor = self.pil_to_tensor(image)
        result = self._process_tensor_core(tensor)
        return self.tensor_to_pil(result)

    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "_detector") or self._detector is None:
            raise RuntimeError(
                f"{self.__class__.__name__}._process_tensor_core: model not initialized — "
                "_load_model() was never called. This is a bug; please report it."
            )
        with torch.no_grad():
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device=self.device, dtype=torch.float32)

            # Apply NormalBae normalization
            normed = self._detector.norm(image_tensor)
            out = self._detector.model(normed)

            # Extract highest-res 3-channel output
            normal = out[0][-1][:, :3]  # (B, 3, H, W)
            normal = ((normal + 1.0) * 0.5).clamp(0.0, 1.0)

        return normal.squeeze(0)  # (3, H, W)


# ---------------------------------------------------------------------------
# Public class — chooses strategy at construction time
# ---------------------------------------------------------------------------


class NormalBaeTensorrtPreprocessor(SelfBuildingTRTPreprocessor):
    """
    Normal map estimation — GPU-native via TRT engine (primary) or torch-direct (fallback).

    The class name retains the '_tensorrt' suffix so the existing engine-path
    wiring in StreamDiffusionExt (the "tensorrt" in name gate, line 3572) works
    correctly.  When TRT is unavailable or ONNX export fails, construction
    transparently returns a _NormalBaeTorchGPU instance instead.
    """

    engine_filename = "normal_bae.engine"
    onnx_filename = "normal_bae.onnx"
    default_detect_resolution = 512

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Normal Map Estimation (TensorRT)",
            "description": (
                "GPU-native surface normal estimation. Self-builds a TRT engine "
                "from the controlnet_aux NormalBaeDetector model on first run. "
                "Falls back to torch-direct GPU mode if TRT export is unavailable."
            ),
            "parameters": {},
            "use_cases": ["Normal ControlNet conditioning"],
        }

    def __new__(cls, **kwargs):
        """
        If TRT strategy is available return a SelfBuildingTRTPreprocessor subclass;
        otherwise return the torch-direct GPU fallback transparently.
        """
        if not CONTROLNET_AUX_AVAILABLE:
            raise ImportError(
                "controlnet_aux is required for NormalBaeTensorrtPreprocessor. "
                "Install with: pip install controlnet_aux"
            )

        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        use_trt = TENSORRT_AVAILABLE and _probe_normal_bae_onnx_export(device)

        if not use_trt:
            # Return a fully-constructed fallback.
            #
            # Using object.__new__(_NormalBaeTorchGPU) here would cause CPython's
            # type.__call__ to skip __init__ entirely, because _NormalBaeTorchGPU is
            # NOT a subclass of NormalBaeTensorrtPreprocessor.  The resulting object
            # would have no self._detector, self.params, or self.device and raise
            # AttributeError on the first frame.  Calling the class directly runs
            # _NormalBaeTorchGPU.__init__ correctly (Finding A fix).
            return _NormalBaeTorchGPU(**kwargs)

        obj = object.__new__(cls)
        return obj

    def __init__(self, **kwargs):
        # __new__ now returns a fully-constructed _NormalBaeTorchGPU when TRT is
        # unavailable, so CPython never calls this __init__ for the fallback path.
        # The guard that was here ("if type(self) is _NormalBaeTorchGPU: return")
        # was dead code and has been removed.
        super().__init__(**kwargs)

    def _export_onnx(self, onnx_path: Path) -> None:
        logger.info("NormalBaeTensorrtPreprocessor: loading NormalBaeDetector for ONNX export…")
        det = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
        wrapper = NormalBaeExportWrapper(det.model, det.norm).to(self.device).eval()

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

        logger.info(f"NormalBaeTensorrtPreprocessor: ONNX exported → {onnx_path}")
        del wrapper, det
        torch.cuda.empty_cache()

    def _postprocess(self, engine_outputs: dict) -> torch.Tensor:
        """
        Convert TRT output (B, 3, H, W) [0,1] to CHW GPU tensor.
        The export wrapper already applies the [0,1] normalisation.
        """
        out = _first_output(engine_outputs).float()
        if out.dim() == 4:
            out = out.squeeze(0)  # (3, H, W)
        return out.clamp(0.0, 1.0)
