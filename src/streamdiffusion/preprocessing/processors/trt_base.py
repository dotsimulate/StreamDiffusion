"""
Shared TensorRT infrastructure for ControlNet preprocessors.

Provides:
  TensorRTEngine              — low-level TRT engine wrapper (load/activate/infer).
                                Extracted from the verbatim copies in depth_tensorrt.py
                                and pose_tensorrt.py; those files now import from here.

  SelfBuildingTRTPreprocessor — base class for preprocessors that self-build their TRT
                                engine from a torch model at first use.  Subclasses only
                                need to implement two hooks:
                                  _export_onnx(onnx_path)           — model-specific ONNX export
                                  _postprocess(engine_outputs) -> T  — GPU-only output shaping
                                plus three class attributes:
                                  engine_filename, onnx_filename, default_detect_resolution
"""

import logging
import threading
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from .base import BasePreprocessor


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional TRT / polygraphy imports
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import tensorrt as trt
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import engine_from_bytes

    numpy_to_torch_dtype_dict: Dict = {
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
    }
    if np.version.full_version >= "1.24.0":
        numpy_to_torch_dtype_dict[np.bool_] = torch.bool
    else:
        numpy_to_torch_dtype_dict[np.bool] = torch.bool  # type: ignore[attr-defined]

    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    numpy_to_torch_dtype_dict = {}


# ---------------------------------------------------------------------------
# Shared TensorRT engine wrapper
# ---------------------------------------------------------------------------


class TensorRTEngine:
    """
    Thin wrapper around a TensorRT ICudaEngine + IExecutionContext.

    Identical to the copies in depth_tensorrt.py and pose_tensorrt.py;
    those modules import this class instead of redefining it.
    """

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.tensors = OrderedDict()
        self._cuda_stream = None  # cached CUDA stream

    def load(self):
        logger.info(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self):
        self.context = self.engine.create_execution_context()
        self._cuda_stream = torch.cuda.current_stream().cuda_stream

    def allocate_buffers(self, device: str = "cuda"):
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)

            tensor = torch.empty(
                tuple(shape),
                dtype=numpy_to_torch_dtype_dict[dtype],
            ).to(device=device)
            self.tensors[name] = tensor

    def infer(self, feed_dict: dict, stream=None) -> OrderedDict:
        if stream is None:
            stream = self._cuda_stream

        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        success = self.context.execute_async_v3(stream)
        if not success:
            raise ValueError("TensorRTEngine: inference failed.")

        return self.tensors


# ---------------------------------------------------------------------------
# Self-building TRT preprocessor base
# ---------------------------------------------------------------------------


class SelfBuildingTRTPreprocessor(BasePreprocessor):
    """
    Base class for TRT preprocessors that build their own engine on first use.

    Subclass interface
    ------------------
    Class attributes (override in subclass):
        engine_filename           : str  = "engine.engine"
        onnx_filename             : str  = "engine.onnx"
        default_detect_resolution : int  = 512

    Abstract methods (implement in subclass):
        _export_onnx(onnx_path: Path) -> None
            Export the underlying torch model to ONNX at onnx_path.

        _postprocess(engine_outputs: dict) -> torch.Tensor
            Convert raw TRT output tensors to a CHW GPU tensor in [0, 1].

    Engine-path precedence
    ----------------------
    1. params["engine_path"]  — TD always supplies this via StreamDiffusionExt config-gen
    2. <repo_root>/engines/preprocessors/<engine_filename>  — offline fallback

    Build-registry hook
    -------------------
    td_manager._ensure_preprocessor_engines calls:
        cls.build_engine_for_path(engine_path, device)
    which instantiates the preprocessor and runs _ensure_engine().
    """

    gpu_native = True

    # Subclasses set these:
    engine_filename: str = "engine.engine"
    onnx_filename: str = "engine.onnx"
    default_detect_resolution: int = 512

    def __init__(self, **kwargs):
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT and polygraphy are required for TRT preprocessors. "
                "Install with: pip install tensorrt polygraphy"
            )
        super().__init__(**kwargs)
        self._engine: Optional[TensorRTEngine] = None
        self._engine_lock = threading.Lock()

    # ------------------------------------------------------------------
    # PIL fallback path — goes through tensor for GPU residency
    # ------------------------------------------------------------------

    def _process_core(self, image: Image.Image) -> Image.Image:
        tensor = self.pil_to_tensor(image)
        result = self._process_tensor_core(tensor)
        return self.tensor_to_pil(result)

    # ------------------------------------------------------------------
    # Engine path resolution
    # ------------------------------------------------------------------

    def _get_engine_path(self) -> Path:
        from_params = self.params.get("engine_path")
        if from_params:
            return Path(from_params)
        # Default fallback: <repo_root>/engines/preprocessors/<engine_filename>
        repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        return repo_root / "engines" / "preprocessors" / self.engine_filename

    def _get_onnx_path(self, engine_path: Path) -> Path:
        return engine_path.parent / self.onnx_filename

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    @property
    def engine(self) -> TensorRTEngine:
        """Lazy-load the TRT engine (double-checked locking)."""
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    engine_path = self._get_engine_path()
                    self._ensure_engine()
                    if not engine_path.exists():
                        raise FileNotFoundError(
                            f"{self.__class__.__name__}: engine not found after build: {engine_path}"
                        )
                    self._engine = TensorRTEngine(str(engine_path))
                    self._engine.load()
                    self._engine.activate()
                    self._engine.allocate_buffers(device=self.device)
        return self._engine

    def _ensure_engine(self) -> None:
        """Build the TRT engine from scratch if it doesn't exist yet."""
        engine_path = self._get_engine_path()
        if engine_path.exists():
            return

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_path = self._get_onnx_path(engine_path)

        try:
            logger.info(f"{self.__class__.__name__}: exporting ONNX → {onnx_path}")
            self._export_onnx(onnx_path)
            logger.info(f"{self.__class__.__name__}: building TRT engine → {engine_path}")
            self._build_tensorrt_engine(onnx_path, engine_path)
            logger.info(f"{self.__class__.__name__}: engine built ({engine_path.stat().st_size / 1024 / 1024:.1f} MB)")
        finally:
            # Always clean up the ONNX intermediary
            if onnx_path.exists():
                onnx_path.unlink()

    def _build_tensorrt_engine(self, onnx_path: Path, engine_path: Path) -> None:
        """Build TRT engine from ONNX using trt.Builder with FP16 + dynamic shapes."""
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        network = builder.create_network()
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"{self.__class__.__name__}: ONNX parse failed: {errors}")

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        res = self.default_detect_resolution
        profile.set_shape(
            "input",
            (1, 3, res // 2, res // 2),  # min
            (1, 3, res, res),  # opt
            (1, 3, res * 2, res * 2),  # max
        )
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"{self.__class__.__name__}: TRT engine build returned None")

        with open(engine_path, "wb") as f:
            f.write(serialized)

    # ------------------------------------------------------------------
    # Subclass hooks (must override)
    # ------------------------------------------------------------------

    @abstractmethod
    def _export_onnx(self, onnx_path: Path) -> None:
        """Export the underlying torch model to ONNX at onnx_path."""

    @abstractmethod
    def _postprocess(self, engine_outputs: dict) -> torch.Tensor:
        """Convert raw TRT outputs to a CHW GPU tensor in [0, 1]."""

    # ------------------------------------------------------------------
    # Core tensor processing
    # ------------------------------------------------------------------

    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize → TRT infer → postprocess.  All on GPU, no PIL round-trip.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if not image_tensor.is_cuda:
            image_tensor = image_tensor.to(self.device)

        detect_resolution = self.params.get("detect_resolution", self.default_detect_resolution)
        image_resized = F.interpolate(
            image_tensor.float(),
            size=(detect_resolution, detect_resolution),
            mode="bilinear",
            align_corners=False,
        )

        # Match the dtype the engine expects on its input tensor
        engine_input = self.engine.tensors.get("input")
        if engine_input is not None and image_resized.dtype != engine_input.dtype:
            image_resized = image_resized.to(dtype=engine_input.dtype)

        cuda_stream = torch.cuda.current_stream().cuda_stream
        outputs = self.engine.infer({"input": image_resized}, cuda_stream)
        result = self._postprocess(outputs)

        # Ensure result is CHW (strip batch dim if present)
        if result.dim() == 4:
            result = result.squeeze(0)
        return result

    # ------------------------------------------------------------------
    # Class-level build hook called by td_manager._ensure_preprocessor_engines
    # ------------------------------------------------------------------

    @classmethod
    def build_engine_for_path(cls, engine_path: str, device: str = "cuda") -> bool:
        """
        Build (export + compile) the TRT engine and write it to engine_path.

        Called by td_manager._ensure_preprocessor_engines for preprocessors
        that use the 'self_build' strategy in the build_registry.

        Returns True on success, False on failure.
        """
        try:
            instance = cls(engine_path=engine_path, device=device)
            instance._ensure_engine()
            return Path(engine_path).exists()
        except Exception as exc:
            logger.error(f"{cls.__name__}.build_engine_for_path failed: {exc}")
            import traceback

            traceback.print_exc()
            return False

    def __del__(self):
        if hasattr(self, "_engine") and self._engine is not None:
            del self._engine
