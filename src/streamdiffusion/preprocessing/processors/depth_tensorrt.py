# NOTE: ported from https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import BasePreprocessor
from .trt_base import TENSORRT_AVAILABLE, TensorRTEngine  # shared engine wrapper


class DepthAnythingTensorrtPreprocessor(BasePreprocessor):
    gpu_native = True  # _process_tensor_core runs full pipeline on GPU — no PIL round-trip
    """
    Depth Anything TensorRT preprocessor for ControlNet

    Uses TensorRT-optimized Depth Anything model for fast depth estimation.
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Depth Estimation (TensorRT)",
            "description": "Fast TensorRT-optimized depth estimation using Depth Anything model. Significantly faster than standard depth estimation.",
            "parameters": {},
            "use_cases": ["High-performance depth estimation", "Real-time applications", "3D-aware generation"],
        }

    def __init__(self, engine_path: str = None, detect_resolution: int = 518, image_resolution: int = 512, **kwargs):
        """
        Initialize TensorRT depth preprocessor

        Args:
            engine_path: Path to TensorRT engine file
            detect_resolution: Resolution for depth detection (should match engine input)
            image_resolution: Output image resolution
            **kwargs: Additional parameters
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT and polygraphy libraries are required for TensorRT depth preprocessing. "
                "Install them with: pip install tensorrt polygraphy"
            )

        super().__init__(
            engine_path=engine_path, detect_resolution=detect_resolution, image_resolution=image_resolution, **kwargs
        )

        self._engine = None

    @property
    def engine(self):
        """Lazy loading of the TensorRT engine"""
        if self._engine is None:
            engine_path = self.params.get("engine_path")
            if engine_path is None:
                raise ValueError(
                    "engine_path is required for TensorRT depth preprocessing. "
                    "Please provide it in the preprocessor_params config."
                )

            if not os.path.exists(engine_path):
                raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

            print(f"Loading TensorRT depth estimation engine: {engine_path}")

            self._engine = TensorRTEngine(engine_path)
            self._engine.load()
            self._engine.activate()
            self._engine.allocate_buffers()

        return self._engine

    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Apply TensorRT depth estimation to the input image
        """
        detect_resolution = self.params.get("detect_resolution", 518)

        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

        image_resized = F.interpolate(
            image_tensor, size=(detect_resolution, detect_resolution), mode="bilinear", align_corners=False
        )

        if torch.cuda.is_available():
            image_resized = image_resized.cuda()

        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = self.engine.infer({"input": image_resized}, cuda_stream)
        depth = result["output"]

        depth = np.reshape(depth.cpu().numpy(), (detect_resolution, detect_resolution))
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        original_size = image.size
        depth = cv2.resize(depth, original_size)

        depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        result = Image.fromarray(depth_rgb)

        return result

    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor directly on GPU to avoid CPU transfers
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if not image_tensor.is_cuda:
            image_tensor = image_tensor.cuda()

        detect_resolution = self.params.get("detect_resolution", 518)

        image_resized = torch.nn.functional.interpolate(
            image_tensor, size=(detect_resolution, detect_resolution), mode="bilinear", align_corners=False
        )

        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = self.engine.infer({"input": image_resized}, cuda_stream)
        depth_tensor = result["output"]

        depth_tensor = depth_tensor.squeeze() if depth_tensor.dim() > 2 else depth_tensor
        depth_min, depth_max = depth_tensor.min(), depth_tensor.max()
        depth_normalized = (depth_tensor - depth_min) / (depth_max - depth_min)

        return depth_normalized.repeat(3, 1, 1).unsqueeze(0)
