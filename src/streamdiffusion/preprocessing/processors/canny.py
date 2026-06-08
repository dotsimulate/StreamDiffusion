import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import BasePreprocessor


class CannyPreprocessor(BasePreprocessor):
    """
    Canny edge detection preprocessor for ControlNet

    Detects edges in the input image using the Canny edge detection algorithm.
    """

    gpu_native = True  # _process_tensor_core uses conv2d — no CPU/PIL round-trip

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Canny Edge Detection",
            "description": "Detects edges in the input image using the Canny edge detection algorithm. Good for line art and architectural images.",
            "parameters": {
                "low_threshold": {
                    "type": "int",
                    "default": 100,
                    "range": [1, 255],
                    "description": "Lower threshold for edge detection. Lower values detect more edges.",
                },
                "high_threshold": {
                    "type": "int",
                    "default": 200,
                    "range": [1, 255],
                    "description": "Upper threshold for edge detection. Higher values are more selective.",
                },
            },
            "use_cases": ["Line art", "Architecture", "Technical drawings", "Clean edge detection"],
        }

    def __init__(self, low_threshold: int = 100, high_threshold: int = 200, **kwargs):
        super().__init__(low_threshold=low_threshold, high_threshold=high_threshold, **kwargs)
        # GPU kernel tensors — lazily initialized on first _process_tensor_core call
        self._gauss_k: torch.Tensor | None = None
        self._sobel_x: torch.Tensor | None = None
        self._sobel_y: torch.Tensor | None = None

    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Apply Canny edge detection to the input image
        """
        image_np = np.array(image)

        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        low_threshold = self.params.get("low_threshold", 100)
        high_threshold = self.params.get("high_threshold", 200)

        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(edges_rgb)

    def _build_gpu_kernels(self, device: torch.device) -> None:
        """Lazily build and cache fixed conv kernels on the target device."""
        self._gauss_k = (
            torch.tensor(
                [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]],
                dtype=torch.float32,
                device=device,
            ).view(1, 1, 5, 5)
            / 256.0
        )
        self._sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 3, 3)
        self._sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 3, 3)

    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """GPU-native Canny approximation: Gaussian blur → Sobel magnitude → double-threshold.

        Replaces the GPU→CPU→GPU cv2.Canny round-trip that existed on this path.
        Output is visually comparable to cv2.Canny at the same thresholds; not
        pixel-identical (no full NMS, hysteresis approximated via 3×3 max-pool dilation).
        Grounding: CUDA HB Ch.11 p.353 — avoid CPU round-trip when data is already on device.
        """
        device = image_tensor.device

        if self._gauss_k is None or self._gauss_k.device != device:
            self._build_gpu_kernels(device)

        # Grayscale conversion
        if image_tensor.shape[0] == 3:
            gray = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
        else:
            gray = image_tensor[0] if image_tensor.shape[0] >= 1 else image_tensor

        # (1, 1, H, W) for conv2d; float32 for numerical precision in gradient computation
        gray_4d = gray.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)

        # Gaussian blur (5×5, σ≈1.4) — reduces noise before gradient computation
        blurred = F.conv2d(gray_4d, self._gauss_k, padding=2)

        # Sobel gradient magnitude
        Gx = F.conv2d(blurred, self._sobel_x, padding=1)
        Gy = F.conv2d(blurred, self._sobel_y, padding=1)
        mag = torch.sqrt(Gx * Gx + Gy * Gy).squeeze(0).squeeze(0)
        # Normalize by a fixed reference (≈max Sobel response for a full-contrast step edge
        # in [0,1]-range input) rather than per-frame amax.  Per-frame amax makes the
        # low/high thresholds relative to the strongest gradient in each frame, which
        # causes threshold semantics to shift when a frame has low contrast — inconsistent
        # frame-to-frame and diverging from cv2.Canny's absolute threshold semantics.
        mag = (mag / 4.0).clamp(0.0, 1.0)

        # Double threshold + single-step hysteresis (max-pool dilation of strong edges)
        low_t = self.params.get("low_threshold", 100) / 255.0
        high_t = self.params.get("high_threshold", 200) / 255.0
        strong = (mag >= high_t).float()
        weak = ((mag >= low_t) & (mag < high_t)).float()
        strong_dilated = (
            F.max_pool2d(strong.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
        )
        edges = (strong + weak * strong_dilated).clamp(0.0, 1.0).to(dtype=self.dtype)

        return edges.unsqueeze(0).repeat(3, 1, 1)  # contiguous CHW; expand() is non-contiguous
