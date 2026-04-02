import random
from typing import Optional

import torch
import torch.nn.functional as F


class SimilarImageFilter:
    def __init__(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.threshold = threshold
        self._mse_threshold: float = max(1e-7, 1.0 - threshold)
        self.max_skip_frame = max_skip_frame
        self.skip_count = 0
        self.prev_tensor: Optional[torch.Tensor] = None
        self._skip_prob_pin: Optional[torch.Tensor] = None  # pinned CPU scalar (lazy init)

    def __call__(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        # First frame: allocate buffers, always pass through
        if self.prev_tensor is None:
            self.prev_tensor = x.detach().clone()
            self._skip_prob_pin = torch.zeros(1, dtype=torch.float32, device="cpu").pin_memory()
            return x

        # Step 1: Read PREVIOUS frame's async result (CPU pinned read, no GPU sync).
        # torch.cuda.synchronize() in StreamDiffusion.__call__ (pipeline.py)
        # guarantees the non_blocking copy from the previous frame has completed.
        skip_prob = self._skip_prob_pin.item()

        # Step 2: Launch THIS frame's MSE computation (GPU kernel, no sync).
        mse = F.mse_loss(self.prev_tensor, x)
        if self._mse_threshold < 1e-6:
            # threshold >= 1.0 → "never skip" mode
            gpu_skip = torch.zeros(1, device=x.device, dtype=torch.float32)
        else:
            gpu_skip = torch.clamp(1.0 - mse / self._mse_threshold, min=0.0, max=1.0)
        # Async copy result to pinned CPU buffer for NEXT frame to read
        self._skip_prob_pin.copy_(gpu_skip.view(1), non_blocking=True)

        # Step 3: Decide based on PREVIOUS frame's probability (1-frame delay, no stall)
        if skip_prob < random.random():
            self.prev_tensor.copy_(x)  # in-place update, no allocation
            self.skip_count = 0
            return x
        else:
            if self.skip_count > self.max_skip_frame:
                self.skip_count = 0
                self.prev_tensor.copy_(x)
                return x
            else:
                self.skip_count += 1
                return None

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold
        self._mse_threshold = max(1e-7, 1.0 - threshold)

    def set_max_skip_frame(self, max_skip_frame: float) -> None:
        self.max_skip_frame = max_skip_frame
