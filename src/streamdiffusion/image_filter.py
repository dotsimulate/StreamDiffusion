import random
from typing import Optional

import torch
import torch.nn.functional as F

from streamdiffusion.utils.nan_guard import NanGuard


class SimilarImageFilter:
    """Stochastic frame-skip filter (StreamDiffusion §3.3).

    NOTE: the StreamDiffusion paper describes cosine similarity in latent space to
    compute the skip probability.  This implementation uses pixel-space MSE for
    simplicity and speed.  `threshold` is remapped via ``mse_threshold = 1 − threshold``,
    so ``threshold=0.98`` means frames with MSE < 0.02 have a non-zero skip probability.
    The stochastic skip logic uses a 1-frame delay (skip probability is computed
    asynchronously and applied on the next frame) to avoid GPU stalls.
    """

    def __init__(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.threshold = threshold
        self._mse_threshold: float = max(1e-7, 1.0 - threshold)
        self.max_skip_frame = max_skip_frame
        self.skip_count = 0
        self.prev_tensor: Optional[torch.Tensor] = None
        self._skip_prob_pin: Optional[torch.Tensor] = None  # pinned CPU scalar (lazy init)
        self._skip_evt: Optional[torch.cuda.Event] = None  # marks when the copy_ below has landed
        self._last_skip_prob: float = 0.0  # fallback while that copy is still in flight
        # NaN guard 8: clamp() below does not filter NaN (NaN < x is always False), so a
        # NaN mse used to survive into skip_prob and latch the always-skip branch forever
        # (comment here previously claimed "never garbage" -- see nan_guard.py docstring).
        self._nan_guard_skip_prob = NanGuard("image_filter.skip_prob")

    def __call__(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        # First frame: allocate buffers, always pass through
        if self.prev_tensor is None:
            self.prev_tensor = x.detach().clone()
            self._skip_prob_pin = torch.zeros(1, dtype=torch.float32, device="cpu").pin_memory()
            self._skip_evt = torch.cuda.Event()
            return x

        # Step 1: Read PREVIOUS frame's async result (CPU pinned read, no GPU sync).
        # By design this is read one frame late (see the class docstring). There is no
        # unconditional device sync in StreamDiffusion.__call__ to lean on here: the only
        # host sync in pipeline.py (end.synchronize()) is gated behind both
        # self.similar_image_filter and a 16-frame sample cadence for the
        # inference_time_ema heuristic (pipeline.py ~1430-1440) -- it exists to bound the
        # cost of EMA timing, not to guarantee this copy has retired. So the copy's
        # completion is checked explicitly instead, via _skip_evt (recorded on the
        # producing stream right after the copy_() below) -- .query() is non-blocking, so
        # this adds no host stall. If the copy hasn't landed yet, fall back to the last
        # value that did; _skip_prob_pin is zero-initialised above and only ever overwritten
        # with a value clamped to [0, 1] AND NaN-sanitized (Step 2's NaN guard 8 below) --
        # clamp() alone does not filter NaN, so a fallback read is always a valid, merely
        # stale, probability -- never garbage.
        if self._skip_evt is not None and self._skip_evt.query():
            self._last_skip_prob = self._skip_prob_pin.item()
        skip_prob = self._last_skip_prob

        # Step 2: Launch THIS frame's MSE computation (GPU kernel, no sync).
        mse = F.mse_loss(self.prev_tensor, x)
        if self._mse_threshold < 1e-6:
            # threshold >= 1.0 → "never skip" mode
            gpu_skip = torch.zeros(1, device=x.device, dtype=torch.float32)
        else:
            gpu_skip = torch.clamp(1.0 - mse / self._mse_threshold, min=0.0, max=1.0)
        # NaN guard 8: sanitize in place before the async copy below -- NaN -> 0.0 means
        # "never skip" (self-healing: keeps processing frames) rather than latching into
        # the always-skip branch (Step 3's `skip_prob < random.random()` is always False
        # for NaN). Unconditional, branch-free, no extra sync (mirrors nan_guard.py).
        self._nan_guard_skip_prob.sanitize_(gpu_skip)
        # Async copy result to pinned CPU buffer for NEXT frame to read
        self._skip_prob_pin.copy_(gpu_skip.view(1), non_blocking=True)
        self._skip_evt.record()  # marks when the copy above has actually landed

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
