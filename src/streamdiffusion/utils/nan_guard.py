"""NaN/Inf detection and sanitization for hot-path GPU buffers.

Context: nothing on the streaming hot path ever checked for non-finite values.
A single degenerate input (e.g. all prompt weights summing to zero -- see
_normalize_weights in stream_parameter_updater.py) could turn a tensor NaN,
and several cross-frame recurrences (stock_noise, x_t_latent_buffer, the
ControlNet residual EMA, the FX feedback canvas) then latch that NaN forever:
torch.clamp does not filter NaN, and every comparison against NaN is False, so
existing ".clamp(...)" callers that read as "safety" did not actually guard
anything. The stream kept running with no exception (the only try/except on
this path catches OOM and shape-mismatch RuntimeErrors) and rendered solid
black until a full re-prepare.

Mechanism (mirrors SimilarImageFilter in image_filter.py -- see its docstring
for the full rationale): a per-frame host-side branch/sync on freshly computed
GPU state is a documented anti-pattern in this codebase (see
docs/PMPP_deep_dive_verification_2026-07-29.md and
docs/perf_bestpractices_audit_2026-07-10.md). So:

  1. Sanitize unconditionally -- one `torch.nan_to_num_` elementwise kernel,
     every call, no branch, no readback needed for this part.
  2. Detect on-device (`~isfinite(...).any()`), async-copy the verdict into a
     pinned CPU scalar, and mark completion with a `torch.cuda.Event`.
  3. The NEXT call reads that scalar via a non-blocking `.query()` -- so the
     "was last frame bad" signal used to gate any recovery action (e.g.
     resetting stock_noise from init_noise) is always one frame stale, never
     a host stall.
  4. Log first occurrence at WARNING (loud -- this is exactly the silent
     failure mode that made the bug hard to find), subsequent occurrences at
     DEBUG, mirroring wrapper.py's `_cn_ipc_export_warned` idiom.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class NanGuard:
    """One instance per guarded buffer/site -- state (pinned scalar, event,
    warn-once flag) is not shared across sites, and the constructor's `name`
    is what shows up in the WARNING so the specific poisoned buffer is named.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._bad_pin: Optional[torch.Tensor] = None  # pinned CPU scalar (lazy init)
        self._evt: Optional[torch.cuda.Event] = None
        self._last_bad: bool = False
        self._warned: bool = False

    def sanitize_(self, tensor: torch.Tensor) -> bool:
        """Sanitize `tensor` in place (NaN/+-Inf -> 0.0), unconditionally.

        Returns whether the PREVIOUS call's tensor was non-finite (1-frame
        delayed on CUDA tensors, exactly like SimilarImageFilter.skip_prob) --
        callers use this to gate a cross-frame recovery action without ever
        forcing a fresh sync to know THIS frame's verdict early.
        """
        if not tensor.is_cuda:
            # CPU tensors (unit tests, CPU-only fallback configs): no async
            # plumbing needed or possible (no CUDA events) -- check inline.
            bad = bool((~torch.isfinite(tensor)).any())
            tensor.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if bad:
                self._warn()
            self._last_bad = bad
            return self._last_bad

        bad_pin = self._bad_pin
        evt = self._evt
        if bad_pin is None or evt is None:
            bad_pin = torch.zeros(1, dtype=torch.float32, device="cpu").pin_memory()
            evt = torch.cuda.Event()
            self._bad_pin, self._evt = bad_pin, evt
        elif evt.query():
            self._last_bad = bool(bad_pin.item())
            if self._last_bad:
                self._warn()

        bad_this_frame = (~torch.isfinite(tensor)).any()
        tensor.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        bad_pin.copy_(bad_this_frame.view(1).float(), non_blocking=True)
        evt.record()

        return self._last_bad

    def _warn(self) -> None:
        if not self._warned:
            logger.warning(
                "NanGuard(%s): non-finite values detected and sanitized (NaN/Inf -> 0.0); "
                "further occurrences on this buffer are logged at DEBUG",
                self.name,
            )
            self._warned = True
        else:
            logger.debug("NanGuard(%s): non-finite values detected and sanitized", self.name)
