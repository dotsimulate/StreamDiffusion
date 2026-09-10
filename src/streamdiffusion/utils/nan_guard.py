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
  3. A later call reads that scalar via a non-blocking `.query()`. Pending
     checks accumulate until a readback slot is free, so transient failures
     survive a busy stream. The verdict can lag by several calls; it is not
     a verdict on the tensor being sanitized now.
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
        self._pending = False
        self._bad_accum: Optional[torch.Tensor] = None

    def sanitize_(self, tensor: torch.Tensor) -> bool:
        """Sanitize `tensor` in place (NaN/+-Inf -> 0.0), unconditionally.

        Returns the most recently completed CUDA verdict, or the current
        verdict on CPU. CUDA readbacks may cover multiple calls and lag by
        several calls. Use this for recovery, not to decide whether the
        current tensor is safe to store: sanitization already makes it finite.
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
            self._bad_accum = torch.zeros((), dtype=torch.bool, device=tensor.device)
        elif self._pending and evt.query():
            self._last_bad = bool(bad_pin.item())
            self._pending = False
            if self._last_bad:
                self._warn()

        bad_this_frame = (~torch.isfinite(tensor)).any()
        tensor.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        # Never overwrite/re-record an in-flight readback: a busy GPU could
        # otherwise postpone the verdict forever and lose a transient failure.
        # Accumulate all calls (including residual groups) until it is consumed.
        self._bad_accum.logical_or_(bad_this_frame)
        if not self._pending:
            bad_pin.copy_(self._bad_accum.view(1), non_blocking=True)
            self._bad_accum.zero_()
            evt.record()
            self._pending = True

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
