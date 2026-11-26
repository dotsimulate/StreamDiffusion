"""Source buffers and failure recovery for the CUDA-IPC output exporters (cuda-link >= 2.2.0).

The wrapper owns two cuda-link ``Exporter`` objects: the main output and the ControlNet preview.
Both export from GPU buffers that the wrapper packs every frame. This module holds the two
pieces of state that each exporter needs:

* ``PackRing``: the BGRA source buffers. With blocking export (the default) ``export()`` has
  copied the buffer before it returns, so one buffer is reused every frame. With async export
  (``CUDALINK_EXPORT_SYNC=0``) the copy is only queued, so each ring slot gets its own buffer,
  and a buffer is rewritten only after ``Exporter.slot_copy_done()`` reports that the copy
  from it completed (ADR-0001, source-buffer lifetime).
* ``ExportRecovery``: what to do after ``FrameOutcome.FAILED``. It calls ``Exporter.recover()``
  and follows the policy of the cuda-link 2.2.0 ``.tox`` Sender.
"""

from __future__ import annotations

import enum
import logging
import time
from typing import Any, Callable, List, Optional

import torch

logger = logging.getLogger(__name__)

# CUDA errors that the CUDA runtime header marks as permanent for the context: the process must
# restart (cudaErrorIllegalAddress 700, LaunchTimeout 702, Assert 710, and 714-719).
PERMANENT_CUDA_ERRORS = frozenset({700, 702, 710, 714, 715, 716, 717, 718, 719})


class PackRing:
    """GPU HWC uint8 BGRA source buffers for one Exporter.

    ``acquire()`` returns the buffer to pack the next frame into. The alpha channel is set to
    255 once, when a buffer is allocated.
    """

    def __init__(self) -> None:
        self._bufs: List[torch.Tensor] = []

    def clear(self) -> None:
        self._bufs = []

    def acquire(
        self,
        height: int,
        width: int,
        device: torch.device,
        exporter: Any = None,
        num_slots: int = 1,
        async_export: bool = False,
    ) -> torch.Tensor:
        """Return the buffer for the next frame.

        ``exporter`` is None before the lazy open on the first frame; then nothing is in flight
        and slot 0 is used. With ``async_export`` and an open exporter, the slot is
        ``exporter.write_idx % num_slots``, the slot that the next ``export()`` writes.
        """
        ring_async = bool(async_export) and exporter is not None and num_slots > 1
        count = num_slots if ring_async else 1
        slot = exporter.write_idx % count if ring_async else 0

        if (
            len(self._bufs) != count
            or self._bufs[0].shape[0] != height
            or self._bufs[0].shape[1] != width
            or self._bufs[0].device != device
        ):
            if async_export and exporter is not None and self._bufs:
                # A queued copy can still read a buffer that is about to be released.
                exporter.drain()
            self._bufs = []
            for _ in range(count):
                buf = torch.empty((height, width, 4), dtype=torch.uint8, device=device)
                buf[..., 3] = 255
                self._bufs.append(buf)
        elif ring_async and not exporter.slot_copy_done(slot):
            exporter.drain()
        return self._bufs[slot]


class RecoveryAction(enum.Enum):
    KEEP = "keep"  # keep the exporter; the next export may run when ExportRecovery.ready()
    CLOSE = "close"  # close and drop the exporter; it reopens on a later frame unless stopped


def _recover(exporter: Any) -> int:
    recover = getattr(exporter, "recover", None)
    if not callable(recover):
        return -1  # cuda-link < 2.2.0: nothing can be proved
    try:
        return int(recover())
    except Exception:  # recover() does not raise; a fake or an old build can
        logger.debug("Exporter.recover() raised", exc_info=True)
        return -1


class ExportRecovery:
    """Failure policy for one Exporter, after the cuda-link 2.2.0 ``.tox`` Sender.

    After ``FrameOutcome.FAILED``, ``failed()`` calls ``Exporter.recover()``:

    * 0: no error is latched, every queued copy has completed: keep the exporter.
    * a permanent CUDA error: close it and stop. Exports resume only after ``reset()``
      (a new stream), because the CUDA context needs a process restart.
    * any other value: keep the exporter and try again after ``retry_interval_s``. After
      ``max_attempts`` such failures in a row, close it so that a new one opens, one time for
      each fault. If the new exporter fails the same way, stop.
    """

    def __init__(
        self,
        label: str,
        *,
        max_attempts: int = 3,
        retry_interval_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.label = label
        self.max_attempts = max_attempts
        self.retry_interval_s = retry_interval_s
        self._clock = clock
        self.stopped = False
        self.last_error: Optional[int] = None
        self._attempts = 0
        self._reopened = False
        self._next_try = 0.0
        self._warned = False

    def reset(self) -> None:
        self.stopped = False
        self.last_error = None
        self._attempts = 0
        self._reopened = False
        self._next_try = 0.0
        self._warned = False

    def ready(self) -> bool:
        """True when the next export may run."""
        return not self.stopped and self._clock() >= self._next_try

    def published(self) -> None:
        """Record a published frame: the current fault, if any, is over."""
        if self._attempts or self._reopened or self.last_error is not None:
            logger.info("%s: CUDA IPC export recovered", self.label)
        self._attempts = 0
        self._reopened = False
        self._warned = False
        self.last_error = None

    def failed(self, exporter: Any) -> RecoveryAction:
        code = _recover(exporter)
        self.last_error = code
        if code == 0:
            self._warn("export failed; Exporter.recover() found no CUDA error, continuing")
            return RecoveryAction.KEEP
        if code in PERMANENT_CUDA_ERRORS:
            self.stopped = True
            logger.error("%s: CUDA IPC export fault: CUDA error %d, restart TouchDesigner", self.label, code)
            return RecoveryAction.CLOSE
        self._attempts += 1
        self._next_try = self._clock() + self.retry_interval_s
        if self._attempts < self.max_attempts:
            self._warn(f"export fault: CUDA error {code}, trying again in {self.retry_interval_s:g} s")
            return RecoveryAction.KEEP
        if self._reopened:
            self.stopped = True
            logger.error(
                "%s: CUDA IPC export fault: CUDA error %d after a new stream, restart the stream or TouchDesigner",
                self.label,
                code,
            )
            return RecoveryAction.CLOSE
        self._reopened = True
        self._attempts = 0
        logger.warning("%s: CUDA IPC export fault: CUDA error %d, opening a new stream", self.label, code)
        return RecoveryAction.CLOSE

    def _warn(self, message: str) -> None:
        if self._warned:
            logger.debug("%s: %s", self.label, message)
        else:
            logger.warning("%s: %s", self.label, message)
            self._warned = True
