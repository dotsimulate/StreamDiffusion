"""Registers PyTorch's bundled CUDA/cuDNN DLL directory on Windows' DLL search path.

On Windows, ONNX Runtime's CUDAExecutionProvider loads cudnn64_9.dll / cublas64_12.dll etc.
via the OS DLL search order, not via any Python import machinery. Those DLLs are *not* the
ones the pip nvidia-*-cu12 packages install: those wheels lay their DLLs out under
``nvidia/<pkg>/bin/`` on Windows, which nothing on the default search path (or torch's own
dependency loader -- see ``torch/__init__.py``'s ``_get_cuda_dep_paths``, which only globs
the Linux ``lib/`` layout) ever touches. The DLLs that actually get loaded are the ones
PyTorch bundles directly in ``torch/lib``, and previously those only ended up on the search
path as a side effect of ``import torch`` running first (torch's own loader calls
``os.add_dll_directory`` on its own ``torch/lib`` at import time).

That made ONNX Runtime's GPU availability depend on import order: ``import onnxruntime``
before ``import torch`` silently fell back to CPUExecutionProvider. This module makes the
registration explicit and import-order-independent by locating torch's package directory
via ``importlib.util.find_spec`` (no need to actually import torch -- keeps this cheap and
side-effect-free even in code paths that never touch torch) and registering it directly.

Called automatically at ``import streamdiffusion`` via _patches/__init__.py, before any
other patch that might trigger ``import onnxruntime``.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_APPLIED = False


def apply() -> None:
    """Register torch's bundled CUDA/cuDNN DLL directory. Idempotent, Windows-only, and
    never raises -- a failure here must not break ``import streamdiffusion``."""
    global _APPLIED
    if _APPLIED or sys.platform != "win32":
        return
    _APPLIED = True

    try:
        spec = importlib.util.find_spec("torch")
        if spec is None or not spec.origin:
            logger.debug("cuda_dll_path: torch not found, skipping")
            return

        torch_lib = Path(spec.origin).resolve().parent / "lib"
        if not torch_lib.is_dir():
            logger.debug("cuda_dll_path: no torch/lib at %s, skipping", torch_lib)
            return

        os.add_dll_directory(str(torch_lib))
        logger.debug("cuda_dll_path: registered %s", torch_lib)
    except Exception:
        logger.debug("cuda_dll_path: failed to register torch/lib DLL directory", exc_info=True)
