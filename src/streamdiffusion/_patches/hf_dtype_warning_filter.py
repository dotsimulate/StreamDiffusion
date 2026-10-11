"""Silences transformers' ``torch_dtype`` deprecation warning when diffusers is the caller.

The pinned diffusers fork's ``load_sub_model`` passes ``torch_dtype=`` to every ``nn.Module``
it loads, including transformers' CLIP text encoders. transformers >= 4.56 maps that onto
``dtype`` and logs "`torch_dtype` is deprecated! Use `dtype` instead!" once per process. The
load itself is correct (fp16 is honoured), so the line is only startup noise. Upstream
diffusers passes ``dtype=`` for transformers >= 4.56; remove this filter once the diffusers
pin moves past that change.

The filter drops the record only when a diffusers ``load_sub_model`` frame is on the stack, so
a caller of our own that passes ``torch_dtype`` to transformers still sees the warning.

Called automatically at ``import streamdiffusion`` via _patches/__init__.py.
"""

from __future__ import annotations

import logging
import sys

_LOGGER_NAME = "transformers.modeling_utils"
_MESSAGE = "`torch_dtype` is deprecated! Use `dtype` instead!"


class _DiffusersTorchDtypeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.getMessage() != _MESSAGE:
            return True
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_name == "load_sub_model" and frame.f_globals.get("__name__", "").startswith(
                "diffusers."
            ):
                return False
            frame = frame.f_back
        return True


def apply() -> None:
    """Attach the filter to transformers' modeling_utils logger. Idempotent; does not import
    transformers (the logger is created by name, so it works before transformers loads)."""
    target = logging.getLogger(_LOGGER_NAME)
    if not any(isinstance(f, _DiffusersTorchDtypeFilter) for f in target.filters):
        target.addFilter(_DiffusersTorchDtypeFilter())
