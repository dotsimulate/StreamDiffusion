import logging as _logging

# The package used to call logging.basicConfig(INFO) at import, which pre-empted host apps' own
# setup. Without it a plain script's root stays at WARNING and every streamdiffusion INFO record
# (engine-build progress, load/variant notices, the error-report log tail) is dropped at the call
# site. Give the package logger INFO unless the host already configured it or its root (TD does).
_pkg_logger = _logging.getLogger(__name__)
if _pkg_logger.level == _logging.NOTSET and _logging.getLogger().getEffectiveLevel() > _logging.INFO:
    _pkg_logger.setLevel(_logging.INFO)

from . import _patches
from .config import create_wrapper_from_config, load_config, save_config
from .pipeline import StreamDiffusion
from .preprocessing.processors import list_preprocessors
from .wrapper import StreamDiffusionWrapper

__all__ = [
    "StreamDiffusion",
    "StreamDiffusionWrapper",
    "create_wrapper_from_config",
    "list_preprocessors",
    "load_config",
    "save_config",
]
