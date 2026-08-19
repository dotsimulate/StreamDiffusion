from .cuda_dll_path import apply as _apply_cuda_dll_path
from .diffusers_kvo_patch import apply as _apply_kvo_patch

_apply_cuda_dll_path()
_apply_kvo_patch()
