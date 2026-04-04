"""
L2 Cache Persistence Utility for StreamDiffusion UNet.

Reserves a portion of the GPU's L2 cache for persistent data (UNet weights),
reducing cache evictions for memory-bandwidth-bound layers.

Requires: CUDA >= 11.2, compute capability >= 8.0 (Ampere+).
RTX 5090 has 128MB L2, compute 12.0 — full support.

Environment variables:
    SDTD_L2_PERSIST=1         Enable L2 persistence (default: 1)
    SDTD_L2_PERSIST_MB=64     MB of L2 to reserve for persistent data (default: 64)
    SDTD_L2_PERSIST_LAYERS=   Comma-separated layer names for access policy (default: auto)

Expected impact: 5-16% on memory-bandwidth-bound layers (normalization, small GEMMs).
Hot layers on SDXL: mid_block, up_blocks.1 (most FF hooks + V2V cached attention).
"""

import ctypes
import os
import sys
from typing import Optional

import torch


# =============================================================================
# Environment Controls
# =============================================================================

L2_PERSIST_ENABLED = os.environ.get("SDTD_L2_PERSIST", "1") == "1"
L2_PERSIST_MB = int(os.environ.get("SDTD_L2_PERSIST_MB", "64"))

# Hot layer prefixes — these contain the most attention + FF hook computation.
# mid_block: 1 transformer block, seq_len=1024, 16 FF hooks
# up_blocks.1: up-sampling path, seq_len=4096
_DEFAULT_HOT_LAYER_PREFIXES = ["mid_block", "up_blocks.1"]


# =============================================================================
# CUDA Runtime Access Policy Structs (for Tier 2 per-tensor persistence)
# =============================================================================


class _CudaAccessPolicyWindow(ctypes.Structure):
    """cudaAccessPolicyWindow struct for cudaStreamSetAttribute."""

    _fields_ = [
        ("base_ptr", ctypes.c_void_p),  # void* — start of memory region
        ("num_bytes", ctypes.c_size_t),  # size_t — size of region in bytes
        ("hitRatio", ctypes.c_float),  # float — fraction in [0, 1] to keep persistent
        (
            "hitProp",
            ctypes.c_int,
        ),  # cudaAccessProperty: 2 = cudaAccessPropertyPersisting
        (
            "missProp",
            ctypes.c_int,
        ),  # cudaAccessProperty: 1 = cudaAccessPropertyStreaming
    ]


# cudaAccessProperty enum values
_CUDA_ACCESS_PROPERTY_NORMAL = 0
_CUDA_ACCESS_PROPERTY_STREAMING = 1
_CUDA_ACCESS_PROPERTY_PERSISTING = 2

# cudaStreamAttrID
_CUDA_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1

# cudaLimit enum (CUDA 11.2+)
# 0x06 = cudaLimitPersistingL2CacheSize (size in bytes — correct for L2 persistence)
_CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE = 0x06


# =============================================================================
# CUDA Runtime Handle
# =============================================================================

_cudart: Optional[ctypes.CDLL] = None
_cudart_loaded: bool = False


def _get_cudart() -> Optional[ctypes.CDLL]:
    """Load the CUDA runtime DLL. Cached after first call."""
    global _cudart, _cudart_loaded
    if _cudart_loaded:
        return _cudart

    _cudart_loaded = True

    if sys.platform != "win32":
        # Non-Windows: use libcudart.so — typically already loaded by PyTorch
        try:
            _cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
            return _cudart
        except OSError:
            pass
        try:
            from ctypes.util import find_library

            lib = find_library("cudart")
            if lib:
                _cudart = ctypes.CDLL(lib)
                return _cudart
        except OSError:
            pass
        return None

    # Windows: find cudart64_*.dll shipped with PyTorch or CUDA toolkit
    import glob

    # Option 1: PyTorch ships cudart in torch/lib/
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    candidates = sorted(
        glob.glob(os.path.join(torch_lib, "cudart64_*.dll")), reverse=True
    )

    # Option 2: CUDA toolkit installation
    cuda_path = os.environ.get(
        "CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    )
    candidates += sorted(
        glob.glob(os.path.join(cuda_path, "bin", "cudart64_*.dll")), reverse=True
    )

    for dll_path in candidates:
        try:
            _cudart = ctypes.WinDLL(dll_path)
            return _cudart
        except OSError:
            continue

    return None


# =============================================================================
# Tier 1: Reserve L2 Persisting Cache Size
# =============================================================================


def reserve_l2_persisting_cache(persist_mb: int = L2_PERSIST_MB) -> bool:
    """
    Reserve a portion of L2 cache for persistent data.

    This is Tier 1 of L2 persistence: informs the driver that `persist_mb` MB
    of L2 should not be evicted by regular (streaming) accesses. Hot data set
    via access policy windows will preferentially stay in this reserved region.

    Args:
        persist_mb: Megabytes of L2 to reserve. Should be <= half of total L2.
                    RTX 5090 has 128MB L2 → 64MB is a safe default.

    Returns:
        True if successful, False if unsupported or failed.
    """
    if not torch.cuda.is_available():
        return False

    # Check compute capability — L2 persistence requires Ampere (8.0+)
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    major, minor = props.major, props.minor
    if major < 8:
        print(
            f"[L2] L2 persistence skipped — compute {major}.{minor} < 8.0 (Ampere required)"
        )
        return False

    l2_total_mb = props.L2_cache_size // (1024 * 1024)
    persist_bytes = min(persist_mb * 1024 * 1024, props.L2_cache_size // 2)
    persist_mb_actual = persist_bytes // (1024 * 1024)

    cudart = _get_cudart()
    if cudart is None:
        print("[L2] CUDA runtime not found — L2 persistence unavailable")
        return False

    try:
        result = cudart.cudaDeviceSetLimit(
            ctypes.c_int(_CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE),
            ctypes.c_size_t(persist_bytes),
        )
        # CRITICAL: Always clear CUDA error state after ctypes calls.
        # cudaDeviceSetLimit sets the thread-local CUDA error on failure, and
        # PyTorch's C10_CUDA_KERNEL_LAUNCH_CHECK() reads it on the next kernel
        # launch — causing a stale error to crash an unrelated operation.
        cudart.cudaGetLastError()
        if result != 0:
            print(f"[L2] cudaDeviceSetLimit failed: error {result}")
            return False

        print(
            f"[L2] Reserved {persist_mb_actual}MB of {l2_total_mb}MB L2 for persisting cache "
            f"(compute {major}.{minor}, {props.name})"
        )
        return True

    except (OSError, ctypes.ArgumentError, AttributeError) as e:
        print(f"[L2] L2 reservation failed: {e}")
        return False


# =============================================================================
# Tier 2: Per-Tensor Access Policy (stream attribute window)
# =============================================================================


def set_tensor_persisting(tensor: torch.Tensor, hit_ratio: float = 1.0) -> bool:
    """
    Mark a tensor's memory region as L2-persistent.

    Uses cudaStreamSetAttribute with cudaAccessPolicyWindow to request that
    `hit_ratio` fraction of the tensor's data stays in the L2 persisting region.

    Args:
        tensor: CUDA tensor whose weights should persist in L2.
        hit_ratio: Fraction [0, 1] of accesses to serve from persisting cache.
                   1.0 = always try to keep in L2 (good for weights).

    Returns:
        True if successful.
    """
    if not tensor.is_cuda or not tensor.is_contiguous():
        return False

    cudart = _get_cudart()
    if cudart is None:
        return False

    try:
        stream_ptr = torch.cuda.current_stream().cuda_stream

        window = _CudaAccessPolicyWindow(
            base_ptr=ctypes.c_void_p(tensor.data_ptr()),
            num_bytes=ctypes.c_size_t(tensor.nbytes),
            hitRatio=ctypes.c_float(hit_ratio),
            hitProp=ctypes.c_int(_CUDA_ACCESS_PROPERTY_PERSISTING),
            missProp=ctypes.c_int(_CUDA_ACCESS_PROPERTY_STREAMING),
        )

        result = cudart.cudaStreamSetAttribute(
            ctypes.c_void_p(stream_ptr),
            ctypes.c_int(_CUDA_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW),
            ctypes.byref(window),
        )
        cudart.cudaGetLastError()  # Clear any stale CUDA error from ctypes call
        return result == 0

    except (RuntimeError, OSError, ctypes.ArgumentError, AttributeError):
        return False


def clear_tensor_persisting(tensor: torch.Tensor) -> bool:
    """
    Remove L2 persistence policy for a tensor (reset to normal access).

    Call when a tensor is no longer hot (e.g., model unload) to release
    the L2 persisting budget for other tensors.
    """
    if not tensor.is_cuda:
        return False

    cudart = _get_cudart()
    if cudart is None:
        return False

    try:
        stream_ptr = torch.cuda.current_stream().cuda_stream
        window = _CudaAccessPolicyWindow(
            base_ptr=ctypes.c_void_p(tensor.data_ptr()),
            num_bytes=ctypes.c_size_t(0),
            hitRatio=ctypes.c_float(0.0),
            hitProp=ctypes.c_int(_CUDA_ACCESS_PROPERTY_NORMAL),
            missProp=ctypes.c_int(_CUDA_ACCESS_PROPERTY_STREAMING),
        )
        result = cudart.cudaStreamSetAttribute(
            ctypes.c_void_p(stream_ptr),
            ctypes.c_int(_CUDA_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW),
            ctypes.byref(window),
        )
        cudart.cudaGetLastError()  # Clear any stale CUDA error from ctypes call
        return result == 0
    except (RuntimeError, OSError, ctypes.ArgumentError, AttributeError):
        return False


# =============================================================================
# High-Level: Pin Hot UNet Layer Weights
# =============================================================================


def pin_hot_unet_weights(
    unet: torch.nn.Module,
    hot_prefixes: Optional[list] = None,
    persist_mb: int = L2_PERSIST_MB,
) -> int:
    """
    Mark hot UNet layer weights as L2-persistent.

    Identifies attention Q/K/V/out projection weights in the hottest layers
    (mid_block, up_blocks.1) and requests they persist in L2 cache.

    Args:
        unet: The UNet model (already on CUDA).
        hot_prefixes: Layer name prefixes to target. Defaults to mid_block + up_blocks.1.
        persist_mb: MB of L2 to reserve (passed to reserve_l2_persisting_cache).

    Returns:
        Number of weight tensors successfully pinned.
    """
    if not L2_PERSIST_ENABLED:
        return 0

    if hot_prefixes is None:
        hot_prefixes = _DEFAULT_HOT_LAYER_PREFIXES

    # Tier 1: Reserve L2 persisting region (skip if persist_mb=0, caller already reserved)
    if persist_mb > 0:
        tier1_ok = reserve_l2_persisting_cache(persist_mb)
        if not tier1_ok:
            return 0

    # TRT engine objects don't expose PyTorch parameters — skip Tier 2 gracefully
    if not hasattr(unet, "named_parameters"):
        return 0

    # Tier 2: Set access policy on hot attention weights
    # Target: to_q, to_k, to_v, to_out weights in hot transformer blocks.
    # These are small-to-medium GEMMs that benefit most from L2 hits.
    _hot_weight_keywords = ["to_q", "to_k", "to_v", "to_out"]
    pinned_count = 0
    pinned_bytes = 0

    for name, param in unet.named_parameters():
        if not param.is_cuda:
            continue
        is_hot = any(prefix in name for prefix in hot_prefixes)
        is_attn_weight = any(kw in name for kw in _hot_weight_keywords)
        if is_hot and is_attn_weight:
            if set_tensor_persisting(param.data):
                pinned_count += 1
                pinned_bytes += param.data.nbytes

    if pinned_count > 0:
        print(
            f"[L2] Pinned {pinned_count} attention weight tensors "
            f"({pinned_bytes / 1024 / 1024:.1f}MB) in L2 persisting cache"
        )
    else:
        print(
            "[L2] No tensors pinned (params may require_grad=True before compile — call after freeze)"
        )

    return pinned_count


def set_trt_persistent_cache(unet, persist_mb: int = L2_PERSIST_MB) -> bool:
    """
    Enable TRT activation caching in L2 for a TensorRT UNet engine.

    Sets IExecutionContext.persistent_cache_limit so TRT retains intermediate
    activations in the L2 persisting region already reserved by Tier 1.

    TRT checks the current cudaLimitPersistingL2CacheSize at assignment time
    (not at context creation), so calling this after reserve_l2_persisting_cache()
    is correct — no reordering of engine initialization is needed.

    Args:
        unet: UNet2DConditionModelEngine (must have .engine.context attribute).
        persist_mb: Target L2 budget in MB. Uses Tier 1 reservation size (L2/2),
                    which is guaranteed <= persistingL2CacheMaxSize on Ampere+.

    Returns:
        True if activation caching was enabled successfully.
    """
    if not L2_PERSIST_ENABLED:
        return False

    try:
        context = unet.engine.context
    except AttributeError:
        return False

    if not hasattr(context, "persistent_cache_limit"):
        return False

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    persist_bytes = min(persist_mb * 1024 * 1024, props.L2_cache_size // 2)
    try:
        context.persistent_cache_limit = persist_bytes
        actual = context.persistent_cache_limit
        print(
            f"[L2] TRT UNet activation caching: {actual / (1024 * 1024):.0f}MB "
            f"of L2 persisting region allocated for activation persistence"
        )
        return actual > 0
    except Exception as e:
        print(f"[L2] TRT persistent_cache_limit failed: {e}")
        return False


def setup_l2_persistence(unet) -> bool:
    """
    Main entry point: set up L2 cache persistence for UNet inference.

    Dispatches between two strategies based on UNet type:
    - PyTorch UNet (nn.Module): pin hot attention weight tensors via
      cudaStreamSetAttribute access policy windows (Tier 2).
    - TRT UNet engine: enable TRT's native activation caching in L2 via
      IExecutionContext.persistent_cache_limit.

    Both paths share Tier 1 (cudaDeviceSetLimit L2 reservation).

    Args:
        unet: The UNet model — either a PyTorch nn.Module or a TRT engine wrapper.

    Returns:
        True if at least Tier 1 (L2 reservation) succeeded.
    """
    if not L2_PERSIST_ENABLED:
        return False

    print(
        f"\n[L2] Setting up L2 cache persistence "
        f"(SDTD_L2_PERSIST_MB={L2_PERSIST_MB})..."
    )

    # Tier 1 is the reliable baseline — always attempt
    tier1_ok = reserve_l2_persisting_cache(L2_PERSIST_MB)

    if tier1_ok:
        if hasattr(unet, "named_parameters"):
            # PyTorch path: pin hot attention weight tensors in L2
            pinned = pin_hot_unet_weights(unet, persist_mb=0)  # Tier 1 already reserved
            if pinned == 0:
                print(
                    "[L2] Tier 2 access policy skipped (call pin_hot_unet_weights() "
                    "after compile+freeze for per-tensor control)"
                )
        else:
            # TRT engine path: use TRT's native activation caching instead
            set_trt_persistent_cache(unet, persist_mb=L2_PERSIST_MB)

    return tier1_ok
