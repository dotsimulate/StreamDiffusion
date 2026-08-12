"""
L2 Cache Persistence Utility for StreamDiffusion UNet.

Reserves a portion of the GPU's L2 cache for persistent data (UNet weights),
reducing cache evictions for memory-bandwidth-bound layers.

Requires: CUDA >= 11.2, compute capability >= 8.0 (Ampere+).
RTX 5090 has 128MB L2, compute 12.0 — full support.

Control precedence (highest to lowest):
    1. Explicit kwargs to setup_l2_persistence() (e.g. the wrapper's `l2_persist` config key).
    2. Environment variables below, read at CALL time (not import time, so a value set after
       this module first imports -- e.g. by TouchDesigner's embedded Python -- still applies).
    3. Mode-aware default: off when acceleration=="tensorrt" (Tier 1 is an inert soft
       carve-out on a serialized TRT engine, and Tier 2 requires an nn.Module UNet), on
       otherwise.

Environment variables:
    SDTD_L2_PERSIST=1         Enable L2 persistence (default: on, except TRT mode -- see above)
    SDTD_L2_PERSIST_MB=64     MB of L2 to reserve for persistent data (default: 64)
    SDTD_L2_PERSIST_TIER2=0   Enable per-tensor access policy window (default: 0, nn.Module UNet only)
    SDTD_L2_PERSIST_LAYERS=   Comma-separated layer names for access policy (default: auto)

Expected impact: 5-16% on memory-bandwidth-bound layers (normalization, small GEMMs).
Hot layers on SDXL: mid_block, up_blocks.1 (most FF hooks + V2V cached attention).

CUDA runtime access: this module goes through `cuda.bindings.runtime` (the same binding
`acceleration/tensorrt/utilities.py` imports as `cudart`), not raw ctypes -- no DLL search,
no manual struct/argtypes definitions, and every call's returned `cudaError_t` is checked
individually (see `_cuda_check` below) instead of being swallowed with a blanket
`cudaGetLastError()`.
"""

import os
from typing import Optional

import torch

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

# =============================================================================
# Environment Controls (read at call time -- see module docstring precedence)
# =============================================================================


def _env_enabled() -> Optional[bool]:
    v = os.environ.get("SDTD_L2_PERSIST")
    return None if v is None else v == "1"  # None = "unset" so it can fall through


def _env_persist_mb() -> int:
    return int(os.environ.get("SDTD_L2_PERSIST_MB", "64"))


def _env_tier2() -> bool:
    return os.environ.get("SDTD_L2_PERSIST_TIER2", "0") == "1"


# Hot layer prefixes — these contain the most attention + FF hook computation.
# mid_block: 1 transformer block, seq_len=1024, 16 FF hooks
# up_blocks.1: up-sampling path, seq_len=4096
_DEFAULT_HOT_LAYER_PREFIXES = ["mid_block", "up_blocks.1"]


# =============================================================================
# Module state
# =============================================================================

# Bytes actually granted by the most recent successful reserve_l2_persisting_cache()
# call (post-clamp) -- used by set_tensor_persisting() to scale hitRatio down instead
# of hardcoding 1.0 (see its docstring). 0 means "nothing reserved yet".
_granted_persist_bytes: int = 0


def _cuda_check(result, action: str):
    """Unpack a `cudart` call's `(cudaError_t, *values)` return tuple.

    Every cudart binding call returns its status as element 0 -- there is no
    thread-local "last error" to separately clear or swallow (that was a ctypes-only
    concern; see the module docstring). On `cudaSuccess` this returns the trailing
    payload (`None` for a bare status, the single value, or a tuple of values for
    multi-value returns). On any other status this logs `[L2] <action> failed: <err>`
    and returns `None` -- callers must check `is None`, not truthiness, since a
    legitimate query result (e.g. an attribute value of 0) is also falsy.
    """
    err = result[0]
    if err != cudart.cudaError_t.cudaSuccess:
        print(f"[L2] {action} failed: {err}")
        return None
    if len(result) == 2:
        return result[1]
    if len(result) > 2:
        return result[1:]
    return True


# =============================================================================
# Tier 1: Reserve L2 Persisting Cache Size
# =============================================================================


def reserve_l2_persisting_cache(persist_mb: Optional[int] = None) -> bool:
    """
    Reserve a portion of L2 cache for persistent data.

    This is Tier 1 of L2 persistence: informs the driver that `persist_mb` MB
    of L2 should not be evicted by regular (streaming) accesses. Hot data set
    via access policy windows will preferentially stay in this reserved region.

    The requested size is clamped to both 75% of total L2 and the device's actual
    `cudaDevAttrMaxPersistingL2CacheSize` -- NVIDIA's own guidance
    (`min(0.75 * l2CacheSize, persistingL2CacheMaxSize)`), since the two limits are
    independent and the smaller one governs. A device reporting `persistingL2CacheMaxSize
    == 0` does not support L2 persistence at all and is skipped rather than silently
    over-requesting.

    Args:
        persist_mb: Megabytes of L2 to reserve. Should be <= half of total L2.
                    RTX 5090 has 128MB L2 → 64MB is a safe default.
                    None -> resolved from SDTD_L2_PERSIST_MB at call time.

    Returns:
        True if successful, False if unsupported or failed.
    """
    global _granted_persist_bytes

    if persist_mb is None:
        persist_mb = _env_persist_mb()

    if not torch.cuda.is_available():
        return False

    # Check compute capability — L2 persistence requires Ampere (8.0+)
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    major, minor = props.major, props.minor
    if major < 8:
        print(f"[L2] L2 persistence skipped — compute {major}.{minor} < 8.0 (Ampere required)")
        return False

    l2_cache_size = _cuda_check(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrL2CacheSize, device),
        "cudaDeviceGetAttribute(L2CacheSize)",
    )
    if l2_cache_size is None:
        return False

    max_persisting = _cuda_check(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrMaxPersistingL2CacheSize, device),
        "cudaDeviceGetAttribute(MaxPersistingL2CacheSize)",
    )
    if max_persisting is None:
        return False

    if max_persisting == 0:
        print("[L2] L2 persistence unsupported on this device (persistingL2CacheMaxSize == 0)")
        return False

    requested_bytes = persist_mb * 1024 * 1024
    persist_bytes = min(requested_bytes, int(l2_cache_size * 0.75), max_persisting)

    ok = _cuda_check(
        cudart.cudaDeviceSetLimit(cudart.cudaLimit.cudaLimitPersistingL2CacheSize, persist_bytes),
        "cudaDeviceSetLimit(PersistingL2CacheSize)",
    )
    if ok is None:
        return False

    _granted_persist_bytes = persist_bytes
    print(
        f"[L2] Reserved {persist_bytes / 1024 / 1024:.1f}MB of {l2_cache_size / 1024 / 1024:.1f}MB L2 "
        f"for persisting cache (max persisting {max_persisting / 1024 / 1024:.1f}MB, "
        f"compute {major}.{minor}, {props.name})"
    )
    return True


# =============================================================================
# Tier 2: Per-Tensor Access Policy (stream attribute window)
# =============================================================================

# NOTE on applicability: the access policy window set below is a *per-stream* attribute
# (cudaStreamSetAttribute), applied to whatever `torch.cuda.current_stream()` is at call
# time. It only affects kernels subsequently launched on that stream -- not the whole
# device or process, and not retroactively. This is exactly right for the torch/compile
# Tier 2 path (weights are pinned once, on the stream inference runs on). On the TRT path,
# Tier 2 is never reached at all: pin_hot_unet_weights() requires an nn.Module and a
# serialized TRT engine is not one, so only Tier 1 (which is inert without a Tier 2 window
# to consume the reservation) applies there, and Tier 1 itself defaults off for
# acceleration=="tensorrt" (see setup_l2_persistence). No behavior change from the
# ctypes version -- this is the existing constraint, written down.


def set_tensor_persisting(tensor: torch.Tensor, hit_ratio: Optional[float] = None) -> bool:
    """
    Mark a tensor's memory region as L2-persistent.

    Uses cudaStreamSetAttribute with cudaAccessPolicyWindow to request that `hit_ratio`
    fraction of the tensor's data stays in the L2 persisting region.

    Args:
        tensor: CUDA tensor whose weights should persist in L2.
        hit_ratio: Fraction [0, 1] of accesses to serve from persisting cache.
                   None (default) -> min(1.0, granted_set_aside_bytes / num_bytes), so a
                   window larger than what reserve_l2_persisting_cache() actually set
                   aside scales down instead of thrashing cache lines at a hardcoded 1.0
                   (NVIDIA's own guidance -- a window bigger than the set-aside at
                   hitRatio=1.0 evicts and re-admits persisting lines every pass instead
                   of settling).

    Returns:
        True if successful.
    """
    if not tensor.is_cuda or not tensor.is_contiguous():
        return False

    device = tensor.get_device()
    max_window = _cuda_check(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrMaxAccessPolicyWindowSize, device),
        "cudaDeviceGetAttribute(MaxAccessPolicyWindowSize)",
    )
    if max_window is None:
        return False

    # accessPolicyMaxWindowSize clamp — docs require num_bytes < this limit (measured
    # 134,213,632 B on an RTX 4090). Unclamped, an oversized tensor would trip
    # cudaErrorInvalidValue here instead of silently degrading.
    num_bytes = min(tensor.nbytes, max_window)
    if num_bytes <= 0:
        return False

    if hit_ratio is None:
        hit_ratio = min(1.0, _granted_persist_bytes / num_bytes)
    hit_ratio = max(0.0, min(1.0, hit_ratio))

    try:
        window = cudart.cudaAccessPolicyWindow()
        window.base_ptr = tensor.data_ptr()
        window.num_bytes = num_bytes
        window.hitRatio = hit_ratio
        window.hitProp = cudart.cudaAccessProperty.cudaAccessPropertyPersisting
        window.missProp = cudart.cudaAccessProperty.cudaAccessPropertyStreaming

        attr_value = cudart.cudaStreamAttrValue()
        attr_value.accessPolicyWindow = window

        stream_ptr = torch.cuda.current_stream().cuda_stream
        ok = _cuda_check(
            cudart.cudaStreamSetAttribute(
                stream_ptr,
                cudart.cudaStreamAttrID.cudaLaunchAttributeAccessPolicyWindow,
                attr_value,
            ),
            "cudaStreamSetAttribute(accessPolicyWindow)",
        )
        return ok is not None
    except (RuntimeError, AttributeError, TypeError) as e:
        print(f"[L2] set_tensor_persisting failed: {e}")
        return False


def clear_tensor_persisting(tensor: torch.Tensor) -> bool:
    """
    Remove L2 persistence policy for a tensor (reset to normal access).

    Call when a tensor is no longer hot (e.g., model unload) to release the L2
    persisting budget for other tensors. Also explicitly resets the persisting L2
    cache lines via cudaCtxResetPersistingL2Cache() -- the docs call out that relying
    on the driver's automatic reset instead "is strongly discouraged because of the
    undetermined length of time required", so this never leaves it to chance.
    """
    if not tensor.is_cuda:
        return False

    try:
        window = cudart.cudaAccessPolicyWindow()
        window.base_ptr = tensor.data_ptr()
        window.num_bytes = 0
        window.hitRatio = 0.0
        window.hitProp = cudart.cudaAccessProperty.cudaAccessPropertyNormal
        window.missProp = cudart.cudaAccessProperty.cudaAccessPropertyStreaming

        attr_value = cudart.cudaStreamAttrValue()
        attr_value.accessPolicyWindow = window

        stream_ptr = torch.cuda.current_stream().cuda_stream
        ok = _cuda_check(
            cudart.cudaStreamSetAttribute(
                stream_ptr,
                cudart.cudaStreamAttrID.cudaLaunchAttributeAccessPolicyWindow,
                attr_value,
            ),
            "cudaStreamSetAttribute(clear accessPolicyWindow)",
        )
        if ok is None:
            return False

        reset_ok = _cuda_check(cudart.cudaCtxResetPersistingL2Cache(), "cudaCtxResetPersistingL2Cache")
        return reset_ok is not None
    except (RuntimeError, AttributeError, TypeError) as e:
        print(f"[L2] clear_tensor_persisting failed: {e}")
        return False


# =============================================================================
# High-Level: Pin Hot UNet Layer Weights
# =============================================================================


def pin_hot_unet_weights(
    unet: torch.nn.Module,
    hot_prefixes: Optional[list] = None,
    persist_mb: Optional[int] = None,
) -> int:
    """
    Mark the single largest hot UNet attention weight as L2-persistent.

    CUDA allows only one cudaAccessPolicyWindow per stream at a time — registering
    multiple tensors silently replaces the previous window.  This function correctly
    picks the single largest hot attention weight (by byte size) and registers exactly
    one window for it.

    Args:
        unet: The UNet model (already on CUDA, must be torch.nn.Module).
        hot_prefixes: Layer name prefixes to target. Defaults to mid_block + up_blocks.1.
        persist_mb: MB of L2 to reserve (passed to reserve_l2_persisting_cache).
                    None -> resolved from SDTD_L2_PERSIST_MB at call time.

    Returns:
        1 if a tensor was pinned, 0 otherwise.

    Note:
        No on/off gate here -- the sole caller, setup_l2_persistence, already resolves
        `enabled` with the full precedence (config > env > mode default) before calling
        this. Re-checking a raw env value here would let a stale SDTD_L2_PERSIST=0 veto
        an explicit enabled=True config override.
    """
    if persist_mb is None:
        persist_mb = _env_persist_mb()

    if not isinstance(unet, torch.nn.Module):
        print("[L2] Tier 2 skipped — model is not nn.Module (e.g. TRT engine). Use Tier 1 only.")
        return 0

    if hot_prefixes is None:
        hot_prefixes = _DEFAULT_HOT_LAYER_PREFIXES

    # Tier 1: Reserve L2 persisting region (skip if persist_mb=0, caller already reserved)
    if persist_mb > 0:
        tier1_ok = reserve_l2_persisting_cache(persist_mb)
        if not tier1_ok:
            return 0

    # Tier 2: Find the single largest hot attention weight.
    # CUDA allows only one cudaAccessPolicyWindow per stream — registering N tensors
    # results in only the Nth window being active (each call replaces the previous).
    # Pinning the largest tensor maximises L2 utilization for the one permitted window.
    _hot_weight_keywords = ["to_q", "to_k", "to_v", "to_out"]
    best_tensor = None
    best_bytes = 0
    candidate_count = 0

    for name, param in unet.named_parameters():
        if not param.is_cuda:
            continue
        is_hot = any(prefix in name for prefix in hot_prefixes)
        is_attn_weight = any(kw in name for kw in _hot_weight_keywords)
        if is_hot and is_attn_weight:
            candidate_count += 1
            if param.data.nbytes > best_bytes:
                best_bytes = param.data.nbytes
                best_tensor = param.data

    if best_tensor is not None and set_tensor_persisting(best_tensor):
        print(
            f"[L2] Pinned 1 of {candidate_count} hot tensors (largest, "
            f"{best_bytes / 1024 / 1024:.1f}MB) — single-window CUDA limit applies"
        )
        return 1

    if candidate_count == 0:
        print("[L2] No tensors pinned (params may require_grad=True before compile — call after freeze)")
    return 0


def setup_l2_persistence(
    unet: torch.nn.Module,
    *,
    enabled: Optional[bool] = None,
    acceleration: Optional[str] = None,
    persist_mb: Optional[int] = None,
    tier2: Optional[bool] = None,
) -> bool:
    """
    Main entry point: set up L2 cache persistence for UNet inference.

    Call this AFTER model is loaded and BEFORE torch.compile.
    For best results with frozen weights, call AFTER torch.compile with freezing=True.

    Tier 1 (L2 set-aside via cudaDeviceSetLimit) is a soft carve-out: it reserves a
    portion of L2 for hot data but is INERT on a serialized TRT engine (nothing is ever
    tagged persistent without Tier 2, which requires an nn.Module). Tier 2 (per-tensor
    access policy window) is opt-in via SDTD_L2_PERSIST_TIER2=1 / tier2=True. It only
    works for PyTorch nn.Module UNets (not TRT engines), and CUDA allows only one window
    per stream — this function registers only the single largest hot tensor.

    Precedence (highest to lowest) for `enabled`: explicit kwarg > SDTD_L2_PERSIST env
    (read here, at call time) > mode-aware default (off when acceleration=="tensorrt",
    since Tier 1 is inert there and Tier 2 is impossible; on otherwise). This inverts
    gpu_profiler's env-over-config precedence ON PURPOSE — TouchDesigner's embedded
    Python cannot set shell env vars but can write config, so config must win.

    Args:
        unet: The UNet model on CUDA (nn.Module for Tier-2 to apply; TRT Engine for Tier-1 only).
        enabled: Explicit on/off override (the wrapper's `l2_persist` config key). None
                 falls through to the env/mode-default resolution described above.
        acceleration: The active acceleration backend (e.g. "tensorrt"), used only for
                      the mode-aware default when `enabled` and the env var are both unset.
        persist_mb: MB of L2 to reserve. None -> SDTD_L2_PERSIST_MB (default 64).
        tier2: Explicit Tier-2 on/off override. None -> SDTD_L2_PERSIST_TIER2 (default off).

    Returns:
        True if at least Tier 1 (L2 reservation) succeeded.
    """
    if enabled is None:
        enabled = _env_enabled()
    if enabled is None:
        enabled = acceleration != "tensorrt"
    if not enabled:
        return False  # silent: no [L2] log when off (e.g. TRT/Performance mode by default)

    if persist_mb is None:
        persist_mb = _env_persist_mb()
    if tier2 is None:
        tier2 = _env_tier2()

    print(f"\n[L2] Setting up L2 cache persistence (SDTD_L2_PERSIST_MB={persist_mb})...")

    # Tier 1: Reserve L2 persisting region — works for all GPU modes, always attempt.
    tier1_ok = reserve_l2_persisting_cache(persist_mb)

    if tier1_ok:
        if tier2:
            # Tier 2: per-tensor access policy window — opt-in, nn.Module only.
            pin_hot_unet_weights(unet, persist_mb=0)  # Tier 1 already reserved above
        else:
            print(
                "[L2] Tier 2 access policy disabled (set SDTD_L2_PERSIST_TIER2=1 to enable; nn.Module UNet required)"
            )

    return tier1_ok
