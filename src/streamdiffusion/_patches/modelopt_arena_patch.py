"""Fixes an nvidia-modelopt bug that makes ORT's CUDA BFC arena degenerate during ONNX
FP8/INT8 static calibration, pinning VRAM at (or above) its true high-water mark for the
whole calibration pass instead of releasing between the 8 per-sample runs.

Upstream: ``modelopt.onnx.quantization.ort_patching``

``_create_inference_session_with_ep_config`` hardcodes ``arena_extend_strategy:
kSameAsRequested`` on the CPU/CUDA providers, so every arena growth is an exact-size
``cudaMalloc`` that can never be reused to satisfy a differently-sized request. With this
graph's thousands of distinct symbolic dims / unresolved-shape tensors, the arena's
high-water mark degenerates from "peak live set" toward "sum of every distinct allocation
ever requested" — the difference between a few GB and 22+ GB observed calibrating the
SDXL-Turbo UNet.

``kSameAsRequested`` reads, at first glance, like it exists to set up arena *shrinkage*
between calibration runs — modelopt's ``_collect_data_minmax_calibrator`` does build a
``RunOptions`` with ``memory.enable_memory_arena_shrinkage`` set to ``f"cpu:0;{gpu_str}"``
(discarded by a bug of its own: a fresh, unconfigured ``ort.RunOptions()`` is rebuilt inside
the per-sample loop right before every ``session.run``, so the shrinkage entry never actually
reaches ORT). Round 2 of this fix "repaired" that by reusing the configured ``RunOptions``
across iterations — which surfaced a second, independent bug: ORT's own contract
(``onnxruntime_run_options_config_keys.h``, ``memory.enable_memory_arena_shrinkage``) states
that if ``"cpu"`` is included in the shrink list, ``DisableCpuMemArena()`` must not have been
called — but ``_create_inference_session_with_ep_config`` sets
``sess_options.enable_cpu_mem_arena = False`` unconditionally, in the same code path. Building
the shrink-list ``RunOptions`` correctly does not help either: measured directly against this
stack (ORT 1.24.4 + CUDA EP), a `"gpu:0"`-only list is *also* rejected with
``INVALID_ARGUMENT``, regardless of ``arena_extend_strategy``. Arena shrinkage is unreachable
here, full stop — so ``kSameAsRequested`` is not a real precondition for anything, just a
purposeless constraint that starves the CUDA arena of reuse. The fix is to drop it, not to fix
the shrink-list string.

Not reachable through any public modelopt parameter — the provider config block is hardcoded
in ``_create_inference_session_with_ep_config`` — so this module carries a verbatim, bug-fixed
copy of that one function and swaps it into ``ort_patching``'s own module namespace: the exact
seam ``ort_patching.patch_ort_modules()`` itself reads from when it assigns
``CalibraterBase.create_inference_session`` on every ``quantize()`` call. Patching the module
global (rather than the ORT class directly) is required because ``patch_ort_modules()`` is
called again, unconditionally, on every ``quantize()`` invocation, and would otherwise stomp a
class-level patch right back to the buggy original.

Scoped, not global: ``apply()``/``revert()`` are meant to bracket a single
``quantize_onnx_fp8()`` call (see ``fp8_quantize.py``) inside a try/finally. This module is
intentionally **not** registered in ``_patches/__init__.py`` (unlike ``diffusers_kvo_patch``)
— it must not alter modelopt's behaviour for any runtime (non-build) code path or any other
caller of modelopt in this process.

Version-gated hard, on purpose: this is a copied third-party internal, not a call through a
public API, so a signature probe (the idiom ``diffusers_kvo_patch.py`` uses) cannot tell a
patched modelopt from an unpatched one — fixing this bug upstream would not change the
function's signature, only its body. A prior round of this same fix already hit exactly that
trap elsewhere in this codebase (a ``"name" in signature().parameters`` test that silently
went ``False`` for two modelopt releases and disabled an unrelated workaround). So: pin to the
modelopt version(s) this has been diffed against, and on any mismatch — wrong version or a
missing/renamed attribute — log a loud warning and skip the patch entirely. Never half-apply.

Verified still present on ``main`` as of the nvidia-modelopt 0.45.0 release (``ort_patching.py``
lines ~298, ~336, ~505-519 there) — upgrading past 0.43.0 would not remove the need for this
patch, and this repo pins ``modelopt==0.43.0`` for unrelated reasons anyway (see
``tools/install-tensorrt.py``: unpinned modelopt drags in ``onnx==1.21.0``, which breaks FP8
quant in a different way).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False
_ORIGINALS: dict = {}

# modelopt.__version__ values this patch has been diffed against and verified to still match
# ort_patching's function bodies. Extend only after re-diffing both target functions.
_VERIFIED_VERSIONS = {"0.43.0"}

_PATCHED_ATTRS = ("_create_inference_session_with_ep_config",)


def apply() -> None:
    """Patch modelopt's ``ort_patching`` module in place. Idempotent — safe to call multiple
    times. Never raises: on any mismatch (unexpected modelopt version, missing/renamed
    attribute, modelopt not installed), logs a warning and leaves modelopt untouched.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        import modelopt
        from modelopt.onnx.quantization import ort_patching
    except ImportError:
        logger.debug("modelopt_arena_patch: modelopt not importable, skipping")
        return

    version = getattr(modelopt, "__version__", None)
    if version not in _VERIFIED_VERSIONS:
        logger.warning(
            f"[FP8] modelopt_arena_patch: modelopt=={version} is not one of the versions this "
            f"CUDA-arena patch was verified against ({sorted(_VERIFIED_VERSIONS)}). Skipping — "
            "FP8 calibration may spill VRAM into shared/system memory. See "
            "_patches/modelopt_arena_patch.py for the upstream bug this works around."
        )
        return

    for name in _PATCHED_ATTRS:
        if not hasattr(ort_patching, name):
            logger.warning(
                f"[FP8] modelopt_arena_patch: ort_patching.{name} not found on "
                f"modelopt=={version} — skipping the CUDA-arena patch entirely (no half-apply)."
            )
            return

    for name, replacement in (
        ("_create_inference_session_with_ep_config", _fixed_create_inference_session_with_ep_config),
    ):
        _ORIGINALS[name] = getattr(ort_patching, name)
        setattr(ort_patching, name, replacement)

    _PATCHED = True
    logger.info(
        "[FP8] modelopt_arena_patch: applied (kSameAsRequested removed) — CUDA arena can reuse "
        "regions across calibration runs instead of exact-size-only growth"
    )


def revert() -> None:
    """Restore modelopt's original functions. Safe to call even if apply() was skipped."""
    global _PATCHED
    if not _PATCHED:
        return

    from modelopt.onnx.quantization import ort_patching

    for name, original in _ORIGINALS.items():
        setattr(ort_patching, name, original)
    _ORIGINALS.clear()
    _PATCHED = False
    logger.info("[FP8] modelopt_arena_patch: reverted")


# ---------------------------------------------------------------------------
# Fixed copy of a modelopt.onnx.quantization.ort_patching internal
#
# Source: modelopt/onnx/quantization/ort_patching.py, nvidia-modelopt 0.43.0
# (Apache-2.0 AND MIT — adapted from Microsoft ONNX Runtime; see that file's own header for
# the full upstream attribution.) Re-diffed end-to-end against the installed package on
# 2026-08-08 (upstream function spans ort_patching.py:286-353): the only change from upstream
# is the deletion of the `_update_provider_config` + kSameAsRequested loop, called out in the
# docstring below. An earlier version of this copy also silently dropped the trailing
# `group_qdq_tensors` assignment (upstream :350-353) — unintentionally, not as part of the
# documented fix — which raised AttributeError three minutes into calibration, in
# `compute_data()`. That tail is restored below; keep it in sync if this is ever re-diffed.
# ---------------------------------------------------------------------------


def _fixed_create_inference_session_with_ep_config(calibrator, **kwargs):
    """Identical to upstream except the ``arena_extend_strategy: kSameAsRequested`` override
    is *not* applied to CPU/CUDA providers, so ORT falls back to its default
    ``kNextPowerOfTwo`` — an arena region can then serve any request up to its size instead of
    exactly one.
    """
    import onnxruntime as ort
    from modelopt.onnx.logging_config import logger as _modelopt_logger

    model_path = kwargs.get("model_path")
    _modelopt_logger.debug("Creating inference session with Execution Provider configuration")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    sess_options.enable_cpu_mem_arena = False

    providers = kwargs.get("execution_providers", [])
    _modelopt_logger.debug(f"Execution providers: {providers}")

    # Note. This path can be an empty string, which denotes that the model has custom ops and
    # TRT EP is needed.
    calibrator.trt_extra_plugin_lib_paths = kwargs.get("trt_extra_plugin_lib_paths")

    if calibrator.trt_extra_plugin_lib_paths is not None:
        _modelopt_logger.debug(f"TRT extra plugin paths: {calibrator.trt_extra_plugin_lib_paths}")
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(f"Could not find `TensorrtExecutionProvider`, only {ort.get_available_providers()}")
        trt_ep_options = (
            {"trt_extra_plugin_lib_paths": calibrator.trt_extra_plugin_lib_paths}
            if calibrator.trt_extra_plugin_lib_paths
            else {}
        )

        # Set GPU memory usage limit
        trt_ep_options["trt_max_workspace_size"] = 80 * (1024**3)  # 80GB
        _modelopt_logger.debug(f"TRT EP options: {trt_ep_options}")

        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        providers.insert(0, ("TensorrtExecutionProvider", trt_ep_options))

    # --- upstream applies {"arena_extend_strategy": "kSameAsRequested"} to every CPU/CUDA
    # provider here via a nested `_update_provider_config` + loop. Both are intentionally
    # omitted — providers keep ORT's default arena_extend_strategy (kNextPowerOfTwo) instead
    # of being forced into exact-size-only arena growth. ---

    if model_path is None:
        # Create the inference session with EP configuration on augmented_model
        calibrator.infer_session = ort.InferenceSession(
            calibrator.augmented_model_path,
            sess_options=sess_options,
            providers=providers,
        )
    else:
        # Create the inference session with EP configuration on provided model path
        calibrator.infer_session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )

    # Group qdq tensors will have the same scaling factor.
    calibrator.group_qdq_tensors = kwargs.get("group_qdq_tensors")
    if calibrator.group_qdq_tensors:
        _modelopt_logger.debug(f"Group QDQ tensors: {calibrator.group_qdq_tensors}")
