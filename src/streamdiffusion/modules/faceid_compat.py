"""Compatibility fixes for the vendored ``diffusers_ipadapter`` FaceID path.

``diffusers_ipadapter`` is pip-installed from a pinned SHA (see ``setup.py`` ->
``livepeer/Diffusers_IPAdapter``), so it cannot be edited directly — edits are lost on
reinstall. This module holds the fixes as monkeypatches / free functions, applied from
``IPAdapterModule.install()`` at runtime, and documents the silent-failure defects and
wasted-work findings from ``docs/plans/FaceID_PLAN.md`` (B1, B2, S1).

B1 and B2 produce **no errors or warnings** — FaceID runs end-to-end and looks healthy
in the log while barely transferring any identity. S1 is not a correctness defect, just
a redundant InsightFace pass on every SDXL update.

Also applies a cost-reduction pair not tracked in ``FaceID_PLAN.md`` (that plan's own S2
is an unrelated dead-code finding): the "update image" button freezes streaming output
in FaceID mode because the whole InsightFace pass runs inline on TouchDesigner's render
thread (``td_manager.py``'s ``_process_ipadapter_frame``), and does far more work than
FaceID actually needs. See ``_apply_insightface_loader_patch`` and
``_apply_detect_faces_multires_patch`` below.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared by every patch below. ``ip_adapter.py`` does
# ``from .face_utils import get_insightface_model, prepare_face_conditioning`` at
# module level, which copies each imported function *object* into ``ip_adapter``'s own
# namespace — and ``diffusers_ipadapter/ip_adapter/__init__.py`` does
# ``from .face_utils import *``, which copies every public face_utils name into the
# *package's* namespace too. Patching only ``face_utils`` therefore silently misses any
# call site that resolves the name through one of those copies instead of through
# ``face_utils`` itself.
#
# This is not hypothetical: ``_apply_insightface_loader_patch`` originally patched only
# ``face_utils.get_insightface_model``. It worked perfectly in isolation — but
# ``ip_adapter.py:56`` (the function's only real caller) reads its own copied-in name,
# never the patched one, so ``allowed_modules`` never reached ``FaceAnalysis`` in
# production despite every test passing. Rebind through this helper everywhere a
# face_utils name is patched, so the next addition can't repeat that mistake.
# ---------------------------------------------------------------------------
def _rebind_across_modules(attr_name: str, original: Any, replacement: Any, modules: Iterable[Any]) -> None:
    """Reassign ``attr_name`` to ``replacement`` on every module in ``modules`` that
    currently holds ``original``.

    Guards on identity (``is``) so callers can pass every module that *might* hold a
    stale copy of the name without risk of clobbering an unrelated binding (e.g. a
    module that was never given this name, or already points at ``replacement``).
    """
    for module in modules:
        if getattr(module, attr_name, None) is original:
            setattr(module, attr_name, replacement)


# ---------------------------------------------------------------------------
# B1 — InsightFace is fed RGB, but every InsightFace ONNX model (arcface_onnx.py,
# retinaface.py, scrfd.py) preprocesses with cv2.dnn.blobFromImage(..., swapRB=True) —
# i.e. it swaps R/B internally and therefore contractually expects BGR input.
# face_utils.py hands it a plain RGB numpy array (np.array(PIL_image)), so the ArcFace
# identity embedding is computed from a channel-swapped face. Detection boxes stay
# mostly correct (robust to the swap); the 512-d embedding does not.
# ---------------------------------------------------------------------------
def apply_faceid_patches() -> None:
    """Apply all face_utils patches: InsightFace loader cost cut, B1 + detection cost
    cuts (merged into one replacement of ``detect_faces_multires``), then S1.

    Idempotent: safe to call more than once (e.g. one process installing IP-Adapter for
    more than one stream) — each sub-patch checks its own marker independently, but the
    top-level check below (mirroring the pre-existing convention) is enough since all
    three are always applied together from this one entry point.
    """
    import diffusers_ipadapter.ip_adapter as _ip_adapter_pkg
    from diffusers_ipadapter.ip_adapter import face_utils as _face_utils
    from diffusers_ipadapter.ip_adapter import ip_adapter as _ip_adapter_module

    if getattr(_face_utils.detect_faces_multires, "_sdtd_bgr_patched", False):
        return

    # Every name patched below is re-exported by both ip_adapter.py's own
    # `from .face_utils import ...` and the package __init__'s `from .face_utils
    # import *` — see _rebind_across_modules above. Pass the full set to every patch so
    # a future addition to either import list is covered automatically.
    _aliasing_modules = (_face_utils, _ip_adapter_module, _ip_adapter_pkg)

    _apply_insightface_loader_patch(_face_utils, _aliasing_modules)
    _apply_detect_faces_multires_patch(_face_utils, _aliasing_modules)
    _apply_single_pass_face_conditioning_patch(_aliasing_modules)


# ---------------------------------------------------------------------------
# Cost reduction — get_insightface_model() builds a full FaceAnalysis with all five
# buffalo_l models: detection, recognition, genderage, landmark_2d_106, landmark_3d_68.
# FaceAnalysis.get() (face_analysis.py:72-75) runs every non-detection model on every
# detected face. FaceID only ever reads face.normed_embedding (recognition) and face.kps
# (a byproduct of detection, not a separate model) — see
# _single_pass_prepare_face_conditioning below and faceid_embedding.py. genderage,
# landmark_2d_106, and the 143MB landmark_3d_68 run for zero benefit. Restricting via
# InsightFace's own ``allowed_modules`` kwarg (face_analysis.py:24, :34-37) also saves
# ~150MB of model memory at load.
# ---------------------------------------------------------------------------
_INSIGHTFACE_ALLOWED_MODULES = ["detection", "recognition"]


def _apply_insightface_loader_patch(_face_utils, _aliasing_modules: Iterable[Any]) -> None:
    """Monkeypatch ``get_insightface_model`` so FaceAnalysis loads only detection + recognition.

    Rather than reimplementing ``get_insightface_model``'s body (provider
    auto-detection, model-dir setup, error wrapping — ``face_utils.py:13-53``), this
    patches ``insightface.app.FaceAnalysis`` for the duration of one call and lets the
    original vendored function run unchanged. ``get_insightface_model`` does
    ``from insightface.app import FaceAnalysis`` *inside* its own body
    (``face_utils.py:25``), so it re-resolves that name from ``insightface.app``'s module
    namespace on every call — the same dynamic-lookup property the BGR and single-pass
    patches below rely on — which is what makes a swap-then-restore safe here: the
    vendored function only ever sees the patched name during its own single call, and the
    original is restored immediately after (call sites are all at install time, not on
    a hot path where concurrent calls could interleave).

    Rebinds through every module in ``_aliasing_modules`` (see ``_rebind_across_modules``),
    not just ``face_utils`` — the real caller is ``ip_adapter.py:56``, which reads its own
    copied-in name from ``from .face_utils import get_insightface_model``. Patching
    ``face_utils`` alone leaves that call site on the unpatched original.

    Idempotent: safe to call more than once.
    """
    if getattr(_face_utils.get_insightface_model, "_sdtd_allowed_modules_patched", False):
        return

    _original_get_insightface_model = _face_utils.get_insightface_model

    def _restricted_get_insightface_model(*args, **kwargs):
        import insightface.app as _insightface_app

        _original_face_analysis = _insightface_app.FaceAnalysis

        def _face_analysis_detection_recognition_only(*fa_args, **fa_kwargs):
            fa_kwargs.setdefault("allowed_modules", _INSIGHTFACE_ALLOWED_MODULES)
            return _original_face_analysis(*fa_args, **fa_kwargs)

        _insightface_app.FaceAnalysis = _face_analysis_detection_recognition_only
        try:
            return _original_get_insightface_model(*args, **kwargs)
        finally:
            _insightface_app.FaceAnalysis = _original_face_analysis

    _restricted_get_insightface_model._sdtd_allowed_modules_patched = True
    _restricted_get_insightface_model._sdtd_original = _original_get_insightface_model
    _rebind_across_modules(
        "get_insightface_model",
        _original_get_insightface_model,
        _restricted_get_insightface_model,
        _aliasing_modules,
    )
    logger.info(
        "apply_faceid_patches: patched get_insightface_model to load only "
        f"{_INSIGHTFACE_ALLOWED_MODULES} (skipping genderage/landmark_2d_106/landmark_3d_68)."
    )


# ---------------------------------------------------------------------------
# Cost reduction — detect_faces_multires() sweeps det_size from a fixed 640 down to
# min_size in steps of 64 (up to 7 passes on a frame with no detectable face) and calls
# insightface_model.get(image) with the default max_num=0 — detect *and run every
# allowed model on* every face found, even though prepare_face_conditioning only ever
# keeps faces[0]. On TouchDesigner's 512x512 style image (td_config.yaml width/height)
# the first pass also upscales to 640 before the network sees a real face.
#
# Folded into the same choke point as B1 (both of face_utils.py's call sites,
# extract_face_embeddings and prepare_face_conditioning's re-crop branch, route through
# this one function) rather than layered as a second wrapper: B1's original
# wrap-and-capture-closure approach would make a *later* reassignment of
# detect_faces_multires invisible to it — the closure keeps calling whatever the module
# attribute pointed to when B1 was installed, not whatever it points to now.
# ---------------------------------------------------------------------------
def _apply_detect_faces_multires_patch(_face_utils, _aliasing_modules: Iterable[Any]) -> None:
    """Replace ``detect_faces_multires`` with a BGR-correct, cost-bounded implementation.

    Combines B1 (RGB->BGR swap — see module docstring above) with three cost cuts:
    ``max_num=1`` (stop running every non-detection model on faces
    ``prepare_face_conditioning`` will discard), a sweep that starts at the input
    image's own size instead of a fixed 640 (avoids upscaling a 512px style image before
    detection even runs), and a sweep bounded to at most 3 sizes (avoids the worst case
    — 7 full passes — on a frame with no detectable face). Logs when a smaller size
    succeeds, and when the sweep is exhausted with no face found, so a bounded sweep
    doesn't silently look identical to "no face in the image" from B1's era.

    Rebinds through every module in ``_aliasing_modules`` (see ``_rebind_across_modules``)
    for consistency with the other two patches, even though today's only known callers
    (``face_utils.py``'s own ``extract_face_embeddings`` and this module's single-pass
    replacement, both via ``_face_utils.detect_faces_multires`` dynamic lookup) resolve
    the name through ``face_utils`` directly and would work with a single binding.

    Idempotent: safe to call more than once.
    """
    if getattr(_face_utils.detect_faces_multires, "_sdtd_bgr_patched", False):
        return

    _original_detect_faces_multires = _face_utils.detect_faces_multires

    def _restricted_detect_faces_multires(insightface_model, image, min_size: int = 256):
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
            image = np.ascontiguousarray(image[:, :, ::-1])  # B1: RGB -> BGR

        h, w = image.shape[:2]
        start = min(640, max(h, w))
        start = max(64, (start // 64) * 64)
        sizes = [start] if start <= min_size else list(range(start, min_size - 1, -64))
        sizes = sizes[:3]  # bound the sweep — see docstring

        for size in sizes:
            insightface_model.det_model.input_size = (size, size)
            faces = insightface_model.get(image, max_num=1)
            if faces:
                if size != sizes[0]:
                    logger.info(f"detect_faces_multires: InsightFace detection resolution lowered to {size}x{size}")
                return faces

        logger.info(f"detect_faces_multires: no face found after trying sizes {sizes} (image {w}x{h})")
        return []

    _restricted_detect_faces_multires._sdtd_bgr_patched = True
    _restricted_detect_faces_multires._sdtd_original = _original_detect_faces_multires
    _rebind_across_modules(
        "detect_faces_multires",
        _original_detect_faces_multires,
        _restricted_detect_faces_multires,
        _aliasing_modules,
    )
    logger.info(
        "apply_faceid_patches: patched detect_faces_multires for BGR input (B1), max_num=1, "
        "and a size-bounded sweep starting from the image's own resolution."
    )


# ---------------------------------------------------------------------------
# S1 — prepare_face_conditioning() detects faces twice per update on SDXL/Kolors:
# once inside extract_face_embeddings() (face_utils.py:117, cropped at image_size=224),
# then again itself (face_utils.py:196) purely to get the same face's landmarks a second
# time so it can re-crop at the model's real crop_size (256 for SDXL, 336 for Kolors —
# get_face_crop_size() only returns 224 for plain SD1.5, so the second pass always fires
# for SDXL/Kolors). Not a correctness bug (S1 is latent-cost, not silent-wrong like
# B1/B2) but it doubles InsightFace's cost on every FaceID update for no benefit.
# ---------------------------------------------------------------------------
def _apply_single_pass_face_conditioning_patch(_aliasing_modules: Iterable[Any]) -> None:
    """Monkeypatch ``prepare_face_conditioning`` to detect each face once, not twice.

    Replaces the vendored two-pass implementation (detect at 224 -> re-detect at the
    model's real crop size) with a single detection pass per image, cropped directly at
    the final size. ``prepare_face_conditioning`` has exactly one caller in the vendored
    package (``ip_adapter.py``'s ``_get_faceid_embeds``) and is itself the only caller of
    ``extract_face_embeddings``, so replacing it whole is safe — no other code path
    depends on ``extract_face_embeddings``'s 224-fixed crop.

    Rebinds through every module in ``_aliasing_modules`` (see ``_rebind_across_modules``),
    not just ``face_utils``: ``ip_adapter.py`` does
    ``from .face_utils import ... prepare_face_conditioning``, which copies the function
    object into ``ip_adapter``'s own module namespace at import time. Reassigning
    ``face_utils.prepare_face_conditioning`` alone would not reach the actual call site
    (``ip_adapter.py:206``, ``_get_faceid_embeds``) — unlike ``detect_faces_multires``,
    whose only callers are inside ``face_utils.py`` itself and resolve the name through
    that module's own globals at call time (B1's patch doesn't strictly need this extra
    step, but uses the same helper for consistency — see that patch's docstring).

    Reads ``detect_faces_multires`` off the module dynamically on every call (not a
    captured reference), so this patch composes correctly with the B1 BGR patch
    regardless of which of the two is applied first.

    Idempotent: safe to call more than once.
    """
    from diffusers_ipadapter.ip_adapter import face_utils as _face_utils

    if getattr(_face_utils.prepare_face_conditioning, "_sdtd_single_pass_patched", False):
        return

    _original_prepare_face_conditioning = _face_utils.prepare_face_conditioning

    def _single_pass_prepare_face_conditioning(
        insightface_model,
        images,
        is_sdxl: bool = False,
        is_kolors: bool = False,
        normalize_embeddings: bool = True,
    ):
        from PIL import Image as _PILImage

        try:
            from insightface.utils import face_align
        except ImportError as err:
            raise ImportError("InsightFace face_align utility is required") from err

        if isinstance(images, _PILImage.Image):
            images = [images]

        crop_size = _face_utils.get_face_crop_size(is_sdxl, is_kolors)

        face_embeddings = []
        cropped_faces = []
        for i, image in enumerate(images):
            image_np = np.array(image) if isinstance(image, _PILImage.Image) else image

            # Dynamic lookup so this composes with the B1 BGR patch either order.
            faces = _face_utils.detect_faces_multires(insightface_model, image_np)
            if not faces:
                raise ValueError(f"extract_face_embeddings: No face detected in image {i}")
            face = faces[0]

            if normalize_embeddings:
                embedding = torch.from_numpy(face.normed_embedding).unsqueeze(0)
            else:
                embedding = torch.from_numpy(face.embedding).unsqueeze(0)
            face_embeddings.append(embedding)

            cropped_face = face_align.norm_crop(image_np, landmark=face.kps, image_size=crop_size)
            cropped_faces.append(_PILImage.fromarray(cropped_face))

        return torch.cat(face_embeddings, dim=0), cropped_faces

    _single_pass_prepare_face_conditioning._sdtd_single_pass_patched = True
    _single_pass_prepare_face_conditioning._sdtd_original = _original_prepare_face_conditioning
    _rebind_across_modules(
        "prepare_face_conditioning",
        _original_prepare_face_conditioning,
        _single_pass_prepare_face_conditioning,
        _aliasing_modules,
    )
    logger.info(
        "apply_faceid_patches: patched prepare_face_conditioning to detect faces once per "
        "update instead of twice (S1)."
    )


# ---------------------------------------------------------------------------
# B2 — the FaceID checkpoint's rank-128 LoRA (to_{q,k,v,out}_lora on all 140 attention
# modules) is silently discarded. ip_adapter.py's strict load filters "lora"/"LoRA" keys
# for FaceID models because no LoRA-aware processor exists in attention_processor.py,
# and the TensorRT export path (unet_ipadapter_export.py) rebuilds every processor
# before ONNX export regardless, so a processor-resident LoRA would be lost there too.
# Fusing the LoRA into the base attention linears survives both paths.
# ---------------------------------------------------------------------------
_LORA_TARGET_ATTR: Dict[str, str] = {
    "to_q_lora": "to_q",
    "to_k_lora": "to_k",
    "to_v_lora": "to_v",
    "to_out_lora": "to_out.0",
}


def _load_faceid_state_dict(ckpt_path: str) -> Dict[str, Any]:
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        # Older torch without the mmap kwarg.
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)


def fuse_faceid_lora(unet: torch.nn.Module, ckpt_path: str, lora_scale: float = 1.0) -> int:
    """Fuse the FaceID checkpoint's per-layer LoRA weights into the UNet's attention linears.

    For each of the UNet's ``attn_processors`` (index ``i``, in ``unet.attn_processors``
    iteration order — the same order ``diffusers_ipadapter`` uses to load
    ``to_k_ip``/``to_v_ip``), adds ``(up @ down) * lora_scale`` into ``to_q``, ``to_k``,
    ``to_v`` and ``to_out[0]`` of the corresponding attention module.

    This is mathematically exact at ``lora_scale=1.0``: in h94's LoRA(IP)AttnProcessor,
    each LoRA branch consumes the same input tensor as the base linear it parallels
    (``to_q_lora`` <- ``hidden_states``; ``to_k_lora``/``to_v_lora`` <- the text slice of
    ``encoder_hidden_states``, exactly what ``attn.to_k``/``attn.to_v`` receive;
    ``to_out_lora`` <- the blended hidden states), and h94's ``LoRALinearLayer`` leaves
    ``network_alpha=None`` — plain ``up @ down``, no rank/alpha division.

    Once fused, the LoRA is permanent and is *not* modulated by ``ipadapter_scale`` at
    runtime — this matches h94's reference behaviour (fixed ``lora_scale=1.0``). Callers
    must bump the TensorRT engine cache-key marker (B3) so stale pre-fusion engines are
    never reused.

    Idempotent: a second call on an already-fused UNet is a no-op. Raises ``RuntimeError``
    on any LoRA delta / target weight shape mismatch rather than silently skipping it —
    this defect class (B1/B2) is exactly the silent-no-op failure this fix exists to end.

    Returns the number of attention modules that received a fused update (0 if the
    checkpoint carries no LoRA — e.g. a non-FaceID IP-Adapter — or the UNet was already
    fused).
    """
    if getattr(unet, "_sdtd_faceid_lora_fused", False):
        logger.info("fuse_faceid_lora: UNet already has fused FaceID LoRA — skipping.")
        return 0

    ckpt = _load_faceid_state_dict(ckpt_path)
    ip_adapter_state_dict: Dict[str, torch.Tensor] = ckpt.get("ip_adapter", {})

    if not any("_lora." in key for key in ip_adapter_state_dict):
        logger.info("fuse_faceid_lora: checkpoint has no LoRA keys — nothing to fuse.")
        del ckpt, ip_adapter_state_dict
        return 0

    processor_keys = list(unet.attn_processors.keys())
    fused_layers = 0

    with torch.no_grad():
        for i, proc_key in enumerate(processor_keys):
            if not proc_key.endswith(".processor"):
                continue
            module_path = proc_key[: -len(".processor")]

            layer_fused = False
            for lora_name, target_attr in _LORA_TARGET_ATTR.items():
                down_key = f"{i}.{lora_name}.down.weight"
                up_key = f"{i}.{lora_name}.up.weight"
                if down_key not in ip_adapter_state_dict or up_key not in ip_adapter_state_dict:
                    continue

                down = ip_adapter_state_dict[down_key].float()
                up = ip_adapter_state_dict[up_key].float()
                delta = (up @ down) * lora_scale

                target_linear = unet.get_submodule(f"{module_path}.{target_attr}")
                if delta.shape != target_linear.weight.shape:
                    raise RuntimeError(
                        f"fuse_faceid_lora: shape mismatch at layer {i} "
                        f"({module_path}.{target_attr}): delta {tuple(delta.shape)} "
                        f"vs weight {tuple(target_linear.weight.shape)}"
                    )
                target_linear.weight.add_(
                    delta.to(dtype=target_linear.weight.dtype, device=target_linear.weight.device)
                )
                layer_fused = True

            if layer_fused:
                fused_layers += 1

    del ckpt, ip_adapter_state_dict
    unet._sdtd_faceid_lora_fused = True
    logger.info(
        f"fuse_faceid_lora: fused FaceID LoRA into {fused_layers} attention modules (lora_scale={lora_scale})."
    )
    return fused_layers


# ---------------------------------------------------------------------------
# Cost reduction — even after S1 (single detection pass, restricted model set), the
# *first* "update image" press in a fresh process still measures ~1.7s (vs ~25ms on the
# second and later presses; see docs/plans/FaceID_PLAN.md's Stage 1 measurement table).
# Two lazy one-time costs land on that first press instead of on install:
#   1. ``insightface.utils.face_align`` is imported lazily inside
#      ``_single_pass_prepare_face_conditioning`` (see below), and its own module-level
#      import (``from skimage import transform as trans``) pulls in a cold skimage import
#      — commonly 0.5-1.5s on Windows.
#   2. ONNX Runtime compiles/selects kernels and allocates its CUDA arena lazily on each
#      session's *first* ``session.run()`` call, per input shape — paid once for the
#      detection model and once for the recognition model.
# Both are one-time-per-process, so warming them during ``install()`` (while TD is
# already blocked building TensorRT engines) moves the cost off the user-visible render
# thread entirely instead of just amortizing it.
# ---------------------------------------------------------------------------
def warmup_faceid(ipadapter: Any) -> None:
    """Pay the FaceID cold-start cost (skimage import + ONNX Runtime session warmup)
    during install instead of on the first "update image" press.

    Takes the ``IPAdapter`` instance itself (not its already-loaded pieces) so callers
    don't need to know which attributes matter — mirrors how ``fuse_faceid_lora`` above
    takes the source objects it needs rather than pre-extracted values.

    Each of the four steps is independently timed and guarded: a failure in one (e.g. an
    InsightFace/skimage version that doesn't expose ``arcface_dst``) logs a warning and
    lets the remaining steps still run, rather than aborting the whole warmup. Every step
    is best-effort perf work — a cold first press is a latency regression, not a
    correctness failure, so nothing here may raise into ``install()``.
    """
    import time

    insightface_model = getattr(ipadapter, "insightface_model", None)
    if insightface_model is None:
        return

    _t_total = time.perf_counter()

    face_align = None
    try:
        _t0 = time.perf_counter()
        from insightface.utils import face_align as _face_align

        face_align = _face_align
        logger.info(
            f"warmup_faceid: imported insightface.utils.face_align in {(time.perf_counter() - _t0) * 1000:.1f}ms"
        )
    except Exception as e:
        logger.warning(f"warmup_faceid: face_align import failed: {e}")

    if face_align is not None:
        try:
            _t0 = time.perf_counter()
            from diffusers_ipadapter.ip_adapter import face_utils as _face_utils

            crop_size = _face_utils.get_face_crop_size(getattr(ipadapter, "is_sdxl", False), False)
            dummy_crop_src = np.zeros((256, 256, 3), dtype=np.uint8)
            face_align.norm_crop(dummy_crop_src, landmark=face_align.arcface_dst, image_size=crop_size)
            logger.info(
                f"warmup_faceid: warmed norm_crop (size={crop_size}) in {(time.perf_counter() - _t0) * 1000:.1f}ms"
            )
        except Exception as e:
            logger.warning(f"warmup_faceid: norm_crop warmup failed: {e}")

    det_model = getattr(insightface_model, "det_model", None)
    if det_model is not None:
        try:
            _t0 = time.perf_counter()
            dummy_frame = np.zeros((512, 512, 3), dtype=np.uint8)
            det_model.input_size = (512, 512)
            insightface_model.get(dummy_frame, max_num=1)
            logger.info(f"warmup_faceid: warmed detection session in {(time.perf_counter() - _t0) * 1000:.1f}ms")
        except Exception as e:
            logger.warning(f"warmup_faceid: detection session warmup failed: {e}")

    recognition_model = (
        getattr(insightface_model, "models", {}).get("recognition") if hasattr(insightface_model, "models") else None
    )
    if recognition_model is not None:
        try:
            _t0 = time.perf_counter()
            dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
            recognition_model.get_feat(dummy_face)
            logger.info(f"warmup_faceid: warmed recognition session in {(time.perf_counter() - _t0) * 1000:.1f}ms")
        except Exception as e:
            logger.warning(f"warmup_faceid: recognition session warmup failed: {e}")

    logger.info(f"warmup_faceid: total warmup {(time.perf_counter() - _t_total) * 1000:.1f}ms")
