"""Compatibility fixes for the vendored ``diffusers_ipadapter`` FaceID path.

``diffusers_ipadapter`` is pip-installed from a pinned SHA (see ``setup.py`` ->
``livepeer/Diffusers_IPAdapter``), so it cannot be edited directly — edits are lost on
reinstall. This module holds the fixes as monkeypatches / free functions, applied from
``IPAdapterModule.install()`` at runtime, and documents the silent-failure defects and
wasted-work findings from ``docs/plans/FaceID_PLAN.md`` (B1, B2, S1).

B1 and B2 produce **no errors or warnings** — FaceID runs end-to-end and looks healthy
in the log while barely transferring any identity. S1 is not a correctness defect, just
a redundant InsightFace pass on every SDXL update.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# B1 — InsightFace is fed RGB, but every InsightFace ONNX model (arcface_onnx.py,
# retinaface.py, scrfd.py) preprocesses with cv2.dnn.blobFromImage(..., swapRB=True) —
# i.e. it swaps R/B internally and therefore contractually expects BGR input.
# face_utils.py hands it a plain RGB numpy array (np.array(PIL_image)), so the ArcFace
# identity embedding is computed from a channel-swapped face. Detection boxes stay
# mostly correct (robust to the swap); the 512-d embedding does not.
# ---------------------------------------------------------------------------
def apply_faceid_patches() -> None:
    """Monkeypatch ``detect_faces_multires`` so InsightFace receives BGR, then apply S1.

    In the vendored code, both call sites (``extract_face_embeddings`` and
    ``prepare_face_conditioning``'s re-crop branch, ``face_utils.py:117`` / ``:196``)
    route through this single choke point, so patching it here covers both. Landmark-based
    crops (``face_align.norm_crop``) are left untouched — they stay RGB, which is correct
    for the CLIP encoder path used by FaceID-Plus.

    Also applies the S1 fix (``_apply_single_pass_face_conditioning_patch``, below), which
    replaces ``prepare_face_conditioning`` itself so the second call site above no longer
    exists — one detection pass per image instead of two. Both patches touch the same
    vendored module and are applied together from the same install-time hook.

    Idempotent: safe to call more than once (e.g. one process installing IP-Adapter for
    more than one stream).
    """
    from diffusers_ipadapter.ip_adapter import face_utils as _face_utils

    if getattr(_face_utils.detect_faces_multires, "_sdtd_bgr_patched", False):
        return

    _original_detect_faces_multires = _face_utils.detect_faces_multires

    def _bgr_detect_faces_multires(insightface_model, image, *args, **kwargs):
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
            image = np.ascontiguousarray(image[:, :, ::-1])
        return _original_detect_faces_multires(insightface_model, image, *args, **kwargs)

    _bgr_detect_faces_multires._sdtd_bgr_patched = True
    _bgr_detect_faces_multires._sdtd_original = _original_detect_faces_multires
    _face_utils.detect_faces_multires = _bgr_detect_faces_multires
    logger.info("apply_faceid_patches: patched detect_faces_multires to feed InsightFace BGR (B1).")

    _apply_single_pass_face_conditioning_patch()


# ---------------------------------------------------------------------------
# S1 — prepare_face_conditioning() detects faces twice per update on SDXL/Kolors:
# once inside extract_face_embeddings() (face_utils.py:117, cropped at image_size=224),
# then again itself (face_utils.py:196) purely to get the same face's landmarks a second
# time so it can re-crop at the model's real crop_size (256 for SDXL, 336 for Kolors —
# get_face_crop_size() only returns 224 for plain SD1.5, so the second pass always fires
# for SDXL/Kolors). Not a correctness bug (S1 is latent-cost, not silent-wrong like
# B1/B2) but it doubles InsightFace's cost on every FaceID update for no benefit.
# ---------------------------------------------------------------------------
def _apply_single_pass_face_conditioning_patch() -> None:
    """Monkeypatch ``prepare_face_conditioning`` to detect each face once, not twice.

    Replaces the vendored two-pass implementation (detect at 224 -> re-detect at the
    model's real crop size) with a single detection pass per image, cropped directly at
    the final size. ``prepare_face_conditioning`` has exactly one caller in the vendored
    package (``ip_adapter.py``'s ``_get_faceid_embeds``) and is itself the only caller of
    ``extract_face_embeddings``, so replacing it whole is safe — no other code path
    depends on ``extract_face_embeddings``'s 224-fixed crop.

    Patches **two** bindings, not one: ``ip_adapter.py`` does
    ``from .face_utils import ... prepare_face_conditioning``, which copies the function
    object into ``ip_adapter``'s own module namespace at import time. Reassigning
    ``face_utils.prepare_face_conditioning`` alone would not reach the actual call site
    (``ip_adapter.py:206``, ``_get_faceid_embeds``) — unlike ``detect_faces_multires``,
    whose only callers are inside ``face_utils.py`` itself and resolve the name through
    that module's own globals at call time (B1's patch doesn't need this extra step).

    Reads ``detect_faces_multires`` off the module dynamically on every call (not a
    captured reference), so this patch composes correctly with the B1 BGR patch
    regardless of which of the two is applied first.

    Idempotent: safe to call more than once.
    """
    from diffusers_ipadapter.ip_adapter import face_utils as _face_utils
    from diffusers_ipadapter.ip_adapter import ip_adapter as _ip_adapter_module

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
    _face_utils.prepare_face_conditioning = _single_pass_prepare_face_conditioning
    # The actual call site (ip_adapter.py:206) uses its own copied-in name — see docstring.
    _ip_adapter_module.prepare_face_conditioning = _single_pass_prepare_face_conditioning
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
