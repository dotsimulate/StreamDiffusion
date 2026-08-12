"""Compatibility fixes for the vendored ``diffusers_ipadapter`` FaceID path.

``diffusers_ipadapter`` is pip-installed from a pinned SHA (see ``setup.py`` ->
``livepeer/Diffusers_IPAdapter``), so it cannot be edited directly — edits are lost on
reinstall. This module holds the fixes as monkeypatches / free functions, applied from
``IPAdapterModule.install()`` at runtime, and documents the silent-failure defects and
wasted-work findings from ``docs/plans/FaceID_PLAN.md`` (B1, S1).

B1 produces **no errors or warnings** — FaceID runs end-to-end and looks healthy in the
log while barely transferring any identity. S1 is not a correctness defect, just a
redundant InsightFace pass on every SDXL update.
"""

from __future__ import annotations

import logging

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
