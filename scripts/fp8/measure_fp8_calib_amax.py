"""
fp8-round-9.1 §10 follow-up — real-vs-surrogate calibration token magnitude.

Context: the `h92807e04ef45` build passed its acceptance criterion
(`fp8_uncalibrated_scales` 420 -> 0) but did so via the `image_proj_model(zeros)`
FaceID surrogate (measured |x|max = 29.8438), not the real calibration images —
`generate_td_config_yaml()` in the TouchDesigner extension never emitted
`fp8_calibration_style_image`, so it never reached the backend. That gap is fixed
separately (Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py).

This script answers the question the fix's rebuild decision hinges on: is a real
ArcFace token's magnitude from `images/calibration/faces/face_IP.png` above or
below the surrogate's 29.8438? If materially below, the shipped engine's `attn2`
merge-point scales are inflated relative to real IPA traffic and a rebuild (~49
min) recovers precision; at or above, the surrogate was already a reasonable (or
conservative) stand-in and a rebuild buys shape accuracy, not headroom.

Deliberately loads only what `image_proj_model(face_embeds)` needs -- the
InsightFace/ArcFace detector plus the FaceID projection head loaded straight
from the checkpoint's `image_proj` state dict -- not the full SDXL-Turbo UNet
or diffusers pipeline that `IPAdapterModule.install()` would normally attach it
to. `IPAdapter.__init__`'s `set_ip_adapter()`/LoRA-fusion/UNet wiring have no
bearing on this one number, so skipping them keeps this a read-only, no-GPU-build,
few-minutes-of-InsightFace-cold-start measurement instead of a full model load.

Mirrors the resolution logic in `IPAdapterModule._resolve_model_path` (HF repo/file
spec) and `IPAdapter.__init__`'s cross_attention_dim/is_plus detection and
`_get_faceid_embeds`'s face-embedding path (both in
`venv/Lib/site-packages/diffusers_ipadapter/ip_adapter/ip_adapter.py`), so the
number this prints is the same one the real build would have captured had the
config key reached it. Writes nothing; triggers no engine build.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("measure_fp8_calib_amax")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_IMAGE = REPO_ROOT / "images" / "calibration" / "faces" / "face_IP.png"
DEFAULT_CKPT_SPEC = "h94/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin"
DEFAULT_INSIGHTFACE_MODEL = "buffalo_l"

# Reference points already measured against the two zeros-surrogates (fp8-round-9.1
# plan §2 / this session's build log) -- printed alongside the real number so the
# comparison doesn't require cross-referencing another document.
SURROGATE_AMAX = {"regular": 8.266, "faceid": 29.8438}


def _resolve_ipadapter_ckpt(spec: str) -> str:
    """Mirror `IPAdapterModule._resolve_model_path`'s file branch for an
    `owner/repo/file.bin` HF spec (ipadapter_module.py:502-513) -- a local path
    is returned unchanged, otherwise it's downloaded via hf_hub_download."""
    if Path(spec).exists():
        return spec
    from huggingface_hub import hf_hub_download

    parts = spec.split("/")
    if len(parts) < 3:
        raise ValueError(f"Invalid HF spec: {spec!r} (need owner/repo/file)")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    return hf_hub_download(repo_id=repo_id, filename=filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Face image to encode")
    parser.add_argument(
        "--ckpt", default=DEFAULT_CKPT_SPEC, help="FaceID IP-Adapter checkpoint (HF spec or local path)"
    )
    parser.add_argument("--insightface-model", default=DEFAULT_INSIGHTFACE_MODEL)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"device={device} dtype={dtype}")

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Calibration image not found: {image_path}")
    pil_image = Image.open(image_path).convert("RGB")
    logger.info(f"Loaded {image_path} size={pil_image.size}")

    ckpt_path = _resolve_ipadapter_ckpt(args.ckpt)
    logger.info(f"Resolved checkpoint: {ckpt_path}")
    # weights_only=True: this checkpoint is a plain dict of tensors (image_proj /
    # ip_adapter state dicts), so restrict unpickling to tensors/primitives rather
    # than the torch.load default that permits arbitrary object unpickling.
    ipadapter_model = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Same detection logic as IPAdapter.__init__ (ip_adapter.py:40-45).
    is_plus = "latents" in ipadapter_model["image_proj"]
    output_cross_attention_dim = ipadapter_model["ip_adapter"]["1.to_k_ip.weight"].shape[1]
    is_sdxl = output_cross_attention_dim == 2048
    cross_attention_dim = 1280 if (is_plus and is_sdxl) else output_cross_attention_dim
    logger.info(f"is_plus={is_plus} is_sdxl={is_sdxl} cross_attention_dim={cross_attention_dim}")

    from diffusers_ipadapter.ip_adapter.face_utils import extract_face_embeddings, get_insightface_model
    from diffusers_ipadapter.ip_adapter.projection_models import create_faceid_projection_model

    from streamdiffusion.modules.faceid_compat import apply_faceid_patches

    # B1: without this, InsightFace receives RGB where every ONNX model
    # (arcface_onnx.py etc.) expects BGR -- same patch the real build applies
    # in IPAdapterModule.install() before any detection runs.
    apply_faceid_patches()

    image_proj_model = create_faceid_projection_model(
        ipadapter_model["image_proj"],
        cross_attention_dim=cross_attention_dim,
        clip_embeddings_dim=1280,  # unused by the non-Plus branch; kept for signature parity
        is_sdxl=is_sdxl,
        is_plus=is_plus,
    ).to(device, dtype=dtype)
    image_proj_model.load_state_dict(ipadapter_model["image_proj"])
    image_proj_model.eval()

    logger.info(f"Loading InsightFace model: {args.insightface_model} (cold start, may take a minute)")
    insightface_model = get_insightface_model(model_name=args.insightface_model)

    face_embeds, cropped_faces = extract_face_embeddings(insightface_model, [pil_image])
    logger.info(f"Detected {len(cropped_faces)} face(s); face_embeds shape={tuple(face_embeds.shape)}")
    face_embeds = face_embeds.to(device, dtype=dtype)

    with torch.inference_mode():
        tokens = image_proj_model(face_embeds)

    tokens_np = tokens.detach().float().cpu().numpy()
    amax = float(np.abs(tokens_np).max())
    std = float(tokens_np.std())
    nonzero = int(np.count_nonzero(tokens_np))
    total = int(tokens_np.size)

    print()
    print(f"shape={tokens_np.shape}  |x|max={amax:.4f}  std={std:.4f}  non-zero={nonzero}/{total}")
    print()
    print("Reference points:")
    for name, val in SURROGATE_AMAX.items():
        print(f"  {name} zeros-surrogate  |x|max = {val}")
    print(f"  real ArcFace token ({image_path.name})  |x|max = {amax:.4f}")
    print()
    delta_pct = (amax - SURROGATE_AMAX["faceid"]) / SURROGATE_AMAX["faceid"] * 100.0
    print(
        f"Real vs faceid-surrogate: {delta_pct:+.1f}% "
        f"({'below' if amax < SURROGATE_AMAX['faceid'] else 'at/above'} the surrogate the "
        f"shipped h92807e04ef45 engine was calibrated on)"
    )


if __name__ == "__main__":
    main()
