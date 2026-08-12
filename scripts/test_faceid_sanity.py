"""
FaceID Sanity Test — headless PyTorch baseline vs FaceID IP-Adapter comparison.

Modelled directly on scripts/test_lora_sanity.py (same acceleration='none' two-pass +
side-by-side-PNG structure). Runs two StreamDiffusionWrapper passes from the same seed:
  A) baseline — no IP-Adapter
  B) faceid   — FaceID IP-Adapter conditioned on a face style image

Saves baseline.png, faceid.png, and a side-by-side comparison PNG.

Purpose: confirm FaceID IP-Adapter is visibly effective (identity transfer toward the style
face) BEFORE paying for a TensorRT engine build. Run once after the B1 (BGR feed) fix alone,
and again after B3+B2 (LoRA fusion + engine cache-key bump) land, per FaceID_PLAN.md Step 2 /
Step 6 — B1 alone should already move the needle since a corrupted ArcFace vector can't be
rescued downstream; B2 recovers the ~57% of the adapter's trained parameters (the LoRA) that
are otherwise silently discarded on top of that.

Usage:
    venv/Scripts/python scripts/test_faceid_sanity.py
    venv/Scripts/python scripts/test_faceid_sanity.py --style-image path/to/face.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root on sys.path so `from streamdiffusion` works without install
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from PIL import Image

from streamdiffusion import StreamDiffusionWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("faceid_sanity")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "stabilityai/sdxl-turbo"
DEFAULT_PROMPT = "a portrait photo, detailed face, studio lighting"
DEFAULT_INPUT = str(_REPO_ROOT / "images" / "inputs" / "input.png")
# insightface's own bundled multi-face test photo — real, detectable faces, no extra
# download required (buffalo_l is already cached locally by the time this runs).
DEFAULT_STYLE_IMAGE = str(Path(sys.prefix) / "Lib" / "site-packages" / "insightface" / "data" / "images" / "t1.jpg")
DEFAULT_T_INDEX = [10, 35]
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "outputs" / "faceid_sanity")
DEFAULT_INSIGHTFACE_MODEL = "buffalo_l"


# ---------------------------------------------------------------------------
# Helper: run one inference pass with StreamDiffusionWrapper
# ---------------------------------------------------------------------------
def run_pass(
    model_id: str,
    prompt: str,
    input_image: Image.Image,
    t_index_list: list,
    seed: int,
    use_ipadapter: bool,
    ipadapter_config: dict | None,
    style_image: Image.Image | None,
    label: str,
) -> Image.Image:
    logger.info(f"--- [{label}] Building wrapper (acceleration=none) ---")
    if ipadapter_config:
        logger.info(f"    ipadapter_config = {ipadapter_config}")

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id,
        t_index_list=t_index_list,
        frame_buffer_size=1,
        width=512,
        height=512,
        warmup=1,
        acceleration="none",
        mode="img2img",
        use_denoising_batch=True,
        cfg_type="self",
        seed=seed,
        use_tiny_vae=True,
        use_ipadapter=use_ipadapter,
        ipadapter_config=ipadapter_config,
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=50,
        guidance_scale=1.0,
        delta=1.0,
    )

    if use_ipadapter and style_image is not None:
        logger.info(f"    [{label}] Setting FaceID style image")
        stream.update_style_image(style_image)

    image_tensor = stream.preprocess_image(input_image)

    # Warmup: batch_size - 1 dummy passes (required by StreamDiffusion)
    for _ in range(stream.batch_size - 1):
        stream(image=image_tensor)

    output = stream(image=image_tensor)
    logger.info(f"    [{label}] Done — output type: {type(output)}")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="FaceID sanity: baseline vs FaceID comparison")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input image path (img2img base frame)")
    parser.add_argument("--style-image", default=DEFAULT_STYLE_IMAGE, help="Face photo for FaceID conditioning")
    parser.add_argument(
        "--insightface-model", default=DEFAULT_INSIGHTFACE_MODEL, help="InsightFace model name (e.g. buffalo_l)"
    )
    parser.add_argument(
        "--t-index",
        nargs="+",
        type=int,
        default=DEFAULT_T_INDEX,
        metavar="T",
        help="t_index_list (e.g. --t-index 10 35)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output PNGs")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Prepare output directory and input images
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input image not found: {input_path}")
        return 1

    style_path = Path(args.style_image)
    if not style_path.exists():
        logger.error(f"Style (face) image not found: {style_path}")
        return 1

    input_image = Image.open(input_path).convert("RGB").resize((512, 512))
    style_image = Image.open(style_path).convert("RGB")
    logger.info(f"Input image: {input_path} -> resized to 512x512")
    logger.info(f"Style (face) image: {style_path}")

    t_index_list = args.t_index

    ipadapter_config = {
        "ipadapter_model_path": "h94/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin",
        "image_encoder_path": "h94/IP-Adapter/sdxl_models/image_encoder",
        "type": "faceid",
        "insightface_model_name": args.insightface_model,
        "scale": 1.0,
    }

    # ------------------------------------------------------------------
    # Run A: Baseline (no IP-Adapter)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("RUN A: baseline (no IP-Adapter)")
    logger.info("=" * 60)
    baseline_img = run_pass(
        model_id=args.model,
        prompt=args.prompt,
        input_image=input_image,
        t_index_list=t_index_list,
        seed=args.seed,
        use_ipadapter=False,
        ipadapter_config=None,
        style_image=None,
        label="baseline",
    )
    baseline_path = output_dir / "baseline.png"
    baseline_img.save(baseline_path)
    logger.info(f"Saved baseline: {baseline_path}")

    # ------------------------------------------------------------------
    # Run B: FaceID
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"RUN B: FaceID (insightface_model={args.insightface_model})")
    logger.info("=" * 60)
    try:
        faceid_img = run_pass(
            model_id=args.model,
            prompt=args.prompt,
            input_image=input_image,
            t_index_list=t_index_list,
            seed=args.seed,
            use_ipadapter=True,
            ipadapter_config=ipadapter_config,
            style_image=style_image,
            label="faceid",
        )
    except RuntimeError as e:
        logger.error(f"FaceID run failed: {e}")
        logger.info("Baseline image saved. FaceID run aborted.")
        return 2

    faceid_path = output_dir / "faceid.png"
    faceid_img.save(faceid_path)
    logger.info(f"Saved faceid: {faceid_path}")

    # ------------------------------------------------------------------
    # Side-by-side comparison
    # ------------------------------------------------------------------
    comparison = Image.new("RGB", (1024, 512))
    comparison.paste(baseline_img.resize((512, 512)), (0, 0))
    comparison.paste(faceid_img.resize((512, 512)), (512, 0))
    comparison_path = output_dir / "comparison.png"
    comparison.save(comparison_path)
    logger.info(f"Saved side-by-side: {comparison_path}")

    logger.info("=" * 60)
    logger.info("DONE. Inspect outputs:")
    logger.info(f"  baseline:   {baseline_path}")
    logger.info(f"  faceid:     {faceid_path}")
    logger.info(f"  comparison: {comparison_path}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
