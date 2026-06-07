"""
LoRA Sanity Test — headless PyTorch baseline vs LoRA comparison.

Runs two StreamDiffusionWrapper passes (acceleration='none') from the same seed:
  A) baseline — no LoRA
  B) lora     — with the requested LoRA fused at the given scale

Saves baseline.png, lora.png, and a side-by-side comparison PNG.

Purpose: confirm a LoRA loads correctly on sdxl-turbo and is visibly effective at
2 denoising steps BEFORE paying the expensive fp8 TRT engine build.

Usage:
    venv/Scripts/python scripts/test_lora_sanity.py
    venv/Scripts/python scripts/test_lora_sanity.py --lora nerijs/pixel-art-xl --weight 1.0
    venv/Scripts/python scripts/test_lora_sanity.py --lora not/a-real-lora  # G1 error path test
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

from PIL import Image  # noqa: E402

from streamdiffusion import StreamDiffusionWrapper  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lora_sanity")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "stabilityai/sdxl-turbo"
DEFAULT_LORA = "nerijs/pixel-art-xl"
DEFAULT_WEIGHT = 1.0
DEFAULT_PROMPT = "pixel art style, a beautiful mountain landscape at sunset, detailed"
DEFAULT_INPUT = str(_REPO_ROOT / "images" / "inputs" / "input.png")
DEFAULT_T_INDEX = [10, 35]
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "outputs" / "lora_sanity")


# ---------------------------------------------------------------------------
# Helper: run one inference pass with StreamDiffusionWrapper
# ---------------------------------------------------------------------------
def run_pass(
    model_id: str,
    prompt: str,
    input_image: Image.Image,
    t_index_list: list,
    seed: int,
    lora_dict: dict | None,
    label: str,
) -> Image.Image:
    logger.info(f"--- [{label}] Building wrapper (acceleration=none) ---")
    if lora_dict:
        logger.info(f"    lora_dict = {lora_dict}")

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
        use_tiny_vae=False,
        lora_dict=lora_dict,
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=50,
        guidance_scale=1.0,
        delta=1.0,
    )

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
    parser = argparse.ArgumentParser(description="LoRA sanity: baseline vs LoRA comparison")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
    parser.add_argument("--lora", default=DEFAULT_LORA, help="HF repo id or local .safetensors path")
    parser.add_argument("--weight", type=float, default=DEFAULT_WEIGHT, help="LoRA scale (0–1)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt (include trigger words)")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input image path")
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
    # Prepare output directory and input image
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input image not found: {input_path}")
        return 1

    input_image = Image.open(input_path).convert("RGB").resize((512, 512))
    logger.info(f"Input image: {input_path} → resized to 512×512")

    lora_dict = {args.lora: args.weight}
    t_index_list = args.t_index

    # ------------------------------------------------------------------
    # Run A: Baseline (no LoRA)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("RUN A: baseline (no LoRA)")
    logger.info("=" * 60)
    baseline_img = run_pass(
        model_id=args.model,
        prompt=args.prompt,
        input_image=input_image,
        t_index_list=t_index_list,
        seed=args.seed,
        lora_dict=None,
        label="baseline",
    )
    baseline_path = output_dir / "baseline.png"
    baseline_img.save(baseline_path)
    logger.info(f"Saved baseline: {baseline_path}")

    # ------------------------------------------------------------------
    # Run B: LoRA
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"RUN B: LoRA={args.lora} @ {args.weight}")
    logger.info("=" * 60)
    try:
        lora_img = run_pass(
            model_id=args.model,
            prompt=args.prompt,
            input_image=input_image,
            t_index_list=t_index_list,
            seed=args.seed,
            lora_dict=lora_dict,
            label="lora",
        )
    except RuntimeError as e:
        logger.error(f"LoRA run failed (expected for invalid LoRA ids): {e}")
        logger.info("Baseline image saved. LoRA run aborted cleanly (G1 fix working correctly).")
        return 2

    lora_path = output_dir / "lora.png"
    lora_img.save(lora_path)
    logger.info(f"Saved lora: {lora_path}")

    # ------------------------------------------------------------------
    # Side-by-side comparison
    # ------------------------------------------------------------------
    comparison = Image.new("RGB", (1024, 512))
    comparison.paste(baseline_img.resize((512, 512)), (0, 0))
    comparison.paste(lora_img.resize((512, 512)), (512, 0))
    comparison_path = output_dir / "comparison.png"
    comparison.save(comparison_path)
    logger.info(f"Saved side-by-side: {comparison_path}")

    logger.info("=" * 60)
    logger.info("DONE. Inspect outputs:")
    logger.info(f"  baseline:   {baseline_path}")
    logger.info(f"  lora:       {lora_path}")
    logger.info(f"  comparison: {comparison_path}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
