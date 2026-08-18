"""
Sweep num_inference_steps (S in {10, 15, 20, 25, 30}) across all samplers and both
production models to find where timestep-spacing actually changes output.

NOTE: this script has not been run (images/outputs/spacing_compare/ does not exist).
The numbers below are derived from the LCM scheduler's own algorithm, not measured.

Key insight: pipeline.py's _initialize_scheduler forces original_inference_steps=100,
so the LCM distillation grid is [9, 19, ..., 999] (congruent 9 mod 10), not [19, 39,
..., 999] (mod 20) as an original_inference_steps=50 assumption would give.
  normal:       always subsamples this grid -> always on-grid at every S.
  sgm_uniform:  (trailing) coincides with the distillation grid only when S divides 100
                (divisors in this range: {10, 20, 25}) -> MSE~0 vs normal.
                At non-divisors (15, 30) it steps off-grid and diverges from normal.
  ddim:         (leading) off-grid at essentially all non-trivial S; excludes t=999.
  simple:       (linspace) off-grid; spans full [0, 999].

t_index is scaled proportionally to hold denoising strength constant across S:
  production [32, 45] at S=50 -> fractions 0.64, 0.90
  -> t_index = [round(0.64*S), round(0.90*S)]

Run from the repo's StreamDiffusion/ dir in its venv:
    python examples/txt2img/spacing_compare.py
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from streamdiffusion import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE = os.path.join(CURRENT_DIR, "..", "..", "images", "inputs", "input.png")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "spacing_compare")

MODELS = [
    "stabilityai/sd-turbo",
    "stabilityai/sdxl-turbo",
]
STEP_COUNTS = [10, 15, 20, 25, 30]  # divisors of 100: {10, 20, 25}; non-divisors: {15, 30}
SAMPLERS = [
    ("normal", "LCM native (baseline, always on distillation grid)"),
    ("sgm_uniform", "trailing  — on-grid only when S divides 100"),
    ("ddim", "leading   — off-grid, excludes t=999"),
    ("simple", "linspace  — off-grid, spans [0, 999]"),
]
PROMPT = "a peaceful mountain landscape at golden hour"
SEED = 2
WIDTH = HEIGHT = 512


def proportional_t_index(S: int) -> list[int]:
    """Scale production t_index=[32,45] at S=50 (fractions 0.64, 0.90) to arbitrary S."""
    a = min(round(0.64 * S), S - 2)
    b = min(round(0.90 * S), S - 1)
    if b <= a:
        b = a + 1
    return [a, b]


def mean_brightness(img: Image.Image) -> float:
    return float(np.array(img).astype(np.float32).mean() / 255.0)


def mse(a: Image.Image, b: Image.Image) -> float:
    return float(((np.array(a).astype(np.float32) - np.array(b).astype(np.float32)) ** 2).mean())


def on_grid(sub_ts) -> bool:
    """All sub-timesteps on the LCM distillation grid (congruent 9 mod 10)."""
    return all(int(t) % 10 == 9 for t in sub_ts)


def run_model(model_id: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_tag = model_id.split("/")[-1]

    for S in STEP_COUNTS:
        t_index = proportional_t_index(S)
        is_divisor = 100 % S == 0
        print(f"\n{'=' * 70}")
        print(
            f"Model: {model_id}  S={S}  t_index={t_index}  "
            f"({'divides 100 -> trailing==normal' if is_divisor else 'does NOT divide 100 -> trailing diverges'})"
        )
        print(f"{'=' * 70}")

        results: dict[str, Image.Image] = {}

        for sampler_name, description in SAMPLERS:
            wrapper = StreamDiffusionWrapper(
                model_id_or_path=model_id,
                t_index_list=t_index,
                frame_buffer_size=1,
                width=WIDTH,
                height=HEIGHT,
                warmup=10,
                acceleration="none",
                mode="img2img",
                use_denoising_batch=True,
                cfg_type="none",
                seed=SEED,
                scheduler="lcm",
                sampler=sampler_name,
            )
            wrapper.prepare(
                prompt=PROMPT,
                num_inference_steps=S,
            )

            image_tensor = wrapper.preprocess_image(INPUT_IMAGE)

            sub_ts = wrapper.stream.sub_timesteps
            grid_flag = on_grid(sub_ts)

            for _ in range(wrapper.batch_size - 1):
                wrapper(image=image_tensor)
            img = wrapper(image=image_tensor)

            out_path = os.path.join(OUTPUT_DIR, f"{model_tag}_S{S:02d}_{sampler_name}.png")
            img.save(out_path)
            results[sampler_name] = img

            brightness = mean_brightness(img)
            print(
                f"  {sampler_name:12s}  on_grid={grid_flag!s:5s}  sub_ts={[int(t) for t in sub_ts]}\n"
                f"    brightness={brightness:.4f}  ({description})"
            )

            del wrapper
            torch.cuda.empty_cache()

        print(f"\n  MSE vs 'normal' baseline (S={S}):")
        baseline = results["normal"]
        for sampler_name, _ in SAMPLERS:
            if sampler_name != "normal":
                err = mse(results[sampler_name], baseline)
                print(f"    {sampler_name:12s}  MSE={err:.2f}")

    print(f"\n  Images saved to: {OUTPUT_DIR}/{model_tag}_S*_*.png")


if __name__ == "__main__":
    for model_id in MODELS:
        run_model(model_id)
