"""GPU e2e regression: RCFG 'self' with a single denoising step must not go black.

Locks down the bug fixed in pipeline.py's predict_x0_batch: at
denoising_steps_num == 1 the ping-pong reseed never runs while unet_step's
cross-frame recurrence has growth factor |A| > 1, so stock_noise diverged
geometrically -> fp16 Inf -> NaN in the CFG combine -> black frames within
seconds whenever guidance_scale > 1. The fix reseeds stock_noise from
init_noise every frame (per paper Eq. 5 the Self-Negative residual at the
first step is exactly init_noise).

Pre-fix, divergence hit fp16 Inf within ~5 frames; 120 frames is a wide
red-capable margin. Uses stabilityai/sd-turbo + taesd (both HF-cached),
acceleration="none" to avoid TRT engine builds.

Run: venv/Scripts/python.exe -m pytest tests/gpu -v
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
INPUT_IMAGE = os.path.join(REPO_ROOT, "images", "inputs", "input.png")

N_FRAMES = 120


class TestRcfgSelfSingleStepE2E:
    def test_single_step_guidance_above_one_stays_visible(self):
        from streamdiffusion import StreamDiffusionWrapper

        stream = StreamDiffusionWrapper(
            model_id_or_path="stabilityai/sd-turbo",
            t_index_list=[16],
            frame_buffer_size=1,
            width=512,
            height=512,
            warmup=1,
            acceleration="none",
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="self",
            output_type="pt",
            seed=42,
        )
        stream.prepare(
            prompt="a photograph of a cat",
            negative_prompt="",
            num_inference_steps=50,
            guidance_scale=1.4,
            delta=1.0,
        )

        image_tensor = stream.preprocess_image(INPUT_IMAGE)
        for frame in range(N_FRAMES):
            output = stream(image=image_tensor)
            assert torch.isfinite(output).all(), f"non-finite output at frame {frame}"

        # Mean luminance over 0-255: a black (NaN-clamped) frame sits at ~0.
        luminance = output.float().mean().item() * 255.0
        assert luminance > 10.0, f"output collapsed to black: mean luminance {luminance:.2f}"
