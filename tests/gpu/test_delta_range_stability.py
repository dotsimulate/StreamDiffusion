"""GPU stability sweep: the max clamp value delta=5.0 must not NaN the
fp16 R-CFG recurrence.

delta enters the CFG combine as gamma*eps_text - (gamma-1)*delta*stock_noise
(pipeline.unet_step) and, at denoising_steps_num > 1, scales the stock_noise
recurrence coefficient through model_pred. The ping-pong rotation reseeds each
row every n frames, bounding growth at ~|A|^(n-1) — but at the clamp corner
(delta=5, gamma=2) that coefficient is ~15x the long-verified delta=1/gamma=1.4
case. This test locks "clamp maximum cannot overflow fp16": finite output only,
no luminance assert — past the gamma/(gamma-1) noise-cancellation ceiling the
output is degraded by design, and that is allowed.

Run: venv/Scripts/python.exe -m pytest tests/gpu -v
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
INPUT_IMAGE = os.path.join(REPO_ROOT, "images", "inputs", "input.png")

N_FRAMES = 60


def _run_frames(t_index_list, guidance_scale, delta):
    from streamdiffusion import StreamDiffusionWrapper

    stream = StreamDiffusionWrapper(
        model_id_or_path="stabilityai/sd-turbo",
        t_index_list=t_index_list,
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
        guidance_scale=guidance_scale,
        delta=delta,
    )

    image_tensor = stream.preprocess_image(INPUT_IMAGE)
    for frame in range(N_FRAMES):
        output = stream(image=image_tensor)
        assert torch.isfinite(output).all(), (
            f"non-finite output at frame {frame} (t_index_list={t_index_list}, gamma={guidance_scale}, delta={delta})"
        )


class TestDeltaMaxStability:
    def test_delta_max_single_step(self):
        # n==1: per-frame reseed path, no recurrence — delta only scales the
        # combine's init_noise subtraction.
        _run_frames([16], guidance_scale=1.4, delta=5.0)

    def test_delta_max_two_step_recurrence(self):
        # n>1 recurrence at the largest allowed (gamma-1)*delta product.
        _run_frames([16, 32], guidance_scale=2.0, delta=5.0)
