"""
Regression tests for fp8-round-10's fix to capture_calibration_data's missing
height/width threading -- without them, diffusers falls back to
pipe.unet.config.sample_size (512 for SDXL-Turbo) regardless of the engine's
actual build resolution, and nothing downstream can correct it because
`sample`'s ONNX H/W axes are exported dynamic. The statically-declared
kvo_cache_in_*/fio_cache_in_* inputs then get zero-padded by _reconcile_2d,
with a real-vs-padded seq fraction of (512/R)^2 -- 25% real at 1024, 11% at
1536. See fp8_quantize.py's _build_capture_call_kwargs /
capture_calibration_data docstrings and the fp8-round-10 plan for the full
derivation.

These tests exercise _build_capture_call_kwargs directly -- a pure,
module-level helper with no CUDA/pipeline/torch dependency -- rather than
capture_calibration_data itself, which needs a real diffusers pipeline.

Run with: pytest tests/unit/test_fp8_calibration_capture_resolution.py -v
"""

import pytest

from streamdiffusion.acceleration.tensorrt.fp8_quantize import _build_capture_call_kwargs


def test_no_resolution_matches_prior_behavior():
    """Omitting image_height/image_width (the default, and every pre-fp8-round-10
    caller's behavior) must not add height/width keys -- 512-native builds stay
    byte-for-byte unaffected by this change."""
    kwargs = _build_capture_call_kwargs(
        batch=["a portrait"],
        guidance_scale=7.5,
        use_explicit_schedule=False,
        timesteps=None,
        num_inference_steps=20,
    )
    assert kwargs == {
        "prompt": "a portrait",
        "output_type": "latent",
        "guidance_scale": 7.5,
        "num_inference_steps": 20,
    }
    assert "height" not in kwargs
    assert "width" not in kwargs


@pytest.mark.parametrize("image_height, image_width", [(1024, None), (None, 1024)])
def test_resolution_omitted_when_only_one_given(image_height, image_width):
    """A lone height or width is ambiguous -- mirrors capture_calibration_data's
    existing timesteps/scheduler_ref one-of-two handling. Neither key should
    appear rather than guessing the missing dimension."""
    kwargs = _build_capture_call_kwargs(
        batch=["a portrait"],
        guidance_scale=7.5,
        use_explicit_schedule=False,
        timesteps=None,
        num_inference_steps=20,
        image_height=image_height,
        image_width=image_width,
    )
    assert "height" not in kwargs
    assert "width" not in kwargs


def test_resolution_included_when_both_given_non_square():
    """Non-square case (1024x576): pins that the two axes thread through
    independently rather than one silently mirroring the other."""
    kwargs = _build_capture_call_kwargs(
        batch=["a portrait"],
        guidance_scale=7.5,
        use_explicit_schedule=False,
        timesteps=None,
        num_inference_steps=20,
        image_height=1024,
        image_width=576,
    )
    assert kwargs["height"] == 1024
    assert kwargs["width"] == 576


def test_explicit_schedule_uses_timesteps_not_num_inference_steps():
    kwargs = _build_capture_call_kwargs(
        batch=["a portrait"],
        guidance_scale=1.0,
        use_explicit_schedule=True,
        timesteps=[999, 761, 499],
        num_inference_steps=999,  # must be ignored
    )
    assert kwargs["timesteps"] == [999, 761, 499]
    assert "num_inference_steps" not in kwargs


def test_non_explicit_schedule_uses_num_inference_steps_not_timesteps():
    kwargs = _build_capture_call_kwargs(
        batch=["a portrait"],
        guidance_scale=1.0,
        use_explicit_schedule=False,
        timesteps=[999, 761, 499],  # must be ignored
        num_inference_steps=20,
    )
    assert kwargs["num_inference_steps"] == 20
    assert "timesteps" not in kwargs


@pytest.mark.parametrize("use_explicit_schedule", [True, False])
def test_resolution_orthogonal_to_schedule_branch(use_explicit_schedule):
    """height/width must thread through regardless of which of the two
    mutually-exclusive schedule branches (timesteps vs num_inference_steps) is
    active -- they are independent axes of the same call, so the resolution fix
    must not depend on which schedule mode a given build uses."""
    kwargs = _build_capture_call_kwargs(
        batch=["a portrait"],
        guidance_scale=1.0,
        use_explicit_schedule=use_explicit_schedule,
        timesteps=[999, 761],
        num_inference_steps=20,
        image_height=1024,
        image_width=1024,
    )
    assert kwargs["height"] == 1024
    assert kwargs["width"] == 1024
    if use_explicit_schedule:
        assert kwargs["timesteps"] == [999, 761]
        assert "num_inference_steps" not in kwargs
    else:
        assert kwargs["num_inference_steps"] == 20
        assert "timesteps" not in kwargs


def test_batch_of_multiple_prompts_passed_as_list():
    """batch_size > 1: prompt must be the full list, not just batch[0] (existing
    pre-fp8-round-10 behavior, pinned here since it moved into the new helper)."""
    kwargs = _build_capture_call_kwargs(
        batch=["prompt a", "prompt b"],
        guidance_scale=7.5,
        use_explicit_schedule=False,
        timesteps=None,
        num_inference_steps=20,
    )
    assert kwargs["prompt"] == ["prompt a", "prompt b"]


def test_batch_of_single_prompt_passed_as_string():
    kwargs = _build_capture_call_kwargs(
        batch=["prompt a"],
        guidance_scale=7.5,
        use_explicit_schedule=False,
        timesteps=None,
        num_inference_steps=20,
    )
    assert kwargs["prompt"] == "prompt a"
