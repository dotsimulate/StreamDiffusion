"""Golden-snapshot test for config.py's construction-time extraction functions.

Captures the exact output of _extract_wrapper_params / _extract_prepare_params
for a minimal config, BEFORE Stage 2 Increment 3 delegates their literal
defaults to param_schema.DEFAULTS. If Inc 3 changes any default value (instead
of just its source), this test catches the divergence — the output must stay
byte-identical.

t_index_list is asserted to remain a `list` (not param_schema.DEFAULTS'
internal tuple) since StreamDiffusionWrapper.__init__ types it List[int] and
downstream consumers (e.g. save_config/JSON) expect a list.
"""

import torch

from streamdiffusion.config import _extract_prepare_params, _extract_wrapper_params

MINIMAL_CONFIG = {"model_id": "stabilityai/sd-turbo"}

EXPECTED_WRAPPER_PARAMS = {
    "model_id_or_path": "stabilityai/sd-turbo",
    "t_index_list": [0, 16, 32, 45],
    "mode": "img2img",
    "output_type": "pil",
    "device": "cuda",
    "dtype": torch.float16,
    "frame_buffer_size": 1,
    "width": 512,
    "height": 512,
    "warmup": 10,
    "acceleration": "tensorrt",
    "do_add_noise": True,
    "use_tiny_vae": True,
    "enable_similar_image_filter": False,
    "similar_image_filter_threshold": 0.98,
    "similar_image_filter_max_skip_frame": 10,
    "similar_filter_sleep_fraction": 0.025,
    "use_denoising_batch": True,
    "cfg_type": "self",
    "seed": 2,
    "use_safety_checker": False,
    "skip_diffusion": False,
    "engine_dir": "engines",
    "normalize_prompt_weights": True,
    "normalize_seed_weights": True,
    "scheduler": "lcm",
    "sampler": "normal",
    "compile_engines_only": False,
    "static_shapes": False,
    "fp8": False,
    "vae_builder_optimization_level": 3,
    "build_engines_if_missing": True,
    "fp8_allow_fp16_fallback": False,
    "fp8_mha_qdq": False,
    "use_controlnet": False,
    "use_ipadapter": False,
    "use_cached_attn": False,
    "cache_maxframes": 1,
    "cache_interval": 1,
    "cn_cache_interval": 1,
    "cn_cache_decay": 0.0,
    "use_feature_injection": True,
    "fi_strength": 0.75,
    "fi_threshold": 0.98,
    "max_cache_maxframes": 4,
    "pin_cache_frames": False,
    "use_cuda_ipc_output": False,
    "cuda_ipc_num_slots": 2,
    "controlnet_preview_passthrough": False,
    "debug_mode": False,
}

EXPECTED_PREPARE_PARAMS = {
    "prompt": "",
    "negative_prompt": "",
    "num_inference_steps": 50,
    "guidance_scale": 1.2,
    "delta": 1.0,
}


def test_extract_wrapper_params_minimal_config_byte_identical():
    result = _extract_wrapper_params(MINIMAL_CONFIG)
    assert result == EXPECTED_WRAPPER_PARAMS


def test_extract_wrapper_params_t_index_list_is_a_list():
    """Must stay a `list`, not param_schema.DEFAULTS' internal immutable tuple."""
    result = _extract_wrapper_params(MINIMAL_CONFIG)
    assert isinstance(result["t_index_list"], list)


def test_extract_wrapper_params_t_index_list_not_aliased_to_schema_default():
    """Mutating the returned list must not corrupt param_schema.DEFAULTS or a
    second call's output (guards against a shared-mutable-default regression)."""
    result = _extract_wrapper_params(MINIMAL_CONFIG)
    result["t_index_list"].append(999)
    second_result = _extract_wrapper_params(MINIMAL_CONFIG)
    assert second_result["t_index_list"] == [0, 16, 32, 45]


def test_extract_prepare_params_minimal_config_byte_identical():
    result = _extract_prepare_params(MINIMAL_CONFIG)
    assert result == EXPECTED_PREPARE_PARAMS
