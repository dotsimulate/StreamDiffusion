import os
import warnings

import torch
import torch.nn as nn


os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from .builder import EngineBuilder
from .models.models import BaseModel


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


class StableDiffusionSafetyCheckerWrapper(StableDiffusionSafetyChecker):
    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def forward(self, clip_input):
        pooled_output = self.vision_model(clip_input)[1]
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        adjustment = 0.0

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)

        return has_nsfw_concepts


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor):
        return retrieve_latents(self.vae.encode(x))


def compile_vae_encoder(
    vae: TorchVAEEncoder,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    vae = vae.to(torch.device("cuda"))
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )


def compile_vae_decoder(
    vae: AutoencoderKL,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    vae = vae.to(torch.device("cuda"))
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )


def compile_safety_checker(
    safety_checker: StableDiffusionSafetyCheckerWrapper,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    safety_checker = safety_checker.to(torch.device("cuda"))
    builder = EngineBuilder(model_data, safety_checker, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )


def compile_unet(
    unet: UNet2DConditionModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    # Extract FP8-specific options before passing the rest to EngineBuilder.build().
    # These are not valid kwargs for build_engine() and must be handled here.
    build_options = dict(engine_build_options)
    fp8 = build_options.pop("fp8", False)
    pipe_ref = build_options.pop("pipe_ref", None)
    calibration_prompts = build_options.pop("calibration_prompts", None)
    calibration_steps = build_options.pop("calibration_steps", 20)
    fp8_allow_fp16_fallback = build_options.pop("fp8_allow_fp16_fallback", False)
    fp8_use_cached_attn = build_options.pop("fp8_use_cached_attn", False)
    fp8_use_feature_injection = build_options.pop("fp8_use_feature_injection", False)
    fp8_use_controlnet = build_options.pop("fp8_use_controlnet", False)
    fp8_num_ip_layers = build_options.pop("fp8_num_ip_layers", 0)
    for _legacy in ("calibration_data_fn", "amax_save_path", "fp8_alpha"):
        if _legacy in build_options:
            warnings.warn(
                f"engine_build_options['{_legacy}'] is deprecated and ignored — the FP8 path "
                "switched to ONNX-level quantization. Remove this kwarg from your config.",
                DeprecationWarning,
                stacklevel=2,
            )
            build_options.pop(_legacy)

    unet = unet.to(torch.device("cuda"), dtype=torch.float16)
    builder = EngineBuilder(model_data, unet, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        fp8=fp8,
        pipe_ref=pipe_ref,
        calibration_prompts=calibration_prompts,
        calibration_steps=calibration_steps,
        fp8_allow_fp16_fallback=fp8_allow_fp16_fallback,
        fp8_use_cached_attn=fp8_use_cached_attn,
        fp8_use_feature_injection=fp8_use_feature_injection,
        fp8_use_controlnet=fp8_use_controlnet,
        fp8_num_ip_layers=fp8_num_ip_layers,
        **build_options,
    )


def compile_controlnet(
    controlnet: ControlNetModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    controlnet = controlnet.to(torch.device("cuda"), dtype=torch.float16)
    builder = EngineBuilder(model_data, controlnet, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        **engine_build_options,
    )
