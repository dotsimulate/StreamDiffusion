import logging
import math
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from diffusers import LCMScheduler, StableDiffusionPipeline, TCDScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

from streamdiffusion.hooks import (
    EmbeddingHook,
    EmbedsCtx,
    ImageCtx,
    ImageHook,
    LatentCtx,
    LatentHook,
    StepCtx,
    UnetHook,
    UnetKwargsDelta,
)
from streamdiffusion.image_filter import SimilarImageFilter
from streamdiffusion.model_detection import detect_model, resolve_is_turbo
from streamdiffusion.param_schema import (
    VALID_CFG_TYPES,
    bleed_risk_message,
    clamp_delta,
    compute_sub_timesteps,
    delta_noise_cancellation_ceiling,
    materialise_timestep_grid,
)
from streamdiffusion.stream_parameter_updater import StreamParameterUpdater
from streamdiffusion.tools.gpu_profiler import profiler
from streamdiffusion.utils.nan_guard import NanGuard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamDiffusion:
    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        t_index_list: List[int],
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        lora_dict: Optional[Dict[str, float]] = None,
        normalize_prompt_weights: bool = True,
        normalize_seed_weights: bool = True,
        scheduler: Literal["lcm", "tcd"] = "lcm",
        sampler: Literal["simple", "sgm_uniform", "normal", "ddim", "beta", "karras"] = "normal",
        kvo_cache: Optional[List[torch.Tensor]] = None,
        cache_interval: int = 1,
        cache_maxframes: int = 1,
        fio_cache: Optional[List[torch.Tensor]] = None,
        use_feature_injection: bool = False,
        fi_strength: float = 0.75,
        fi_threshold: float = 0.98,
        is_turbo: Optional[bool] = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = torch_dtype
        # P1: TF32/cuDNN performance flags — idempotent, set once per process per CUDA device.
        # Enables TF32 accumulation on Ampere+ for fp32 matmuls/convs (~3× throughput) and
        # cuDNN autotuning for fixed-size conv inputs.  Safe: UNet runs in fp16 via TRT;
        # only PyTorch VAE encoder/decoder and preprocessing convs are affected.
        # Grounding: PMPP Ch.5 roofline; CUDA HB Ch.6 stream overlap.
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        self.generator = None
        self._input_staging: Optional[torch.Tensor] = None

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)

        # G6: cfg_type is Literal-hinted but that's not enforced at runtime — a typo
        # previously behaved like "none" until the guidance combine, then raised an
        # opaque UnboundLocalError. Validate here, the one choke point every
        # construction path (direct + wrapper.py) flows through.
        if cfg_type not in VALID_CFG_TYPES:
            raise ValueError(f"cfg_type must be one of {VALID_CFG_TYPES}, got {cfg_type!r}")
        self.cfg_type = cfg_type
        self.scheduler_type = scheduler
        self.sampler_type = sampler

        # Detect model type. Turbo status is resolved once, authoritatively, by the caller
        # (StreamDiffusionWrapper._load_model, via resolve_is_turbo) and passed in as
        # `is_turbo` -- detect_model only reports architecture. The fallback below keeps
        # direct StreamDiffusion(...) construction (examples, tests) working without a
        # wrapper: it re-derives the scheduler-only signal via resolve_is_turbo itself,
        # same as _load_model would for a non-single-file load.
        detection_result = detect_model(pipe.unet, pipe)
        self.model_type = detection_result["model_type"]
        self.is_sdxl = detection_result["is_sdxl"]
        self.is_turbo = is_turbo if is_turbo is not None else resolve_is_turbo(pipe=pipe)[0]
        self.detection_confidence = detection_result["confidence"]

        # TCD scheduler is incompatible with denoising batch optimization due to Strategic Stochastic Sampling
        # Force sequential processing for TCD
        if scheduler == "tcd":
            logger.info("TCD scheduler detected: Disabling denoising batch optimization for compatibility")
            logger.info("TCD now supports ControlNet through proper hook processing")
            self.use_denoising_batch = False
            self.batch_size = frame_buffer_size
            self.trt_unet_batch_size = frame_buffer_size
        elif use_denoising_batch:
            self.use_denoising_batch = True
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (self.denoising_steps_num + 1) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = 2 * self.denoising_steps_num * self.frame_bff_size
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.use_denoising_batch = False
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.t_list = t_index_list
        self.do_add_noise = do_add_noise

        # Schedule-diagnostics dedup state — see _log_schedule_diagnostics.
        self._last_schedule_signature = None
        self._last_bleed_message: Optional[str] = None

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None
        self.prev_latent_result = None
        # Published each frame by geometric FX processors (e.g. feedback_loop) as the
        # [B,2,3] affine theta they applied, or None when identity. Read by unet_step()
        # to attenuate Feature Injection in proportion to warp magnitude — see
        # _fi_warp_attenuation().
        self.fx_frame_transform: Optional[torch.Tensor] = None
        self._latent_cache = None  # pre-allocated buffer; avoids per-frame CUDA malloc for prev_latent_result
        self._noise_buf = None  # pre-allocated buffer for per-step noise in TCD non-batched path
        self._image_decode_buf = None  # pre-allocated buffer for VAE decode output (avoids .clone())
        self._prev_image_buf = None  # pre-allocated buffer for skip-frame image cache
        self._combined_latent_buf = None  # pre-allocated: avoids torch.cat in predict_x0_batch
        self._alpha_next = None  # pre-computed: cat([alpha_prod_t_sqrt[1:], ones[0:1]])
        self._beta_next = None  # pre-computed: cat([beta_prod_t_sqrt[1:], ones[0:1]])
        self._init_noise_rotated = None  # pre-computed: cat([init_noise[1:], init_noise[0:1]])
        self._unet_kwargs: dict = {"return_dict": False}  # pre-allocated: avoids per-frame dict creation
        self._cfg_latent_buf = None  # pre-allocated: avoids torch.concat for CFG latent doubling
        self._cfg_t_buf = None  # pre-allocated: avoids torch.concat for CFG timestep doubling

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)
        self.scheduler = self._initialize_scheduler(scheduler, sampler, pipe.scheduler.config)

        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        self.inference_time_ema = 0
        self._sync_counter = 0  # counts __call__ invocations for timing-sync cadence (P2)
        self.similar_filter_sleep_fraction = 0.025
        self.last_frame_was_skipped = False  # True when similar filter skipped inference this frame

        # NaN/Inf guards on the hot path — see utils/nan_guard.py docstring. One
        # instance per guarded buffer so each keeps its own warn-once state and
        # names itself in the WARNING. Sanitizing is unconditional every call
        # (branch-free); only the recovery ACTION below (stock_noise reseed) is
        # gated on a 1-frame-delayed, non-blocking verdict.
        self._nan_guard_model_pred = NanGuard("unet_step.model_pred")
        self._nan_guard_image_input = NanGuard("__call__.image_input")
        self._nan_guard_prev_image = NanGuard("_prev_image_buf")
        self._nan_guard_latent_cache = NanGuard("_latent_cache")
        # Guard 7 (_fi_warp_attenuation) already forces a host sync via float() every call,
        # so it checks isfinite() inline rather than through NanGuard -- but still needs its
        # own warn-once flag to avoid flooding if fx_frame_transform stays NaN.
        self._warned_fi_warp_attenuation: bool = False

        # Initialize SDXL-specific attributes
        if self.is_sdxl:
            self.add_text_embeds = None
            self.add_time_ids = None

        # Initialize parameter updater
        self._param_updater = StreamParameterUpdater(
            self,
            normalize_prompt_weights=normalize_prompt_weights,
            normalize_seed_weights=normalize_seed_weights,
        )

        # Hook containers (step 1: introduced but initially no-op)
        self.embedding_hooks: List[EmbeddingHook] = []
        self.unet_hooks: List[UnetHook] = []

        # Phase 1: Core Pipeline Hooks (Immediate Priority)
        self.image_preprocessing_hooks: List[ImageHook] = []
        self.latent_preprocessing_hooks: List[LatentHook] = []
        self.latent_postprocessing_hooks: List[LatentHook] = []

        # Phase 2: Quality & Performance Hooks
        self.image_postprocessing_hooks: List[ImageHook] = []
        self.image_filtering_hooks: List[ImageHook] = []

        # Cache TensorRT detection to avoid repeated hasattr checks
        self._is_unet_tensorrt = None

        # Cache SDXL conditioning tensors to avoid repeated torch.cat/repeat operations
        self._sdxl_conditioning_cache: Dict[str, torch.Tensor] = {}
        self._cached_batch_size: Optional[int] = None
        self._cached_cfg_type: Optional[str] = None
        self._cached_guidance_scale: Optional[float] = None

        self.kvo_cache = kvo_cache if kvo_cache is not None else []
        self._kvo_buckets = None
        self._kvo_outputs_by_bucket = None
        self.cache_interval = cache_interval
        self.cache_maxframes = cache_maxframes
        self.frame_idx = 0

        # Feature Injection (FI) — output cache (O-cache) for StreamV2V thesis §3.4.2.
        # fio_cache[i] shape: (cache_maxframes, B, S, H) — pre-allocated by wrapper.py.
        # Persistent [1] fp32 scalars are updated in-place by param_updater (CUDA-graph-safe).
        self.fio_cache: List[torch.Tensor] = fio_cache if fio_cache is not None else []
        self.use_feature_injection: bool = use_feature_injection and bool(self.fio_cache)
        if self.use_feature_injection:
            self._fi_strength_tensor: Optional[torch.Tensor] = torch.tensor(
                [fi_strength], dtype=torch.float32, device=self.device
            )
            self._fi_threshold_tensor: Optional[torch.Tensor] = torch.tensor(
                [fi_threshold], dtype=torch.float32, device=self.device
            )
        else:
            self._fi_strength_tensor = None
            self._fi_threshold_tensor = None
        # User-set Fistrength slider value. unet_step() writes _fi_strength_tensor as
        # _fi_strength_base * _fi_warp_attenuation() every frame, so a live slider
        # change (which updates this base, not the tensor) is never clobbered by the
        # warp attenuation and vice versa.
        self._fi_strength_base: float = fi_strength

        # Runtime-adjustable LoRA weight — a [num_loras] fp32 vector, one entry per
        # loaded adapter, index-aligned with stream._lora_order. LoRA loading happens
        # in wrapper.py._load_model *after* this constructor returns (adapters are
        # discovered from lora_dict there, not here), so this starts None and is set
        # for real by that call once _lora_order is known. Unlike fi_strength there is
        # no per-frame recomputation in unet_step — only stream_parameter_updater
        # mutates it in place (CUDA-graph-safe: same device address across frames).
        self._lora_scale_tensor: Optional[torch.Tensor] = None

        # Ordered (path, adapter_name) list wrapper.py._load_model populates once LoRA
        # adapters are discovered from lora_dict — the index<->path mapping consumed by
        # unet_unified_export.py and stream_parameter_updater.py. Declared here (not just
        # assigned dynamically in wrapper.py) so static type checking can resolve it.
        self._lora_order: List[Tuple[str, str]] = []

        # Pre-allocated CUDA timing events — reused every frame via .record()
        self._timing_start = torch.cuda.Event(enable_timing=True)
        self._timing_end = torch.cuda.Event(enable_timing=True)

    def _initialize_scheduler(self, scheduler_type: str, sampler_type: str, config):
        """Initialize scheduler based on type and sampler configuration."""

        # TODO: More testing and validation required on samplers.
        # Map sampler types to configuration parameters
        sampler_config = {
            "simple": {"timestep_spacing": "linspace"},
            "sgm_uniform": {"timestep_spacing": "trailing"},
            "normal": {},  # Default configuration
            "ddim": {"timestep_spacing": "leading"},
            "beta": {"beta_schedule": "scaled_linear"},  # no-op: equals SD/SDXL default
            "karras": {},  # no-op: LCM/TCD have no karras-sigma logic
        }

        # Get sampler-specific configuration
        sampler_params = sampler_config.get(sampler_type, {})

        # Set original_inference_steps to 100 to allow flexible num_inference_steps updates
        # This prevents "original_steps x strength < num_inference_steps" errors when
        # dynamically changing num_inference_steps in production without pipeline restarts
        sampler_params["original_inference_steps"] = 100

        if scheduler_type == "lcm":
            return LCMScheduler.from_config(config, **sampler_params)
        elif scheduler_type == "tcd":
            return TCDScheduler.from_config(config, **sampler_params)
        else:
            logger.warning(f"Unknown scheduler type '{scheduler_type}', falling back to LCM")
            return LCMScheduler.from_config(config, **sampler_params)

    def _get_spaced_timesteps(self, spacing: str, num_inference_steps: int) -> torch.Tensor:
        """Return a descending timestep schedule per the paper's Table 2 spacing strategies.

        LCMScheduler and TCDScheduler store timestep_spacing in config but never consume it
        in set_timesteps (they always build a linearly-spaced grid). This produces the
        correct leading/linspace/trailing schedules so the sampler option actually takes
        effect. 'normal' bypasses this and uses the LCM/TCD native grid.
        """
        T = self.scheduler.config.num_train_timesteps
        S = num_inference_steps
        if spacing == "trailing":
            # round(arange(T, 0, -T/S)) - 1 — descending, includes T-1
            ts = torch.arange(T, 0, -T / S).round().long() - 1
        elif spacing == "linspace":
            # linspace(0, T-1, S) reversed — includes both 0 and T-1
            ts = torch.linspace(0, T - 1, S).round().long().flip(0)
        else:  # "leading"
            # arange(S) * (T//S) reversed — includes 0, excludes T-1
            ts = (torch.arange(S, dtype=torch.float32) * (T // S)).round().long().flip(0)
        return ts.clamp(0, T - 1)

    def _check_unet_tensorrt(self) -> bool:
        """Cache TensorRT detection to avoid repeated hasattr calls"""
        if self._is_unet_tensorrt is None:
            self._is_unet_tensorrt = hasattr(self.unet, "engine") and hasattr(self.unet, "stream")
        return self._is_unet_tensorrt

    def _get_cached_sdxl_conditioning(
        self, batch_size: int, cfg_type: str, guidance_scale: float
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve cached SDXL conditioning tensors if configuration matches"""
        if (
            self._cached_batch_size == batch_size
            and self._cached_cfg_type == cfg_type
            and self._cached_guidance_scale == guidance_scale
            and len(self._sdxl_conditioning_cache) > 0
        ):
            return {
                "text_embeds": self._sdxl_conditioning_cache.get("text_embeds"),
                "time_ids": self._sdxl_conditioning_cache.get("time_ids"),
            }
        return None

    def _cache_sdxl_conditioning(
        self, batch_size: int, cfg_type: str, guidance_scale: float, text_embeds: torch.Tensor, time_ids: torch.Tensor
    ) -> None:
        """Cache SDXL conditioning tensors for reuse"""
        self._cached_batch_size = batch_size
        self._cached_cfg_type = cfg_type
        self._cached_guidance_scale = guidance_scale
        self._sdxl_conditioning_cache["text_embeds"] = text_embeds.clone()
        self._sdxl_conditioning_cache["time_ids"] = time_ids.clone()

    def _build_sdxl_conditioning(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Build SDXL conditioning tensors with optimized tensor operations"""
        # Replicate add_text_embeds and add_time_ids to match the batch size
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            # For initialize mode: [uncond, cond, cond, ...]
            # Use more efficient tensor operations
            uncond_text = self.add_text_embeds[0:1]
            cond_text = self.add_text_embeds[1:2]
            uncond_time = self.add_time_ids[0:1]
            cond_time = self.add_time_ids[1:2]

            if batch_size > 1:
                cond_text_repeated = cond_text.expand(batch_size - 1, -1).contiguous()
                cond_time_repeated = cond_time.expand(batch_size - 1, -1).contiguous()
                add_text_embeds = torch.cat([uncond_text, cond_text_repeated], dim=0)
                add_time_ids = torch.cat([uncond_time, cond_time_repeated], dim=0)
            else:
                add_text_embeds = uncond_text
                add_time_ids = uncond_time

        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            # For full mode: repeat both uncond and cond for each latent
            repeat_factor = batch_size // 2
            add_text_embeds = self.add_text_embeds.expand(repeat_factor, -1).contiguous()
            add_time_ids = self.add_time_ids.expand(repeat_factor, -1).contiguous()
        else:
            # No CFG: just repeat the conditioning
            source_text = self.add_text_embeds[1:2] if self.add_text_embeds.shape[0] > 1 else self.add_text_embeds
            source_time = self.add_time_ids[1:2] if self.add_time_ids.shape[0] > 1 else self.add_time_ids
            add_text_embeds = source_text.expand(batch_size, -1).contiguous()
            add_time_ids = source_time.expand(batch_size, -1).contiguous()
        return {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self._load_lora_with_offline_fallback(pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs)

    def _load_lora_with_offline_fallback(
        self,
        pretrained: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any],
        **kwargs,
    ) -> None:
        """
        Load LoRA weights, auto-guessing common weight filenames when HF offline mode is enabled.
        """
        try:
            self.pipe.load_lora_weights(pretrained, adapter_name, **kwargs)
            return
        except Exception as e:
            message = str(e)
            is_offline_weight_error = isinstance(e, ValueError) and "must specify a `weight_name`" in message
            if not is_offline_weight_error:
                raise

        candidate_weight_names = (
            "pytorch_lora_weights.safetensors",
            "pytorch_lora_weights.bin",
            "diffusion_pytorch_model.safetensors",
            "adapter_model.safetensors",
            "lora.safetensors",
        )

        last_err: Optional[Exception] = None
        for weight_name in candidate_weight_names:
            try:
                self.pipe.load_lora_weights(
                    pretrained,
                    adapter_name,
                    **{**kwargs, "weight_name": weight_name},
                )
                return
            except Exception as e:
                last_err = e
                logger.debug(f"load_lora_weights: candidate {weight_name!r} failed: {e}", exc_info=True)
                continue

        if last_err is not None:
            raise last_err

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    def _nan_guard(self, attr: str, name: str) -> NanGuard:
        """Lazily fetch/create a NanGuard stored at `attr`.

        __init__ already sets the standard guards up front (self-documenting,
        the normal path), but some tests construct StreamDiffusion via
        object.__new__ to drive predict_x0_batch/unet_step directly without a
        real model (test_rcfg_self_single_step_reseed.py, test_derived_tensor_sync.py),
        bypassing __init__ entirely. Falling back to lazy creation here keeps
        the guards resilient to that construction path too.
        """
        guard = getattr(self, attr, None)
        if guard is None:
            guard = NanGuard(name)
            setattr(self, attr, guard)
        return guard

    @torch.inference_mode()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = None,
        seed: int = 2,
    ) -> None:
        if generator is None:
            generator = torch.Generator()
        self.generator = generator
        self.generator.manual_seed(seed)
        self.current_seed = seed

        # Pinned CPU staging buffer for async H2D DMA via non_blocking=True.
        # Pageable memory silently falls back to synchronous double-buffered staging (~½ bandwidth).
        # dtype=self.dtype: avoids a transient GPU fp32 buffer that PyTorch would allocate
        # when casting pageable float32 input to fp16 on the device.
        self._input_staging = torch.zeros(1, 3, self.height, self.width, dtype=self.dtype).pin_memory()

        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        clamped_delta, was_clamped = clamp_delta(delta)
        if was_clamped:
            logger.warning(
                f"prepare: delta={delta} outside the valid R-CFG range [1.0, 5.0]; clamped to {clamped_delta}"
            )
        self.delta = clamped_delta
        _ceiling = delta_noise_cancellation_ceiling(self.guidance_scale)
        if self.delta > _ceiling:
            logger.warning(
                f"prepare: delta={self.delta} exceeds the noise-cancellation ceiling "
                f"gamma/(gamma-1)={_ceiling:.2f} at guidance_scale={self.guidance_scale} — "
                "output will re-inject noise"
            )

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True

        # G5 (not fixed — see plan): cfg_type="full" has been broken at baseline since
        # _apply_prompt_blending's live-update path builds batch_size//2 uncond+cond rows
        # (stream_parameter_updater.py, batch_size == frame_bff_size collapses that to 0
        # at denoising_steps_num == 1) while unet_step doubles the latent to 2*batch_size.
        # No shipped config uses cfg_type="full", so this has never been observed in
        # practice. Warn loudly rather than silently producing garbage/shape-error output.
        if self.cfg_type == "full" and self.guidance_scale > 1.0:
            logger.warning(
                "prepare: cfg_type='full' is known-broken (embeds row count vs. UNet batch "
                "mismatch after the first prompt update) — not repaired by this change; "
                "use cfg_type='self' or 'initialize' instead."
            )

        # Handle SDXL vs SD1.5/SD2.1 text encoding differently
        if self.is_sdxl:
            # SDXL encode_prompt returns 4 values:
            # (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
            encoder_output = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,  # Use same prompt for both encoders
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=None,  # Use same negative prompt for both encoders
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=None,
                clip_skip=None,
            )

            if len(encoder_output) >= 4:
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                    encoder_output[:4]
                )

                # Set up prompt embeddings for the UNet (base before hooks)
                base_prompt_embeds = prompt_embeds.repeat(self.batch_size, 1, 1)

                # Handle CFG for prompt embeddings
                # G4: uncond_prompt_embeds is only assigned under specific
                # use_denoising_batch/cfg_type combos but consumed unconditionally below
                # whenever guidance_scale>1 and cfg_type is initialize/full. TCD forces
                # use_denoising_batch=False, so {scheduler: tcd, cfg_type: full} left this
                # unbound. Initialize it and turn the unreachable-combo case into a legible
                # error instead of UnboundLocalError.
                uncond_prompt_embeds = None
                if self.use_denoising_batch and self.cfg_type == "full":
                    uncond_prompt_embeds = negative_prompt_embeds.repeat(self.batch_size, 1, 1)
                elif self.cfg_type == "initialize":
                    uncond_prompt_embeds = negative_prompt_embeds.repeat(self.frame_bff_size, 1, 1)

                if self.guidance_scale > 1.0 and (self.cfg_type == "initialize" or self.cfg_type == "full"):
                    if uncond_prompt_embeds is None:
                        raise ValueError(
                            f"prepare: cfg_type={self.cfg_type!r} with guidance_scale>1.0 requires "
                            f"use_denoising_batch=True (got {self.use_denoising_batch}); cfg_type='full' "
                            "is not supported with use_denoising_batch=False (e.g. the TCD scheduler, "
                            "which forces it off)."
                        )
                    base_prompt_embeds = torch.cat([uncond_prompt_embeds, base_prompt_embeds], dim=0)

                # Set up SDXL-specific conditioning (added_cond_kwargs)
                if do_classifier_free_guidance:
                    self.add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                else:
                    self.add_text_embeds = pooled_prompt_embeds

                # Create time conditioning for SDXL micro-conditioning
                original_size = (self.height, self.width)
                target_size = (self.height, self.width)
                crops_coords_top_left = (0, 0)

                add_time_ids = list(original_size + crops_coords_top_left + target_size)
                add_time_ids = torch.tensor([add_time_ids], dtype=self.dtype, device=self.device)

                if do_classifier_free_guidance:
                    self.add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
                else:
                    self.add_time_ids = add_time_ids
            else:
                raise ValueError(f"SDXL encode_prompt returned {len(encoder_output)} outputs, expected at least 4")
            # Run embedding hooks (no-op unless modules register)
            embeds_ctx = EmbedsCtx(prompt_embeds=base_prompt_embeds, negative_prompt_embeds=None)
            for hook in self.embedding_hooks:
                try:
                    embeds_ctx = hook(embeds_ctx)
                except Exception as e:
                    logger.error(f"prepare: embedding hook failed: {e}")
                    raise
            self.prompt_embeds = embeds_ctx.prompt_embeds
        else:
            # SD1.5/SD2.1 encode_prompt returns 2 values: (prompt_embeds, negative_prompt_embeds)
            encoder_output = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
            )
            base_prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

            # G4 (see the SDXL branch above for the full explanation).
            uncond_prompt_embeds = None
            if self.use_denoising_batch and self.cfg_type == "full":
                uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
            elif self.cfg_type == "initialize":
                uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

            if self.guidance_scale > 1.0 and (self.cfg_type == "initialize" or self.cfg_type == "full"):
                if uncond_prompt_embeds is None:
                    raise ValueError(
                        f"prepare: cfg_type={self.cfg_type!r} with guidance_scale>1.0 requires "
                        f"use_denoising_batch=True (got {self.use_denoising_batch}); cfg_type='full' "
                        "is not supported with use_denoising_batch=False (e.g. the TCD scheduler, "
                        "which forces it off)."
                    )
                base_prompt_embeds = torch.cat([uncond_prompt_embeds, base_prompt_embeds], dim=0)

            # Run embedding hooks (no-op unless modules register)
            embeds_ctx = EmbedsCtx(prompt_embeds=base_prompt_embeds, negative_prompt_embeds=None)
            for hook in self.embedding_hooks:
                try:
                    embeds_ctx = hook(embeds_ctx)
                except Exception as e:
                    logger.error(f"prepare: embedding hook failed: {e}")
                    raise
            self.prompt_embeds = embeds_ctx.prompt_embeds

        # LCM/TCD ignore timestep_spacing natively. Apply the correct grid for samplers
        # that explicitly request a spacing; leave "normal" on the LCM native schedule.
        # Shared with wrapper.py's fp8-calibration path via materialise_timestep_grid so
        # the two can never silently drift on what "spacing override" means.
        self.timesteps = materialise_timestep_grid(
            self.scheduler, num_inference_steps, self.sampler_type, self.device, self._get_spaced_timesteps
        )

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = compute_sub_timesteps(self.timesteps, self.t_list)

        sub_timesteps_tensor = torch.tensor(self.sub_timesteps, dtype=torch.long, device=self.device)
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        # Clone init_noise rather than zeros so stock_noise starts coherent when CFG is active.
        # Initialising with zeros forces a cold-restart warm-up of the RCFG residual evolution;
        # the reference fork (pipeline_td.py:1103-1105) uses clone() to avoid that.
        self.stock_noise = self.init_noise.clone()

        # Ping-pong buffers for stock_noise rotation.
        # Replaces the per-frame torch.cat((init_noise[0:1], stock_noise[:-1]))
        # which allocates a new tensor every frame.  Two preallocated buffers
        # are alternated so src and dst never alias.
        self._stock_noise_bufs = [self.stock_noise.clone(), torch.empty_like(self.stock_noise)]
        self._stock_noise_pong = 0  # next write-target index (0 or 1)

        # Handle scheduler-specific scaling calculations
        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self._get_scheduler_scalings(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = torch.stack(c_skip_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)
        self.c_out = torch.stack(c_out_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        # alpha_prod_t_sqrt = √ᾱₜ  (cumulative signal-retention std)
        # beta_prod_t_sqrt  = √(1−ᾱₜ) (cumulative MARGINAL noise std, NOT per-step βₜ)
        # Notation follows diffusers convention; ᾱₜ = Πᵢ(1−βᵢ) is computed by the scheduler.
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        # NOTE: this is a hack. Pipeline needs a major refactor along with stream parameter updater.
        self.update_prompt(prompt)

        # Only collapse tensors to scalars for LCM non-batched mode
        # TCD needs to keep tensor dimensions for iteration
        if not self.use_denoising_batch and isinstance(self.scheduler, LCMScheduler):
            self.sub_timesteps_tensor = self.sub_timesteps_tensor[0]
            self.alpha_prod_t_sqrt = self.alpha_prod_t_sqrt[0]
            self.beta_prod_t_sqrt = self.beta_prod_t_sqrt[0]

        self.sub_timesteps_tensor = self.sub_timesteps_tensor.to(self.device)
        self.c_skip = self.c_skip.to(self.device)
        self.c_out = self.c_out.to(self.device)

        # Precompute per-step expanded timestep tensors for the TCD / non-batched sequential loop (G1).
        self._rebuild_sub_timesteps_expanded()

        # Pre-compute shifted alpha/beta/init_noise (eliminates 5 mallocs + 8 kernel launches per frame)
        if self.use_denoising_batch and (self.cfg_type == "self" or self.cfg_type == "initialize"):
            self._alpha_next = torch.cat(
                [self.alpha_prod_t_sqrt[1:], torch.ones_like(self.alpha_prod_t_sqrt[0:1])], dim=0
            )
            self._beta_next = torch.cat(
                [self.beta_prod_t_sqrt[1:], torch.ones_like(self.beta_prod_t_sqrt[0:1])], dim=0
            )
            self._init_noise_rotated = torch.cat([self.init_noise[1:], self.init_noise[0:1]], dim=0)
        else:
            self._alpha_next = None
            self._beta_next = None
            self._init_noise_rotated = None

        # Pre-allocate combined latent buffer (eliminates 1 malloc + copy kernel per frame)
        if self.denoising_steps_num > 1:
            self._combined_latent_buf = torch.empty(
                (self.batch_size, 4, self.latent_height, self.latent_width),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self._combined_latent_buf = None

        # Pre-allocate CFG expansion buffers (eliminates torch.concat mallocs for latent/timestep doubling)
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize" or self.cfg_type == "full"):
            cfg_batch = (1 + self.batch_size) if self.cfg_type == "initialize" else (2 * self.batch_size)
            self._cfg_latent_buf = torch.empty(
                (cfg_batch, 4, self.latent_height, self.latent_width),
                dtype=self.dtype,
                device=self.device,
            )
            self._cfg_t_buf = torch.empty(cfg_batch, dtype=self.sub_timesteps_tensor.dtype, device=self.device)
        else:
            self._cfg_latent_buf = None
            self._cfg_t_buf = None

        # Seed _unet_kwargs with the constant key so per-frame code only updates values
        self._unet_kwargs = {"return_dict": False}

        # Diagnostics-only: never let a logging failure abort prepare() itself
        # (e.g. a partially-constructed test/debug shell missing an attribute
        # this dump reads should not turn a successful prepare() into a crash).
        try:
            self._log_schedule_diagnostics("prepare")
        except Exception as exc:
            logger.debug("schedule diagnostics (prepare) skipped: %r", exc)

    def _rebuild_sub_timesteps_expanded(self) -> None:
        """(Re)build the per-step expanded timestep table consumed by the TCD /
        non-batched sequential loop in predict_x0_batch. Avoids per-step
        t.view(1).repeat(frame_bff_size) tensor allocations. Only valid when
        sub_timesteps_tensor is a 1-D sequence (not the collapsed-scalar LCM path).

        Called from prepare() and from _refresh_derived_tensors() (G1) — a live
        t_index_list length change previously left this table at its old length,
        raising IndexError on the sequential path (predict_x0_batch indexes it by
        loop position) rather than the RuntimeError __call__'s fallback catches.
        Idempotent, so the extra call from that fallback is harmless.
        """
        _use_seq_loop = not (self.use_denoising_batch and isinstance(self.scheduler, LCMScheduler))
        if _use_seq_loop and self.sub_timesteps_tensor.dim() >= 1:
            self._sub_timesteps_expanded = (
                self.sub_timesteps_tensor.view(-1).unsqueeze(1).expand(-1, self.frame_bff_size).contiguous()
            )  # shape [loop_steps, frame_bff_size]
        else:
            self._sub_timesteps_expanded = None

    def _refresh_derived_tensors(self) -> None:
        """Re-create tensors derived from batch_size/scheduler state but NOT rebuilt by
        StreamParameterUpdater._update_timestep_calculations.

        Called after _param_updater._update_timestep_calculations() has already
        refreshed sub_timesteps_tensor, c_skip/c_out, alpha/beta_prod_t_sqrt — by
        the error-fallback handler in __call__ and by the updater's live
        t_index_list length-change path (_recalculate_timestep_dependent_params).
        """
        # G1: keep the sequential-loop timestep table in sync with the just-refreshed
        # sub_timesteps_tensor (its own length may have just changed).
        self._rebuild_sub_timesteps_expanded()

        # init_noise + stock_noise — re-sampled so the next frame is coherent
        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=self.generator,
        ).to(device=self.device, dtype=self.dtype)
        self.stock_noise = self.init_noise.clone()

        # Ping-pong buffers for the stock_noise rotation (see prepare()) — batch-sized,
        # so any batch_size change without this rebuild leaves predict_x0_batch's
        # copy_ failing every frame with no way to self-heal.
        self._stock_noise_bufs = [self.stock_noise.clone(), torch.empty_like(self.stock_noise)]
        self._stock_noise_pong = 0

        # Pre-computed shifted tensors (depend on alpha/beta already refreshed above)
        if self.use_denoising_batch and (self.cfg_type == "self" or self.cfg_type == "initialize"):
            self._alpha_next = torch.cat(
                [self.alpha_prod_t_sqrt[1:], torch.ones_like(self.alpha_prod_t_sqrt[0:1])], dim=0
            )
            self._beta_next = torch.cat(
                [self.beta_prod_t_sqrt[1:], torch.ones_like(self.beta_prod_t_sqrt[0:1])], dim=0
            )
            self._init_noise_rotated = torch.cat([self.init_noise[1:], self.init_noise[0:1]], dim=0)
        else:
            self._alpha_next = None
            self._beta_next = None
            self._init_noise_rotated = None

        # Pre-allocated latent scratch buffers (batch-size-dependent)
        if self.denoising_steps_num > 1:
            self._combined_latent_buf = torch.empty(
                (self.batch_size, 4, self.latent_height, self.latent_width),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self._combined_latent_buf = None

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize" or self.cfg_type == "full"):
            cfg_batch = (1 + self.batch_size) if self.cfg_type == "initialize" else (2 * self.batch_size)
            self._cfg_latent_buf = torch.empty(
                (cfg_batch, 4, self.latent_height, self.latent_width),
                dtype=self.dtype,
                device=self.device,
            )
            self._cfg_t_buf = torch.empty(cfg_batch, dtype=self.sub_timesteps_tensor.dtype, device=self.device)
        else:
            self._cfg_latent_buf = None
            self._cfg_t_buf = None

        # Diagnostics-only: never let a logging failure abort the __call__
        # RuntimeError-fallback rebuild this method also serves as (see the
        # signature-gate note in _log_schedule_diagnostics's own docstring).
        try:
            self._log_schedule_diagnostics("refresh")
        except Exception as exc:
            logger.debug("schedule diagnostics (refresh) skipped: %r", exc)

    def _get_scheduler_scalings(self, timestep: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get LCM/TCD-specific scaling factors for boundary conditions."""
        if isinstance(self.scheduler, LCMScheduler):
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            return c_skip, c_out
        else:
            # TCD and other schedulers don't use boundary condition scaling like LCM
            # They handle scaling internally in their step() method
            # Return tensors that are compatible with torch.stack()
            c_skip = torch.tensor(1.0, device=self.device, dtype=self.dtype)
            c_out = torch.tensor(1.0, device=self.device, dtype=self.dtype)
            return c_skip, c_out

    def _log_schedule_diagnostics(self, tag: str) -> None:
        """Log the realised schedule so a do_add_noise / t_index investigation
        doesn't have to hand-derive its numbers: the grid, the per-index
        sub_timestep table (t, alpha_sqrt, beta_sqrt, c_skip, c_out), the
        do_add_noise reachability verdict, RCFG activity, and cross-frame
        state. Called from the end of prepare() (tag="prepare") and the end
        of _refresh_derived_tensors() (tag="refresh").

        Gated on a schedule signature — (t_list, num_inference_steps,
        sampler_type, scheduler class, batch_size, do_add_noise) — because
        _refresh_derived_tensors() is not only reached at stream start and on
        a live schedule change: its third caller is __call__'s RuntimeError
        fallback (an "expanded size ... must match" recovery, __call__
        ~:1527-1537), which re-runs every frame if the mismatch is
        persistent rather than transient. Without this gate that path would
        flood the log with an identical dump every frame; with it, a repeat
        is one skipped call and a genuine change still re-logs.
        """
        num_inference_steps = len(self.timesteps)
        signature = (
            tuple(self.t_list),
            num_inference_steps,
            self.sampler_type,
            type(self.scheduler).__name__,
            self.batch_size,
            self.do_add_noise,
        )
        if signature == self._last_schedule_signature:
            return
        self._last_schedule_signature = signature

        n_steps = len(self.t_list)

        def _row(flat: torch.Tensor, i: int) -> float:
            # alpha/beta/c_skip/c_out may or may not be repeat_interleave'd by
            # frame_bff_size depending on which code path last touched them —
            # prepare() never repeat_interleaves c_skip/c_out, but
            # stream_parameter_updater._update_timestep_calculations does — so
            # read by size rather than assume one fixed layout. Falls back to
            # the last element when i is out of range (e.g. the non-batched
            # LCM scalar collapse in prepare(), :695-699).
            flat = flat.reshape(-1)
            n = flat.shape[0]
            if n == n_steps * self.frame_bff_size and self.frame_bff_size > 1:
                idx = i * self.frame_bff_size
            elif n == n_steps:
                idx = i
            else:
                idx = min(i, n - 1)
            return float(flat[min(idx, n - 1)])

        lines = [
            f"[schedule:{tag}] grid: scheduler={type(self.scheduler).__name__} "
            f"sampler={self.sampler_type} steps={num_inference_steps} t_index_list={list(self.t_list)}"
        ]
        if tag == "refresh":
            lines.append(
                "  note: _refresh_derived_tensors() just re-sampled init_noise -- "
                "cross-frame grain is NOT preserved across this refresh"
            )
        lines.append(" idx |    t |   a_sqrt |  b_sqrt  |   b/a   |   c_skip |    c_out")
        for i, t_idx in enumerate(self.t_list):
            sub_t = int(self.sub_timesteps[i])
            a = _row(self.alpha_prod_t_sqrt, i)
            b = _row(self.beta_prod_t_sqrt, i)
            cs = _row(self.c_skip, i)
            co = _row(self.c_out, i)
            ratio = (b / a) if a else float("inf")
            lines.append(f" {t_idx:>3} | {sub_t:>4} | {a:>8.4f} | {b:>8.4f} | {ratio:>7.4f} | {cs:>8.2e} | {co:>8.6f}")

        lines.append(
            f"denoising_steps_num={self.denoising_steps_num} use_denoising_batch={self.use_denoising_batch} "
            f"frame_buffer_size={self.frame_bff_size} batch_size={self.batch_size} "
            f"trt_unet_batch_size={self.trt_unet_batch_size}"
        )

        # do_add_noise reachability verdict — see predict_x0_batch's three branches.
        if isinstance(self.scheduler, TCDScheduler):
            reachable = False
            reason = "TCD scheduler never reads do_add_noise (predict_x0_batch TCD branch, pipeline.py:1448-1458)"
        elif self.use_denoising_batch and isinstance(self.scheduler, LCMScheduler):
            reachable = self.denoising_steps_num > 1
            reason = (
                "batched-LCM buffer rebuild only reads do_add_noise when denoising_steps_num > 1 "
                f"(pipeline.py:1393,1420); denoising_steps_num={self.denoising_steps_num}"
            )
        else:
            reachable = n_steps > 1
            reason = (
                "sequential/non-batched loop only reads do_add_noise on a non-final step "
                f"(pipeline.py:1463-1464); t_index_list has {n_steps} "
                f"entr{'y' if n_steps == 1 else 'ies'}"
            )
        lines.append(
            f"do_add_noise: configured={self.do_add_noise} EFFECTIVE={reachable}"
            + ("" if reachable else f"  unreachable: {reason}")
        )

        rcfg_active = self.guidance_scale > 1.0 and self.cfg_type != "none"
        lines.append(
            f"cfg_type={self.cfg_type!r} guidance_scale={self.guidance_scale} delta={self.delta} "
            f"RCFG_active={rcfg_active}"
            + ("" if rcfg_active else "  (guidance_scale<=1.0 or cfg_type='none': RCFG combine skipped)")
        )

        x_t_buf = self.x_t_latent_buffer
        lines.append(
            "x_t_latent_buffer="
            + ("None" if x_t_buf is None else f"shape={tuple(x_t_buf.shape)}")
            + f" cached_attn_active(kvo_cache)={bool(self.kvo_cache)}"
            f" use_feature_injection={self.use_feature_injection}"
            f" similar_image_filter={self.similar_image_filter}"
        )

        # Ghost-bleed check (param_schema.bleed_risk_message) — the same helper
        # stream_parameter_updater._update_timestep_calculations calls, so this
        # is the boot-time counterpart: prepare() never calls that method, so
        # today a config that boots straight into the bad regime never warns.
        bleed_msg = bleed_risk_message(
            [_row(self.beta_prod_t_sqrt, i) for i in range(1, n_steps)],
            self.t_list,
            self.use_denoising_batch,
            self.do_add_noise,
        )
        if bleed_msg is not None:
            lines.append(f"ghost-bleed risk: {bleed_msg}")
            # Dedup against _update_timestep_calculations's own warning: a live
            # length-changed t_index update calls both that method and this one
            # for the same schedule build, and should warn once, not twice.
            if bleed_msg != self._last_bleed_message:
                logger.warning(bleed_msg)
        self._last_bleed_message = bleed_msg

        logger.info("\n".join(lines))

    @torch.inference_mode()
    def update_prompt(self, prompt: str) -> None:
        self._param_updater.update_stream_params(prompt_list=[(prompt, 1.0)], prompt_interpolation_method="average")

    def get_normalize_prompt_weights(self) -> bool:
        """Get the current prompt weight normalization setting."""
        return self._param_updater.get_normalize_prompt_weights()

    def get_normalize_seed_weights(self) -> bool:
        """Get the current seed weight normalization setting."""
        return self._param_updater.get_normalize_seed_weights()

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        noisy_samples = self.alpha_prod_t_sqrt[t_index] * original_samples + self.beta_prod_t_sqrt[t_index] * noise
        return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        with profiler.region("sched.step_batch"):
            if idx is None:
                # Upcast division to fp32 — alpha_prod_t_sqrt can be small at early timesteps,
                # causing fp16 rounding artifacts; cast result back to original dtype.
                F_theta = (
                    (x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch).float()
                    / self.alpha_prod_t_sqrt.float()
                ).to(x_t_latent_batch.dtype)
                denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
            else:
                F_theta = (
                    (x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch).float()
                    / self.alpha_prod_t_sqrt[idx].float()
                ).to(x_t_latent_batch.dtype)
                denoised_batch = self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
            return denoised_batch

    # Empirical: at zoom=1.02 (warp magnitude ~0.028) attenuates FI to ~0.66x; scales
    # down smoothly to ~0 by zoom~1.1. FI is a hard per-token gate that fights the warp
    # (see fx_frame_transform docs) — EA is left untouched since it's soft/attention-weighted.
    _FI_WARP_ATTENUATION_K = 15.0

    def _fi_warp_attenuation(self) -> float:
        theta = self.fx_frame_transform
        if theta is None:
            return 1.0
        identity = torch.eye(2, device=theta.device, dtype=theta.dtype)
        linear_dev = (theta[:, :, :2] - identity).flatten(1).norm(dim=1)
        translation_dev = theta[:, :, 2].norm(dim=1)
        warp_mag = (linear_dev + translation_dev).max()
        attenuation = float(torch.exp(-self._FI_WARP_ATTENUATION_K * warp_mag).clamp(0.0, 1.0))
        # NaN guard 7: this already forces a host read via float() above (no extra sync
        # cost), so check inline rather than through NanGuard. A non-finite fx_frame_transform
        # (e.g. a NaN FX zoom/rotation) would otherwise become a NaN FI strength at the call
        # site below and NaN the entire UNet -- fall back to no attenuation instead.
        if not math.isfinite(attenuation):
            # getattr fallback: tests construct StreamDiffusion via object.__new__,
            # bypassing __init__ (see pipeline._nan_guard docstring for the same concern).
            if not getattr(self, "_warned_fi_warp_attenuation", False):
                logger.warning(
                    "pipeline._fi_warp_attenuation: non-finite result (fx_frame_transform has "
                    "NaN/Inf) -- falling back to 1.0 (no attenuation); further occurrences "
                    "logged at DEBUG"
                )
                self._warned_fi_warp_attenuation = True
            else:
                logger.debug("pipeline._fi_warp_attenuation: non-finite result -- falling back to 1.0")
            return 1.0
        return attenuation

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with profiler.region("unet_step.fi_attenuation"):
            if self.use_feature_injection and self._fi_strength_tensor is not None:
                self._fi_strength_tensor.fill_(self._fi_strength_base * self._fi_warp_attenuation())

        with profiler.region("unet_step.prep"):
            if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
                # Pre-allocated buf avoids torch.concat malloc for CFG latent doubling
                self._cfg_latent_buf[:1].copy_(x_t_latent[0:1])
                self._cfg_latent_buf[1:].copy_(x_t_latent)
                x_t_latent_plus_uc = self._cfg_latent_buf
                self._cfg_t_buf[:1].copy_(t_list[0:1])
                self._cfg_t_buf[1:].copy_(t_list)
                t_list = self._cfg_t_buf
            elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
                self._cfg_latent_buf[: len(x_t_latent)].copy_(x_t_latent)
                self._cfg_latent_buf[len(x_t_latent) :].copy_(x_t_latent)
                x_t_latent_plus_uc = self._cfg_latent_buf
                self._cfg_t_buf[: len(t_list)].copy_(t_list)
                self._cfg_t_buf[len(t_list) :].copy_(t_list)
                t_list = self._cfg_t_buf
            else:
                x_t_latent_plus_uc = x_t_latent

            # Prepare UNet call arguments (update pre-allocated dict in-place: avoids per-frame dict malloc)
            self._unet_kwargs["sample"] = x_t_latent_plus_uc
            self._unet_kwargs["timestep"] = t_list
            self._unet_kwargs["encoder_hidden_states"] = self.prompt_embeds
            unet_kwargs = self._unet_kwargs

        # Add SDXL-specific conditioning if this is an SDXL model
        with profiler.region("unet_step.sdxl_cond"):
            if self.is_sdxl and hasattr(self, "add_text_embeds") and hasattr(self, "add_time_ids"):
                if self.add_text_embeds is not None and self.add_time_ids is not None:
                    # Handle batching for CFG - replicate conditioning to match batch size
                    batch_size = x_t_latent_plus_uc.shape[0]

                    # Use optimized caching system for SDXL conditioning tensors
                    cached_conditioning = self._get_cached_sdxl_conditioning(
                        batch_size, self.cfg_type, self.guidance_scale
                    )
                    if cached_conditioning is not None:
                        # Cache hit - reuse existing tensors
                        add_text_embeds = cached_conditioning["text_embeds"]
                        add_time_ids = cached_conditioning["time_ids"]
                    else:
                        # Cache miss - build new tensors using optimized operations
                        conditioning = self._build_sdxl_conditioning(batch_size)
                        add_text_embeds = conditioning["text_embeds"]
                        add_time_ids = conditioning["time_ids"]
                        # Cache for future use
                        self._cache_sdxl_conditioning(
                            batch_size, self.cfg_type, self.guidance_scale, add_text_embeds, add_time_ids
                        )

                    unet_kwargs["added_cond_kwargs"] = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        # Allow modules to contribute additional UNet kwargs via hooks
        with profiler.region("unet_step.hooks"):
            if self.unet_hooks:
                try:
                    step_ctx = StepCtx(
                        x_t_latent=x_t_latent_plus_uc,
                        t_list=t_list,
                        step_index=idx if isinstance(idx, int) else (int(idx) if idx is not None else None),
                        guidance_mode=self.cfg_type if self.guidance_scale > 1.0 else "none",
                        sdxl_cond=unet_kwargs.get("added_cond_kwargs", None),
                    )
                    extra_from_hooks = {}
                    for hook in self.unet_hooks:
                        delta: UnetKwargsDelta = hook(step_ctx)
                        if delta is None:
                            continue
                        if delta.down_block_additional_residuals is not None:
                            unet_kwargs["down_block_additional_residuals"] = delta.down_block_additional_residuals
                        if delta.mid_block_additional_residual is not None:
                            unet_kwargs["mid_block_additional_residual"] = delta.mid_block_additional_residual
                        if delta.added_cond_kwargs is not None:
                            # Merge SDXL cond if both exist
                            base_added = unet_kwargs.get("added_cond_kwargs", {})
                            base_added.update(delta.added_cond_kwargs)
                            unet_kwargs["added_cond_kwargs"] = base_added
                        if getattr(delta, "extra_unet_kwargs", None):
                            # Merge extra kwargs from hooks (e.g., ipadapter_scale)
                            try:
                                extra_from_hooks.update(delta.extra_unet_kwargs)
                            except Exception as e:
                                logger.debug(
                                    f"unet_step: failed to merge extra_unet_kwargs from hook: {e}", exc_info=True
                                )
                    if extra_from_hooks:
                        unet_kwargs["extra_unet_kwargs"] = extra_from_hooks
                except Exception as e:
                    logger.error(f"unet_step: unet hook failed: {e}")
                    raise

        # Extract potential ControlNet residual kwargs / extra kwargs (e.g., ipadapter_scale), then call UNet
        with profiler.region("unet_step.engine"):
            # Guard 1's verdict is read unconditionally in unet_step.post (:1375), so it must be
            # bound on EVERY branch below -- the two SDXL branches previously left it unbound.
            nan_guard_bad_last_frame = False
            hook_down_res = unet_kwargs.get("down_block_additional_residuals", None)
            hook_mid_res = unet_kwargs.get("mid_block_additional_residual", None)
            hook_extra_kwargs = (
                unet_kwargs.get("extra_unet_kwargs", None) if "extra_unet_kwargs" in unet_kwargs else None
            )

            if self.is_sdxl:
                try:
                    # Detect UNet type and use appropriate calling convention
                    added_cond_kwargs = unet_kwargs.get("added_cond_kwargs", {})

                    # Check if this is a TensorRT engine or PyTorch UNet
                    is_tensorrt_engine = self._check_unet_tensorrt()

                    if is_tensorrt_engine:
                        # TensorRT engine expects positional args + kwargs. IP-Adapter scale vector, if any, is provided by hooks via extra_unet_kwargs
                        extra_kwargs = {}
                        if isinstance(hook_extra_kwargs, dict):
                            extra_kwargs.update(hook_extra_kwargs)

                        # Include ControlNet residuals if provided by hooks
                        if hook_down_res is not None:
                            extra_kwargs["down_block_additional_residuals"] = hook_down_res
                        if hook_mid_res is not None:
                            extra_kwargs["mid_block_additional_residual"] = hook_mid_res

                        _unet_result = self.unet(
                            unet_kwargs["sample"],  # latent_model_input (positional)
                            unet_kwargs["timestep"],  # timestep (positional)
                            unet_kwargs["encoder_hidden_states"],  # encoder_hidden_states (positional)
                            kvo_cache=self.kvo_cache,
                            fio_cache=self.fio_cache,
                            fi_strength=self._fi_strength_tensor,
                            fi_threshold=self._fi_threshold_tensor,
                            lora_scale=self._lora_scale_tensor,
                            **extra_kwargs,
                            # For TRT engines, ensure SDXL cond shapes match engine builds; if engine expects 81 tokens (77+4), append dummy image tokens when none
                            **added_cond_kwargs,  # SDXL conditioning as kwargs
                        )
                        model_pred = _unet_result[0]
                        kvo_cache_out = _unet_result[1] if len(_unet_result) > 1 else []
                        fio_cache_out = _unet_result[2] if len(_unet_result) > 2 else []
                        # Sanitize predictions now; cache ingress has its own guards.
                        # A delayed model verdict cannot identify this call's cache values.
                        nan_guard_bad_last_frame = self._nan_guard(
                            "_nan_guard_model_pred", "unet_step.model_pred"
                        ).sanitize_(model_pred)
                        self.update_kvo_cache(kvo_cache_out, fio_cache_out)
                    else:
                        # PyTorch UNet expects diffusers-style named arguments. Any processor scaling is handled by IP-Adapter hook

                        call_kwargs = {
                            "sample": unet_kwargs["sample"],
                            "timestep": unet_kwargs["timestep"],
                            "encoder_hidden_states": unet_kwargs["encoder_hidden_states"],
                            "added_cond_kwargs": added_cond_kwargs,
                            "return_dict": False,
                        }
                        # Include ControlNet residuals if present
                        if hook_down_res is not None:
                            call_kwargs["down_block_additional_residuals"] = hook_down_res
                        if hook_mid_res is not None:
                            call_kwargs["mid_block_additional_residual"] = hook_mid_res
                        model_pred = self.unet(**call_kwargs)[0]
                        # No restoration for per-layer scale; next step will set again via updater/time factor

                        # NaN guard 1: this branch never calls update_kvo_cache, so sanitize only.
                        nan_guard_bad_last_frame = self._nan_guard(
                            "_nan_guard_model_pred", "unet_step.model_pred"
                        ).sanitize_(model_pred)

                except Exception as e:
                    logger.error(f"[PIPELINE] unet_step: *** ERROR: SDXL UNet call failed: {e} ***")
                    import traceback

                    traceback.print_exc()
                    raise
            else:
                # For SD1.5/SD2.1, use the old calling convention for compatibility
                # Build kwargs from hooks and include residuals
                ip_scale_kw = {}
                if isinstance(hook_extra_kwargs, dict):
                    ip_scale_kw.update(hook_extra_kwargs)

                # PyTorch processor time scaling is handled by the IP-Adapter hook

                # Include ControlNet residuals if present
                if hook_down_res is not None:
                    ip_scale_kw["down_block_additional_residuals"] = hook_down_res
                if hook_mid_res is not None:
                    ip_scale_kw["mid_block_additional_residual"] = hook_mid_res

                # kvo_cache (StreamV2V cached-attention) is understood by both the
                # TensorRT engine and the diffusers UNet (patched at import time via
                # _patches/diffusers_kvo_patch.py), so it's always safe to pass. Feature
                # injection (fio_cache/fi_strength/fi_threshold), however, is a
                # TensorRT-engine-only capability — the patched PyTorch UNet's forward()
                # has no such parameters and raises TypeError if passed one.
                is_tensorrt_engine = self._check_unet_tensorrt()
                if is_tensorrt_engine:
                    ip_scale_kw["fio_cache"] = self.fio_cache
                    ip_scale_kw["fi_strength"] = self._fi_strength_tensor
                    ip_scale_kw["fi_threshold"] = self._fi_threshold_tensor
                    ip_scale_kw["lora_scale"] = self._lora_scale_tensor

                _unet_result = self.unet(
                    x_t_latent_plus_uc,
                    t_list,
                    encoder_hidden_states=self.prompt_embeds,
                    kvo_cache=self.kvo_cache,
                    return_dict=False,
                    **ip_scale_kw,
                )
                model_pred = _unet_result[0]
                kvo_cache_out = _unet_result[1] if len(_unet_result) > 1 else []
                fio_cache_out = _unet_result[2] if len(_unet_result) > 2 else []

                # NaN guard 1 (highest value): sanitizes model_pred unconditionally so a
                # poisoned UNet output can't propagate into the CFG combine, the scheduler
                # step, or the self/initialize stock_noise recurrence below. The returned
                # CUDA verdict is delayed (possibly several calls on a busy stream).
                # Use it below to reseed stock_noise, not to discard current cache
                # outputs: sanitized cache writes must continue replacing old history.
                nan_guard_bad_last_frame = self._nan_guard("_nan_guard_model_pred", "unet_step.model_pred").sanitize_(
                    model_pred
                )
                self.update_kvo_cache(kvo_cache_out, fio_cache_out)

        with profiler.region("unet_step.post"):
            if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
                noise_pred_text = model_pred[1:]
                self.stock_noise[0:1].copy_(model_pred[0:1])  # in-place: eliminates 1 malloc + copy kernel
            elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
                noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
            else:
                noise_pred_text = model_pred

            if self.guidance_scale > 1.0 and (self.cfg_type == "self" or self.cfg_type == "initialize"):
                noise_pred_uncond = self.stock_noise * self.delta

            if self.guidance_scale > 1.0 and self.cfg_type != "none":
                model_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

            # Residual propagation only matters when a next step exists to consume it.
            # At denoising_steps_num == 1 the write would persist a value the n>1 design
            # treats as a throwaway (_alpha_next/_beta_next degenerate to 1.0), and
            # predict_x0_batch reseeds stock_noise from init_noise every frame anyway.
            if (self.cfg_type == "self" or self.cfg_type == "initialize") and self.denoising_steps_num > 1:
                if nan_guard_bad_last_frame:
                    # NaN guard 1 recovery: the previous frame poisoned model_pred, which
                    # (pre-guard) would have poisoned self.stock_noise via this same
                    # recurrence -- reseed from the clean init_noise before continuing it,
                    # reusing predict_x0_batch's n==1 recovery (pipeline.py's per-frame
                    # reseed comment above `self.stock_noise.copy_(self.init_noise)`).
                    self.stock_noise.copy_(self.init_noise)
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                delta_x = self._alpha_next * delta_x
                delta_x = delta_x / self._beta_next
                self.stock_noise = self._init_noise_rotated + delta_x

        else:
            # denoised_batch = self.scheduler.step(model_pred, t_list[0], x_t_latent).denoised
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        return denoised_batch, model_pred

    def update_kvo_cache(
        self,
        kvo_cache_out: List[torch.Tensor],
        fio_cache_out: Optional[List[torch.Tensor]] = None,
    ) -> None:
        fio_cache_out = fio_cache_out or []
        has_kvo = bool(self.kvo_cache)
        has_fio = bool(fio_cache_out) and bool(self.fio_cache)

        if not has_kvo and not has_fio:
            return

        self.frame_idx += 1

        # Slot-latch circular buffer: written on EVERY unet_step call, but the write
        # pointer advances only every cache_interval calls. The latched slot tracks the
        # newest output (always exactly one call old — a smooth temporal anchor for
        # EA/FI), while the remaining slots freeze at ~cache_interval spacing, keeping
        # the long style-consistency window (worst-case span loss vs the old
        # write-skipping scheme is cache_interval - 1 calls). Skipping writes instead
        # made the whole bank's age oscillate and shift by cache_interval at once —
        # visible judder at FPS/interval.
        # The attention processor reads all slots as an unordered K/V bag, so slot order is irrelevant.
        # Use self.cache_maxframes (not tensor shape) so that when the buffer is allocated at
        # max_cache_maxframes but the logical window is smaller, writes stay within the active range.
        write_slot = (self.frame_idx // self.cache_interval - 1) % self.cache_maxframes

        if has_kvo:
            if self._kvo_buckets is not None:
                # Bucketed path: N stacks + N copies (N≈3) instead of ~70 individual copies.
                # Each kvo_cache_out[i] arrives shape (2, 1, B, S, H); squeeze(1) → (2, B, S, H).
                for bucket_idx, output_indices in enumerate(self._kvo_outputs_by_bucket):
                    stacked = torch.stack([kvo_cache_out[i].squeeze(1) for i in output_indices], dim=0)
                    stacked.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    self._kvo_buckets[bucket_idx][:, :, write_slot].copy_(stacked)
            else:
                # Fallback path — used when buckets were dropped after a kvo_cache resize
                # (stream_parameter_updater rebinds kvo_cache[i] to standalone tensors).
                for i, new_kv in enumerate(kvo_cache_out):
                    new_kv.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    self.kvo_cache[i][:, write_slot].copy_(new_kv.squeeze(1))

        if has_fio:
            # FI output cache update — each fio_cache_out[i] has shape (1, B, S, H).
            # Squeeze dim 0 to (B, S, H) and write into the circular buffer slot.
            for i, new_fi in enumerate(fio_cache_out):
                new_fi.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                self.fio_cache[i][write_slot].copy_(new_fi.squeeze(0))

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        with profiler.region("encode_image"):
            image_tensors = image_tensors.to(
                device=self.device,
                dtype=self.vae.dtype,
                non_blocking=True,
            )
            with torch.autocast("cuda", dtype=self.dtype, enabled=self.vae.dtype != torch.float32):
                img_latent = retrieve_latents(self.vae.encode(image_tensors), self.generator)

            img_latent = (img_latent * self.vae.config.scaling_factor).to(self.dtype)

            x_t_latent = self.add_noise(img_latent, self.init_noise[0], 0)
            return x_t_latent

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        with profiler.region("decode_image"):
            scaled_latent = x_0_pred_out.to(self.vae.dtype) / self.vae.config.scaling_factor
            with torch.autocast("cuda", dtype=self.dtype, enabled=self.vae.dtype != torch.float32):
                output_latent = self.vae.decode(scaled_latent, return_dict=False)[0]
            return output_latent

    def predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        with profiler.region("predict_x0_batch"):
            prev_latent_batch = self.x_t_latent_buffer
            # Repair recurrent inputs as well as outputs: overflow in scheduler
            # arithmetic or a latent hook can poison these independently of UNet.
            x_t_latent.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if prev_latent_batch is not None:
                prev_latent_batch.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            self.stock_noise.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

            # LCM supports our denoising-batch trick. TCD must use standard scheduler.step() sequentially
            # but now properly processes ControlNet hooks through unet_step()
            if self.use_denoising_batch and isinstance(self.scheduler, LCMScheduler):
                t_list = self.sub_timesteps_tensor
                if self.denoising_steps_num > 1:
                    # Copy into pre-allocated buffer: eliminates 1 malloc + copy kernel vs torch.cat
                    self._combined_latent_buf[: self.frame_bff_size].copy_(x_t_latent)
                    self._combined_latent_buf[self.frame_bff_size :].copy_(prev_latent_batch)
                    x_t_latent = self._combined_latent_buf
                    # Ping-pong rotation: eliminates torch.cat malloc every frame.
                    # _stock_noise_bufs[pong] is always a different tensor from self.stock_noise,
                    # so src and dst never alias.
                    _sn_dst = self._stock_noise_bufs[self._stock_noise_pong]
                    _sn_dst[0].copy_(self.init_noise[0])
                    _sn_dst[1:].copy_(self.stock_noise[:-1])
                    self.stock_noise = _sn_dst
                    self._stock_noise_pong = 1 - self._stock_noise_pong
                elif self.cfg_type == "self" or self.cfg_type == "initialize":
                    # denoising_steps_num == 1: no ping-pong slot exists to reseed, and the
                    # RCFG cross-frame recurrence has growth factor |A| > 1 at typical
                    # single-step timesteps — without this per-frame reseed stock_noise
                    # diverges geometrically and guidance > 1 renders black (fp16 Inf→NaN).
                    # Per paper Eq. 5, the Self-Negative residual at the first (only) step
                    # is exactly init_noise.
                    self.stock_noise.copy_(self.init_noise)
                with profiler.region("unet_step"):
                    x_0_pred_batch, model_pred = self.unet_step(x_t_latent, t_list)

                if self.denoising_steps_num > 1:
                    x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                    with profiler.region("sched.rebuild"):
                        if self.do_add_noise:
                            self.x_t_latent_buffer = (
                                self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                                + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                            )
                        else:
                            self.x_t_latent_buffer = self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                else:
                    x_0_pred_out = x_0_pred_batch
                    self.x_t_latent_buffer = None
            else:
                # Standard scheduler loop for TCD and non-batched LCM
                sample = x_t_latent
                for idx, timestep in enumerate(self.sub_timesteps_tensor):
                    # Resolve scalar timestep tensor on device.
                    # Avoid per-step t.view(1).repeat() allocations by using the
                    # precomputed _sub_timesteps_expanded table from prepare().
                    t = (
                        timestep
                        if isinstance(timestep, torch.Tensor)
                        else torch.tensor(timestep, device=self.device, dtype=torch.long)
                    )
                    if self._sub_timesteps_expanded is not None:
                        t_expanded = self._sub_timesteps_expanded[idx]  # [frame_bff_size], no alloc
                    else:
                        t_expanded = t.view(1).repeat(self.frame_bff_size)  # fallback

                    # For TCD, use the same UNet calling logic as LCM to ensure ControlNet hooks are processed
                    if isinstance(self.scheduler, TCDScheduler):
                        # Use unet_step to process ControlNet hooks and get proper noise prediction
                        with profiler.region("unet_step"):
                            x_0_pred, model_pred = self.unet_step(sample, t_expanded, idx)

                        # Apply TCD scheduler step to the guided noise prediction
                        with profiler.region("scheduler_step"):
                            step_out = self.scheduler.step(model_pred, t, sample)
                        sample = getattr(
                            step_out, "prev_sample", step_out[0] if isinstance(step_out, (tuple, list)) else step_out
                        )
                    else:
                        # Original LCM logic for non-batched mode
                        with profiler.region("unet_step"):
                            x_0_pred, model_pred = self.unet_step(sample, t_expanded, idx)
                        if idx < len(self.sub_timesteps_tensor) - 1:
                            if self.do_add_noise:
                                if self._noise_buf is None:
                                    self._noise_buf = torch.empty_like(x_0_pred)
                                self._noise_buf.normal_()
                                sample = (
                                    self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
                                    + self.beta_prod_t_sqrt[idx + 1] * self._noise_buf
                                )
                            else:
                                sample = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
                        else:
                            sample = x_0_pred

                x_0_pred_out = sample
            return x_0_pred_out

    @torch.inference_mode()
    def __call__(self, x: Optional[Union[torch.Tensor, PIL.Image.Image, np.ndarray]] = None) -> torch.Tensor:
        start = self._timing_start
        end = self._timing_end
        if self.similar_image_filter:
            start.record()

        if x is not None:
            # Fast path: already a normalized GPU tensor with the right shape/dtype.
            # Skips image_processor.preprocess (and its per-frame image.min() GPU scan).
            # PIL/numpy inputs still take the full preprocessing path below.
            if not (
                isinstance(x, torch.Tensor)
                and x.is_cuda
                and x.dtype == self.dtype
                and x.shape[-2:] == (self.height, self.width)
            ):
                _raw = self.image_processor.preprocess(x, self.height, self.width)
                if not _raw.is_cuda and self._input_staging is not None and _raw.shape == self._input_staging.shape:
                    self._input_staging.copy_(_raw)  # CPU fp32→dtype cast into pinned memory
                    x = self._input_staging.to(device=self.device, non_blocking=True)
                else:
                    x = _raw.to(device=self.device, dtype=self.dtype, non_blocking=True)

            # IMAGE PREPROCESSING HOOKS: After built-in preprocessing, before filtering
            x = self._apply_image_preprocessing_hooks(x)

            # NaN guard 2: single choke point for all image input, including anything an
            # image-preprocessing hook (FX chain) may have injected. Sanitizes in place,
            # unconditionally -- see utils/nan_guard.py. No recovery action needed here
            # (unlike guard 1's stock_noise reseed): cutting off propagation into
            # encode_image is the whole job.
            self._nan_guard("_nan_guard_image_input", "__call__.image_input").sanitize_(x)

            if self.similar_image_filter:
                x = self.similar_filter(x)
                if x is None:
                    self.last_frame_was_skipped = True
                    time.sleep(self.inference_time_ema * self.similar_filter_sleep_fraction)
                    return self.prev_image_result

            self.last_frame_was_skipped = False
            x_t_latent = self.encode_image(x)

            # LATENT PREPROCESSING HOOKS: After VAE encoding, before diffusion
            x_t_latent = self._apply_latent_preprocessing_hooks(x_t_latent)
        else:
            # TODO: check the dimension of x_t_latent
            x_t_latent = torch.randn((1, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )

        try:
            x_0_pred_out = self.predict_x0_batch(x_t_latent)
        except RuntimeError as _e:
            _msg = str(_e).lower()
            if "out of memory" in _msg:
                logger.error("StreamDiffusion.__call__: OOM — clearing cache and re-raising: %s", _e)
                torch.cuda.empty_cache()
                raise
            if "expanded size" in _msg and "must match" in _msg:
                logger.error("StreamDiffusion.__call__: tensor size mismatch — attempting scheduler rebuild: %s", _e)
                try:
                    self._param_updater._update_timestep_calculations()
                    self._refresh_derived_tensors()
                except Exception as _fix_err:
                    logger.error("StreamDiffusion.__call__: rebuild failed: %s", _fix_err)
                torch.cuda.empty_cache()
                logger.warning("StreamDiffusion.__call__: returning safe fallback frame")
                return self.decode_image(
                    torch.randn(
                        (1, 4, self.latent_height, self.latent_width),
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
            raise

        # LATENT POSTPROCESSING HOOKS: After diffusion, before VAE decoding
        x_0_pred_out = self._apply_latent_postprocessing_hooks(x_0_pred_out)

        # NaN guard 4: sanitizes in place before both consumers below -- the persistent
        # _latent_cache (read by latent-feedback processors on the NEXT frame) and
        # decode_image just below it. Covers anything a latent-postprocessing hook (e.g.
        # latent_feedback.py) may have injected, in addition to guard 1's UNet coverage.
        self._nan_guard("_nan_guard_latent_cache", "_latent_cache").sanitize_(x_0_pred_out)

        # Store latent result for latent feedback processors (reuse pre-allocated buffer)
        if self._latent_cache is None:
            self._latent_cache = torch.empty_like(x_0_pred_out)
        self._latent_cache.copy_(x_0_pred_out)
        self.prev_latent_result = self._latent_cache

        _decoded = self.decode_image(x_0_pred_out)
        if self._image_decode_buf is None:
            self._image_decode_buf = torch.empty_like(_decoded)
        self._image_decode_buf.copy_(_decoded)
        x_output = self._image_decode_buf

        # IMAGE POSTPROCESSING HOOKS: After VAE decoding, before final output
        x_output = self._apply_image_postprocessing_hooks(x_output)

        # NaN guard 3: sanitizes in place before the persistent _prev_image_buf (read by
        # image-feedback processors, e.g. feedback_loop.py, on the NEXT frame) and before
        # this frame's returned output -- covers anything an image-postprocessing hook may
        # have injected, in addition to guards 1/2/4's upstream coverage.
        self._nan_guard("_nan_guard_prev_image", "_prev_image_buf").sanitize_(x_output)

        # Copy into pre-allocated skip-frame cache — TRT VAE buffer is reused on next decode call
        if self._prev_image_buf is None:
            self._prev_image_buf = torch.empty_like(x_output)
        self._prev_image_buf.copy_(x_output)
        self.prev_image_result = self._prev_image_buf
        # P2: the timing path exists only to maintain inference_time_ema, which is consumed
        # solely by the similar-filter sleep heuristic (see the similar_image_filter branch
        # above). When the filter is off the EMA has no reader, so skip the records AND the
        # blocking end.synchronize() entirely. When on, retain the 16-frame sample cadence
        # (eliminates ~15/16 per-frame host stalls).
        # Grounding: CUDA HB §6.1 — per-frame device sync ~100 µs vs ~3.4 µs amortized.
        if self.similar_image_filter:
            end.record()
            self._sync_counter += 1
            if self._sync_counter % 16 == 0:
                end.synchronize()
                inference_time = start.elapsed_time(end) / 1000
                self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time

        # NOTE: x_output aliases self._image_decode_buf, a persistent buffer reused every
        # frame to avoid a per-frame .clone() on this hot path (see buffer init above).
        # Intentional: callers needing an independent copy must clone it themselves.
        # StreamDiffusionWrapper.postprocess_image() clones at the public API boundary.
        return x_output

    # =========================================================================
    # Pipeline Hook Helper Methods (Phase 3: Performance-Optimized Hot Path)
    # =========================================================================

    def _apply_image_preprocessing_hooks(self, x: torch.Tensor) -> torch.Tensor:
        """Apply image preprocessing hooks with minimal hot path overhead."""
        # Early exit - zero overhead when no hooks registered
        if not self.image_preprocessing_hooks:
            return x

        # Single context object creation to minimize allocation overhead
        image_ctx = ImageCtx(image=x, width=self.width, height=self.height)

        # Direct iteration - no additional function calls
        for hook in self.image_preprocessing_hooks:
            image_ctx = hook(image_ctx)

        return image_ctx.image

    def _apply_image_postprocessing_hooks(self, x: torch.Tensor) -> torch.Tensor:
        """Apply image postprocessing hooks with minimal hot path overhead."""
        # Early exit - zero overhead when no hooks registered
        if not self.image_postprocessing_hooks:
            return x

        # Single context object creation to minimize allocation overhead
        image_ctx = ImageCtx(image=x, width=self.width, height=self.height)

        # Direct iteration - no additional function calls
        for hook in self.image_postprocessing_hooks:
            image_ctx = hook(image_ctx)

        return image_ctx.image

    def _apply_latent_preprocessing_hooks(self, latent: torch.Tensor) -> torch.Tensor:
        """Apply latent preprocessing hooks with minimal hot path overhead."""
        # Early exit - zero overhead when no hooks registered
        if not self.latent_preprocessing_hooks:
            return latent

        # Single context object creation to minimize allocation overhead
        latent_ctx = LatentCtx(latent=latent)

        # Direct iteration - no additional function calls
        for hook in self.latent_preprocessing_hooks:
            latent_ctx = hook(latent_ctx)

        return latent_ctx.latent

    def _apply_latent_postprocessing_hooks(self, latent: torch.Tensor) -> torch.Tensor:
        """Apply latent postprocessing hooks with minimal hot path overhead."""
        # Early exit - zero overhead when no hooks registered
        if not self.latent_postprocessing_hooks:
            return latent

        # Single context object creation to minimize allocation overhead
        latent_ctx = LatentCtx(latent=latent)

        # Direct iteration - no additional function calls
        for hook in self.latent_postprocessing_hooks:
            latent_ctx = hook(latent_ctx)

        return latent_ctx.latent

    @torch.inference_mode()
    def txt2img(self, batch_size: int = 1) -> torch.Tensor:
        x_0_pred_out = self.predict_x0_batch(
            torch.randn((batch_size, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
        )

        # LATENT POSTPROCESSING HOOKS: After diffusion, before VAE decoding
        x_0_pred_out = self._apply_latent_postprocessing_hooks(x_0_pred_out)

        # NaN guard 4 (see __call__ for full rationale): same persistent _latent_cache.
        self._nan_guard("_nan_guard_latent_cache", "_latent_cache").sanitize_(x_0_pred_out)

        # Store latent result for latent feedback processors (reuse pre-allocated buffer)
        if self._latent_cache is None:
            self._latent_cache = torch.empty_like(x_0_pred_out)
        self._latent_cache.copy_(x_0_pred_out)
        self.prev_latent_result = self._latent_cache

        _decoded = self.decode_image(x_0_pred_out)
        if self._image_decode_buf is None:
            self._image_decode_buf = torch.empty_like(_decoded)
        self._image_decode_buf.copy_(_decoded)
        x_output = self._image_decode_buf

        # IMAGE POSTPROCESSING HOOKS: After VAE decoding, before final output
        x_output = self._apply_image_postprocessing_hooks(x_output)

        # NaN guard 3 (see __call__ for full rationale): sanitize final output in place.
        self._nan_guard("_nan_guard_prev_image", "_prev_image_buf").sanitize_(x_output)

        # NOTE: x_output aliases self._image_decode_buf, a persistent buffer reused every
        # call to avoid a per-frame .clone() (see buffer init above). Intentional: callers
        # needing an independent copy must clone it themselves.
        # StreamDiffusionWrapper.postprocess_image() clones at the public API boundary.
        return x_output

    @torch.inference_mode()
    def txt2img_sd_turbo(self, batch_size: int = 1) -> torch.Tensor:
        x_t_latent = torch.randn(
            (batch_size, 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype,
        )

        # Prepare UNet call arguments
        unet_kwargs = {
            "sample": x_t_latent,
            "timestep": self.sub_timesteps_tensor,
            "encoder_hidden_states": self.prompt_embeds,
            "return_dict": False,
        }

        # Add SDXL-specific conditioning if this is an SDXL model
        if self.is_sdxl and hasattr(self, "add_text_embeds") and hasattr(self, "add_time_ids"):
            if self.add_text_embeds is not None and self.add_time_ids is not None:
                # For txt2img, replicate conditioning to match batch size
                add_text_embeds = (
                    self.add_text_embeds[1:2].repeat(batch_size, 1)
                    if self.add_text_embeds.shape[0] > 1
                    else self.add_text_embeds.repeat(batch_size, 1)
                )
                add_time_ids = (
                    self.add_time_ids[1:2].repeat(batch_size, 1)
                    if self.add_time_ids.shape[0] > 1
                    else self.add_time_ids.repeat(batch_size, 1)
                )

                unet_kwargs["added_cond_kwargs"] = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        # Call UNet with appropriate conditioning
        if self.is_sdxl:
            model_pred = self.unet(**unet_kwargs)[0]
        else:
            # For SD1.5/SD2.1, use the old calling convention for compatibility
            model_pred = self.unet(
                x_t_latent,
                self.sub_timesteps_tensor,
                encoder_hidden_states=self.prompt_embeds,
                return_dict=False,
            )[0]

        # NaN guard (mirrors guard 1 -- this path doesn't go through unet_step, so
        # model_pred isn't covered there): sanitize before it feeds the x_0 formula below.
        self._nan_guard("_nan_guard_model_pred", "unet_step.model_pred").sanitize_(model_pred)

        x_0_pred_out = ((x_t_latent - self.beta_prod_t_sqrt * model_pred).float() / self.alpha_prod_t_sqrt.float()).to(
            x_t_latent.dtype
        )

        # LATENT POSTPROCESSING HOOKS: After diffusion, before VAE decoding
        x_0_pred_out = self._apply_latent_postprocessing_hooks(x_0_pred_out)

        # NaN guard 4 (see __call__ for full rationale): same persistent _latent_cache.
        self._nan_guard("_nan_guard_latent_cache", "_latent_cache").sanitize_(x_0_pred_out)

        # Store latent result for latent feedback processors (reuse pre-allocated buffer)
        if self._latent_cache is None:
            self._latent_cache = torch.empty_like(x_0_pred_out)
        self._latent_cache.copy_(x_0_pred_out)
        self.prev_latent_result = self._latent_cache

        x_output = self.decode_image(x_0_pred_out)

        # IMAGE POSTPROCESSING HOOKS: After VAE decoding, before final output
        x_output = self._apply_image_postprocessing_hooks(x_output)

        # NaN guard 3 (see __call__ for full rationale): sanitize final output in place.
        self._nan_guard("_nan_guard_prev_image", "_prev_image_buf").sanitize_(x_output)

        return x_output
