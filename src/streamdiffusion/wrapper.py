import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import AutoencoderTiny, AutoPipelineForText2Image, StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image

from .image_utils import postprocess_image
from .model_detection import detect_model
from .pipeline import StreamDiffusion
from .tools.gpu_profiler import configure as _configure_profiler
from .tools.gpu_profiler import profiler


logger = logging.getLogger(__name__)

# Text-encoder CPU offload frees ~1.6 GB VRAM but each prompt update pays a
# CPU<->GPU round-trip plus torch.cuda.empty_cache() — a measurable stall
# mid-stream on high-VRAM GPUs. Default off (encoders stay resident on GPU);
# set SD_TEXT_ENCODER_OFFLOAD=1 to restore offloading on VRAM-constrained GPUs.
_TEXT_ENCODER_OFFLOAD: bool = os.environ.get("SD_TEXT_ENCODER_OFFLOAD", "0") == "1"

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


class StreamDiffusionWrapper:
    """
    StreamDiffusionWrapper for real-time image generation.

    This wrapper provides a unified interface for both single prompts and prompt blending:

    ## Unified Interface:
    ```python
    # Single prompt
    wrapper.prepare("a beautiful cat")

    # Prompt blending
    wrapper.prepare([("cat", 0.7), ("dog", 0.3)])

    # Prompt + seed blending
    wrapper.prepare(
        prompt=[("style1", 0.6), ("style2", 0.4)],
        seed_list=[(123, 0.8), (456, 0.2)]
    )
    ```

    ## Runtime Updates:
    ```python
    # Update single prompt
    wrapper.update_prompt("new prompt")

    # Update prompt blending
    wrapper.update_prompt([("new1", 0.5), ("new2", 0.5)])

    # Update combined parameters
    wrapper.update_stream_params(
        prompt_list=[("bird", 0.6), ("fish", 0.4)],
        seed_list=[(789, 0.3), (101, 0.7)]
    )
    ```

    ## Weight Management:
    - Prompt weights are normalized by default (sum to 1.0) unless normalize_prompt_weights=False
    - Seed weights are normalized by default (sum to 1.0) unless normalize_seed_weights=False
    - To change blend weights, pass the full prompt_list/seed_list to update_stream_params —
      unchanged texts/seeds hit the embedding/noise cache, so only re-blending occurs (no re-encode)

    ## Cache Management:
    - Prompt embeddings and seed noise tensors are automatically cached for performance
    - Use get_cache_info() to inspect cache statistics
    - Use clear_caches() to free memory
    """

    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        min_batch_size: int = 1,
        max_batch_size: int = 4,
        lora_dict: Optional[Dict[str, float]] = None,
        mode: Literal["img2img", "txt2img"] = "img2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: Optional[bool] = None,  # DEPRECATED: Backwards compatibility parameter
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        similar_filter_sleep_fraction: float = 0.025,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        use_safety_checker: bool = False,
        skip_diffusion: bool = False,
        engine_dir: Optional[Union[str, Path]] = "engines",
        compile_engines_only: bool = False,
        build_engines_if_missing: bool = True,
        normalize_prompt_weights: bool = True,
        normalize_seed_weights: bool = True,
        # Scheduler and sampler options
        scheduler: Literal["lcm", "tcd"] = "lcm",
        sampler: Literal["simple", "sgm_uniform", "normal", "ddim", "beta", "karras"] = "normal",
        # ControlNet options
        use_controlnet: bool = False,
        controlnet_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        # IPAdapter options
        use_ipadapter: bool = False,
        ipadapter_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        # Pipeline hook configurations
        image_preprocessing_config: Optional[Dict[str, Any]] = None,
        image_postprocessing_config: Optional[Dict[str, Any]] = None,
        latent_preprocessing_config: Optional[Dict[str, Any]] = None,
        latent_postprocessing_config: Optional[Dict[str, Any]] = None,
        safety_checker_fallback_type: Literal["blank", "previous"] = "previous",
        safety_checker_threshold: float = 0.5,
        use_cached_attn: bool = False,
        cache_maxframes: int = 1,
        cache_interval: int = 1,
        min_cache_maxframes: int = 1,
        max_cache_maxframes: int = 4,
        cn_cache_interval: int = 1,
        use_feature_injection: bool = False,
        fi_strength: float = 0.75,
        fi_threshold: float = 0.98,
        fp8: bool = False,
        static_shapes: bool = False,
        fp8_allow_fp16_fallback: bool = False,
        builder_optimization_level: Optional[int] = None,
        # CUDA IPC output (SD→TD zero-copy GPU transport via cuda-link)
        use_cuda_ipc_output: bool = False,
        cuda_ipc_shm_name: Optional[str] = None,
        cuda_ipc_num_slots: int = 2,
        # CUDA IPC CN-preview (SD→TD zero-copy preprocessor output display, fixed name, display-only)
        cuda_ipc_cn_processed_shm_name: Optional[str] = None,
        # When True, forces the preprocessor to run even when conditioning_scale==0 so that
        # controlnet_images[index] is populated for the preview. No diffusion effect.
        controlnet_preview_passthrough: bool = False,
        # Debug mode — gates IPC health tracking and other diagnostic instrumentation
        debug_mode: bool = False,
        vae_builder_optimization_level: Optional[int] = None,
    ):
        """
        Initializes the StreamDiffusionWrapper.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        min_batch_size : int, optional
            The minimum batch size for inference, by default 1.
        max_batch_size : int, optional
            The maximum batch size for inference, by default 4.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        mode : Literal["img2img", "txt2img"], optional
            txt2img or img2img, by default "img2img".
        output_type : Literal["pil", "pt", "np", "latent"], optional
            The output type of image, by default "pil".
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
            If None, the default TinyVAE
            ("madebyollin/taesd") will be used.
        device : Literal["cpu", "cuda"], optional
            The device to use for inference, by default "cuda".
        device_ids : Optional[List[int]], optional
            The device ids to use for DataParallel, by default None.
        dtype : torch.dtype, optional
            The dtype for inference, by default torch.float16.
        frame_buffer_size : int, optional
            The frame buffer size for denoising batch, by default 1.
        width : int, optional
            The width of the image, by default 512.
        height : int, optional
            The height of the image, by default 512.
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        acceleration : Literal["none", "xformers", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        device_ids : Optional[List[int]], optional
            The device ids to use for DataParallel, by default None.
        use_lcm_lora : Optional[bool], optional
            DEPRECATED: Use lora_dict instead. For backwards compatibility only.
            If True, automatically adds appropriate LCM LoRA to lora_dict based on model type.
            SDXL models get "latent-consistency/lcm-lora-sdxl", others get "latent-consistency/lcm-lora-sdv1-5".
            By default None (ignored).
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        enable_similar_image_filter : bool, optional
            Whether to enable similar image filter or not,
            by default False.
        similar_image_filter_threshold : float, optional
            The threshold for similar image filter, by default 0.98.
        similar_image_filter_max_skip_frame : int, optional
            The max skip frame for similar image filter, by default 10.
        use_denoising_batch : bool, optional
            Whether to use denoising batch or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.
        use_safety_checker : bool, optional
            Whether to use safety checker or not, by default False.
        skip_diffusion : bool, optional
            Whether to skip diffusion and apply only preprocessing/postprocessing hooks, by default False.
        engine_dir : Optional[Union[str, Path]], optional
            Directory path for storing/loading TensorRT engines, by default "engines".
        build_engines_if_missing : bool, optional
            Whether to build TensorRT engines if they don't exist, by default True.
        normalize_prompt_weights : bool, optional
            Whether to normalize prompt weights in blending to sum to 1,
            by default True. When False, weights > 1 will amplify embeddings.
        normalize_seed_weights : bool, optional
            Whether to normalize seed weights in blending to sum to 1,
            by default True. When False, weights > 1 will amplify noise.
        scheduler : Literal["lcm", "tcd"], optional
            The scheduler type to use for denoising, by default "lcm".
        sampler : Literal["simple", "sgm_uniform", "normal", "ddim", "beta", "karras"], optional
            The sampler type to use for noise scheduling, by default "normal".
        use_controlnet : bool, optional
            Whether to enable ControlNet support, by default False.
        controlnet_config : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional
            ControlNet configuration(s), by default None.
            Can be a single config dict or list of config dicts for multiple ControlNets.
            Each config should contain: model_id, preprocessor (optional), conditioning_scale, etc.
        use_ipadapter : bool, optional
            Whether to enable IPAdapter support, by default False.
        ipadapter_config : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional
            IPAdapter configuration(s), by default None. Can be a single config dict
            or list of config dicts for multiple IPAdapters.
        image_preprocessing_config : Optional[Dict[str, Any]], optional
            Configuration for image preprocessing hooks, by default None.
        image_postprocessing_config : Optional[Dict[str, Any]], optional
            Configuration for image postprocessing hooks, by default None.
        latent_preprocessing_config : Optional[Dict[str, Any]], optional
            Configuration for latent preprocessing hooks, by default None.
        latent_postprocessing_config : Optional[Dict[str, Any]], optional
            Configuration for latent postprocessing hooks, by default None.
        safety_checker_fallback_type : Literal["blank", "previous"], optional
            Whether to use a blank image or the previous image as a fallback, by default "previous".
        safety_checker_threshold: float, optional
            The threshold for the safety checker, by default 0.5.
        compile_engines_only : bool, optional
            Whether to only compile engines and not load the model, by default False.
        use_cached_attn : bool, optional
            Whether to use cached attention or not, by default True.
        cache_maxframes : int, optional
            The maximum number of frames to cache, by default 1.
        cache_interval : int, optional
            The interval to cache the frames, by default 1.
        builder_optimization_level : Optional[int], optional
            TensorRT IBuilderConfig.builder_optimization_level (range 0-5,
            TRT default 3). When set, overrides the per-GPU auto-detect default
            in ``acceleration/tensorrt/utilities.py::detect_gpu_profile()``.

            TouchDesigner TrtProfile mapping aligned with NVIDIA reference
            pipelines (demoDiffusion: level 3 for FP16; TensorRT-Model-Optimizer:
            level 4 for FP8/INT8 quantized)::

              0 = Flexible   static_shapes=False + level 3 — FP16 dynamic;
                             matches NVIDIA demoDiffusion default.
              2 = Fast Build static_shapes=True  + level 2 — heuristic-sorted
                             fastest tactics; ~30-40% faster build with minimal
                             runtime loss (build-time tradeoff).
              4 = Quality    static_shapes=True  + level 3 — FP16 static;
                             matches NVIDIA demoDiffusion default (level 4 has
                             no NVIDIA-validated benefit for unquantized FP16).
              Performance    static_shapes=True  + level 4 + fp8=True —
                             matches NVIDIA TensorRT-Model-Optimizer default
                             for quantized diffusion (RTX 40+ only).

            Levels 1 and 5 are valid TRT values but not exposed via TrtProfile
            UI (1 = degraded; 5 = used by no NVIDIA reference pipeline). Set to
            None to auto-detect per GPU (Ada/Ampere/Blackwell → 4, pre-Ampere
            → 3). Default None.
        """
        if compile_engines_only:
            logger.info("compile_engines_only is True, will only compile engines and not load the model")

        # Store use_lcm_lora for backwards compatibility processing in _load_model
        self.use_lcm_lora = use_lcm_lora

        self.sd_turbo = "turbo" in model_id_or_path
        self.use_controlnet = use_controlnet
        self.use_ipadapter = use_ipadapter
        self.ipadapter_config = ipadapter_config

        # Store pipeline hook configurations
        self.image_preprocessing_config = image_preprocessing_config
        self.image_postprocessing_config = image_postprocessing_config
        self.latent_preprocessing_config = latent_preprocessing_config
        self.latent_postprocessing_config = latent_postprocessing_config

        if mode == "txt2img":
            if cfg_type != "none":
                raise ValueError(f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}")
            if use_denoising_batch and frame_buffer_size > 1:
                if not self.sd_turbo:
                    raise ValueError("txt2img mode cannot use denoising batch with frame_buffer_size > 1.")

        if mode == "img2img":
            if not use_denoising_batch:
                raise NotImplementedError("img2img mode must use denoising batch for now.")

        _configure_profiler()  # activates via GPU_PROFILER=1 env var; no-op otherwise
        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.mode = mode
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self._output_pin_buf: Optional[torch.Tensor] = None  # pinned CPU buffer for async D2H output
        self._output_gpu_buf: Optional[torch.Tensor] = None  # persistent GPU fp32 staging (avoids per-frame alloc)
        self._d2h_event: Optional[torch.cuda.Event] = None  # event for fine-grained D2H sync
        self.use_cuda_ipc_output = use_cuda_ipc_output
        self._cuda_ipc_shm_name = cuda_ipc_shm_name
        self._cuda_ipc_num_slots = cuda_ipc_num_slots
        self._cuda_ipc_exporter = None  # lazy-init on first frame via _lazy_init_ipc_exporter
        self._cuda_ipc_cn_processed_shm_name = cuda_ipc_cn_processed_shm_name
        self._cuda_ipc_cn_exporter = None  # lazy-init on first CN frame via _lazy_init_cn_ipc_exporter
        self._controlnet_preview_passthrough = controlnet_preview_passthrough
        self.debug_mode = debug_mode
        # IPC health tracking — updated per-frame only when debug_mode is True
        self._ipc_consecutive_failures: int = 0
        self._ipc_barrier_skip_count: int = 0
        self._ipc_graphs_degraded: bool = False
        self.batch_size = len(t_index_list) * frame_buffer_size if use_denoising_batch else frame_buffer_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        self.use_denoising_batch = use_denoising_batch
        # safety checker is only supported for TensorRT acceleration
        self.use_safety_checker = use_safety_checker and (acceleration == "tensorrt")
        self.safety_checker_fallback_type = safety_checker_fallback_type
        self.safety_checker_threshold = safety_checker_threshold
        # Caches the last clean (non-flagged) pipeline tensor for the "previous" fallback strategy.
        # Operates in diffusion range [-1, 1]; set by _apply_safety_checker().
        self._prev_clean_tensor: Optional[torch.Tensor] = None
        self.fp8 = fp8
        self.static_shapes = static_shapes
        self.fp8_allow_fp16_fallback = fp8_allow_fp16_fallback
        self.builder_optimization_level = builder_optimization_level
        # Per-engine VAE optlvl (None → inherit builder_optimization_level).
        # Tiny-VAE engines are small and gain little from optlvl 4 — defaulting to
        # optlvl 3 via config.py shaves VAE encoder build time without affecting UNet quality.
        self.vae_builder_optimization_level = vae_builder_optimization_level

        self.stream: StreamDiffusion = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,  # Deprecated:Backwards compatibility
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            engine_dir=engine_dir,
            build_engines_if_missing=build_engines_if_missing,
            normalize_prompt_weights=normalize_prompt_weights,
            normalize_seed_weights=normalize_seed_weights,
            scheduler=scheduler,
            sampler=sampler,
            use_controlnet=use_controlnet,
            controlnet_config=controlnet_config,
            use_ipadapter=use_ipadapter,
            ipadapter_config=ipadapter_config,
            # Pipeline hook configurations
            image_preprocessing_config=image_preprocessing_config,
            image_postprocessing_config=image_postprocessing_config,
            latent_preprocessing_config=latent_preprocessing_config,
            latent_postprocessing_config=latent_postprocessing_config,
            compile_engines_only=compile_engines_only,
            use_cached_attn=use_cached_attn,
            cache_maxframes=cache_maxframes,
            cache_interval=cache_interval,
            min_cache_maxframes=min_cache_maxframes,
            max_cache_maxframes=max_cache_maxframes,
            cn_cache_interval=cn_cache_interval,
            use_feature_injection=use_feature_injection,
            fi_strength=fi_strength,
            fi_threshold=fi_threshold,
            fp8=fp8,
        )

        # Store skip_diffusion on wrapper for execution flow control
        self.skip_diffusion = skip_diffusion

        if compile_engines_only:
            return

        if seed < 0:  # Random seed
            seed = np.random.randint(0, 1000000)

        self.stream.prepare(
            "",
            "",
            num_inference_steps=50,
            guidance_scale=1.1 if self.stream.cfg_type in ["full", "self", "initialize"] else 1.0,
            generator=torch.manual_seed(seed),
            seed=seed,
        )

        # Offload text encoders to CPU after initial encoding to free ~1.6 GB VRAM (SDXL).
        # They are reloaded on-demand before each prompt re-encoding call.
        if acceleration == "tensorrt":
            self._offload_text_encoders()

        # Set wrapper reference on parameter updater so it can access pipeline structure
        self.stream._param_updater.wrapper = self

        # Store acceleration settings for ControlNet integration
        self._acceleration = acceleration
        self._engine_dir = engine_dir

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(self.stream.unet, device_ids=device_ids)

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(
                similar_image_filter_threshold, similar_image_filter_max_skip_frame
            )
        self.stream.similar_filter_sleep_fraction = similar_filter_sleep_fraction

    def prepare(
        self,
        prompt: Union[str, List[Tuple[str, float]]],
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        # Blending-specific parameters (only used when prompt is a list)
        prompt_interpolation_method: Literal["linear", "slerp", "cosine_weighted"] = "slerp",
        seed_list: Optional[List[Tuple[int, float]]] = None,
        seed_interpolation_method: Literal["linear", "slerp"] = "linear",
    ) -> None:
        """
        Prepares the model for inference.

        Supports both single prompts and prompt blending based on the prompt parameter type.

        Parameters
        ----------
        prompt : Union[str, List[Tuple[str, float]]]
            Either a single prompt string or a list of (prompt, weight) tuples for blending.
            Examples:
            - Single: "a beautiful cat"
            - Blending: [("cat", 0.7), ("dog", 0.3)]
        negative_prompt : str, optional
            The negative prompt, by default "".
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise, by default 1.0.
        prompt_interpolation_method : Literal["linear", "slerp"], optional
            Method for interpolating between prompt embeddings (only used for prompt blending),
            by default "slerp".
        seed_list : Optional[List[Tuple[int, float]]], optional
            List of seeds with weights for blending, by default None.
        seed_interpolation_method : Literal["linear", "slerp"], optional
            Method for interpolating between seed noise tensors, by default "linear".
        """

        # Handle both single prompt and prompt blending
        if isinstance(prompt, str):
            # Single prompt mode (legacy interface)
            self._reload_text_encoders()
            try:
                self.stream.prepare(
                    prompt,
                    negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    delta=delta,
                )
            finally:
                self._offload_text_encoders()

            # Apply seed blending if provided
            if seed_list is not None:
                self.update_stream_params(
                    seed_list=seed_list,
                    seed_interpolation_method=seed_interpolation_method,
                )

        elif isinstance(prompt, list):
            # Prompt blending mode
            if not prompt:
                raise ValueError("prepare: prompt list cannot be empty")

            # Prepare with first prompt to initialize the pipeline
            first_prompt = prompt[0][0]
            self._reload_text_encoders()
            try:
                self.stream.prepare(
                    first_prompt,
                    negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    delta=delta,
                )
            finally:
                self._offload_text_encoders()

            # Then apply prompt blending (and seed blending if provided)
            # update_stream_params handles its own reload/offload
            self.update_stream_params(
                prompt_list=prompt,
                negative_prompt=negative_prompt,
                prompt_interpolation_method=prompt_interpolation_method,
                seed_list=seed_list,
                seed_interpolation_method=seed_interpolation_method,
            )

        else:
            raise TypeError(f"prepare: prompt must be str or List[Tuple[str, float]], got {type(prompt)}")

    def _offload_text_encoders(self) -> None:
        """Move text encoders to CPU to free VRAM (~1.6 GB for SDXL).

        Called automatically after initial prepare() when using TRT acceleration.
        Text encoders are reloaded to GPU before each prompt re-encoding call.

        No-op when SD_TEXT_ENCODER_OFFLOAD is not set (default). High-VRAM GPUs
        (RTX 4090, A100…) benefit from keeping encoders resident to avoid the
        CPU<->GPU transfer + empty_cache() stall on every prompt update.
        """
        if not _TEXT_ENCODER_OFFLOAD:
            return
        pipe = self.stream.pipe
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            if next(pipe.text_encoder.parameters(), None) is not None:
                pipe.text_encoder = pipe.text_encoder.to("cpu")
        if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
            if next(pipe.text_encoder_2.parameters(), None) is not None:
                pipe.text_encoder_2 = pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()
        logger.debug("[VRAM] Text encoders offloaded to CPU")

    def _reload_text_encoders(self) -> None:
        """Move text encoders back to GPU before prompt re-encoding.

        No-op when SD_TEXT_ENCODER_OFFLOAD is not set (default) because
        encoders were never offloaded.
        """
        if not _TEXT_ENCODER_OFFLOAD:
            return
        pipe = self.stream.pipe
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            pipe.text_encoder = pipe.text_encoder.to(self.device)
        if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
            pipe.text_encoder_2 = pipe.text_encoder_2.to(self.device)
        logger.debug("[VRAM] Text encoders reloaded to GPU")

    def update_prompt(
        self,
        prompt: Union[str, List[Tuple[str, float]]],
        negative_prompt: str = "",
        prompt_interpolation_method: Literal["linear", "slerp", "cosine_weighted"] = "slerp",
        clear_blending: bool = True,
        warn_about_conflicts: bool = True,
    ) -> None:
        """
        Update to a new prompt or prompt blending configuration.

        Supports both single prompts and prompt blending based on the prompt parameter type.

        This is for legacy compatibility, use update_stream_params instead

        Parameters
        ----------
        prompt : Union[str, List[Tuple[str, float]]]
            Either a single prompt string or a list of (prompt, weight) tuples for blending.
            Examples:
            - Single: "a beautiful cat"
            - Blending: [("cat", 0.7), ("dog", 0.3)]
        negative_prompt : str, optional
            The negative prompt (used with blending), by default "".
        prompt_interpolation_method : Literal["linear", "slerp", "cosine_weighted"], optional
            Method for interpolating between prompt embeddings (used with blending), by default "slerp".
        clear_blending : bool, optional
            Whether to clear existing blending when switching to single prompt, by default True.
        warn_about_conflicts : bool, optional
            Whether to warn about conflicts when switching between modes, by default True.
        """
        # Handle both single prompt and prompt blending
        if isinstance(prompt, str):
            # Single prompt mode
            current_prompts = self.stream._param_updater.get_current_prompts()
            if current_prompts and len(current_prompts) > 1 and warn_about_conflicts:
                logger.warning("update_prompt: WARNING: Active prompt blending detected!")
                logger.warning(f"  Current blended prompts: {len(current_prompts)} prompts")
                logger.warning("  Switching to single prompt mode.")
                if clear_blending:
                    logger.warning("  Clearing prompt blending cache...")

            if clear_blending:
                # Clear the blending caches to avoid conflicts
                self.stream._param_updater.clear_caches()

            # Reload text encoders to GPU for re-encoding, then offload when done.
            self._reload_text_encoders()
            try:
                self.stream.update_prompt(prompt)
            finally:
                self._offload_text_encoders()

        elif isinstance(prompt, list):
            # Prompt blending mode
            if not prompt:
                raise ValueError("update_prompt: prompt list cannot be empty")

            current_prompts = self.stream._param_updater.get_current_prompts()
            if len(current_prompts) <= 1 and warn_about_conflicts:
                logger.warning("update_prompt: Switching from single prompt to prompt blending mode.")

            # Apply prompt blending (update_stream_params handles reload/offload internally)
            self.update_stream_params(
                prompt_list=prompt,
                negative_prompt=negative_prompt,
                prompt_interpolation_method=prompt_interpolation_method,
            )

        else:
            raise TypeError(f"update_prompt: prompt must be str or List[Tuple[str, float]], got {type(prompt)}")

    def update_stream_params(
        self,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        delta: Optional[float] = None,
        t_index_list: Optional[List[int]] = None,
        seed: Optional[int] = None,
        # Prompt blending parameters
        prompt_list: Optional[List[Tuple[str, float]]] = None,
        negative_prompt: Optional[str] = None,
        prompt_interpolation_method: Literal["linear", "slerp", "cosine_weighted"] = "slerp",
        normalize_prompt_weights: Optional[bool] = None,
        # Seed blending parameters
        seed_list: Optional[List[Tuple[int, float]]] = None,
        seed_interpolation_method: Literal["linear", "slerp"] = "linear",
        normalize_seed_weights: Optional[bool] = None,
        # ControlNet configuration
        controlnet_config: Optional[List[Dict[str, Any]]] = None,
        # IPAdapter configuration
        ipadapter_config: Optional[Dict[str, Any]] = None,
        # Hook configurations
        image_preprocessing_config: Optional[List[Dict[str, Any]]] = None,
        image_postprocessing_config: Optional[List[Dict[str, Any]]] = None,
        latent_preprocessing_config: Optional[List[Dict[str, Any]]] = None,
        latent_postprocessing_config: Optional[List[Dict[str, Any]]] = None,
        use_safety_checker: Optional[bool] = None,
        safety_checker_threshold: Optional[float] = None,
        cache_maxframes: Optional[int] = None,
        cache_interval: Optional[int] = None,
        # ControlNet residual cache interval (1=off, N>1=reuse residuals for N-1 frames)
        cn_cache_interval: Optional[int] = None,
        # Feature Injection live-tunable params (in-place tensor update, no engine rebuild)
        fi_strength: Optional[float] = None,
        fi_threshold: Optional[float] = None,
    ) -> None:
        """
        Update streaming parameters efficiently in a single call.

        Parameters
        ----------
        num_inference_steps : Optional[int]
            The number of inference steps to perform.
        guidance_scale : Optional[float]
            The guidance scale to use for CFG.
        delta : Optional[float]
            The delta multiplier of virtual residual noise.
        t_index_list : Optional[List[int]]
            The t_index_list to use for inference.
        seed : Optional[int]
            The random seed to use for noise generation.
        prompt_list : Optional[List[Tuple[str, float]]]
            List of prompts with weights for blending. Each tuple contains (prompt_text, weight).
            Example: [("cat", 0.7), ("dog", 0.3)]
        negative_prompt : Optional[str]
            The negative prompt to apply to all blended prompts.
        prompt_interpolation_method : Literal["linear", "slerp", "cosine_weighted"]
            Method for interpolating between prompt embeddings, by default "slerp".
        normalize_prompt_weights : Optional[bool]
            Whether to normalize prompt weights in blending to sum to 1, by default None (no change).
            When False, weights > 1 will amplify embeddings.
        seed_list : Optional[List[Tuple[int, float]]]
            List of seeds with weights for blending. Each tuple contains (seed_value, weight).
            Example: [(123, 0.6), (456, 0.4)]
        seed_interpolation_method : Literal["linear", "slerp"]
            Method for interpolating between seed noise tensors, by default "linear".
        normalize_seed_weights : Optional[bool]
            Whether to normalize seed weights in blending to sum to 1, by default None (no change).
            When False, weights > 1 will amplify noise.
        controlnet_config : Optional[List[Dict[str, Any]]]
            Complete ControlNet configuration list defining the desired state.
            Each dict contains: model_id, preprocessor, conditioning_scale, enabled,
            preprocessor_params, etc. System will diff current vs desired state and
            perform minimal add/remove/update operations.
        ipadapter_config : Optional[Dict[str, Any]]
            IPAdapter configuration dict containing scale, style_image, etc.
        use_safety_checker : Optional[bool]
            Whether to use the safety checker. Only supported for TensorRT acceleration.
        safety_checker_threshold : Optional[float]
            The threshold for the safety checker.
        """
        # Skip re-encoding if the incoming prompt_list is identical to the cached one.
        # OSC delivers list-of-lists from JSON; normalise to (str, float) tuples before
        # comparing so type mismatches don't cause spurious cache misses.
        if prompt_list is not None:
            _normalized = [(str(p), float(w)) for p, w in prompt_list]
            _current = self.stream._param_updater.get_current_prompts()
            _neg_unchanged = (
                negative_prompt is None
                or negative_prompt == self.stream._param_updater._current_negative_prompt
            )
            if _normalized == _current and _neg_unchanged:
                logger.debug(
                    "update_stream_params: prompt_list unchanged (%d prompt(s)) -- skipping re-encode",
                    len(_normalized),
                )
                prompt_list = None

        # Reload text encoders to GPU if a new prompt needs encoding.
        needs_encoding = prompt_list is not None or negative_prompt is not None
        if needs_encoding:
            self._reload_text_encoders()
        try:
            # Handle all parameters via parameter updater (including ControlNet)
            self.stream._param_updater.update_stream_params(
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                delta=delta,
                t_index_list=t_index_list,
                seed=seed,
                prompt_list=prompt_list,
                negative_prompt=negative_prompt,
                prompt_interpolation_method=prompt_interpolation_method,
                seed_list=seed_list,
                seed_interpolation_method=seed_interpolation_method,
                normalize_prompt_weights=normalize_prompt_weights,
                normalize_seed_weights=normalize_seed_weights,
                controlnet_config=controlnet_config,
                ipadapter_config=ipadapter_config,
                image_preprocessing_config=image_preprocessing_config,
                image_postprocessing_config=image_postprocessing_config,
                latent_preprocessing_config=latent_preprocessing_config,
                latent_postprocessing_config=latent_postprocessing_config,
                cache_maxframes=cache_maxframes,
                cache_interval=cache_interval,
                cn_cache_interval=cn_cache_interval,
                fi_strength=fi_strength,
                fi_threshold=fi_threshold,
            )
        finally:
            if needs_encoding:
                self._offload_text_encoders()
        if use_safety_checker is not None:
            self.use_safety_checker = use_safety_checker and (self._acceleration == "tensorrt")
        if safety_checker_threshold is not None:
            self.safety_checker_threshold = safety_checker_threshold

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Performs img2img or txt2img based on the mode.

        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if self.skip_diffusion:
            return self._process_skip_diffusion(image, prompt)

        if self.mode == "img2img":
            return self.img2img(image, prompt)
        else:
            return self.txt2img(prompt)

    def _process_skip_diffusion(
        self, image: Optional[Union[str, Image.Image, torch.Tensor]] = None, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Process input directly without diffusion, applying pre/post processing hooks.

        This method bypasses VAE encoding, diffusion, and VAE decoding, but still
        applies image preprocessing and postprocessing hooks for consistent processing.

        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to process directly.
        prompt : Optional[str]
            Prompt (ignored in skip mode, but kept for API consistency).

        Returns
        -------
        Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]
            The processed image with hooks applied.
        """

        if self.mode == "txt2img":
            raise RuntimeError(
                "_process_skip_diffusion: skip_diffusion mode not applicable for txt2img - no input image"
            )

        if image is None:
            raise ValueError("_process_skip_diffusion: image required for skip diffusion mode")

        # Handle input tensor normalization to [-1,1] pipeline range
        if isinstance(image, str) or isinstance(image, Image.Image):
            processed_tensor = self.preprocess_image(image)
            preprocessor_input = self._denormalize_on_gpu(processed_tensor)
        elif isinstance(image, torch.Tensor):
            # Ensure tensor is on correct device and dtype first
            preprocessor_input = image.to(device=self.device, dtype=self.dtype)
        else:
            preprocessor_input = image

        preprocessor_output = self.stream._apply_image_preprocessing_hooks(preprocessor_input)

        # Convert [0,1] -> [-1,1] back to pipeline range for postprocessing hooks
        processed_tensor = self._normalize_on_gpu(preprocessor_output)

        # Apply image postprocessing hooks (expect [-1,1] range - post-VAE decoding)
        processed_tensor = self.stream._apply_image_postprocessing_hooks(processed_tensor)

        # Screen skip-diffusion output too (raw [-1, 1] tensor, before postprocess/IPC export).
        processed_tensor = self._apply_safety_checker(processed_tensor)

        # Final postprocessing for output format
        return self.postprocess_image(processed_tensor, output_type=self.output_type)

    def txt2img(self, prompt: Optional[str] = None) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs txt2img.

        Parameters
        ----------
        prompt : Optional[str]
            The prompt to generate images from. If provided, will update to single prompt mode
            and may conflict with active prompt blending.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if prompt is not None:
            self.update_prompt(prompt, warn_about_conflicts=True)

        if self.sd_turbo:
            image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
        else:
            image_tensor = self.stream.txt2img(self.frame_buffer_size)

        image_tensor = self._apply_safety_checker(image_tensor)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        return image

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs img2img.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from. If provided, will update to single prompt mode
            and may conflict with active prompt blending.

        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            self.update_prompt(prompt, warn_about_conflicts=True)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        # Full pipeline with diffusion
        image_tensor = self.stream(image)
        image_tensor = self._apply_safety_checker(image_tensor)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        return image

    def preprocess_image(self, image: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Preprocesses the image.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.

        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
        # Use stream's current resolution instead of wrapper's cached values
        current_width = self.stream.width
        current_height = self.stream.height

        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((current_width, current_height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((current_width, current_height))

        return self.stream.image_processor.preprocess(image, current_height, current_width).to(
            device=self.device, dtype=self.dtype
        )

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Postprocesses the image (OPTIMIZED VERSION)

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The postprocessed image.
        """
        # CUDA IPC fast-path: export to TD via zero-copy GPU IPC (cuda-link Exporter v1.5.0+ API).
        # Skips D2H, CPU repack, and CPU SHM write. Returns None to let the TD-side
        # _send_output_frame early-exit (it already guards on output_image is None).
        if self.use_cuda_ipc_output and self._cuda_ipc_shm_name:
            from cuda_link import FrameOutcome, GpuFrame

            bgra = self._ipc_pack_rgba(image_tensor)
            exporter = self._lazy_init_ipc_exporter(bgra.shape[0], bgra.shape[1])
            # Pass the producer stream so the Exporter issues a GPU-side stream_wait_event
            # before the D2D memcpy. Without this the high-priority non-blocking IPC stream
            # can launch the memcpy before the default-stream pack kernels finish, reading
            # a half-written BGRA buffer and producing a gray-washed torn frame every frame
            # when blend-weight updates are in flight (they extend default-stream work).
            # producer_stream=0 (legacy default stream) is valid: 0 is not None, and
            # cudaEventRecord(event, 0) captures all prior default-stream work correctly.
            outcome = exporter.export(
                GpuFrame(
                    ptr=bgra.data_ptr(),
                    size=bgra.numel(),
                    producer_stream=torch.cuda.current_stream().cuda_stream,
                )
            )
            if self.debug_mode:
                # Health tracking — diagnostic only; gated behind debug_mode (par.Debugmode in TD UI).
                # Reads private attr defensively; safe if vendored exporter.py is re-synced.
                self._ipc_graphs_degraded = getattr(exporter, "_graphs_disabled", False)
                if outcome == FrameOutcome.PUBLISHED:
                    self._ipc_consecutive_failures = 0
                elif outcome == FrameOutcome.FAILED:
                    self._ipc_consecutive_failures += 1
                    logger.warning(
                        "CUDA IPC export failed (consecutive=%d); check GPU/SHM state",
                        self._ipc_consecutive_failures,
                    )
                elif outcome == FrameOutcome.SKIPPED_BARRIER:
                    self._ipc_barrier_skip_count += 1
            return None

        # Fast paths for non-PIL outputs (avoid unnecessary conversions)
        if output_type == "latent":
            return image_tensor
        elif output_type == "pt":
            # Denormalize on GPU, return tensor
            return self._denormalize_on_gpu(image_tensor)
        elif output_type == "np":
            # GPU uint8 conversion + single async DMA to pinned host buffer.
            # uint8 is 4× smaller than fp32, so PCIe transfer time is 4× shorter.
            # Eliminates the intermediate fp32 GPU staging buffer and the PIL round-trip
            # that was needed when callers immediately called np.array(pil_image).
            denormalized = self._denormalize_on_gpu(image_tensor)
            uint8_nhwc = (denormalized * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
            if (
                self._output_pin_buf is None
                or self._output_pin_buf.shape != uint8_nhwc.shape
                or self._output_pin_buf.dtype != torch.uint8
            ):
                self._output_pin_buf = torch.empty(uint8_nhwc.shape, dtype=torch.uint8, pin_memory=True)
                self._d2h_event = torch.cuda.Event()
            self._output_pin_buf.copy_(uint8_nhwc, non_blocking=True)
            with profiler.region("d2h_sync"):
                self._d2h_event.record()
                self._d2h_event.synchronize()
            out = self._output_pin_buf.numpy()
            return out if self.frame_buffer_size > 1 else out[0]

        # PIL output path (optimized)
        if output_type == "pil":
            if self.frame_buffer_size > 1:
                return self._tensor_to_pil_optimized(image_tensor)
            else:
                return self._tensor_to_pil_optimized(image_tensor)[0]

        # Fallback to original method for any unexpected output types
        if self.frame_buffer_size > 1:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)
        else:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)[0]

    def _ipc_pack_rgba(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Convert pipeline output to HWC uint8 BGRA on GPU for cuda-link wire contract."""
        with profiler.region("glue.ipc_pack_rgba"):
            denorm = self._denormalize_on_gpu(image_tensor)  # NCHW [0,1]
            if denorm.dim() == 4:
                denorm = denorm[0]  # CHW [0,1]
            rgb_u8 = (denorm * 255).clamp(0, 255).to(torch.uint8)  # CHW uint8
            rgb_hwc = rgb_u8.permute(1, 2, 0).contiguous()  # HWC RGB
            alpha = torch.full_like(rgb_hwc[..., :1], 255)
            return torch.cat([rgb_hwc[..., 2:3], rgb_hwc[..., 1:2], rgb_hwc[..., 0:1], alpha], dim=-1).contiguous()

    def _lazy_init_ipc_exporter(self, height: int, width: int):
        """Initialize Exporter on first frame (lazy to defer CUDA IPC SHM creation)."""
        if self._cuda_ipc_exporter is not None:
            return self._cuda_ipc_exporter
        from dataclasses import replace as _dc_replace

        from cuda_link import Exporter, ExportPolicy, FrameSpec

        # SD source buffers (_ipc_pack_rgba output) are transient torch tensors recycled by
        # PyTorch's caching allocator the moment the caller returns.  Async export would read
        # recycled memory → torn frames (ADR-0001 source-buffer lifetime race).  Force
        # blocking unless the user explicitly opted into async with CUDALINK_EXPORT_SYNC=0.
        policy = ExportPolicy.from_env()
        if os.environ.get("CUDALINK_EXPORT_SYNC") is None:
            policy = _dc_replace(policy, export_sync=True)

        self._cuda_ipc_exporter = Exporter.open(
            FrameSpec(
                shm_name=self._cuda_ipc_shm_name,
                height=height,
                width=width,
                channels=4,
                dtype="uint8",
                num_slots=self._cuda_ipc_num_slots,
            ),
            policy=policy,
        )
        return self._cuda_ipc_exporter

    def _ipc_pack_unit_rgba(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Convert a [0,1] CHW/NCHW tensor to HWC uint8 BGRA on GPU for cuda-link wire contract.

        Like _ipc_pack_rgba but skips _denormalize_on_gpu — the ControlNet preprocessor
        output is already in [0, 1] (not the diffusion [-1, 1] range).
        """
        with profiler.region("glue.ipc_pack_unit_rgba"):
            t = image_tensor
            if t.dim() == 4:
                t = t[0]  # NCHW → CHW [0,1]
            rgb_u8 = (t * 255).clamp(0, 255).to(torch.uint8)  # CHW uint8
            rgb_hwc = rgb_u8.permute(1, 2, 0).contiguous()  # HWC RGB
            alpha = torch.full_like(rgb_hwc[..., :1], 255)
            return torch.cat([rgb_hwc[..., 2:3], rgb_hwc[..., 1:2], rgb_hwc[..., 0:1], alpha], dim=-1).contiguous()

    def _lazy_init_cn_ipc_exporter(self, height: int, width: int):
        """Initialize the CN-preview Exporter on first frame (lazy, mirrors _lazy_init_ipc_exporter)."""
        if self._cuda_ipc_cn_exporter is not None:
            return self._cuda_ipc_cn_exporter
        from dataclasses import replace as _dc_replace

        from cuda_link import Exporter, ExportPolicy, FrameSpec

        # Same source-buffer lifetime race as _lazy_init_ipc_exporter: _ipc_pack_unit_rgba
        # returns a transient tensor that the caching allocator recycles on return.
        # Force blocking unless CUDALINK_EXPORT_SYNC=0 explicitly opts into async.
        policy = ExportPolicy.from_env()
        if os.environ.get("CUDALINK_EXPORT_SYNC") is None:
            policy = _dc_replace(policy, export_sync=True)

        self._cuda_ipc_cn_exporter = Exporter.open(
            FrameSpec(
                shm_name=self._cuda_ipc_cn_processed_shm_name,
                height=height,
                width=width,
                channels=4,
                dtype="uint8",
                num_slots=self._cuda_ipc_num_slots,
            ),
            policy=policy,
        )
        return self._cuda_ipc_cn_exporter

    def export_controlnet_preview_ipc(self, tensor: torch.Tensor) -> None:
        """Export a ControlNet preprocessor output tensor to TD via zero-copy GPU IPC.

        The tensor must be in [0, 1] range (CHW or NCHW); it is NOT denormalized.
        This is a display-only path — no health tracking, no return value.
        No-op if cuda_ipc_cn_processed_shm_name was not configured.
        """
        logger.info(  # [DEBUG-cnprev] — remove after preview confirmed
            "[DEBUG-cnprev] export_controlnet_preview_ipc called, shm_name=%r",
            self._cuda_ipc_cn_processed_shm_name,
        )
        if not self._cuda_ipc_cn_processed_shm_name:
            return
        try:
            from cuda_link import GpuFrame

            bgra = self._ipc_pack_unit_rgba(tensor)
            exporter = self._lazy_init_cn_ipc_exporter(bgra.shape[0], bgra.shape[1])
            exporter.export(
                GpuFrame(
                    ptr=bgra.data_ptr(),
                    size=bgra.numel(),
                    producer_stream=torch.cuda.current_stream().cuda_stream,
                )
            )
        except Exception:
            logger.debug("export_controlnet_preview_ipc: export failed", exc_info=True)

    def get_ipc_health_status(self) -> str:
        """Return a short health string for the CUDA-IPC zero-copy output path.

        Designed for the 1 Hz status loop — reads only Python counters and one private attr;
        no GPU calls, no locks.  Returns one of:
          'disabled'           – use_cuda_ipc_output is off
          'not-init'           – exporter not yet constructed (first frame not processed)
          'FAILED(N)'          – N consecutive per-frame export failures
          'barrier-skip(N)'    – activation-barrier skips (normal during startup settle)
          'ok/graph-fallback'  – exporter running but CUDA graphs fell back to legacy memcpy
          'ok'                 – all clear, zero-copy graphs active
        """
        if not (self.use_cuda_ipc_output and self._cuda_ipc_shm_name):
            return "disabled"
        exporter = self._cuda_ipc_exporter
        if exporter is None:
            return "not-init"
        if self._ipc_consecutive_failures > 0:
            return f"FAILED({self._ipc_consecutive_failures})"
        if self._ipc_barrier_skip_count > 0:
            s = f"barrier-skip({self._ipc_barrier_skip_count})"
            self._ipc_barrier_skip_count = 0  # reset: startup transient, report once then clear
            return s
        if self._ipc_graphs_degraded:
            return "ok/graph-fallback"
        return "ok"

    def cleanup_cuda_ipc(self) -> None:
        """Tear down the CUDA IPC exporters and release their SHM + GPU resources."""
        if self._cuda_ipc_exporter is not None:
            try:
                self._cuda_ipc_exporter.close()
            except Exception:
                logger.debug("cleanup_cuda_ipc: _cuda_ipc_exporter.close() failed", exc_info=True)
            self._cuda_ipc_exporter = None
        if self._cuda_ipc_cn_exporter is not None:
            try:
                self._cuda_ipc_cn_exporter.close()
            except Exception:
                logger.debug("cleanup_cuda_ipc: _cuda_ipc_cn_exporter.close() failed", exc_info=True)
            self._cuda_ipc_cn_exporter = None

    def _denormalize_on_gpu(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize image tensor on GPU for efficiency.

        Converts image tensor from diffusion range [-1, 1] to standard image range [0, 1].

        Parameters
        ----------
        image_tensor : torch.Tensor
            Input tensor in diffusion range [-1, 1], expected to be on GPU.

        Returns
        -------
        torch.Tensor
            Denormalized tensor in range [0, 1], clamped and on GPU.
        """
        return (image_tensor / 2 + 0.5).clamp(0, 1)

    def _normalize_on_gpu(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor from processor range to diffusion range.

        Converts image tensor from standard image range [0, 1] to diffusion range [-1, 1].

        Parameters
        ----------
        image_tensor : torch.Tensor
            Input tensor in standard image range [0, 1], expected to be on GPU.

        Returns
        -------
        torch.Tensor
            Normalized tensor in diffusion range [-1, 1], clamped and on GPU.
        """
        return (image_tensor * 2 - 1).clamp(-1, 1)

    def _tensor_to_pil_optimized(self, image_tensor: torch.Tensor) -> List[Image.Image]:
        """
        Optimized tensor to PIL conversion with minimal CPU transfers.

        Efficiently converts a batch of GPU tensors to PIL Images with minimal
        CPU-GPU transfers and memory allocations.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Input tensor in diffusion range [-1, 1], expected to be on GPU.
            Shape should be (batch_size, channels, height, width).

        Returns
        -------
        List[Image.Image]
            List of PIL RGB images, one for each item in the batch.
        """
        # Denormalize on GPU first
        denormalized = self._denormalize_on_gpu(image_tensor)

        # Convert to uint8 on GPU to reduce transfer size
        # Scale to [0, 255] and convert to uint8
        # Scale to [0, 255] and convert to uint8
        uint8_tensor = (denormalized * 255).clamp(0, 255).to(torch.uint8)

        # Single efficient CPU transfer
        cpu_tensor = uint8_tensor.cpu()

        # Convert to HWC format for PIL
        # From BCHW to BHWC
        cpu_tensor = cpu_tensor.permute(0, 2, 3, 1)

        # Convert to PIL images efficiently
        pil_images = []
        for i in range(cpu_tensor.shape[0]):
            img_array = cpu_tensor[i].numpy()

            if img_array.shape[-1] == 1:
                # Grayscale
                pil_images.append(Image.fromarray(img_array.squeeze(-1), mode="L"))
            else:
                # RGB
                pil_images.append(Image.fromarray(img_array))

        return pil_images

    def _apply_safety_checker(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Run the NSFW check on the raw pipeline tensor and substitute a fallback if flagged.

        This MUST be called *before* postprocess_image so that the substitution also covers the
        CUDA-IPC export path: postprocess_image() exports the frame inside its own body and
        returns None, making any post-hoc substitution unreachable and unsafe.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Raw pipeline output in diffusion range [-1, 1], NCHW.

        Returns
        -------
        torch.Tensor
            The original tensor when content is clean, or a fallback tensor (previous clean
            frame, or all-black encoded as -1.0 in diffusion range) when content is flagged.
        """
        if not self.use_safety_checker:
            return image_tensor

        # Denormalize to [0, 1] NCHW for the classifier; stays on GPU.
        denormalized = self._denormalize_on_gpu(image_tensor)

        if self.safety_checker(denormalized, self.safety_checker_threshold):
            logger.info("NSFW content detected, applying safety fallback frame")
            if self.safety_checker_fallback_type == "previous" and self._prev_clean_tensor is not None:
                return self._prev_clean_tensor
            # -1.0 in diffusion range → 0.0 after denormalization → true black on every output
            # path (pt, np, pil, CUDA-IPC).
            return torch.full_like(image_tensor, -1.0)

        # Content is clean — cache it for the "previous" fallback strategy.
        if self.safety_checker_fallback_type == "previous":
            self._prev_clean_tensor = image_tensor.clone()
        return image_tensor

    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        engine_dir: Optional[Union[str, Path]] = "engines",
        build_engines_if_missing: bool = True,
        normalize_prompt_weights: bool = True,
        normalize_seed_weights: bool = True,
        scheduler: Literal["lcm", "tcd"] = "lcm",
        sampler: Literal["simple", "sgm_uniform", "normal", "ddim", "beta", "karras"] = "normal",
        use_controlnet: bool = False,
        controlnet_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        use_ipadapter: bool = False,
        ipadapter_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        # Pipeline hook configurations (Phase 4: Configuration Integration)
        image_preprocessing_config: Optional[Dict[str, Any]] = None,
        image_postprocessing_config: Optional[Dict[str, Any]] = None,
        latent_preprocessing_config: Optional[Dict[str, Any]] = None,
        latent_postprocessing_config: Optional[Dict[str, Any]] = None,
        safety_checker_model_id: Optional[str] = "Freepik/nsfw_image_detector",
        compile_engines_only: bool = False,
        use_cached_attn: bool = False,
        cache_maxframes: int = 1,
        cache_interval: int = 1,
        min_cache_maxframes: int = 1,
        max_cache_maxframes: int = 4,
        cn_cache_interval: int = 1,
        use_feature_injection: bool = False,
        fi_strength: float = 0.75,
        fi_threshold: float = 0.98,
        fp8: bool = False,
    ) -> StreamDiffusion:
        """
        Loads the model.

        This method does the following:

        1. Loads the model from the model_id_or_path.
        2. Loads and fuses LoRA models from lora_dict if provided.
        3. Loads the VAE model from the vae_id if needed.
        4. Enables acceleration if needed.
        5. Prepares the model for inference.
        6. Load the safety checker if needed.
        7. Apply ControlNet patch if needed.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load. Can be a Hugging Face model ID, local path to
            safetensors/ckpt file, or directory containing model files.
        t_index_list : List[int]
            The t_index_list to use for inference. Specifies which denoising timesteps
            to use from the diffusion schedule.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
            Use this to load LCM LoRA: {'latent-consistency/lcm-lora-sdv1-5': 1.0}
        vae_id : Optional[str], optional
            The vae_id to load, by default None. If None, uses default TinyVAE
            ("madebyollin/taesd" for SD1.5, "madebyollin/taesdxl" for SDXL).
        acceleration : Literal["none", "xformers", "tensorrt"], optional
            The acceleration method, by default "tensorrt". Note: docstring shows
            "xfomers" and "sfast" but code uses "xformers".
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        use_lcm_lora : bool, optional
            DEPRECATED: Use lora_dict instead. For backwards compatibility only.
            If True, automatically adds appropriate LCM LoRA to lora_dict based on model type.
            SDXL models get "latent-consistency/lcm-lora-sdxl", others get "latent-consistency/lcm-lora-sdv1-5".
            By default None (ignored).
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True. TinyVAE is a distilled,
            smaller VAE model that provides faster encoding/decoding with minimal quality loss.
        cfg_type : Literal["none", "full", "self", "initialize"], optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        engine_dir : Optional[Union[str, Path]], optional
            Directory path for storing/loading TensorRT engines, by default "engines".
        build_engines_if_missing : bool, optional
            Whether to build TensorRT engines if they don't exist, by default True.
        normalize_prompt_weights : bool, optional
            Whether to normalize prompt weights in blending to sum to 1, by default True.
            When False, weights > 1 will amplify embeddings.
        normalize_seed_weights : bool, optional
            Whether to normalize seed weights in blending to sum to 1, by default True.
            When False, weights > 1 will amplify noise.
        scheduler : Literal["lcm", "tcd"], optional
            The scheduler type to use for denoising, by default "lcm".
        sampler : Literal["simple", "sgm_uniform", "normal", "ddim", "beta", "karras"], optional
            The sampler type to use for noise scheduling, by default "normal".
        use_controlnet : bool, optional
            Whether to enable ControlNet support, by default False.
        controlnet_config : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional
            ControlNet configuration(s), by default None. Can be a single config dict
            or list of config dicts for multiple ControlNets.
        use_ipadapter : bool, optional
            Whether to enable IPAdapter support, by default False.
        ipadapter_config : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional
            IPAdapter configuration(s), by default None. Can be a single config dict
            or list of config dicts for multiple IPAdapters.
        image_preprocessing_config : Optional[Dict[str, Any]], optional
            Configuration for image preprocessing hooks, by default None.
        image_postprocessing_config : Optional[Dict[str, Any]], optional
            Configuration for image postprocessing hooks, by default None.
        latent_preprocessing_config : Optional[Dict[str, Any]], optional
            Configuration for latent preprocessing hooks, by default None.
        latent_postprocessing_config : Optional[Dict[str, Any]], optional
            Configuration for latent postprocessing hooks, by default None.
        safety_checker_model_id : Optional[str], optional
            Model ID for the safety checker, by default "Freepik/nsfw_image_detector".
        compile_engines_only : bool, optional
            Whether to only compile engines and not load the model, by default False.

        Returns
        -------
        StreamDiffusion
            The loaded model (potentially wrapped with ControlNet pipeline).
        """

        # Clean up GPU memory before loading new model to prevent OOM errors
        try:
            self.cleanup_gpu_memory()
        except Exception as e:
            logger.warning(f"GPU cleanup warning: {e}")

        # Reset CUDA context to prevent corruption from previous runs
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force CUDA context reset by creating and destroying a small tensor
        temp_tensor = torch.zeros(1, device=self.device)
        del temp_tensor
        logger.info("_load_model: CUDA context reset completed")

        # First, try to detect if this is an SDXL model before loading
        # TODO: CAN we do this step with model_detection.py?
        is_sdxl_model = False
        model_path_lower = model_id_or_path.lower()

        # Check path for SDXL indicators
        if any(indicator in model_path_lower for indicator in ["sdxl", "xl", "1024"]):
            is_sdxl_model = True
            logger.info(f"_load_model: Path suggests SDXL model: {model_id_or_path}")

        # For .safetensor files, we need to be more careful about pipeline selection
        if model_id_or_path.endswith(".safetensors"):
            # For .safetensor files, try SDXL pipeline first if path suggests SDXL
            if is_sdxl_model:
                loading_methods = [
                    (StableDiffusionXLPipeline.from_single_file, "SDXL from_single_file"),
                    (AutoPipelineForText2Image.from_pretrained, "AutoPipeline from_pretrained"),
                    (StableDiffusionPipeline.from_single_file, "SD from_single_file"),
                ]
            else:
                loading_methods = [
                    (AutoPipelineForText2Image.from_pretrained, "AutoPipeline from_pretrained"),
                    (StableDiffusionPipeline.from_single_file, "SD from_single_file"),
                    (StableDiffusionXLPipeline.from_single_file, "SDXL from_single_file"),
                ]
        else:
            # For regular model directories or checkpoints, use the original order
            loading_methods = [
                (AutoPipelineForText2Image.from_pretrained, "AutoPipeline from_pretrained"),
                (StableDiffusionPipeline.from_single_file, "SD from_single_file"),
                (StableDiffusionXLPipeline.from_single_file, "SDXL from_single_file"),
            ]

        pipe = None
        last_error = None
        for method, method_name in loading_methods:
            try:
                logger.info(f"_load_model: Attempting to load with {method_name}...")
                pipe = method(model_id_or_path).to(dtype=self.dtype)
                logger.info(f"_load_model: Successfully loaded using {method_name}")

                # Verify that we have the right pipeline type for SDXL models
                if is_sdxl_model and not isinstance(pipe, StableDiffusionXLPipeline):
                    logger.warning(f"_load_model: SDXL model detected but loaded with non-SDXL pipeline: {type(pipe)}")
                    # Try to explicitly load with SDXL pipeline instead
                    try:
                        logger.info("_load_model: Retrying with StableDiffusionXLPipeline...")
                        pipe = StableDiffusionXLPipeline.from_single_file(model_id_or_path).to(dtype=self.dtype)
                        logger.info("_load_model: Successfully loaded using SDXL pipeline on retry")
                    except Exception as retry_error:
                        logger.warning(f"_load_model: SDXL pipeline retry failed: {retry_error}")
                        # Continue with the originally loaded pipeline

                break
            except Exception as e:
                logger.warning(f"_load_model: {method_name} failed: {e}")
                last_error = e
                continue

        if pipe is None:
            error_msg = (
                f"_load_model: All loading methods failed for model '{model_id_or_path}'. Last error: {last_error}"
            )
            logger.error(error_msg)
            if last_error:
                logger.warning("Full traceback of last error:")
                import traceback

                traceback.print_exc()
            raise RuntimeError(error_msg)
        else:
            if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                pipe.text_encoder = pipe.text_encoder.to(device=self.device)
            if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
                pipe.text_encoder_2 = pipe.text_encoder_2.to(device=self.device)
            # Move main pipeline components to device, but skip UNet for TensorRT
            if hasattr(pipe, "unet") and pipe.unet is not None and acceleration != "tensorrt":
                pipe.unet = pipe.unet.to(device=self.device)
            if hasattr(pipe, "vae") and pipe.vae is not None and acceleration != "tensorrt":
                pipe.vae = pipe.vae.to(device=self.device)

        # If we get here, the model loaded successfully - break out of retry loop
        logger.info("Model loading succeeded")

        # Use comprehensive model detection instead of basic detection
        detection_result = detect_model(pipe.unet, pipe)
        model_type = detection_result["model_type"]
        is_sdxl = detection_result["is_sdxl"]
        is_turbo = detection_result["is_turbo"]
        confidence = detection_result["confidence"]

        # Store comprehensive model info for later use (after TensorRT conversion)
        self._detected_model_type = model_type
        self._detection_confidence = confidence
        self._is_turbo = is_turbo
        self._is_sdxl = is_sdxl

        logger.info(f"_load_model: Detected model type: {model_type} (confidence: {confidence:.2f})")

        # Auto-resolve IP-Adapter model/encoder paths for detected architecture.
        # Runs once here so both pre-TRT and post-TRT installation paths see the resolved cfg.
        if use_ipadapter and ipadapter_config:
            from streamdiffusion.modules.ipadapter_module import resolve_ipadapter_paths

            _ip_cfgs = ipadapter_config if isinstance(ipadapter_config, list) else [ipadapter_config]
            for _ip_cfg in _ip_cfgs:
                resolve_ipadapter_paths(_ip_cfg, model_type, is_sdxl)

        # DEPRECATED: THIS WILL LOAD LCM_LORA IF USE_LCM_LORA IS TRUE
        # Validate backwards compatibility LCM LoRA selection using proper model detection
        if hasattr(self, "use_lcm_lora") and self.use_lcm_lora is not None:
            if self.use_lcm_lora and not self.sd_turbo:
                if lora_dict is None:
                    lora_dict = {}

                # Determine correct LCM LoRA based on actual model detection
                lcm_lora = "latent-consistency/lcm-lora-sdxl" if is_sdxl else "latent-consistency/lcm-lora-sdv1-5"

                # Add to lora_dict if not already present
                if lcm_lora not in lora_dict:
                    lora_dict[lcm_lora] = 1.0
                    logger.info(f"Added {lcm_lora} with scale 1.0 to lora_dict")
                else:
                    logger.info(f"LCM LoRA {lcm_lora} already present in lora_dict with scale {lora_dict[lcm_lora]}")
            else:
                logger.info(
                    f"LCM LoRA will not be loaded because use_lcm_lora is {self.use_lcm_lora} and sd_turbo is {self.sd_turbo}"
                )

                # Remove use_lcm_lora from self
                self.use_lcm_lora = None
                logger.info("use_lcm_lora has been removed from self")

        # Get kvo_cache_structure before stream init (needed for TRT export wrapper).
        # Actual cache tensors are created AFTER stream init so we can use
        # stream.trt_unet_batch_size, which accounts for scheduler overrides
        # (e.g. TCD sets trt_unet_batch_size = frame_buffer_size, not denoising_steps * frame_buffer_size).
        if use_cached_attn:
            from streamdiffusion.acceleration.tensorrt.models.utils import get_kvo_cache_info

            _, kvo_cache_structure, _ = get_kvo_cache_info(pipe.unet, self.height, self.width)
        else:
            kvo_cache_structure = []

        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            device=self.device,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
            lora_dict=lora_dict,  # We pass this to include loras in engine path names
            normalize_prompt_weights=normalize_prompt_weights,
            normalize_seed_weights=normalize_seed_weights,
            scheduler=scheduler,
            sampler=sampler,
            kvo_cache=[],  # Set below after stream init with the correct batch size
            cache_interval=cache_interval,
            cache_maxframes=cache_maxframes,
            fio_cache=[],  # Set below if FI is enabled
            use_feature_injection=use_feature_injection and use_cached_attn,
        )

        # Create KVO cache tensors using the pipeline's actual runtime batch size.
        # pipeline.py overrides trt_unet_batch_size for TCD (= frame_buffer_size),
        # so this must happen after StreamDiffusion.__init__ to get the correct value.
        if use_cached_attn:
            from streamdiffusion.acceleration.tensorrt.models.utils import create_kvo_cache

            kvo_cache, _, kvo_buckets, kvo_outputs_by_bucket = create_kvo_cache(
                pipe.unet,
                batch_size=stream.trt_unet_batch_size,
                cache_maxframes=max_cache_maxframes,  # Allocate at max to avoid runtime resize race
                height=self.height,
                width=self.width,
                device=self.device,
                dtype=self.dtype,
            )
            stream.kvo_cache = kvo_cache
            stream._kvo_buckets = kvo_buckets
            stream._kvo_outputs_by_bucket = kvo_outputs_by_bucket

        # Allocate FI output-cache (O-cache) for Feature Injection (Phase 2, StreamV2V §3.4.2).
        # Must happen after kvo_cache so create_fi_cache can call get_kvo_cache_info for alignment.
        if use_feature_injection and use_cached_attn:
            from streamdiffusion.acceleration.tensorrt.models.utils import create_fi_cache

            fio_cache, _, _, _ = create_fi_cache(
                pipe.unet,
                batch_size=stream.trt_unet_batch_size,
                cache_maxframes=max_cache_maxframes,
                height=self.height,
                width=self.width,
                device=self.device,
                dtype=self.dtype,
            )
            stream.fio_cache = fio_cache
            stream.use_feature_injection = True
            # Persistent fp32 [1] tensors — updated in-place by stream_parameter_updater
            # (CUDA-graph-safe: same device address across frames).
            stream._fi_strength_tensor = torch.tensor([float(fi_strength)], dtype=torch.float32, device=stream.device)
            stream._fi_threshold_tensor = torch.tensor(
                [float(fi_threshold)], dtype=torch.float32, device=stream.device
            )

        # Load and properly merge LoRA weights using the standard diffusers approach
        lora_adapters_to_merge = []
        lora_scales_to_merge = []
        # adapter_name → (lora_name, lora_scale) for only successfully loaded adapters (G1 fix)
        _loaded_adapter_names: dict = {}

        # Collect all LoRA adapters and their scales from lora_dict
        if lora_dict is not None:
            for i, (lora_name, lora_scale) in enumerate(lora_dict.items()):
                adapter_name = f"custom_lora_{i}"
                logger.info(f"_load_model: Loading LoRA '{lora_name}' with scale {lora_scale}")

                # G8 fix: scale-0 fuse is a mathematical no-op (W + 0·ΔW = W), so skip
                # loading and fusing entirely.  The entry is also excluded from
                # _loaded_adapter_names so the G1 block at the end of the loop naturally
                # drops it from the engine cache signature — a lora_dict with only
                # zero-scale entries collapses to None and reuses the baseline UNet engine.
                # Note: negative scales are valid (subtract the LoRA delta), so skip == 0
                # exactly, not <= 0.
                if lora_scale == 0:
                    logger.info(
                        f"_load_model: Skipping zero-scale LoRA '{lora_name}' — "
                        "no effect on weights; engine will match baseline cache"
                    )
                    continue

                try:
                    # Load LoRA weights with unique adapter name
                    stream.load_lora(lora_name, adapter_name=adapter_name)
                    lora_adapters_to_merge.append(adapter_name)
                    lora_scales_to_merge.append(lora_scale)
                    _loaded_adapter_names[adapter_name] = (lora_name, lora_scale)
                    logger.info(f"Successfully loaded LoRA adapter: {adapter_name}")
                except Exception as e:
                    logger.error(f"Failed to load LoRA {lora_name}: {e}")
                    # Drop this entry — do NOT carry it into the engine cache key (G1 fix)
                    continue

        # Merge all LoRA adapters using the proper diffusers method
        if lora_adapters_to_merge:
            try:
                for adapter_name, scale in zip(lora_adapters_to_merge, lora_scales_to_merge):
                    logger.info(f"Merging individual LoRA: {adapter_name} with scale {scale}")
                    stream.pipe.fuse_lora(lora_scale=scale, adapter_names=[adapter_name])

                # Clean up after individual merging
                stream.pipe.unload_lora_weights()
                logger.info("Successfully merged LoRAs individually")

            except Exception as fuse_error:
                # Partial fusion leaves UNet weights in an ambiguous state; baking a TRT engine
                # from this state creates a permanently mislabeled or corrupted engine (G1 fix).
                try:
                    stream.pipe.unload_lora_weights()
                except Exception:
                    logger.debug("LoRA cleanup: unload_lora_weights() failed after merge failure", exc_info=True)
                raise RuntimeError(
                    f"LoRA fusion failed — cannot build TRT engine with partial UNet state. Error: {fuse_error}"
                ) from fuse_error

        # G1 fix: Correct lora_dict to only contain successfully fused LoRAs so that
        # get_engine_path() computes the correct engine cache signature.  Any LoRA that
        # failed to load was never merged into UNet weights; the engine must NOT carry
        # its signature in the cache path.
        if lora_dict is not None:
            fused_lora_dict = {
                lora_name: lora_scale for _adapter, (lora_name, lora_scale) in _loaded_adapter_names.items()
            }
            lora_dict = fused_lora_dict if fused_lora_dict else None

        if use_tiny_vae:
            if vae_id is not None:
                stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(device=self.device, dtype=self.dtype)
            else:
                # Use TAESD XL for SDXL models, regular TAESD for SD 1.5
                taesd_model = "madebyollin/taesdxl" if is_sdxl else "madebyollin/taesd"
                stream.vae = AutoencoderTiny.from_pretrained(taesd_model).to(device=self.device, dtype=self.dtype)
        elif acceleration != "tensorrt":
            # For non-TensorRT acceleration, ensure VAE is on device if it wasn't moved earlier
            if hasattr(pipe, "vae") and pipe.vae is not None:
                pipe.vae = pipe.vae.to(device=self.device)

        try:
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
            if acceleration == "tensorrt":
                from polygraphy import cuda

                from streamdiffusion.acceleration.tensorrt import TorchVAEEncoder
                from streamdiffusion.acceleration.tensorrt.engine_manager import EngineManager, EngineType
                from streamdiffusion.acceleration.tensorrt.models.models import (
                    VAE,
                    NSFWDetector,
                    UNet,
                    VAEEncoder,
                )
                from streamdiffusion.acceleration.tensorrt.runtime_engines.unet_engine import (
                    AutoencoderKLEngine,
                    NSFWDetectorEngine,
                )

                # Add ControlNet detection and support
                from streamdiffusion.model_detection import extract_unet_architecture, validate_architecture

                # Legacy TensorRT implementation (fallback)
                # Initialize engine manager
                engine_manager = EngineManager(engine_dir)

                # Enhanced SDXL and ControlNet TensorRT support
                use_controlnet_trt = False
                use_ipadapter_trt = False
                unet_arch = {}
                is_sdxl_model = False
                load_engine = not compile_engines_only

                # Use the explicit use_ipadapter parameter
                has_ipadapter = use_ipadapter

                # Determine IP-Adapter presence and token count directly from config (no legacy pipeline)
                if has_ipadapter and not ipadapter_config:
                    has_ipadapter = False

                try:
                    # Use model detection results already computed during model loading
                    model_type = getattr(self, "_detected_model_type", "SD15")
                    is_sdxl = getattr(self, "_is_sdxl", False)
                    is_turbo = getattr(self, "_is_turbo", False)
                    confidence = getattr(self, "_detection_confidence", 0.0)

                    if is_sdxl:
                        logger.info(f"Building TensorRT engines for SDXL model: {model_type}")
                        logger.info(f"   Turbo variant: {is_turbo}")
                        logger.info(f"   Detection confidence: {confidence:.2f}")
                    else:
                        logger.info(f"Building TensorRT engines for {model_type}")

                    # Enable IPAdapter TensorRT if configured and available
                    if has_ipadapter:
                        use_ipadapter_trt = True

                    # Only enable ControlNet for legacy TensorRT if ControlNet is actually being used
                    if self.use_controlnet:
                        try:
                            unet_arch = extract_unet_architecture(stream.unet)
                            unet_arch = validate_architecture(unet_arch, model_type)
                            use_controlnet_trt = True
                            logger.info(f"   Including ControlNet support for {model_type}")
                        except Exception as e:
                            logger.warning(f"   ControlNet architecture detection failed: {e}")
                            use_controlnet_trt = False

                    # Set up architecture info for enabled modes
                    if use_controlnet_trt and not use_ipadapter_trt:
                        # ControlNet only: Full architecture needed
                        if not unet_arch:
                            unet_arch = extract_unet_architecture(stream.unet)
                            unet_arch = validate_architecture(unet_arch, model_type)
                    elif use_ipadapter_trt and not use_controlnet_trt:
                        # IPAdapter only: Cross-attention dim needed
                        unet_arch = {"context_dim": stream.unet.config.cross_attention_dim}
                    elif use_controlnet_trt and use_ipadapter_trt:
                        # Combined mode: Full architecture + cross-attention dim
                        if not unet_arch:
                            unet_arch = extract_unet_architecture(stream.unet)
                            unet_arch = validate_architecture(unet_arch, model_type)
                        unet_arch["context_dim"] = stream.unet.config.cross_attention_dim
                    else:
                        # Neither enabled: Standard UNet
                        unet_arch = {}

                except Exception as e:
                    logger.error(f"Advanced model detection failed: {e}")
                    logger.error("   Falling back to basic TensorRT")

                    # Fallback to basic detection
                    try:
                        detection_result = detect_model(stream.unet, None)
                        model_type = detection_result["model_type"]
                        is_sdxl = detection_result["is_sdxl"]
                        if self.use_controlnet:
                            unet_arch = extract_unet_architecture(stream.unet)
                            unet_arch = validate_architecture(unet_arch, model_type)
                            use_controlnet_trt = True
                    except Exception:
                        pass

                if not use_controlnet_trt and not self.use_controlnet:
                    logger.info("ControlNet not enabled, building engines without ControlNet support")

                # Use the engine_dir parameter passed to this function, with fallback to instance variable
                engine_dir = engine_dir if engine_dir else getattr(self, "_engine_dir", "engines")

                # Resolve IP-Adapter runtime params from config
                # Strength is now a runtime input, so we do NOT bake scale into engine identity
                ipadapter_scale = None
                ipadapter_tokens = None
                if use_ipadapter_trt and has_ipadapter and ipadapter_config:
                    cfg0 = ipadapter_config[0] if isinstance(ipadapter_config, list) else ipadapter_config
                    # scale omitted from engine naming; runtime will pass ipadapter_scale vector
                    ipadapter_tokens = cfg0.get("num_image_tokens", 4)
                    # Determine FaceID type from config for engine naming
                    is_faceid = cfg0["type"] == "faceid"
                # Generate engine paths using EngineManager
                unet_path = engine_manager.get_engine_path(
                    EngineType.UNET,
                    model_id_or_path=model_id_or_path,
                    max_batch_size=self.max_batch_size,
                    min_batch_size=self.min_batch_size,
                    mode=self.mode,
                    use_tiny_vae=use_tiny_vae,
                    lora_dict=lora_dict,
                    ipadapter_scale=ipadapter_scale,
                    ipadapter_tokens=ipadapter_tokens,
                    is_faceid=is_faceid if use_ipadapter_trt else None,
                    use_cached_attn=use_cached_attn,
                    use_feature_injection=use_feature_injection,
                    use_controlnet=use_controlnet_trt,
                    fp8=fp8,
                    resolution=(self.height, self.width),
                    builder_optimization_level=self.builder_optimization_level,
                    # Must match the hardcoded build_static_batch value below so the cache
                    # key reflects the actual TRT profile policy (static vs dynamic batch).
                    build_static_batch=True,
                )
                # Effective VAE optlvl: per-engine override first, then global fallback.
                _vae_optlvl = (
                    self.vae_builder_optimization_level
                    if self.vae_builder_optimization_level is not None
                    else self.builder_optimization_level
                )
                vae_encoder_path = engine_manager.get_engine_path(
                    EngineType.VAE_ENCODER,
                    model_id_or_path=model_id_or_path,
                    max_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    mode=self.mode,
                    use_tiny_vae=use_tiny_vae,
                    lora_dict=lora_dict,
                    ipadapter_scale=ipadapter_scale,
                    ipadapter_tokens=ipadapter_tokens,
                    is_faceid=is_faceid if use_ipadapter_trt else None,
                    resolution=(self.height, self.width),
                    builder_optimization_level=_vae_optlvl,
                )
                vae_decoder_path = engine_manager.get_engine_path(
                    EngineType.VAE_DECODER,
                    model_id_or_path=model_id_or_path,
                    max_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    mode=self.mode,
                    use_tiny_vae=use_tiny_vae,
                    lora_dict=lora_dict,
                    ipadapter_scale=ipadapter_scale,
                    ipadapter_tokens=ipadapter_tokens,
                    is_faceid=is_faceid if use_ipadapter_trt else None,
                    resolution=(self.height, self.width),
                    builder_optimization_level=_vae_optlvl,
                )

                # Check if all required engines exist
                missing_engines = []
                if not unet_path.exists():
                    missing_engines.append(f"UNet engine: {unet_path}")
                if not vae_decoder_path.exists():
                    missing_engines.append(f"VAE decoder engine: {vae_decoder_path}")
                if not vae_encoder_path.exists():
                    missing_engines.append(f"VAE encoder engine: {vae_encoder_path}")

                if missing_engines:
                    if build_engines_if_missing:
                        logger.info("Missing TensorRT engines, building them...")
                        for engine in missing_engines:
                            logger.info(f"  - {engine}")
                    else:
                        error_msg = "Required TensorRT engines are missing and build_engines_if_missing=False:\n"
                        for engine in missing_engines:
                            error_msg += f"  - {engine}\n"
                        error_msg += (
                            "\nTo build engines, set build_engines_if_missing=True or run the build script manually."
                        )
                        raise RuntimeError(error_msg)

                # Determine correct embedding dimension based on model type
                if is_sdxl:
                    # SDXL uses concatenated embeddings from dual text encoders (768 + 1280 = 2048)
                    embedding_dim = 2048
                    logger.info(f"SDXL model detected! Using embedding_dim = {embedding_dim}")
                else:
                    # SD1.5, SD2.1, etc. use single text encoder
                    embedding_dim = stream.text_encoder.config.hidden_size
                    logger.info(f"Non-SDXL model ({model_type}) detected! Using embedding_dim = {embedding_dim}")

                # Gather parameters for unified wrapper - validate IPAdapter first for consistent token count
                control_input_names = None
                num_tokens = 4  # Default for non-IPAdapter mode

                if use_ipadapter_trt:
                    # Use token count resolved from configuration (default to 4)
                    num_tokens = ipadapter_tokens if isinstance(ipadapter_tokens, int) else 4

                # Compile UNet engine using EngineManager
                logger.info(
                    f"compile_and_load_engine: Compiling UNet engine for image size: {self.width}x{self.height}"
                )
                logger.debug(f"compile_and_load_engine: use_ipadapter_trt={use_ipadapter_trt}, tokens={num_tokens}")

                # Note: LoRA weights have already been merged permanently during model loading

                # CRITICAL: Install IPAdapter module BEFORE TensorRT compilation to ensure processors are baked into engines
                if use_ipadapter and ipadapter_config and not hasattr(stream, "_ipadapter_module"):
                    # Check if auto-resolution disabled IP-Adapter (e.g. no adapter released for this arch)
                    _cfg_check = ipadapter_config[0] if isinstance(ipadapter_config, list) else ipadapter_config
                    if _cfg_check.get("enabled", True) is False:
                        logger.info(
                            "IP-Adapter disabled by auto-resolution (no compatible adapter for this model). Skipping."
                        )
                        use_ipadapter_trt = False
                    else:
                        try:
                            from streamdiffusion.modules.ipadapter_module import (
                                IPAdapterConfig,
                                IPAdapterModule,
                                IPAdapterType,
                            )

                            logger.info("Installing IPAdapter module before TensorRT compilation...")

                            # Snapshot processors before install — IPAdapter.set_ip_adapter() replaces them
                            # before load_state_dict(), so a failure leaves the UNet in corrupted state
                            _saved_unet_processors = dict(stream.unet.attn_processors)

                            # Use first config if list provided
                            cfg = ipadapter_config[0] if isinstance(ipadapter_config, list) else ipadapter_config
                            ip_cfg = IPAdapterConfig(
                                style_image_key=cfg.get("style_image_key") or "ipadapter_main",
                                num_image_tokens=cfg.get("num_image_tokens", 4),
                                ipadapter_model_path=cfg["ipadapter_model_path"],
                                image_encoder_path=cfg["image_encoder_path"],
                                style_image=cfg.get("style_image"),
                                scale=cfg.get("scale", 1.0),
                                type=IPAdapterType(cfg.get("type", "regular")),
                                insightface_model_name=cfg.get("insightface_model_name"),
                            )
                            ip_module = IPAdapterModule(ip_cfg)
                            ip_module.install(stream)
                            # Expose for later updates
                            stream._ipadapter_module = ip_module
                            logger.info("IPAdapter module installed successfully before TensorRT compilation")

                            # Cleanup after IPAdapter installation
                            import gc

                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                        except torch.cuda.OutOfMemoryError as oom_error:
                            logger.error(f"CUDA Out of Memory during early IPAdapter installation: {oom_error}")
                            logger.error("Try reducing batch size, using smaller models, or increasing GPU memory")
                            raise RuntimeError(
                                "Insufficient VRAM for IPAdapter installation. Consider using a GPU with more memory or reducing model complexity."
                            )

                        except RuntimeError as rt_error:
                            if "size mismatch" in str(rt_error):
                                unet_dim = getattr(getattr(stream, "unet", None), "config", None)
                                unet_cross_attn = (
                                    getattr(unet_dim, "cross_attention_dim", "unknown") if unet_dim else "unknown"
                                )
                                logger.warning(
                                    f"IP-Adapter weights are incompatible with this model "
                                    f"(UNet cross_attention_dim={unet_cross_attn}). "
                                    f"Checkpoint dimension does not match — this may be a custom model path "
                                    f"that could not be auto-resolved. "
                                    f"Check ipadapter_model_path in td_config.yaml. "
                                    f"Skipping IP-Adapter and continuing without it."
                                )
                                # Restore original processors — IPAdapter.set_ip_adapter() already replaced
                                # them before load_state_dict() failed, leaving the UNet in a corrupted state
                                try:
                                    stream.unet.set_attn_processor(_saved_unet_processors)
                                    logger.info(
                                        "Restored original UNet attention processors after IP-Adapter failure."
                                    )
                                except Exception as restore_err:
                                    logger.warning(f"Could not restore UNet processors: {restore_err}")
                                use_ipadapter_trt = False
                            else:
                                import traceback

                                traceback.print_exc()
                                logger.error("Failed to install IPAdapterModule before TensorRT compilation")
                                raise

                        except Exception as e:
                            import traceback

                            traceback.print_exc()
                            logger.warning(f"Failed to install IPAdapterModule: {e}. Continuing without IP-Adapter.")
                            try:
                                stream.unet.set_attn_processor(_saved_unet_processors)
                                logger.info("Restored original UNet attention processors after IP-Adapter failure.")
                            except Exception as restore_err:
                                logger.warning(f"Could not restore UNet processors: {restore_err}")
                            use_ipadapter_trt = False

                # NOTE: When IPAdapter is enabled, we must pass num_ip_layers. We cannot know it until after
                # installing processors in the export wrapper. We construct the wrapper first to discover it,
                # then construct UNet model with that value.

                # Build a temporary unified wrapper to install processors and discover num_ip_layers
                from streamdiffusion.acceleration.tensorrt.export_wrappers.unet_unified_export import (
                    UnifiedExportWrapper,
                )

                temp_wrapped_unet = UnifiedExportWrapper(
                    stream.unet,
                    use_controlnet=use_controlnet_trt,
                    use_ipadapter=use_ipadapter_trt,
                    control_input_names=None,
                    num_tokens=num_tokens,
                )

                num_ip_layers = None
                if use_ipadapter_trt:
                    # Access underlying IPAdapter wrapper
                    if hasattr(temp_wrapped_unet, "ipadapter_wrapper") and temp_wrapped_unet.ipadapter_wrapper:
                        num_ip_layers = getattr(temp_wrapped_unet.ipadapter_wrapper, "num_ip_layers", None)
                        if not isinstance(num_ip_layers, int) or num_ip_layers <= 0:
                            raise RuntimeError("Failed to determine num_ip_layers for IP-Adapter")
                        try:
                            logger.info(f"compile_and_load_engine: discovered num_ip_layers={num_ip_layers}")
                        except Exception:
                            pass

                unet_model = UNet(
                    stream.unet,
                    fp16=True,
                    device=self.device,
                    max_batch_size=self.max_batch_size,
                    min_batch_size=self.min_batch_size,
                    embedding_dim=embedding_dim,
                    unet_dim=stream.unet.config.in_channels,
                    use_control=use_controlnet_trt,
                    unet_arch=unet_arch if use_controlnet_trt else None,
                    use_ipadapter=use_ipadapter_trt,
                    num_image_tokens=num_tokens,
                    num_ip_layers=num_ip_layers if use_ipadapter_trt else None,
                    image_height=self.height,
                    image_width=self.width,
                    use_cached_attn=use_cached_attn,
                    cache_maxframes=cache_maxframes,
                    min_cache_maxframes=min_cache_maxframes,
                    max_cache_maxframes=max_cache_maxframes,
                    use_feature_injection=use_feature_injection,
                )

                # Use ControlNet wrapper if ControlNet support is enabled
                if use_controlnet_trt:
                    # Build control_input_names excluding ipadapter_scale so indices align to 3-base offset
                    all_input_names = unet_model.get_input_names()
                    control_input_names = [name for name in all_input_names if name != "ipadapter_scale"]

                # Unified compilation path
                # Recreate wrapped_unet with control input names if needed (after unet_model is ready)
                wrapped_unet = UnifiedExportWrapper(
                    stream.unet,
                    use_controlnet=use_controlnet_trt,
                    use_ipadapter=use_ipadapter_trt,
                    control_input_names=control_input_names,
                    num_tokens=num_tokens,
                    kvo_cache_structure=kvo_cache_structure,
                    fi_layer_count=getattr(unet_model, "fi_cache_count", 0),
                )

                if use_cached_attn:
                    from .acceleration.tensorrt.models.attention_processors import CachedSTAttnProcessor2_0

                    # Walk the UNet in kvo-cache order (down→mid→up) and install
                    # CachedSTAttnProcessor2_0 on each self-attn (attn1) layer.
                    # fi_eligible_mask is in the same walk order so index alignment is exact.
                    fi_mask = getattr(unet_model, "fi_eligible_mask", None) if use_feature_injection else None
                    _global_idx = 0

                    def _install_cached_proc(attn_module):
                        nonlocal _global_idx
                        fi_eligible = bool(fi_mask[_global_idx]) if fi_mask is not None else False
                        if not isinstance(attn_module.processor, CachedSTAttnProcessor2_0):
                            attn_module.set_processor(CachedSTAttnProcessor2_0(fi_eligible=fi_eligible))
                        else:
                            attn_module.processor.fi_eligible = fi_eligible
                        _global_idx += 1

                    _unet = stream.unet
                    for _block in _unet.down_blocks:
                        if hasattr(_block, "attentions") and _block.attentions is not None:
                            for _attn in _block.attentions:
                                for _tf in _attn.transformer_blocks:
                                    _install_cached_proc(_tf.attn1)
                    if hasattr(_unet.mid_block, "attentions") and _unet.mid_block.attentions is not None:
                        for _attn in _unet.mid_block.attentions:
                            for _tf in _attn.transformer_blocks:
                                _install_cached_proc(_tf.attn1)
                    for _block in _unet.up_blocks:
                        if hasattr(_block, "attentions") and _block.attentions is not None:
                            for _attn in _block.attentions:
                                for _tf in _attn.transformer_blocks:
                                    _install_cached_proc(_tf.attn1)

                    # Re-collect FI processors now that CachedSTAttnProcessor2_0 is
                    # installed.  UnifiedExportWrapper.__init__ ran _collect_fi_processors
                    # before this block — at that point processors were still stock diffusers
                    # so _fi_procs was [].  refresh_fi_procs() is the authoritative call and
                    # fail-fasts if the count doesn't match fi_layer_count.
                    if use_feature_injection:
                        wrapped_unet.refresh_fi_procs()

                # Effective VAE optlvl for both decoder and encoder compile calls.
                # Mirrors the _vae_optlvl computed for get_engine_path above.
                _vae_build_optlvl = (
                    self.vae_builder_optimization_level
                    if self.vae_builder_optimization_level is not None
                    else self.builder_optimization_level
                )

                # Compile VAE decoder engine using EngineManager
                vae_decoder_model = VAE(
                    device=self.device,
                    max_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                )

                engine_manager.compile_and_load_engine(
                    EngineType.VAE_DECODER,
                    vae_decoder_path,
                    load_engine=False,
                    model=stream.vae,
                    model_config=vae_decoder_model,
                    batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    cuda_stream=None,
                    stream_vae=stream.vae,
                    engine_build_options={
                        "opt_image_height": self.height,
                        "opt_image_width": self.width,
                        "build_dynamic_shape": not self.static_shapes,
                        "build_static_batch": self.static_shapes,
                        **(
                            {"min_image_resolution": 384, "max_image_resolution": 1024, "build_all_tactics": True}
                            if not self.static_shapes
                            else {}
                        ),
                        **({"builder_optimization_level": _vae_build_optlvl} if _vae_build_optlvl is not None else {}),
                    },
                )

                # Compile VAE encoder engine using EngineManager
                vae_encoder = TorchVAEEncoder(stream.vae)
                vae_encoder_model = VAEEncoder(
                    device=self.device,
                    max_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                )

                engine_manager.compile_and_load_engine(
                    EngineType.VAE_ENCODER,
                    vae_encoder_path,
                    load_engine=False,
                    model=vae_encoder,
                    model_config=vae_encoder_model,
                    batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    cuda_stream=None,
                    engine_build_options={
                        "opt_image_height": self.height,
                        "opt_image_width": self.width,
                        "build_dynamic_shape": not self.static_shapes,
                        "build_static_batch": self.static_shapes,
                        **(
                            {"min_image_resolution": 384, "max_image_resolution": 1024, "build_all_tactics": True}
                            if not self.static_shapes
                            else {}
                        ),
                        **({"builder_optimization_level": _vae_build_optlvl} if _vae_build_optlvl is not None else {}),
                    },
                )

                # Use polygraphy's default Blocking stream. A NonBlocking engine stream
                # would skip the legacy/per-thread NULL-stream auto-sync that the rest of
                # the pipeline relies on (PyTorch ops run on stream 0x0), creating a data
                # race where the engine reads stale inputs and writes outputs that
                # downstream PyTorch never observes — symptom is black/zero output frames.
                cuda_stream = cuda.Stream()

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype

                try:
                    # Note: the UNet always builds with build_static_batch=True /
                    # build_dynamic_shape=False regardless of self.static_shapes.
                    # static_shapes only controls the VAE enc/dec build flags.
                    logger.warning(
                        f"[TRT] UNet engine: fp8={fp8}, "
                        f"build_static_batch=True, build_dynamic_shape=False, "
                        f"engine_path={unet_path}"
                    )
                    _unet_build_opts = {
                        "opt_image_height": self.height,
                        "opt_image_width": self.width,
                        "build_dynamic_shape": False,
                        "build_static_batch": True,
                    }
                    if self.builder_optimization_level is not None:
                        _unet_build_opts["builder_optimization_level"] = self.builder_optimization_level
                    if fp8:
                        _is_turbo = getattr(self, "_is_turbo", False)
                        _unet_build_opts["fp8"] = True
                        _unet_build_opts["onnx_opset"] = 19  # FP8 Q/DQ scales require opset ≥19
                        _unet_build_opts["pipe_ref"] = stream.pipe
                        # SDXL-Turbo: 4 steps, guidance_scale=0.0 (matches inference);
                        # SDXL base: 20 steps, guidance_scale=7.5.
                        # Calibration activations must match inference-time ranges.
                        _unet_build_opts["calibration_steps"] = 4 if _is_turbo else 20
                        _unet_build_opts["fp8_guidance_scale"] = 0.0 if _is_turbo else 7.5
                        _unet_build_opts["fp8_allow_fp16_fallback"] = self.fp8_allow_fp16_fallback
                        _unet_build_opts["fp8_use_cached_attn"] = use_cached_attn
                        _unet_build_opts["fp8_use_feature_injection"] = use_feature_injection
                        _unet_build_opts["fp8_use_controlnet"] = use_controlnet_trt
                        _unet_build_opts["fp8_num_ip_layers"] = num_ip_layers if use_ipadapter_trt else 0
                        logger.warning(
                            f"[TRT] FP8 build opts: turbo={_is_turbo}, "
                            f"steps={_unet_build_opts['calibration_steps']}, "
                            f"guidance={_unet_build_opts['fp8_guidance_scale']}"
                        )

                    # Compile and load UNet engine using EngineManager
                    stream.unet = engine_manager.compile_and_load_engine(
                        EngineType.UNET,
                        unet_path,
                        load_engine=load_engine,
                        model=wrapped_unet,
                        model_config=unet_model,
                        batch_size=stream.trt_unet_batch_size,
                        cuda_stream=cuda_stream,
                        use_controlnet_trt=use_controlnet_trt,
                        use_ipadapter_trt=use_ipadapter_trt,
                        unet_arch=unet_arch,
                        num_ip_layers=num_ip_layers if use_ipadapter_trt else None,
                        engine_build_options=_unet_build_opts,
                    )
                    if load_engine:
                        logger.info("TensorRT UNet engine loaded successfully")

                except Exception as e:
                    error_msg = str(e).lower()
                    is_oom_error = (
                        "out of memory" in error_msg
                        or "outofmemory" in error_msg
                        or "oom" in error_msg
                        or "cuda error" in error_msg
                    )

                    if is_oom_error:
                        logger.error(f"TensorRT UNet engine OOM: {e}")
                        logger.info("Falling back to PyTorch UNet (no TensorRT acceleration)")
                        logger.info("This will be slower but should work with less memory")

                        # Clean up any partial TensorRT state
                        if hasattr(stream, "unet"):
                            try:
                                del stream.unet
                            except Exception:
                                pass

                        self.cleanup_gpu_memory()

                        # Fall back to original PyTorch UNet
                        try:
                            logger.info("Loading PyTorch UNet as fallback...")
                            # Keep the original UNet from the pipe
                            if hasattr(stream, "pipe") and hasattr(stream.pipe, "unet"):
                                stream.unet = stream.pipe.unet
                                logger.info("PyTorch UNet fallback successful")
                            else:
                                raise RuntimeError("No PyTorch UNet available for fallback")
                        except Exception as fallback_error:
                            logger.error(f"PyTorch UNet fallback also failed: {fallback_error}")
                            raise RuntimeError(
                                f"Both TensorRT and PyTorch UNet loading failed. TensorRT error: {e}, Fallback error: {fallback_error}"
                            )
                    else:
                        # Non-OOM error, re-raise
                        logger.error(f"TensorRT UNet engine loading failed (non-OOM): {e}")
                        raise e

                if load_engine:
                    try:
                        logger.info(
                            f"Loading TensorRT VAE engines vae_encoder_path: {vae_encoder_path}, vae_decoder_path: {vae_decoder_path}"
                        )
                        stream.vae = AutoencoderKLEngine(
                            str(vae_encoder_path),
                            str(vae_decoder_path),
                            cuda_stream,
                            stream.pipe.vae_scale_factor,
                            use_cuda_graph=True,
                        )
                        stream.vae.config = vae_config
                        stream.vae.dtype = vae_dtype
                        logger.info("TensorRT VAE engines loaded successfully")

                    except Exception as e:
                        error_msg = str(e).lower()
                        is_oom_error = (
                            "out of memory" in error_msg
                            or "outofmemory" in error_msg
                            or "oom" in error_msg
                            or "cuda error" in error_msg
                        )

                        if is_oom_error:
                            logger.error(f"TensorRT VAE engine OOM: {e}")
                            logger.info("Falling back to PyTorch VAE (no TensorRT acceleration)")
                            logger.info("This will be slower but should work with less memory")

                            # Clean up any partial TensorRT state
                            if hasattr(stream, "vae"):
                                try:
                                    del stream.vae
                                except Exception:
                                    pass

                            self.cleanup_gpu_memory()

                            # Fall back to original PyTorch VAE
                            try:
                                logger.info("Loading PyTorch VAE as fallback...")
                                # Keep the original VAE from the pipe
                                if hasattr(stream, "pipe") and hasattr(stream.pipe, "vae"):
                                    stream.vae = stream.pipe.vae
                                    logger.info("PyTorch VAE fallback successful")
                                else:
                                    raise RuntimeError("No PyTorch VAE available for fallback")
                            except Exception as fallback_error:
                                logger.error(f"PyTorch VAE fallback also failed: {fallback_error}")
                                raise RuntimeError(
                                    f"Both TensorRT and PyTorch VAE loading failed. TensorRT error: {e}, Fallback error: {fallback_error}"
                                )
                        else:
                            # Non-OOM error, re-raise
                            logger.error(f"TensorRT VAE engine loading failed (non-OOM): {e}")
                            raise e

                # Safety checker engine (TensorRT-specific)
                safety_checker_path = engine_manager.get_engine_path(
                    EngineType.SAFETY_CHECKER,
                    model_id_or_path=safety_checker_model_id,
                    max_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                    mode=self.mode,
                    use_tiny_vae=use_tiny_vae,
                )
                safety_checker_engine_exists = os.path.exists(safety_checker_path)

                # Always load the safety checker if the engine exists. The model is really small and may be toggled later.
                if self.use_safety_checker or safety_checker_engine_exists:
                    if not safety_checker_engine_exists:
                        from transformers import AutoModelForImageClassification

                        self.safety_checker = AutoModelForImageClassification.from_pretrained(safety_checker_model_id)

                        safety_checker_model = NSFWDetector(
                            device=self.device,
                            max_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                            min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                        )

                        engine_manager.compile_and_load_engine(
                            EngineType.SAFETY_CHECKER,
                            safety_checker_path,
                            model=self.safety_checker,
                            model_config=safety_checker_model,
                            batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                            cuda_stream=None,
                            load_engine=False,
                        )

                    if load_engine:
                        self.safety_checker = NSFWDetectorEngine(
                            safety_checker_path,
                            cuda_stream,
                            use_cuda_graph=True,
                        )
                        logger.info("Safety Checker engine loaded successfully")

            if acceleration == "sfast":
                from streamdiffusion.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

                stream = accelerate_with_stable_fast(stream)
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise Exception(f"Acceleration has failed: {e}") from e

        # Install modules via hooks instead of patching (wrapper keeps forwarding updates only)
        if use_controlnet:
            try:
                from streamdiffusion.modules.controlnet_module import ControlNetConfig, ControlNetModule

                cn_module = ControlNetModule(device=self.device, dtype=self.dtype)
                cn_module.install(stream)
                # Normalize to list of configs
                configs = (
                    controlnet_config
                    if isinstance(controlnet_config, list)
                    else [controlnet_config]
                    if isinstance(controlnet_config, dict)
                    else []
                )
                for cfg in configs:
                    if not cfg.get("model_id"):
                        continue
                    cn_cfg = ControlNetConfig(
                        model_id=cfg["model_id"],
                        preprocessor=cfg.get("preprocessor"),
                        conditioning_scale=cfg.get("conditioning_scale", 1.0),
                        enabled=cfg.get("enabled", True),
                        conditioning_channels=cfg.get("conditioning_channels"),
                        preprocessor_params=cfg.get("preprocessor_params"),
                    )
                    cn_module.add_controlnet(cn_cfg, control_image=cfg.get("control_image"))
                # Expose for later updates if needed by caller code
                stream._controlnet_module = cn_module
                # Enable always_preprocess so controlnet_images is populated even at scale==0,
                # which is required for the IPC/CPU preview path to have something to export.
                if self._controlnet_preview_passthrough:
                    cn_module.always_preprocess = True
                # Apply startup cache interval from config (1 = disabled, no-op).
                if cn_cache_interval > 1:
                    cn_module.set_cn_cache_interval(cn_cache_interval)

                if acceleration == "tensorrt":
                    try:
                        compiled_cn_engines = []
                        for cfg, cn_model in zip(configs, cn_module.controlnets):
                            if not cfg or not cfg.get("model_id") or cn_model is None:
                                continue
                            try:
                                engine = engine_manager.get_or_load_controlnet_engine(
                                    model_id=cfg["model_id"],
                                    pytorch_model=cn_model,
                                    model_type=model_type,
                                    batch_size=stream.trt_unet_batch_size,
                                    max_batch_size=self.max_batch_size,
                                    min_batch_size=self.min_batch_size,
                                    cuda_stream=cuda_stream,
                                    use_cuda_graph=False,  # TRT's genericReformat uses legacy stream during execute_async_v3 — incompatible with graph capture (901)
                                    unet=None,
                                    model_path=cfg["model_id"],
                                    opt_image_height=self.height,
                                    opt_image_width=self.width,
                                    load_engine=load_engine,
                                    conditioning_channels=cfg.get("conditioning_channels", 3),
                                    builder_optimization_level=self.builder_optimization_level,
                                    fp8=fp8 or bool(cfg.get("fp8", False)),
                                )
                                try:
                                    setattr(engine, "model_id", cfg["model_id"])
                                except Exception:
                                    pass
                                compiled_cn_engines.append(engine)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to compile/load ControlNet engine for {cfg.get('model_id')}: {e}"
                                )
                        if compiled_cn_engines:
                            setattr(stream, "controlnet_engines", compiled_cn_engines)
                            try:
                                logger.info(
                                    f"Compiled/loaded {len(compiled_cn_engines)} ControlNet TensorRT engine(s)"
                                )
                            except Exception:
                                pass
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        logger.warning(
                            "ControlNet TensorRT engine build step encountered an issue; continuing with PyTorch ControlNet"
                        )
            except Exception:
                import traceback

                traceback.print_exc()
                logger.error("Failed to install ControlNetModule")
                raise

        # IPAdapter module installation has been moved to before TensorRT compilation (see lines 1307-1345)
        # This ensures processors are properly baked into the TensorRT engines
        # After TRT compilation, stream.unet is a UNet2DConditionModelEngine with no attn_processors —
        # skip IP-Adapter install entirely in that case.
        if (
            use_ipadapter
            and ipadapter_config
            and not hasattr(stream, "_ipadapter_module")
            and hasattr(stream.unet, "attn_processors")
        ):
            try:
                from streamdiffusion.modules.ipadapter_module import IPAdapterConfig, IPAdapterModule, IPAdapterType

                # Use first config if list provided
                cfg = ipadapter_config[0] if isinstance(ipadapter_config, list) else ipadapter_config

                # Get adapter type from config
                ipadapter_type = IPAdapterType(cfg["type"])

                ip_cfg = IPAdapterConfig(
                    style_image_key=cfg.get("style_image_key") or "ipadapter_main",
                    num_image_tokens=cfg.get("num_image_tokens", 4),
                    ipadapter_model_path=cfg["ipadapter_model_path"],
                    image_encoder_path=cfg["image_encoder_path"],
                    style_image=cfg.get("style_image"),
                    scale=cfg.get("scale", 1.0),
                    type=ipadapter_type,
                    insightface_model_name=cfg.get("insightface_model_name"),
                )
                ip_module = IPAdapterModule(ip_cfg)
                _saved_unet_processors_post = dict(stream.unet.attn_processors)
                ip_module.install(stream)
                # Expose for later updates
                stream._ipadapter_module = ip_module

            except RuntimeError as rt_error:
                if "size mismatch" in str(rt_error):
                    unet_dim = getattr(getattr(stream, "unet", None), "config", None)
                    unet_cross_attn = getattr(unet_dim, "cross_attention_dim", "unknown") if unet_dim else "unknown"
                    logger.warning(
                        f"IP-Adapter weights are incompatible with this model "
                        f"(UNet cross_attention_dim={unet_cross_attn}). "
                        f"Skipping post-TRT IP-Adapter installation and continuing without it."
                    )
                    try:
                        stream.unet.set_attn_processor(_saved_unet_processors_post)
                    except Exception as restore_err:
                        logger.warning(f"Could not restore UNet processors: {restore_err}")
                else:
                    import traceback

                    traceback.print_exc()
                    logger.error("Failed to install IPAdapterModule")
                    raise

            except Exception:
                import traceback

                traceback.print_exc()
                logger.error("Failed to install IPAdapterModule")
                raise

        # Note: LoRA weights have already been merged permanently during model loading

        # Install pipeline hook modules (Phase 4: Configuration Integration)
        if image_preprocessing_config and image_preprocessing_config.get("enabled", True):
            try:
                from streamdiffusion.modules.image_processing_module import ImagePreprocessingModule

                img_pre_module = ImagePreprocessingModule()
                img_pre_module.install(stream)
                for proc_config in image_preprocessing_config.get("processors", []):
                    img_pre_module.add_processor(proc_config)
                stream._image_preprocessing_module = img_pre_module
            except Exception as e:
                logger.error(f"Failed to install ImagePreprocessingModule: {e}")

        if image_postprocessing_config and image_postprocessing_config.get("enabled", True):
            try:
                from streamdiffusion.modules.image_processing_module import ImagePostprocessingModule

                img_post_module = ImagePostprocessingModule()
                img_post_module.install(stream)
                for proc_config in image_postprocessing_config.get("processors", []):
                    img_post_module.add_processor(proc_config)
                stream._image_postprocessing_module = img_post_module
            except Exception as e:
                logger.error(f"Failed to install ImagePostprocessingModule: {e}")

        if latent_preprocessing_config and latent_preprocessing_config.get("enabled", True):
            try:
                from streamdiffusion.modules.latent_processing_module import LatentPreprocessingModule

                latent_pre_module = LatentPreprocessingModule()
                latent_pre_module.install(stream)
                for proc_config in latent_preprocessing_config.get("processors", []):
                    latent_pre_module.add_processor(proc_config)
                stream._latent_preprocessing_module = latent_pre_module
            except Exception as e:
                logger.error(f"Failed to install LatentPreprocessingModule: {e}")

        if latent_postprocessing_config and latent_postprocessing_config.get("enabled", True):
            try:
                from streamdiffusion.modules.latent_processing_module import LatentPostprocessingModule

                latent_post_module = LatentPostprocessingModule()
                latent_post_module.install(stream)
                for proc_config in latent_postprocessing_config.get("processors", []):
                    latent_post_module.add_processor(proc_config)
                stream._latent_postprocessing_module = latent_post_module
            except Exception as e:
                logger.error(f"Failed to install LatentPostprocessingModule: {e}")

        # L2 cache persistence: pin hot UNet attention weights in GPU L2 cache.
        # Gated by SDTD_L2_PERSIST=1 (default on). Silent fallback on unsupported GPUs.
        # Requires Ampere+ (compute 8.0+). Expected gain: 2-5% end-to-end on SD1.5/SD-Turbo.
        try:
            from streamdiffusion.tools.cuda_l2_cache import setup_l2_persistence

            setup_l2_persistence(stream.unet)
        except Exception as e:
            logger.debug(f"L2 cache persistence setup skipped: {e}")

        return stream

    def get_last_processed_image(self, index: int) -> Optional[Image.Image]:
        """Forward get_last_processed_image call to the underlying ControlNet pipeline"""
        if not self.use_controlnet:
            raise RuntimeError(
                "get_last_processed_image: ControlNet support not enabled. Set use_controlnet=True in constructor."
            )

        return self.stream.get_last_processed_image(index)

    def cleanup_controlnets(self) -> None:
        """Cleanup ControlNet resources including background threads and VRAM"""
        if not self.use_controlnet:
            return

        if hasattr(self, "stream") and self.stream and hasattr(self.stream, "cleanup"):
            self.stream.cleanup_controlnets()

    def update_control_image(self, index: int, image: Union[str, Image.Image, torch.Tensor]) -> None:
        """Update control image for specific ControlNet index"""
        if not self.use_controlnet:
            raise RuntimeError(
                "update_control_image: ControlNet support not enabled. Set use_controlnet=True in constructor."
            )
        if not self.skip_diffusion:
            self.stream._controlnet_module.update_control_image_efficient(image, index=index)
        else:
            logger.debug("update_control_image: Skipping ControlNet update in skip diffusion mode")

    def update_style_image(
        self, image: Union[str, Image.Image, torch.Tensor], is_stream: bool = False, style_key="ipadapter_main"
    ) -> None:
        """Update IPAdapter style image"""
        if not self.use_ipadapter:
            raise RuntimeError(
                "update_style_image: IPAdapter support not enabled. Set use_ipadapter=True in constructor."
            )

        if not self.skip_diffusion:
            self.stream._param_updater.update_style_image(style_key, image, is_stream=is_stream)
        else:
            logger.debug("update_style_image: Skipping IPAdapter update in skip diffusion mode")

    def clear_caches(self) -> None:
        """Clear all cached prompt embeddings and seed noise tensors."""
        self.stream._param_updater.clear_caches()

    def get_stream_state(self, include_caches: bool = False) -> Dict[str, Any]:
        """Get a unified snapshot of the current stream state.

        Args:
            include_caches: When True, include cache statistics in the response

        Returns:
            Dict[str, Any]: Consolidated state including prompts/seeds, runtime settings,
                            module configs, and basic pipeline info.
        """
        stream = self.stream
        updater = stream._param_updater

        # Prompts / Seeds
        prompts = updater.get_current_prompts()
        seeds = updater.get_current_seeds()

        # Normalization flags
        normalize_prompt_weights = updater.get_normalize_prompt_weights()
        normalize_seed_weights = updater.get_normalize_seed_weights()

        # Core runtime params
        guidance_scale = getattr(stream, "guidance_scale", None)
        delta = getattr(stream, "delta", None)
        t_index_list = list(getattr(stream, "t_list", []))
        current_seed = getattr(stream, "current_seed", None)
        num_inference_steps = None
        try:
            if hasattr(stream, "timesteps") and stream.timesteps is not None:
                num_inference_steps = int(len(stream.timesteps))
        except Exception:
            pass

        # Resolution and model/pipeline info
        state: Dict[str, Any] = {
            "width": getattr(stream, "width", None),
            "height": getattr(stream, "height", None),
            "latent_width": getattr(stream, "latent_width", None),
            "latent_height": getattr(stream, "latent_height", None),
            "device": getattr(stream, "device", None).type
            if hasattr(getattr(stream, "device", None), "type")
            else getattr(stream, "device", None),
            "dtype": str(getattr(stream, "dtype", None)),
            "model_type": getattr(stream, "model_type", None),
            "is_sdxl": getattr(stream, "is_sdxl", None),
            "is_turbo": getattr(stream, "is_turbo", None),
            "cfg_type": getattr(stream, "cfg_type", None),
            "use_denoising_batch": getattr(stream, "use_denoising_batch", None),
            "batch_size": getattr(stream, "batch_size", None),
            "min_batch_size": getattr(stream, "min_batch_size", None),
            "max_batch_size": getattr(stream, "max_batch_size", None),
        }

        # Blending state
        state.update(
            {
                "prompt_list": prompts,
                "seed_list": seeds,
                "normalize_prompt_weights": normalize_prompt_weights,
                "normalize_seed_weights": normalize_seed_weights,
                "negative_prompt": getattr(updater, "_current_negative_prompt", ""),
            }
        )

        # Core runtime knobs
        state.update(
            {
                "guidance_scale": guidance_scale,
                "delta": delta,
                "t_index_list": t_index_list,
                "current_seed": current_seed,
                "num_inference_steps": num_inference_steps,
            }
        )

        # Module configs (ControlNet, IP-Adapter)
        try:
            controlnet_config = updater._get_current_controlnet_config()
        except Exception:
            controlnet_config = []
        try:
            ipadapter_config = updater._get_current_ipadapter_config()
        except Exception:
            ipadapter_config = None
        # Hook configs
        try:
            image_preprocessing_config = updater._get_current_hook_config("image_preprocessing")
        except Exception:
            image_preprocessing_config = []
        try:
            image_postprocessing_config = updater._get_current_hook_config("image_postprocessing")
        except Exception:
            image_postprocessing_config = []
        try:
            latent_preprocessing_config = updater._get_current_hook_config("latent_preprocessing")
        except Exception:
            latent_preprocessing_config = []
        try:
            latent_postprocessing_config = updater._get_current_hook_config("latent_postprocessing")
        except Exception:
            latent_postprocessing_config = []

        state.update(
            {
                "controlnet_config": controlnet_config,
                "ipadapter_config": ipadapter_config,
                "image_preprocessing_config": image_preprocessing_config,
                "image_postprocessing_config": image_postprocessing_config,
                "latent_preprocessing_config": latent_preprocessing_config,
                "latent_postprocessing_config": latent_postprocessing_config,
            }
        )

        # Optional caches
        if include_caches:
            try:
                state["caches"] = updater.get_cache_info()
            except Exception:
                state["caches"] = None

        return state

    def cleanup_gpu_memory(self) -> None:
        """Comprehensive GPU memory cleanup for model switching."""
        import gc

        import torch

        logger.info("Cleaning up GPU memory...")

        # Clear prompt caches
        if hasattr(self, "stream") and self.stream:
            try:
                self.stream._param_updater.clear_caches()
                logger.info("   Cleared prompt caches")
            except Exception:
                logger.debug("cleanup_gpu_memory: clear_caches() failed", exc_info=True)

        # Enhanced TensorRT engine cleanup
        if hasattr(self, "stream") and self.stream:
            try:
                # Cleanup UNet TensorRT engine
                if hasattr(self.stream, "unet"):
                    unet_engine = self.stream.unet
                    logger.info("   Cleaning up TensorRT UNet engine...")

                    # Check if it's a TensorRT engine and cleanup properly
                    if hasattr(unet_engine, "engine") and hasattr(unet_engine.engine, "__del__"):
                        try:
                            # Call the engine's destructor explicitly
                            unet_engine.engine.__del__()
                        except Exception:
                            pass

                    # Clear all engine-related attributes
                    if hasattr(unet_engine, "context"):
                        try:
                            del unet_engine.context
                        except Exception:
                            pass
                    if hasattr(unet_engine, "engine"):
                        try:
                            del unet_engine.engine.engine  # TensorRT runtime engine
                            del unet_engine.engine
                        except Exception:
                            pass

                    del self.stream.unet
                    logger.info("   UNet engine cleanup completed")

                # Cleanup VAE TensorRT engines
                if hasattr(self.stream, "vae"):
                    vae_engine = self.stream.vae
                    logger.info("   Cleaning up TensorRT VAE engines...")

                    # VAE has encoder and decoder engines
                    for engine_name in ["vae_encoder", "vae_decoder"]:
                        if hasattr(vae_engine, engine_name):
                            engine = getattr(vae_engine, engine_name)
                            if hasattr(engine, "engine") and hasattr(engine.engine, "__del__"):
                                try:
                                    engine.engine.__del__()
                                except Exception:
                                    pass
                            try:
                                delattr(vae_engine, engine_name)
                            except Exception:
                                pass

                    del self.stream.vae
                    logger.info("   VAE engines cleanup completed")

                # Cleanup ControlNet engine pool if it exists
                if hasattr(self.stream, "controlnet_engine_pool"):
                    logger.info("   Cleaning up ControlNet engine pool...")
                    try:
                        self.stream.controlnet_engine_pool.cleanup()
                        del self.stream.controlnet_engine_pool
                        logger.info("   ControlNet engine pool cleanup completed")
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"   TensorRT cleanup warning: {e}")

        # Clear the entire stream object to free all models
        if hasattr(self, "stream"):
            try:
                del self.stream
                logger.info("   Cleared stream object")
            except Exception:
                logger.debug("cleanup_gpu_memory: del self.stream failed", exc_info=True)
            self.stream = None

        # Release wrapper-level frame buffers so the next model swap allocates fresh
        # for the new output shape and pinned host memory is returned to the OS.
        self._output_pin_buf = None
        self._output_gpu_buf = None
        self._d2h_event = None

        # Force multiple garbage collection cycles for thorough cleanup
        for i in range(3):
            gc.collect()

        # Clear CUDA cache and cleanup IPC handles
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Force additional memory cleanup
        torch.cuda.ipc_collect()

        # Get memory info
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        cached = torch.cuda.memory_reserved() / (1024**3)  # GB
        logger.info(f"   GPU Memory after cleanup: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

        logger.info("   Enhanced GPU memory cleanup complete")

    def check_gpu_memory_for_engine(self, engine_size_gb: float) -> bool:
        """
        Check if there's enough GPU memory to load a TensorRT engine.

        Args:
            engine_size_gb: Expected engine size in GB

        Returns:
            True if enough memory is available, False otherwise
        """
        if not torch.cuda.is_available():
            return True  # Assume OK if CUDA not available

        try:
            # Get current memory status
            allocated = torch.cuda.memory_allocated() / (1024**3)
            cached = torch.cuda.memory_reserved() / (1024**3)

            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free_memory = total_memory - allocated

            # Add 20% overhead for safety
            required_memory = engine_size_gb * 1.2

            logger.info("GPU Memory Check:")
            logger.info(f"   Total: {total_memory:.2f}GB")
            logger.info(f"   Allocated: {allocated:.2f}GB")
            logger.info(f"   Cached: {cached:.2f}GB")
            logger.info(f"   Free: {free_memory:.2f}GB")
            logger.info(f"   Required: {required_memory:.2f}GB (engine: {engine_size_gb:.2f}GB + 20% overhead)")

            if free_memory >= required_memory:
                logger.info("   Sufficient memory available")
                return True
            else:
                logger.error(
                    f"   Insufficient memory! Need {required_memory:.2f}GB but only {free_memory:.2f}GB available"
                )
                return False

        except Exception as e:
            logger.error(f"   Memory check failed: {e}")
            return True  # Assume OK if check fails

    def cleanup_engines_and_rebuild(self, reduce_batch_size: bool = True, reduce_resolution: bool = False) -> None:
        """
        Clean up TensorRT engines and rebuild with smaller settings to fix OOM issues.

        Parameters:
        -----------
        reduce_batch_size : bool
            If True, reduce batch size to 1
        reduce_resolution : bool
            If True, reduce resolution by half
        """
        import os
        import shutil

        logger.info("Cleaning up engines and rebuilding with smaller settings...")

        # Clean up GPU memory first
        self.cleanup_gpu_memory()

        # Remove engines directory
        engines_dir = "engines"
        if os.path.exists(engines_dir):
            try:
                shutil.rmtree(engines_dir)
                logger.info(f"   Removed engines directory: {engines_dir}")
            except Exception as e:
                logger.error(f"   Failed to remove engines: {e}")

        # Reduce settings
        if reduce_batch_size:
            if hasattr(self, "batch_size") and self.batch_size > 1:
                old_batch = self.batch_size
                self.batch_size = 1
                logger.info(f"   Reduced batch size: {old_batch} -> {self.batch_size}")

            # Also reduce frame buffer size if needed
            if hasattr(self, "frame_buffer_size") and self.frame_buffer_size > 1:
                old_buffer = self.frame_buffer_size
                self.frame_buffer_size = 1
                logger.info(f"   Reduced frame buffer size: {old_buffer} -> {self.frame_buffer_size}")

        if reduce_resolution:
            if hasattr(self, "width") and hasattr(self, "height"):
                old_width, old_height = self.width, self.height
                self.width = max(512, self.width // 2)
                self.height = max(512, self.height // 2)
                # Round to multiples of 64 for compatibility
                self.width = (self.width // 64) * 64
                self.height = (self.height // 64) * 64
                logger.info(f"   Reduced resolution: {old_width}x{old_height} -> {self.width}x{self.height}")

        logger.info("   Next model load will rebuild engines with these smaller settings")
