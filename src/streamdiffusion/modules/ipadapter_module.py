from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch

from streamdiffusion.hooks import EmbeddingHook, EmbedsCtx, StepCtx, UnetHook, UnetKwargsDelta
from streamdiffusion.preprocessing.orchestrator_user import OrchestratorUser
from streamdiffusion.utils.reporting import report_error

logger = logging.getLogger(__name__)


class IPAdapterType(Enum):
    REGULAR = "regular"
    PLUS = "plus"
    FACEID = "faceid"


@dataclass
class IPAdapterConfig:
    """Minimal config for constructing an IP-Adapter module instance.

    This module focuses only on embedding composition (step 2 of migration).
    Runtime installation and wrapper wiring will come in later steps.
    """

    style_image_key: Optional[str] = None
    num_image_tokens: int = 4  # e.g., 4 for standard, 16 for plus
    ipadapter_model_path: Optional[str] = None
    image_encoder_path: Optional[str] = None
    style_image: Optional[Any] = None
    scale: float = 1.0
    weight_type: Optional[str] = None  # Weight type for per-layer scaling
    enabled: bool = True  # Runtime enable/disable state

    type: IPAdapterType = IPAdapterType.REGULAR
    insightface_model_name: Optional[str] = None


# ---------------------------------------------------------------------------
# IP-Adapter model path mapping by base model architecture and adapter type
# ---------------------------------------------------------------------------
# None means the variant is unavailable for that architecture — callers fall
# back to REGULAR automatically.
IPADAPTER_MODEL_MAP: Dict[tuple, Optional[Dict[str, str]]] = {
    ("SD1.5", IPAdapterType.REGULAR): {
        "model_path": "h94/IP-Adapter/models/ip-adapter_sd15.bin",
        "image_encoder_path": "h94/IP-Adapter/models/image_encoder",
    },
    ("SD1.5", IPAdapterType.PLUS): {
        "model_path": "h94/IP-Adapter/models/ip-adapter-plus_sd15.safetensors",
        "image_encoder_path": "h94/IP-Adapter/models/image_encoder",
    },
    ("SD1.5", IPAdapterType.FACEID): {
        "model_path": "h94/IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin",
        "image_encoder_path": "h94/IP-Adapter/models/image_encoder",
    },
    ("SD2.1", IPAdapterType.REGULAR): None,  # not available from h94 (ip-adapter_sd21.bin was never released)
    ("SD2.1", IPAdapterType.PLUS): None,  # not available from h94
    ("SD2.1", IPAdapterType.FACEID): None,  # not available from h94
    ("SDXL", IPAdapterType.REGULAR): {
        "model_path": "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin",
        "image_encoder_path": "h94/IP-Adapter/sdxl_models/image_encoder",
    },
    ("SDXL", IPAdapterType.PLUS): {
        "model_path": "h94/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
        "image_encoder_path": "h94/IP-Adapter/sdxl_models/image_encoder",
    },
    ("SDXL", IPAdapterType.FACEID): {
        "model_path": "h94/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin",
        "image_encoder_path": "h94/IP-Adapter/sdxl_models/image_encoder",
    },
}

# Set of all known HF model paths — used to distinguish known vs custom paths.
# Custom/local paths are never overridden.
_KNOWN_IPADAPTER_PATHS: frozenset = frozenset(
    entry["model_path"] for entry in IPADAPTER_MODEL_MAP.values() if entry is not None
)

_KNOWN_ENCODER_PATHS: frozenset = frozenset(
    {
        "h94/IP-Adapter/models/image_encoder",
        "h94/IP-Adapter/sdxl_models/image_encoder",
    }
)


def _normalize_model_type(detected_model_type: str, is_sdxl: bool) -> Optional[str]:
    """Map model detection strings to IPADAPTER_MODEL_MAP keys."""
    if is_sdxl:
        return "SDXL"
    return {
        "SD1.5": "SD1.5",
        "SD15": "SD1.5",
        "SD2.1": "SD2.1",
        "SD21": "SD2.1",
        "SDXL": "SDXL",
    }.get(detected_model_type)


def resolve_ipadapter_paths(
    cfg: Dict[str, Any],
    detected_model_type: str,
    is_sdxl: bool,
) -> Dict[str, Any]:
    """Validate and auto-resolve IP-Adapter model/encoder paths for the detected base model.

    Mutates *cfg* in-place and returns it. Custom/local paths are never overridden.

    Args:
        cfg: Single IP-Adapter config dict (keys: ipadapter_model_path, image_encoder_path, type, ...).
        detected_model_type: Value from detect_model() e.g. "SD1.5", "SD2.1", "SDXL".
        is_sdxl: Whether the base model is SDXL-family (takes precedence over detected_model_type).

    Returns:
        The (potentially mutated) cfg dict.
    """
    current_model_path = cfg.get("ipadapter_model_path") or ""
    current_encoder_path = cfg.get("image_encoder_path") or ""

    # Parse adapter type, default to REGULAR
    try:
        adapter_type = IPAdapterType(cfg.get("type", "regular"))
    except ValueError:
        adapter_type = IPAdapterType.REGULAR

    # Normalize to map key; unknown types are left unchanged
    norm_type = _normalize_model_type(detected_model_type, is_sdxl)
    if norm_type is None:
        logger.warning(
            f"IP-Adapter auto-resolution: unknown model type '{detected_model_type}' — "
            f"cannot validate compatibility. Ensure ipadapter_model_path is correct for this model."
        )
        return cfg

    # Custom/local path — respect it, only log info
    if current_model_path and current_model_path not in _KNOWN_IPADAPTER_PATHS:
        logger.info(
            f"IP-Adapter: custom model path '{current_model_path}' — "
            f"skipping auto-resolution (manual compatibility check required for {detected_model_type})."
        )
        return cfg

    # Look up the correct entry for this architecture + type
    target_entry = IPADAPTER_MODEL_MAP.get((norm_type, adapter_type))

    # Variant unavailable for this architecture — fall back to REGULAR with warning
    if target_entry is None:
        logger.warning(
            f"IP-Adapter type '{adapter_type.value}' is not available for {detected_model_type}. "
            f"Falling back to 'regular' adapter type."
        )
        adapter_type = IPAdapterType.REGULAR
        cfg["type"] = adapter_type.value
        target_entry = IPADAPTER_MODEL_MAP.get((norm_type, adapter_type))

    if target_entry is None:
        logger.warning(
            f"IP-Adapter: no compatible adapter exists for {detected_model_type} "
            f"(type='{adapter_type.value}'). No IP-Adapter was released for this architecture. "
            f"IP-Adapter will be disabled for this model."
        )
        cfg["enabled"] = False
        return cfg

    correct_model_path = target_entry["model_path"]
    correct_encoder_path = target_entry["image_encoder_path"]

    # Resolve model path
    if current_model_path != correct_model_path:
        logger.warning(
            f"IP-Adapter auto-resolution: '{current_model_path}' is incompatible with "
            f"{detected_model_type} (cross_attention_dim mismatch). "
            f"Resolving to '{correct_model_path}'."
        )
        cfg["ipadapter_model_path"] = correct_model_path
    else:
        logger.info(f"IP-Adapter: '{current_model_path}' is compatible with {detected_model_type}.")

    # Resolve encoder path (only if it's a known HF encoder — custom encoders untouched)
    if current_encoder_path in _KNOWN_ENCODER_PATHS and current_encoder_path != correct_encoder_path:
        logger.info(f"IP-Adapter: resolving image encoder '{current_encoder_path}' → '{correct_encoder_path}'.")
        cfg["image_encoder_path"] = correct_encoder_path

    return cfg


class IPAdapterModule(OrchestratorUser):
    """IP-Adapter embedding hook provider.

    Produces an embedding hook that concatenates cached image tokens (from
    StreamParameterUpdater's embedding cache) to the current text embeddings.
    """

    def __init__(self, config: IPAdapterConfig) -> None:
        self.config = config
        self.ipadapter: Optional[Any] = None

    def build_embedding_hook(self, stream) -> EmbeddingHook:
        # S4: must match install()'s default (below) — a mismatch here would silently
        # miss the cache under the key install() actually populated, hitting the
        # zero-fill path below on every frame. Latent today only because both wrapper
        # call sites (wrapper.py:2263, :2864) pin style_image_key explicitly.
        style_key = self.config.style_image_key or "ipadapter_main"
        num_tokens = int(self.config.num_image_tokens)
        warned_zero_fill = False

        def _embedding_hook(ctx: EmbedsCtx) -> EmbedsCtx:
            nonlocal warned_zero_fill
            # Fetch cached image token embeddings (prompt, negative)
            cached: Optional[Tuple[torch.Tensor, torch.Tensor]] = stream._param_updater.get_cached_embeddings(
                style_key
            )
            image_prompt_tokens: Optional[torch.Tensor] = None
            image_negative_tokens: Optional[torch.Tensor] = None
            if cached is not None:
                image_prompt_tokens, image_negative_tokens = cached

            # Validate or synthesize tokens when missing to satisfy engine shape (e.g., TRT expects 77+num_tokens)
            hidden_dim = ctx.prompt_embeds.shape[2]
            batch_size = ctx.prompt_embeds.shape[0]
            if image_prompt_tokens is None:
                # S4: silent zero-fill is the same failure class as B1/B2 — it disables
                # image conditioning without ever raising or failing a shape check.
                # Warn once per hook instance so a real-time streaming loop doesn't
                # spam this every frame.
                if not warned_zero_fill:
                    logger.warning(
                        "IPAdapterModule: no cached image-token embeddings for "
                        f"style_image_key='{style_key}' — falling back to zero-filled "
                        "tokens (image conditioning is a no-op until a style image is "
                        "processed for this key). Logged once per hook instance."
                    )
                    warned_zero_fill = True
                image_prompt_tokens = torch.zeros(
                    (batch_size, num_tokens, hidden_dim),
                    dtype=ctx.prompt_embeds.dtype,
                    device=ctx.prompt_embeds.device,
                )
            else:
                if image_prompt_tokens.shape[1] != num_tokens:
                    raise ValueError(
                        f"IPAdapterModule: Expected {num_tokens} image tokens, got {image_prompt_tokens.shape[1]}"
                    )

            # Concatenate image tokens to the right of text tokens
            prompt_with_image = ctx.prompt_embeds
            if image_prompt_tokens is not None:
                # Repeat to match batch size if needed
                if image_prompt_tokens.shape[0] != prompt_with_image.shape[0]:
                    image_prompt_tokens = image_prompt_tokens.repeat_interleave(
                        repeats=prompt_with_image.shape[0] // max(image_prompt_tokens.shape[0], 1), dim=0
                    )
                prompt_with_image = torch.cat([prompt_with_image, image_prompt_tokens], dim=1)

            neg_with_image = ctx.negative_prompt_embeds
            if neg_with_image is not None:
                if image_negative_tokens is None:
                    image_negative_tokens = torch.zeros(
                        (neg_with_image.shape[0], num_tokens, hidden_dim),
                        dtype=neg_with_image.dtype,
                        device=neg_with_image.device,
                    )
                else:
                    if image_negative_tokens.shape[0] != neg_with_image.shape[0]:
                        image_negative_tokens = image_negative_tokens.repeat_interleave(
                            repeats=neg_with_image.shape[0] // max(image_negative_tokens.shape[0], 1), dim=0
                        )
                neg_with_image = torch.cat([neg_with_image, image_negative_tokens], dim=1)

            return EmbedsCtx(prompt_embeds=prompt_with_image, negative_prompt_embeds=neg_with_image)

        return _embedding_hook

    def install(self, stream) -> None:
        """Install IP-Adapter processors and register embedding hook and preprocessor.

        - Instantiates IP-Adapter with model and encoder paths
        - Registers IPAdapterEmbeddingPreprocessor with StreamParameterUpdater using style_image_key
        - Optionally processes provided style image to populate the embedding cache
        - Registers the embedding hook onto stream.embedding_hooks
        - Sets the initial scale on the IPAdapter instance
        """
        style_key = self.config.style_image_key or "ipadapter_main"

        # Attach shared orchestrator to ensure consistent reuse across modules
        self.attach_orchestrator(stream)

        # Validate required paths
        if not self.config.ipadapter_model_path or not self.config.image_encoder_path:
            raise ValueError("IPAdapterModule.install: ipadapter_model_path and image_encoder_path are required")

        # Lazy import to avoid hard dependency unless used
        try:
            from diffusers_ipadapter import IPAdapter  # type: ignore
        except Exception as e:
            logger.error(f"IPAdapterModule.install: Failed to import IPAdapter: {e}")
            raise
        try:
            from streamdiffusion.preprocessing.processors.ipadapter_embedding import IPAdapterEmbeddingPreprocessor
        except Exception as e:
            logger.error(f"IPAdapterModule.install: Failed to import IPAdapterEmbeddingPreprocessor: {e}")
            raise

        # Resolve model paths (HF repo file or local path)
        resolved_ip_path = self._resolve_model_path(self.config.ipadapter_model_path)
        resolved_encoder_path = self._resolve_model_path(self.config.image_encoder_path)

        # Create IP-Adapter and install processors into UNet (FaceID-aware)
        ip_kwargs = {
            "pipe": stream.pipe,
            "ipadapter_ckpt_path": resolved_ip_path,
            "image_encoder_path": resolved_encoder_path,
            "device": stream.device,
            "dtype": stream.dtype,
        }
        if self.config.type == IPAdapterType.FACEID:
            # B1: diffusers_ipadapter feeds InsightFace RGB, but every InsightFace ONNX
            # model expects BGR (swapRB=True internally) — patch the single choke point
            # (detect_faces_multires) before any detection can run.
            try:
                from streamdiffusion.modules.faceid_compat import apply_faceid_patches

                apply_faceid_patches()
            except Exception as e:
                logger.warning(f"IPAdapterModule.install: apply_faceid_patches (B1) failed: {e}")

            if self.config.insightface_model_name:
                ip_kwargs["insightface_model_name"] = self.config.insightface_model_name
                print(
                    f"IPAdapterModule.install: Initializing FaceID IP-Adapter with InsightFace model: {self.config.insightface_model_name}"
                )
        ipadapter = IPAdapter(**ip_kwargs)
        self.ipadapter = ipadapter

        # B2: the FaceID checkpoint's rank-128 LoRA is silently discarded by the vendored
        # strict load (it filters "lora"/"LoRA" keys — no LoRA-aware processor exists) and
        # would be lost again by the TensorRT export path even if it did load. Fuse it
        # directly into the UNet's attention linears instead, where no processor rebuild
        # can touch it. B3 (engine cache-key marker) MUST accompany this — see
        # engine_manager.py's EngineType.UNET branch.
        if self.config.type == IPAdapterType.FACEID:
            try:
                from streamdiffusion.modules.faceid_compat import fuse_faceid_lora

                fuse_faceid_lora(stream.pipe.unet, resolved_ip_path, lora_scale=1.0)
            except Exception as e:
                report_error(f"IPAdapterModule.install: fuse_faceid_lora (B2) failed: {e}")
                raise

        # Fix kvo_cache incompatibility: diffusers_ipadapter sets old AttnProcessor on
        # self-attention blocks (attn1) that doesn't accept the kvo_cache kwarg passed by
        # newer diffusers Attention.forward(). Replace them with diffusers' native
        # AttnProcessor2_0 which accepts kvo_cache and returns (hidden_states, kvo_cache).
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0 as NativeAttnProcessor2_0

            attn_procs = stream.pipe.unet.attn_processors
            for name in attn_procs:
                if name.endswith("attn1.processor"):
                    attn_procs[name] = NativeAttnProcessor2_0()
            stream.pipe.unet.set_attn_processor(attn_procs)
        except Exception as e:
            logger.warning(f"IPAdapterModule.install: kvo_cache processor patch failed: {e}")

        # Register embedding preprocessor for this style key
        # Use FaceID preprocessor if applicable
        if self.config.type == IPAdapterType.FACEID:
            try:
                from streamdiffusion.preprocessing.processors.faceid_embedding import FaceIDEmbeddingPreprocessor

                embedding_preprocessor = FaceIDEmbeddingPreprocessor(
                    ipadapter=ipadapter,
                    device=stream.device,
                    dtype=stream.dtype,
                )
                print("IPAdapterModule.install: Using FaceIDEmbeddingPreprocessor for FaceID model")
            except Exception as e:
                report_error(f"IPAdapterModule.install: Failed to initialize FaceIDEmbeddingPreprocessor: {e}")
                raise
        else:
            embedding_preprocessor = IPAdapterEmbeddingPreprocessor(
                ipadapter=ipadapter,
                device=stream.device,
                dtype=stream.dtype,
            )
        stream._param_updater.register_embedding_preprocessor(embedding_preprocessor, style_key)

        # Process initial style image if provided
        if self.config.style_image is not None:
            try:
                stream._param_updater.update_style_image(style_key, self.config.style_image, is_stream=False)
            except Exception as e:
                logger.error(f"IPAdapterModule.install: Failed to process style image: {e}")
                raise

        # Set initial scale on the IPAdapter instance
        try:
            ipadapter.set_scale(float(self.config.scale))
        except Exception as e:
            logger.debug(f"Failed to set initial IPAdapter scale: {e}", exc_info=True)

        # Expose IPAdapter instance as single source of truth
        try:
            stream.ipadapter = ipadapter
            # Extend IPAdapter with our custom attributes since diffusers IPAdapter doesn't expose current state
            ipadapter.weight_type = self.config.weight_type  # For build_layer_weights
            ipadapter.scale = float(self.config.scale)  # Track current scale
            ipadapter.enabled = bool(self.config.enabled)  # Track enabled state
        except Exception as e:
            logger.debug(f"Failed to attach custom attributes to ipadapter instance: {e}", exc_info=True)

        # Register embedding hook for concatenation of image tokens
        stream.embedding_hooks.append(self.build_embedding_hook(stream))

        # Register UNet hook to supply per-step IP-Adapter scale via extra kwargs
        stream.unet_hooks.append(self.build_unet_hook(stream))

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Resolve a model path.

        Accepts either a local filesystem path or a Hugging Face repo/file spec like
        "h94/IP-Adapter/models/ip-adapter-plus_sd15.safetensors" or a directory path
        such as "h94/IP-Adapter/models/image_encoder".
        """
        if not model_path:
            raise ValueError("IPAdapterModule._resolve_model_path: model_path is required")

        if os.path.exists(model_path):
            return model_path

        # Treat as HF repo path
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except Exception as e:
            import logging

            logging.getLogger(__name__).error(
                f"IPAdapterModule: huggingface_hub required to resolve '{model_path}': {e}"
            )
            raise

        parts = model_path.split("/")
        if len(parts) < 3:
            raise ValueError(f"IPAdapterModule._resolve_model_path: Invalid Hugging Face spec: '{model_path}'")

        repo_id = "/".join(parts[:2])
        subpath = "/".join(parts[2:])

        # File if last component has an extension; otherwise treat as directory
        if "." in parts[-1]:
            # File download
            local_path = hf_hub_download(repo_id=repo_id, filename=subpath)
            return local_path
        else:
            # Directory download
            repo_root = snapshot_download(repo_id=repo_id, allow_patterns=[f"{subpath}/*"])
            full_path = os.path.join(repo_root, subpath)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"IPAdapterModule._resolve_model_path: Downloaded path not found: {full_path}")
            return full_path

    def build_unet_hook(self, stream) -> UnetHook:
        """Provide per-step ipadapter_scale vector via UNet hook extra kwargs.

        - For TensorRT UNet engines compiled with IP-Adapter, pass a per-layer vector in extra kwargs
        - For PyTorch UNet with installed IP processors, modulate per-layer processor scale by time factor
        """
        _last_enabled_state = None  # Track previous enabled state to avoid redundant updates

        # Resolve the optional diffusers_ipadapter helpers once, at hook-build time,
        # instead of importing them on every frame inside the per-step hook below.
        build_time_weight_factor = None
        build_layer_weights = None
        try:
            from diffusers_ipadapter.ip_adapter.attention_processor import (
                build_layer_weights as _build_layer_weights,
            )
            from diffusers_ipadapter.ip_adapter.attention_processor import (
                build_time_weight_factor as _build_time_weight_factor,
            )

            build_time_weight_factor = _build_time_weight_factor
            build_layer_weights = _build_layer_weights
        except Exception as import_error:
            logger.debug(
                f"IPAdapterModule.build_unet_hook: diffusers_ipadapter helpers unavailable: {import_error}",
                exc_info=True,
            )

        def _unet_hook(ctx: StepCtx) -> UnetKwargsDelta:
            # If no IP-Adapter installed, do nothing
            if not hasattr(stream, "ipadapter") or stream.ipadapter is None:
                return UnetKwargsDelta()

            # Check if IPAdapter is enabled
            enabled = getattr(stream.ipadapter, "enabled", True)

            # Read base weight and weight type from IPAdapter instance
            try:
                base_weight = float(getattr(stream.ipadapter, "scale", 1.0)) if enabled else 0.0
            except Exception as scale_error:
                logger.debug(f"IPAdapterModule: failed to read ipadapter.scale: {scale_error}", exc_info=True)
                base_weight = 0.0 if not enabled else 1.0
            weight_type = getattr(stream.ipadapter, "weight_type", None)

            # Determine total steps and current step index for time scheduling
            total_steps = None
            try:
                if hasattr(stream, "denoising_steps_num") and isinstance(stream.denoising_steps_num, int):
                    total_steps = int(stream.denoising_steps_num)
                elif hasattr(stream, "t_list") and stream.t_list is not None:
                    total_steps = len(stream.t_list)
            except Exception as steps_error:
                logger.debug(f"IPAdapterModule: failed to determine total_steps: {steps_error}", exc_info=True)
                total_steps = None

            time_factor = 1.0
            if total_steps is not None and ctx.step_index is not None and build_time_weight_factor is not None:
                try:
                    time_factor = float(build_time_weight_factor(weight_type, int(ctx.step_index), int(total_steps)))
                except Exception as scale_factor_error:
                    # Do not add fallback mechanisms
                    logger.debug(
                        f"IPAdapterModule: build_time_weight_factor failed: {scale_factor_error}", exc_info=True
                    )

            # TensorRT engine path: supply ipadapter_scale vector via extra kwargs
            try:
                is_trt_unet = (
                    hasattr(stream, "unet") and hasattr(stream.unet, "engine") and hasattr(stream.unet, "stream")
                )
            except Exception as trt_check_error:
                logger.debug(f"IPAdapterModule: TensorRT UNet detection failed: {trt_check_error}", exc_info=True)
                is_trt_unet = False

            if is_trt_unet and getattr(stream.unet, "use_ipadapter", False):
                num_ip_layers = getattr(stream.unet, "num_ip_layers", None)
                if isinstance(num_ip_layers, int) and num_ip_layers > 0:
                    weights_tensor = None
                    try:
                        if build_layer_weights is not None:
                            weights_tensor = build_layer_weights(num_ip_layers, float(base_weight), weight_type)
                    except Exception as weights_error:
                        logger.debug(f"IPAdapterModule: build_layer_weights failed: {weights_error}", exc_info=True)
                        weights_tensor = None
                    if weights_tensor is None:
                        weights_tensor = torch.full(
                            (num_ip_layers,), float(base_weight), dtype=torch.float32, device=stream.device
                        )
                    # Apply per-step time factor
                    try:
                        weights_tensor = weights_tensor * float(time_factor)
                    except Exception as time_scale_error:
                        logger.debug(
                            f"IPAdapterModule: applying time_factor to weights_tensor failed: {time_scale_error}",
                            exc_info=True,
                        )
                    return UnetKwargsDelta(extra_unet_kwargs={"ipadapter_scale": weights_tensor})

            # PyTorch UNet path: modulate installed processor scales by time factor and enabled state
            try:
                nonlocal _last_enabled_state
                # Only process if we need to make changes (time scaling or state transition)
                needs_update = time_factor != 1.0 or enabled != _last_enabled_state
                if needs_update and hasattr(stream.pipe, "unet") and hasattr(stream.pipe.unet, "attn_processors"):
                    _last_enabled_state = enabled
                    for proc in stream.pipe.unet.attn_processors.values():
                        if hasattr(proc, "scale") and hasattr(proc, "_ip_layer_index"):
                            base_val = getattr(proc, "_base_scale", proc.scale)
                            # Apply both enabled state and time factor
                            final_scale = float(base_val) * float(time_factor) if enabled else 0.0
                            proc.scale = final_scale
            except Exception as processor_scale_error:
                logger.debug(
                    f"IPAdapterModule: PyTorch UNet processor scale update failed: {processor_scale_error}",
                    exc_info=True,
                )

            return UnetKwargsDelta()

        return _unet_hook
