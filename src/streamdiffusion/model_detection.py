"""Comprehensive model detection for TensorRT and pipeline support"""

import os
from typing import Any, Dict, Optional, Tuple

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

# Gracefully import the SD3 model class; it might not exist in older diffusers versions.
try:
    from diffusers.models.transformers.mm_dit import MMDiTTransformer2DModel

    HAS_MMDIT = True
except ImportError:
    # Create a dummy class if the import fails to prevent runtime errors.
    MMDiTTransformer2DModel = type("MMDiTTransformer2DModel", (torch.nn.Module,), {})
    HAS_MMDIT = False

import logging

logger = logging.getLogger(__name__)


def _detect_turbo_from_scheduler(pipe: Optional[Any]) -> Optional[bool]:
    """Discriminate ADD-distilled Turbo checkpoints from their base counterparts via the
    pipeline's *source* scheduler config, evaluated before StreamDiffusion swaps in its own
    LCMScheduler/TCDScheduler (pipeline.py calls this at __init__ time, ahead of
    _initialize_scheduler).

    Both sd-turbo and sdxl-turbo ship an EulerAncestralDiscreteScheduler with
    timestep_spacing="trailing" (see docs/plans/SDXL-Turbo_Research_Verification.md). Their
    non-Turbo counterparts (SD2.1, SDXL-Base-1.0) do not. This replaces a UNet-config
    heuristic that keyed off `time_cond_proj_dim`, which is actually the LCM-distillation
    guidance-embedding dim: stock SDXL-Base-1.0 also has it `None` (false positive for
    Turbo), and it says nothing about non-SDXL UNets at all (sd-turbo was never flagged
    Turbo).

    Returns None (undecided) rather than False when no pipe/scheduler is available to
    check, so callers can tell "confirmed non-Turbo" apart from "couldn't check" and fall
    back to a corroborating hint (e.g. the model-id string) if they have one.
    """
    if pipe is None:
        return None
    scheduler = getattr(pipe, "scheduler", None)
    scheduler_config = getattr(scheduler, "config", None)
    if scheduler_config is None:
        return None
    # `_class_name` is only present on configs loaded from a JSON scheduler_config.json
    # (from_pretrained). A scheduler built via `.from_config(...)` -- e.g. diffusers'
    # single-file `_legacy_load_scheduler` synthesized-defaults path -- has no
    # `_class_name` at all, which used to silently degrade this check to False. The
    # runtime class name is always correct regardless of construction path.
    class_name = getattr(scheduler_config, "_class_name", "") or type(scheduler).__name__
    spacing = getattr(scheduler_config, "timestep_spacing", None)
    return "EulerAncestralDiscreteScheduler" in class_name and spacing == "trailing"


def read_safetensors_metadata(path: Optional[str]) -> Dict[str, str]:
    """Header-only read of a `.safetensors` file's `__metadata__` block. No tensor
    weights are loaded -- `safe_open` reads only the JSON header. Returns `{}` on any
    failure (missing file, not a local path (repo id/URL), not a safetensors file,
    corrupt header) rather than raising, so callers can treat metadata as
    always-available-but-possibly-empty.
    """
    if not path:
        return {}
    try:
        from safetensors import safe_open
    except ImportError:
        return {}
    try:
        with safe_open(path, framework="pt") as f:
            return dict(f.metadata() or {})
    except Exception:
        return {}


def turbo_from_checkpoint_metadata(path: Optional[str]) -> Optional[bool]:
    """True/False when the checkpoint carries a SAI Model Spec
    `modelspec.architecture` tag (e.g. "stable-diffusion-xl-turbo-v1" vs
    "stable-diffusion-xl-v1-base"). None when the tag is absent, which is the common
    case for community merges -- callers should fall back to a weaker signal rather
    than treat None as a negative.
    """
    architecture = read_safetensors_metadata(path).get("modelspec.architecture")
    if not architecture:
        return None
    return "turbo" in architecture.lower()


def turbo_hint_from_model_id(model_id_or_path: Optional[str]) -> bool:
    """Case-insensitive "turbo" match on the checkpoint's file *basename* only.
    Deliberately ignores parent directories, so a base model staged under a path
    like `D:/turbo_tests/sdxl_base.safetensors` is not misclassified. This is the
    weakest signal in resolve_is_turbo's precedence -- a naming convention, not a
    guarantee -- so it is consulted last.
    """
    if not model_id_or_path:
        return False
    basename = os.path.basename(str(model_id_or_path))
    return "turbo" in basename.lower()


def resolve_is_turbo(
    *,
    pipe: Optional[Any] = None,
    explicit: Optional[bool] = None,
    model_id_or_path: Optional[str] = None,
    loaded_via_single_file: bool = False,
) -> Tuple[bool, str]:
    """Resolve whether the loaded checkpoint is an ADD-distilled Turbo variant.

    Closes the gap flagged in dotsimulate's PR #58 review: `_detect_turbo_from_scheduler`
    depends on `pipe.scheduler.config` reflecting the checkpoint's own published
    scheduler_config.json. That holds for `from_pretrained` loads (repo id or a local
    diffusers-format directory) but not for `from_single_file` -- diffusers borrows a
    *reference repo's* scheduler config for the checkpoint's declared architecture family
    (e.g. an SDXL-Turbo `.safetensors` merge inherits SDXL-Base's `EulerDiscreteScheduler`),
    so the scheduler check returns a confident but wrong verdict there, not an "undecided"
    one. This function therefore only trusts the scheduler for non-single-file loads and
    falls back to checkpoint-embedded metadata, then a filename hint, for single-file ones.

    Precedence (highest to lowest):
      1. `explicit` -- caller-supplied override, wins outright.
      2. Scheduler config -- only when `loaded_via_single_file` is False.
      3. `.safetensors` `modelspec.architecture` metadata -- single-file loads only; a
         present verdict is authoritative and can VETO a misleading filename.
      4. "turbo" substring in the file basename -- single-file loads only, last resort.
      5. Nothing decisive -- False, tagged "undecided" so callers can warn that no
         automatic signal exists and an explicit override is needed.

    Returns `(is_turbo, reason)` where `reason` is one of "explicit", "scheduler",
    "modelspec", "filename", "undecided".
    """
    if explicit is not None:
        return bool(explicit), "explicit"

    if not loaded_via_single_file:
        scheduler_verdict = _detect_turbo_from_scheduler(pipe)
        if scheduler_verdict is not None:
            return scheduler_verdict, "scheduler"
        return False, "undecided"

    metadata_verdict = turbo_from_checkpoint_metadata(model_id_or_path)
    if metadata_verdict is not None:
        return metadata_verdict, "modelspec"

    if turbo_hint_from_model_id(model_id_or_path):
        return True, "filename"

    return False, "undecided"


def detect_model(model: torch.nn.Module, pipe: Optional[Any] = None) -> Dict[str, Any]:
    """
    Comprehensive and robust model detection using definitive architectural features.

    This function replaces heuristic-based analysis with a deterministic,
    rule-based approach by first inspecting the model's class and then its key
    configuration parameters that define the architecture.

    Args:
        model: The model to analyze (e.g., UNet or MMDiT).
        pipe: Optional pipeline for additional context (e.g., detecting Turbo via scheduler).

    Returns:
        A dictionary with detailed information about the detected model.
    """
    model_type = "Unknown"
    # Optional[bool]: None = undecided (no pipe / scheduler available to check), True/False =
    # a decision based on the *source* scheduler. This is a signal, not a verdict -- Turbo
    # detection is resolved once, authoritatively, by resolve_is_turbo(); detect_model no
    # longer collapses this into a bool of its own (see resolve_is_turbo's docstring for why
    # a pipe-less scheduler check must never be read as "confirmed non-Turbo").
    turbo_from_scheduler = None
    is_sdxl = False
    is_sd3 = False
    confidence = 0.0

    # 1. SD3 Detection (based on MMDiT Architecture)
    # NOTE THAT THIS IS NOT IMPLEMNTED AT THIS TIME
    if HAS_MMDIT and isinstance(model, MMDiTTransformer2DModel):
        config = model.config
        is_sd3 = True
        # Definitive fingerprints for SD3 Medium
        if config.get("in_channels") == 16 and config.get("joint_attention_dim") == 4096:
            model_type = "SD3"
            confidence = 1.0
            # Differentiating SD3 vs. SD3-Turbo from the MMDiT config alone is currently
            # speculative. A check on the pipeline's scheduler is a reasonable proxy.
            if pipe and hasattr(pipe, "scheduler"):
                scheduler_name = getattr(pipe.scheduler.config, "_class_name", "").lower()
                if "lcm" in scheduler_name or "turbo" in scheduler_name:
                    turbo_from_scheduler = True
                    model_type = "SD3-Turbo"
        else:
            model_type = "Unknown MMDiT"
            confidence = 0.6

    # 2. UNet-based Model Detection (SDXL, SD2.1, SD1.5)
    elif isinstance(model, UNet2DConditionModel):
        config = model.config

        # `time_cond_proj_dim` is the LCM-distillation guidance-embedding dim, not a
        # Turbo/Base signal (see _detect_turbo_from_scheduler's docstring). Turbo detection
        # uses the source scheduler instead; `time_cond_proj_dim` is surfaced separately as
        # `has_time_conditioning` via detect_unet_characteristics() below.
        turbo_from_scheduler = _detect_turbo_from_scheduler(pipe)

        # 2a. SDXL vs. non-SDXL
        # The `addition_embed_type` is the clearest indicator for the SDXL architecture.
        if config.get("addition_embed_type") is not None:
            model_type = "SDXL"
            is_sdxl = True
            confidence = 1.0

        # 2b. SD2.1 vs. SD1.5 (if not SDXL)
        # Differentiate based on the text encoder's projection dimension.
        else:
            cross_attention_dim = config.get("cross_attention_dim")
            if cross_attention_dim == 1024:
                model_type = "SD2.1"
                confidence = 1.0
                # sd-turbo is SD2.1-architecture (cross_attention_dim=1024); the old
                # heuristic never set is_turbo on this branch at all.
            elif cross_attention_dim == 768:
                model_type = "SD1.5"
                confidence = 1.0
            else:
                # Fallback for fine-tunes with non-standard dimensions.
                model_type = "SD-finetune"
                confidence = 0.7

    # 3. ControlNet Model Detection (detect underlying architecture)
    elif hasattr(model, "config") and hasattr(model.config, "cross_attention_dim"):
        # ControlNet models have UNet-like configs, detect their base architecture
        config = model.config
        turbo_from_scheduler = _detect_turbo_from_scheduler(pipe)

        # Apply same detection logic as UNet models
        if config.get("addition_embed_type") is not None:
            model_type = "SDXL"
            is_sdxl = True
            confidence = 0.95  # Slightly lower confidence for ControlNet
        else:
            cross_attention_dim = config.get("cross_attention_dim")
            if cross_attention_dim == 1024:
                model_type = "SD2.1"
                confidence = 0.95
            elif cross_attention_dim == 768:
                model_type = "SD1.5"
                confidence = 0.95
            else:
                model_type = "SD-finetune"
                confidence = 0.7

    else:
        # The model is not a known UNet or MMDiT class.
        confidence = 0.0
        model_type = f"Unknown ({model.__class__.__name__})"

    # Populate architecture and compatibility details (can be expanded as needed)
    architecture_details = {
        "model_class": model.__class__.__name__,
        "in_channels": getattr(model.config, "in_channels", "N/A"),
        "cross_attention_dim": getattr(model.config, "cross_attention_dim", "N/A"),
        "block_out_channels": getattr(model.config, "block_out_channels", "N/A"),
    }

    # For UNet models, add detailed characteristics that SDXL code expects
    if isinstance(model, UNet2DConditionModel):
        unet_chars = detect_unet_characteristics(model)
        architecture_details.update(
            {
                "has_time_conditioning": unet_chars["has_time_cond"],
                "has_addition_embeds": unet_chars["has_addition_embed"],
            }
        )

    # For ControlNet models, add similar characteristics
    elif hasattr(model, "config") and hasattr(model.config, "cross_attention_dim"):
        # ControlNet models have similar config structure to UNet
        config = model.config
        has_addition_embed = config.get("addition_embed_type") is not None
        has_time_cond = hasattr(config, "time_cond_proj_dim") and config.time_cond_proj_dim is not None

        architecture_details.update(
            {
                "has_time_conditioning": has_time_cond,
                "has_addition_embeds": has_addition_embed,
            }
        )

    compatibility_info = {"notes": f"Detected as {model_type} with {confidence:.2f} confidence based on architecture."}

    result = {
        "model_type": model_type,
        "turbo_from_scheduler": turbo_from_scheduler,
        "is_sdxl": is_sdxl,
        "is_sd3": is_sd3,
        "confidence": confidence,
        "architecture_details": architecture_details,
        "compatibility_info": compatibility_info,
    }

    return result


def detect_unet_characteristics(unet: UNet2DConditionModel) -> Dict[str, Any]:
    """Detect detailed UNet characteristics including SDXL-specific features"""
    config = unet.config

    # Get cross attention dimensions to detect model type
    cross_attention_dim = getattr(config, "cross_attention_dim", None)

    # Detect SDXL by multiple indicators
    is_sdxl = False

    # Check cross attention dimension
    if isinstance(cross_attention_dim, (list, tuple)):
        # SDXL typically has [1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280]
        is_sdxl = any(dim >= 1280 for dim in cross_attention_dim)
    elif isinstance(cross_attention_dim, int):
        # Single value - SDXL uses 2048 for concatenated embeddings, or 1280+ for individual encoders
        is_sdxl = cross_attention_dim >= 1280

    # Check addition_embed_type for SDXL detection (strong indicator)
    addition_embed_type = getattr(config, "addition_embed_type", None)
    has_addition_embed = addition_embed_type is not None

    if addition_embed_type in ["text_time", "text_time_guidance"]:
        is_sdxl = True  # This is a definitive SDXL indicator

    # Check if model has time conditioning projection (SDXL feature)
    has_time_cond = hasattr(config, "time_cond_proj_dim") and config.time_cond_proj_dim is not None

    # Additional SDXL detection checks
    if hasattr(config, "num_class_embeds") and config.num_class_embeds is not None:
        is_sdxl = True  # SDXL often has class embeddings

    # Check sample size (SDXL typically uses 128 vs 64 for SD1.5)
    sample_size = getattr(config, "sample_size", 64)
    if sample_size >= 128:
        is_sdxl = True

    return {
        "is_sdxl": is_sdxl,
        "has_time_cond": has_time_cond,
        "has_addition_embed": has_addition_embed,
        "cross_attention_dim": cross_attention_dim,
        "addition_embed_type": addition_embed_type,
        "in_channels": getattr(config, "in_channels", 4),
        "sample_size": getattr(config, "sample_size", 64 if not is_sdxl else 128),
        "block_out_channels": tuple(getattr(config, "block_out_channels", [])),
        "attention_head_dim": getattr(config, "attention_head_dim", None),
    }


# This is used for controlnet/ipadapter model detection - can be deprecated (along with detect_unet_characteristics)
def detect_model_from_diffusers_unet(unet: UNet2DConditionModel) -> str:
    """Detect model type from diffusers UNet configuration"""
    characteristics = detect_unet_characteristics(unet)

    in_channels = characteristics["in_channels"]
    block_out_channels = characteristics["block_out_channels"]
    cross_attention_dim = characteristics["cross_attention_dim"]
    is_sdxl = characteristics["is_sdxl"]

    # Use enhanced SDXL detection
    if is_sdxl:
        return "SDXL"

    # Original detection logic for other models
    if cross_attention_dim == 768 and block_out_channels == (320, 640, 1280, 1280) and in_channels == 4:
        return "SD15"

    elif cross_attention_dim == 1024 and block_out_channels == (320, 640, 1280, 1280) and in_channels == 4:
        return "SD21"

    elif cross_attention_dim == 768 and in_channels == 4:
        return "SD15"
    elif cross_attention_dim == 1024 and in_channels == 4:
        return "SD21"

    if cross_attention_dim == 768:
        logger.warning(
            f"detect_model_from_diffusers_unet: Unknown SD1.5-like model with channels {block_out_channels}, defaulting to SD15"
        )
        return "SD15"
    elif cross_attention_dim == 1024:
        logger.warning(
            f"detect_model_from_diffusers_unet: Unknown SD2.1-like model with channels {block_out_channels}, defaulting to SD21"
        )
        return "SD21"
    else:
        raise ValueError(
            f"Unknown model architecture: "
            f"cross_attention_dim={cross_attention_dim}, "
            f"block_out_channels={block_out_channels}, "
            f"in_channels={in_channels}"
        )


def extract_unet_architecture(unet: UNet2DConditionModel) -> Dict[str, Any]:
    """
    Extract UNet architecture details needed for TensorRT engine building.

    This function provides the essential architecture information needed
    for TensorRT engine compilation in a clean, structured format.

    Args:
        unet: The UNet model to analyze

    Returns:
        Dict with architecture parameters for TensorRT engine building
    """
    config = unet.config

    # Basic model parameters
    model_channels = config.block_out_channels[0] if config.block_out_channels else 320
    block_out_channels = tuple(config.block_out_channels)
    channel_mult = tuple(ch // model_channels for ch in block_out_channels)

    # Resolution blocks
    if hasattr(config, "layers_per_block"):
        if isinstance(config.layers_per_block, (list, tuple)):
            num_res_blocks = tuple(config.layers_per_block)
        else:
            num_res_blocks = tuple([config.layers_per_block] * len(block_out_channels))
    else:
        num_res_blocks = tuple([2] * len(block_out_channels))

    # Attention and context dimensions
    context_dim = config.cross_attention_dim
    in_channels = config.in_channels

    # Attention head configuration
    attention_head_dim = getattr(config, "attention_head_dim", 8)
    if isinstance(attention_head_dim, (list, tuple)):
        attention_head_dim = attention_head_dim[0]

    # Transformer depth
    transformer_depth = getattr(config, "transformer_layers_per_block", 1)
    if isinstance(transformer_depth, (list, tuple)):
        transformer_depth = tuple(transformer_depth)
    else:
        transformer_depth = tuple([transformer_depth] * len(block_out_channels))

    # Time embedding
    time_embed_dim = getattr(config, "time_embedding_dim", None)
    if time_embed_dim is None:
        time_embed_dim = model_channels * 4

    # Build architecture dictionary
    architecture_dict = {
        "model_channels": model_channels,
        "in_channels": in_channels,
        "out_channels": getattr(config, "out_channels", in_channels),
        "num_res_blocks": num_res_blocks,
        "channel_mult": channel_mult,
        "context_dim": context_dim,
        "attention_head_dim": attention_head_dim,
        "transformer_depth": transformer_depth,
        "time_embed_dim": time_embed_dim,
        "block_out_channels": block_out_channels,
        # Additional configuration
        "use_linear_in_transformer": getattr(config, "use_linear_in_transformer", False),
        "conv_in_kernel": getattr(config, "conv_in_kernel", 3),
        "conv_out_kernel": getattr(config, "conv_out_kernel", 3),
        "resnet_time_scale_shift": getattr(config, "resnet_time_scale_shift", "default"),
        "class_embed_type": getattr(config, "class_embed_type", None),
        "num_class_embeds": getattr(config, "num_class_embeds", None),
        # Block types
        "down_block_types": getattr(config, "down_block_types", []),
        "up_block_types": getattr(config, "up_block_types", []),
    }

    return architecture_dict


def validate_architecture(arch_dict: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    Validate and fix architecture dictionary using model type presets.

    Ensures that all required architecture parameters are present and
    have reasonable values for the specified model type.

    Args:
        arch_dict: Architecture dictionary to validate
        model_type: Expected model type for validation

    Returns:
        Validated and corrected architecture dictionary
    """

    # Check for required keys
    required_keys = [
        "model_channels",
        "channel_mult",
        "num_res_blocks",
        "context_dim",
        "in_channels",
        "block_out_channels",
    ]

    for key in required_keys:
        if key not in arch_dict:
            raise ValueError(f"Missing required architecture parameter: {key}")

    # Ensure tuple format for sequence parameters
    for key in ["channel_mult", "num_res_blocks", "transformer_depth", "block_out_channels"]:
        if key in arch_dict and not isinstance(arch_dict[key], tuple):
            if isinstance(arch_dict[key], (list, int)):
                if isinstance(arch_dict[key], int):
                    arch_dict[key] = tuple([arch_dict[key]] * len(arch_dict["channel_mult"]))
                else:
                    arch_dict[key] = tuple(arch_dict[key])
            else:
                raise ValueError(
                    f"validate_architecture: '{key}' has unsupported type {type(arch_dict[key]).__name__!r}; "
                    f"expected list, int, or tuple"
                )

    # Validate sequence lengths match
    expected_levels = len(arch_dict["channel_mult"])
    for key in ["num_res_blocks", "transformer_depth"]:
        if key in arch_dict and len(arch_dict[key]) != expected_levels:
            raise ValueError(
                f"validate_architecture: '{key}' has {len(arch_dict[key])} levels but "
                f"'channel_mult' has {expected_levels}; they must match"
            )

    return arch_dict
