"""
FP8 Quantization for StreamDiffusion TensorRT UNet engine.

Uses nvidia-modelopt torch-level quantization with a real pipeline forward_loop
to calibrate activation ranges across all denoising timesteps.

Previous approach: modelopt.onnx.quantization.quantize with RandomDataProvider
                   (Gaussian noise only — never saw real activations)
Current approach:  modelopt.torch.quantization.quantize with forward_loop
                   running full pipe(prompt=...) calls on diverse calibration prompts

Requirements:
    nvidia-modelopt[torch] >= 0.19.0
    TensorRT >= 10.0 (FP8 E4M3 hardware support, STRONGLY_TYPED build flag)
    RTX 4090+ (Ada Lovelace, compute capability 8.9)
"""

import copy
import logging
import os
from pathlib import Path
from typing import List, Optional


logger = logging.getLogger(__name__)

_BUNDLED_PROMPTS_PATH = Path(__file__).parent / "calibration_prompts_sdxl.txt"


def _load_calibration_prompts(user_path: Optional[str] = None) -> List[str]:
    """Load calibration prompts from user path (if given) or bundled default."""
    path = Path(user_path) if user_path else _BUNDLED_PROMPTS_PATH
    if not path.exists():
        logger.warning(f"[FP8] Calibration prompts not found: {path}. Using 3-prompt fallback.")
        return [
            "a portrait of a person in soft studio lighting",
            "abstract colorful geometric pattern",
            "landscape photography at golden hour",
        ]
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    logger.info(f"[FP8] Loaded {len(prompts)} calibration prompts from {path.name}")
    return prompts


def _get_fp8_config(alpha: float = 0.8):
    """
    Build an FP8 quantization config, attempting to copy FP8_DEFAULT_CFG from modelopt
    and patch the SmoothQuant alpha.  Falls back to a hand-written equivalent if the
    import path changes between modelopt releases.
    """
    try:
        import modelopt.torch.quantization as mtq

        cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
        if isinstance(cfg.get("algorithm"), dict):
            cfg["algorithm"]["alpha"] = alpha
        return cfg
    except AttributeError:
        pass
    try:
        from modelopt.torch.quantization.config import FP8_DEFAULT_CFG

        cfg = copy.deepcopy(FP8_DEFAULT_CFG)
        if isinstance(cfg.get("algorithm"), dict):
            cfg["algorithm"]["alpha"] = alpha
        return cfg
    except ImportError:
        pass

    logger.warning("[FP8] FP8_DEFAULT_CFG not found in modelopt; using hand-written FP8 E4M3 fallback.")
    return {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
            "*input_quantizer": {"num_bits": (4, 3), "axis": None},
            "default": {"num_bits": (4, 3), "axis": None},
        },
        "algorithm": {"method": "smoothquant", "alpha": alpha},
    }


DEFAULT_FP8_DISABLE_PATTERNS = ["*add_embedding*", "*time_embedding*"]


def calibrate_unet_fp8_torch(
    pipe,
    unet_module,
    prompts: List[str],
    num_inference_steps: int = 20,
    alpha: float = 0.8,
    amax_save_path: Optional[str] = None,
    disable_quantizer_patterns: Optional[List[str]] = None,
    batch_size: int = 1,
) -> None:
    """
    Calibrate UNet for FP8 using torch-level modelopt quantization.

    Runs the full diffusers pipeline as forward_loop so activation ranges are
    collected across all denoising timesteps with real text-conditioned inputs.

    After this call the UNet module has fake-quantizers installed in-place.
    Exporting to ONNX inside `mto.export_torch_mode()` will produce Q/DQ nodes.

    Args:
        pipe: Full StableDiffusionPipeline / StableDiffusionXLPipeline instance.
        unet_module: pipe.unet (the raw torch nn.Module to be quantized).
        prompts: 32–128 diverse calibration texts.
        num_inference_steps: Denoising steps per prompt. 20 for SDXL, 4 for SDXL-Turbo.
        alpha: SmoothQuant alpha. 0.8 for SDXL, 1.0 for SD1.5.
        amax_save_path: Where to persist calibrated scales (*.amax.pt).
        disable_quantizer_patterns: Module name patterns to exclude from quantization.
        batch_size: Prompts per pipe() call. >1 reduces calibration time but uses more VRAM.
    """
    try:
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq
    except ImportError as e:
        raise ImportError(
            "nvidia-modelopt[torch] is required for torch-level FP8 calibration.\n"
            "Install with:  pip install 'nvidia-modelopt[torch]>=0.19.0'\n"
            "Then re-run the engine build."
        ) from e

    import torch

    logger.info(
        f"[FP8] Starting torch-level FP8 calibration: "
        f"{len(prompts)} prompts × {num_inference_steps} steps, alpha={alpha}, batch_size={batch_size}"
    )

    # Fuse active LoRAs before quantization — unfused LoRA breaks TRT kernel fusion.
    # Using try/finally so unfuse always runs even if calibration raises.
    _fused = False
    try:
        active_adapters = pipe.get_active_adapters() if hasattr(pipe, "get_active_adapters") else []
        if active_adapters and hasattr(pipe, "fuse_lora"):
            logger.info("[FP8] Fusing LoRA before calibration")
            pipe.fuse_lora()
            _fused = True
    except AttributeError as e:
        logger.warning(f"[FP8] LoRA fusion skipped: {e}")

    try:
        quant_config = _get_fp8_config(alpha=alpha)
        if disable_quantizer_patterns is None:
            disable_quantizer_patterns = DEFAULT_FP8_DISABLE_PATTERNS
        if disable_quantizer_patterns:
            try:
                mtq.disable_quantizer(unet_module, disable_quantizer_patterns)
            except Exception as e:
                logger.warning(f"[FP8] disable_quantizer skipped: {e}")

        def forward_loop(mod):
            with torch.inference_mode(), torch.autocast("cuda"):
                batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
                for i, batch in enumerate(batches):
                    logger.info(f"[FP8] Calibrating batch {i + 1}/{len(batches)}: {batch[0][:50]}")
                    try:
                        pipe(
                            prompt=batch if len(batch) > 1 else batch[0],
                            num_inference_steps=num_inference_steps,
                            output_type="latent",
                            guidance_scale=7.5,
                        ).images
                    except Exception as e:
                        logger.warning(f"[FP8] Batch {i + 1} failed ({type(e).__name__}): {e}. Skipping.")

        # Snapshot state_dict so we can restore on partial failure (half-quantized is worse than none)
        _sd_backup = {k: v.clone() for k, v in unet_module.state_dict().items()}

        logger.info("[FP8] Running mtq.quantize with pipeline forward_loop (several minutes)...")
        try:
            mtq.quantize(unet_module, quant_config, forward_loop)
        except Exception as quant_err:
            logger.warning(f"[FP8] mtq.quantize failed: {quant_err}. Restoring original UNet state.")
            unet_module.load_state_dict(_sd_backup)
            raise
        finally:
            del _sd_backup

        logger.info("[FP8] mtq.quantize complete — fake-quantizers installed")

        if amax_save_path:
            try:
                mto.save(unet_module, amax_save_path)
                logger.info(f"[FP8] Saved calibration scales to {amax_save_path}")
            except Exception as e:
                logger.warning(f"[FP8] Could not save amax: {e}")

    finally:
        if _fused and hasattr(pipe, "unfuse_lora"):
            try:
                pipe.unfuse_lora()
                logger.info("[FP8] LoRA unfused after calibration")
            except Exception as e:
                logger.warning(f"[FP8] LoRA unfuse failed: {e}")


def load_unet_amax(unet_module, amax_path: str) -> bool:
    """
    Restore previously-saved calibration scales into the UNet.

    Returns True if successful, False if the file is missing or incompatible
    (in which case it is deleted and full calibration should be re-run).
    """
    if not os.path.exists(amax_path):
        return False
    try:
        import modelopt.torch.opt as mto

        mto.restore(unet_module, amax_path)
        logger.info(f"[FP8] Loaded cached calibration scales from {amax_path}")
        return True
    except Exception as e:
        logger.warning(f"[FP8] Cannot load amax from {amax_path}: {e}. Will recalibrate.")
        try:
            os.remove(amax_path)
        except OSError:
            pass
        return False
