"""
FP8 Quantization for StreamDiffusion TensorRT UNet engine.

Uses nvidia-modelopt for ONNX-level FP8 quantization via Q/DQ node insertion.
The quantized ONNX is then compiled to TRT with STRONGLY_TYPED + FP8 builder flags.

Requirements:
    nvidia-modelopt[onnx] >= 0.35.0
    TensorRT >= 10.0 (FP8 support)
    RTX 4090+ (Ada Lovelace, compute 8.9, FP8 E4M3 hardware support)

This module is called from builder.py when fp8=True is passed to EngineBuilder.build().
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def generate_unet_calibration_data(
    model_data,
    opt_batch_size: int,
    opt_image_height: int,
    opt_image_width: int,
    num_batches: int = 8,
) -> List[Dict[str, np.ndarray]]:
    """
    Generate calibration data for SDXL-Turbo UNet FP8 quantization.

    Returns a list of input dicts matching the ONNX model's input names,
    with values as numpy arrays shaped to the TRT optimization profile's opt shapes.

    Args:
        model_data: UNet BaseModel instance (provides input names, kvo_cache_shapes,
                    text_maxlen, embedding_dim, cache_maxframes).
        opt_batch_size: Optimal batch size from TRT profile (typically 1 for
                        frame_buffer_size=1). The UNet input dim is 2*opt_batch_size
                        because cond + uncond are batched together.
        opt_image_height: Optimal image height in pixels (e.g. 512).
        opt_image_width: Optimal image width in pixels (e.g. 512).
        num_batches: Number of calibration batches. Capped at 8 for SDXL-scale
                     models: each batch contains 70 KVO cache tensors (~2.2 GB),
                     so 128 batches would require ~281 GB RAM. FP8 is less
                     sensitive to calibration size than INT8 (wider dynamic range).

    Returns:
        List of dicts: [{input_name: np.ndarray}, ...] — one dict per batch.
    """
    latent_h = opt_image_height // 8
    latent_w = opt_image_width // 8
    # UNet always receives 2× the batch (cond + uncond paired)
    effective_batch = 2 * opt_batch_size

    input_names = model_data.get_input_names()

    # Fixed seed for reproducible calibration
    rng = np.random.default_rng(seed=42)

    # Pre-read model_data properties once to avoid repeated attribute access
    text_maxlen = getattr(model_data, "text_maxlen", 77)
    embedding_dim = getattr(model_data, "embedding_dim", 2048)
    cache_maxframes = getattr(model_data, "cache_maxframes", 4)
    kvo_cache_shapes = getattr(model_data, "kvo_cache_shapes", [])
    num_ip_layers = getattr(model_data, "num_ip_layers", 1)
    control_inputs = getattr(model_data, "control_inputs", {})

    calibration_dataset = []

    for i in range(num_batches):
        batch_data = {}

        for name in input_names:
            if name == "sample":
                # Noisy latents in float32 (UNet ingests fp32 sample before internal autocast)
                # VAE latent scale: 0.18215 for SDXL
                data = (rng.standard_normal((effective_batch, 4, latent_h, latent_w)) * 0.18215)
                batch_data[name] = data.astype(np.float32)

            elif name == "timestep":
                # Timesteps: float32, shape (effective_batch,)
                # Sample broadly across [0, 999] to cover full activation range.
                t = rng.integers(0, 1000, size=(effective_batch,))
                batch_data[name] = t.astype(np.float32)

            elif name == "encoder_hidden_states":
                # CLIP/OpenCLIP text embeddings: float16 for fp16 SDXL models
                # Scale 0.01 approximates typical normalized text embedding magnitude.
                data = (rng.standard_normal((effective_batch, text_maxlen, embedding_dim)) * 0.01)
                batch_data[name] = data.astype(np.float16)

            elif name == "ipadapter_scale":
                # IP-Adapter per-layer scale: float32, shape (num_ip_layers,)
                batch_data[name] = np.ones((num_ip_layers,), dtype=np.float32)

            elif name.startswith("input_control_"):
                # ControlNet residual tensors: float16
                if name in control_inputs:
                    spec = control_inputs[name]
                    data = rng.standard_normal(
                        (effective_batch, spec["channels"], spec["height"], spec["width"])
                    )
                    batch_data[name] = data.astype(np.float16)

            elif name.startswith("kvo_cache_in_"):
                # KVO cached attention inputs: float16
                # shape = (2, cache_maxframes, effective_batch, seq_len, hidden_dim)
                # Zeros = cold cache. Conservative but avoids over-fitting calibration
                # ranges to cached-attention activation patterns.
                idx = int(name.rsplit("_", 1)[-1])
                if idx < len(kvo_cache_shapes):
                    seq_len, hidden_dim = kvo_cache_shapes[idx]
                    batch_data[name] = np.zeros(
                        (2, cache_maxframes, effective_batch, seq_len, hidden_dim),
                        dtype=np.float16,
                    )

        calibration_dataset.append(batch_data)

    logger.info(
        f"[FP8] Generated {num_batches} calibration batches "
        f"(effective_batch={effective_batch}, latent={latent_h}x{latent_w}, "
        f"inputs={len(input_names)}, kvo_count={len(kvo_cache_shapes)})"
    )
    return calibration_dataset


def quantize_onnx_fp8(
    onnx_opt_path: str,
    onnx_fp8_path: str,
    calibration_data: List[Dict[str, np.ndarray]],
    quantize_mha: bool = True,
    percentile: float = 1.0,
    alpha: float = 0.8,
) -> None:
    """
    Insert FP8 Q/DQ nodes into an optimized ONNX model via nvidia-modelopt.

    Takes the FP16-optimized ONNX (*.opt.onnx), runs calibration to collect
    activation ranges, and writes a new ONNX with QuantizeLinear/DequantizeLinear
    nodes annotated for FP8 E4M3 precision. TRT compiles this with
    STRONGLY_TYPED + FP8 builder flags.

    Args:
        onnx_opt_path: Input FP16 optimized ONNX path (*.opt.onnx).
        onnx_fp8_path: Output FP8 quantized ONNX path (*.fp8.onnx).
        calibration_data: List of input dicts from generate_unet_calibration_data().
        quantize_mha: Enable FP8 quantization of multi-head attention ops.
                      Recommended: True. Requires TRT 10+ and compute 8.9+.
        percentile: Percentile for activation range calibration.
                    1.0 = no clipping (safest for first run).
        alpha: SmoothQuant alpha — balances quantization difficulty between
               activations (alpha→0) and weights (alpha→1). 0.8 is optimal
               for transformer attention layers.
    """
    try:
        from modelopt.onnx.quantization import quantize as modelopt_quantize
    except ImportError as e:
        raise ImportError(
            "nvidia-modelopt is required for FP8 quantization. "
            "Install with: pip install 'nvidia-modelopt[onnx]'"
        ) from e

    input_size_mb = os.path.getsize(onnx_opt_path) / (1024 * 1024)
    logger.info(f"[FP8] Starting ONNX FP8 quantization")
    logger.info(f"[FP8]   Input:  {onnx_opt_path} ({input_size_mb:.0f} MB)")
    logger.info(f"[FP8]   Output: {onnx_fp8_path}")
    logger.info(f"[FP8]   Config: quantize_mha={quantize_mha}, percentile={percentile}, alpha={alpha}")
    logger.info(f"[FP8]   Calibration batches: {len(calibration_data)}")

    # Patch ByteSize() for >2GB ONNX models: modelopt calls onnx_model.ByteSize()
    # to auto-detect external data format, but protobuf cannot serialize >2GB protos.
    # Return a large value on failure so modelopt correctly uses external data format.
    import onnx as _onnx
    from google.protobuf.message import EncodeError as _EncodeError

    _orig_byte_size = _onnx.ModelProto.ByteSize

    def _safe_byte_size(self):
        try:
            return _orig_byte_size(self)
        except _EncodeError:
            return 3 * (1024**3)  # >2GB → triggers external data format

    _onnx.ModelProto.ByteSize = _safe_byte_size

    # modelopt expects {name: ndarray} with calibration samples stacked along axis 0,
    # not a list of dicts. Merge: [(name: shape)...] → {name: (N, *shape)}
    if isinstance(calibration_data, list) and calibration_data:
        merged = {}
        for name in calibration_data[0]:
            merged[name] = np.stack([batch[name] for batch in calibration_data if name in batch])
        calibration_data = merged
        logger.info(f"[FP8] Merged calibration data: {len(merged)} inputs, {next(iter(merged.values())).shape[0]} samples")

    quantize_kwargs = {
        "quantize_mode": "fp8",
        "output_path": onnx_fp8_path,
        "calibration_data": calibration_data,
        "calibration_method": "percentile",
        "percentile": percentile,
        "alpha": alpha,
        "use_external_data_format": True,
    }
    if quantize_mha:
        quantize_kwargs["quantize_mha"] = True

    try:
        modelopt_quantize(onnx_opt_path, **quantize_kwargs)
    except TypeError as e:
        # Older nvidia-modelopt versions may not support alpha / quantize_mha.
        # Retry with base parameters only.
        logger.warning(f"[FP8] Retrying without alpha/quantize_mha (API error: {e})")
        quantize_kwargs.pop("alpha", None)
        quantize_kwargs.pop("quantize_mha", None)
        modelopt_quantize(onnx_opt_path, **quantize_kwargs)
    finally:
        _onnx.ModelProto.ByteSize = _orig_byte_size  # Restore original method

    if not os.path.exists(onnx_fp8_path):
        raise RuntimeError(
            f"[FP8] Quantization completed but output file not found: {onnx_fp8_path}"
        )

    output_size_mb = os.path.getsize(onnx_fp8_path) / (1024 * 1024)
    ratio = output_size_mb / input_size_mb if input_size_mb > 0 else 0
    logger.info(
        f"[FP8] Quantization complete: {input_size_mb:.0f} MB → {output_size_mb:.0f} MB "
        f"(ratio: {ratio:.2f}x)"
    )
