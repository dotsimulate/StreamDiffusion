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


def _restore_dynamic_axes(onnx_fp8_path: str, model_data) -> None:
    """Restore dynamic dim_param symbols in FP8 ONNX after ModelOpt quantization.

    ModelOpt's override_shapes replaces dim_param with static dim_value for
    calibration. TRT requires dynamic dims (dim_param) on inputs/outputs to
    accept optimization profiles (min/opt/max ranges). This reads the original
    dynamic_axes from model_data and restores them in the FP8 ONNX.

    Uses load_external_data=False so only the small protobuf is loaded/modified,
    leaving the ~23GB external weight file untouched.
    """
    import onnx

    try:
        dynamic_axes = model_data.get_dynamic_axes()
    except Exception as e:
        logger.warning(f"[FP8] Could not get dynamic_axes from model_data: {e}. Skipping restore.")
        return

    if not dynamic_axes:
        logger.warning("[FP8] dynamic_axes is empty — skipping dynamic dim restore.")
        return

    model = onnx.load(onnx_fp8_path, load_external_data=False)

    restored_count = 0
    for graph_input in model.graph.input:
        name = graph_input.name
        if name not in dynamic_axes:
            continue
        axes = dynamic_axes[name]
        dims = graph_input.type.tensor_type.shape.dim
        for dim_idx, symbolic_name in axes.items():
            if dim_idx < len(dims):
                dim = dims[dim_idx]
                dim.ClearField("dim_value")
                dim.dim_param = symbolic_name
                restored_count += 1

    for graph_output in model.graph.output:
        name = graph_output.name
        if name not in dynamic_axes:
            continue
        axes = dynamic_axes[name]
        dims = graph_output.type.tensor_type.shape.dim
        for dim_idx, symbolic_name in axes.items():
            if dim_idx < len(dims):
                dim = dims[dim_idx]
                dim.ClearField("dim_value")
                dim.dim_param = symbolic_name
                restored_count += 1

    if restored_count == 0:
        logger.warning("[FP8] No dynamic dimensions restored — graph inputs may already be dynamic.")
        return

    # Save only the protobuf (weight data stays in existing external file).
    # load_external_data=False keeps tensor data_location=EXTERNAL references intact,
    # so onnx.save() writes a small protobuf that still points to the existing _data file.
    onnx.save(model, onnx_fp8_path)
    logger.info(
        f"[FP8] Restored {restored_count} dynamic dimensions in {os.path.basename(onnx_fp8_path)}"
    )


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
                # shape = (2, cache_maxframes, kvo_calib_batch, seq_len, hidden_dim)
                # dim[0]=2: K/V pair (must match ONNX trace, which always uses 2).
                # dim[2]: Must equal sample's batch dimension (effective_batch = 2 * opt_batch_size)
                # because both share the ONNX dynamic axis "2B". Using a different value
                # causes Concat dimension mismatches in attention layers during calibration.
                # Zeros = cold cache. Conservative but avoids over-fitting calibration
                # ranges to cached-attention activation patterns.
                idx = int(name.rsplit("_", 1)[-1])
                if idx < len(kvo_cache_shapes):
                    seq_len, hidden_dim = kvo_cache_shapes[idx]
                    kvo_calib_batch = effective_batch  # Must match sample batch (ONNX axis "2B")
                    batch_data[name] = np.zeros(
                        (2, cache_maxframes, kvo_calib_batch, seq_len, hidden_dim),
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
    calibration_data: Optional[List[Dict[str, np.ndarray]]] = None,
    quantize_mha: bool = False,
    percentile: float = 1.0,
    alpha: float = 0.8,
    model_data=None,
    opt_batch_size: int = 1,
    opt_image_height: int = 512,
    opt_image_width: int = 512,
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
        calibration_data: Unused. Kept for backward compatibility.
        quantize_mha: Enable FP8 quantization of multi-head attention ops.
                      Kept False — MHA analysis via ORT inference adds ~3 hours to build.
                      Non-MHA ops (Conv, Gemm, MatMul outside MHA) are still FP8.
        percentile: Unused. Kept for backward compatibility (entropy calibration
                    does not use percentile clipping).
        alpha: SmoothQuant alpha — balances quantization difficulty between
               activations (alpha→0) and weights (alpha→1). 0.8 is optimal
               for transformer attention layers.
        model_data: UNet BaseModel instance for building calibration_shapes.
                    If None, RandomDataProvider defaults all dynamic dims to 1.
        opt_batch_size: Optimal batch size from TRT profile.
        opt_image_height: Optimal image height in pixels.
        opt_image_width: Optimal image width in pixels.
    """
    try:
        from modelopt.onnx.quantization import quantize as modelopt_quantize
    except ImportError as e:
        raise ImportError(
            "nvidia-modelopt is required for FP8 quantization. "
            "Install with: pip install 'nvidia-modelopt[onnx]'"
        ) from e

    # Enable verbose ORT logging so Memcpy node details are visible before the
    # summary warning. Severity 1 = INFO (shows per-node placement decisions).
    try:
        import onnxruntime as _ort
        _ort.set_default_logger_severity(1)
        logger.info("[FP8] ORT log_severity_level set to 1 (INFO) for Memcpy diagnostics")
    except Exception:
        pass

    input_size_mb = os.path.getsize(onnx_opt_path) / (1024 * 1024)
    logger.info(f"[FP8] Starting ONNX FP8 quantization")
    logger.info(f"[FP8]   Input:  {onnx_opt_path} ({input_size_mb:.0f} MB)")
    logger.info(f"[FP8]   Output: {onnx_fp8_path}")
    logger.info(f"[FP8]   Config: quantize_mha={quantize_mha}, calibration=entropy, alpha={alpha}")
    logger.info(f"[FP8]   Calibration: RandomDataProvider with calibration_shapes (model_data={'provided' if model_data is not None else 'none'})")

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

    # Ensure NVIDIA DLLs (cuDNN, cuBLAS, CUDA runtime) are on PATH so modelopt's
    # ORT sessions can use CUDA/TensorRT EPs instead of CPU EP (which is stricter
    # about mixed-precision Cast nodes and fails on FP16 models).
    _nvidia_pkg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), os.pardir, "venv", "Lib",
        "site-packages", "nvidia")
    _nvidia_pkg_dir = os.path.normpath(_nvidia_pkg_dir)
    if not os.path.isdir(_nvidia_pkg_dir):
        # Fallback: find via importlib
        try:
            import nvidia.cudnn
            _nvidia_pkg_dir = os.path.dirname(os.path.dirname(nvidia.cudnn.__file__))
        except ImportError:
            _nvidia_pkg_dir = None

    if _nvidia_pkg_dir and os.path.isdir(_nvidia_pkg_dir):
        _bin_dirs = []
        for _subpkg in ("cudnn", "cublas", "cuda_runtime", "cufft", "curand"):
            _bdir = os.path.join(_nvidia_pkg_dir, _subpkg, "bin")
            if os.path.isdir(_bdir) and _bdir not in os.environ.get("PATH", ""):
                _bin_dirs.append(_bdir)
        if _bin_dirs:
            os.environ["PATH"] = os.pathsep.join(_bin_dirs) + os.pathsep + os.environ.get("PATH", "")
            logger.info(f"[FP8] Added {len(_bin_dirs)} NVIDIA DLL dirs to PATH")

    # Build calibration_shapes string for modelopt's RandomDataProvider.
    # RandomDataProvider calls _get_tensor_shape() which sets ALL dynamic dims to 1.
    # For a 512x512 UNet, sample becomes (1,4,1,1) instead of (2,4,64,64), causing
    # spatial dimension mismatches at UNet skip-connection Concat nodes (up_blocks).
    # calibration_shapes overrides _get_tensor_shape() per input — only specified
    # inputs bypass the default-to-1 fallback.
    #
    # Format: "input0:d0xd1x...,input1:d0xd1x..." (modelopt parse_shapes_spec format)
    calibration_shapes_str: Optional[str] = None
    if model_data is not None:
        latent_h = opt_image_height // 8
        latent_w = opt_image_width // 8
        effective_batch = 2 * opt_batch_size
        text_maxlen = getattr(model_data, "text_maxlen", 77)
        embedding_dim = getattr(model_data, "embedding_dim", 2048)
        # Use cache_maxframes=1 for calibration. The attention processor does:
        #   kvo_cache[0] → (cache_maxframes, batch, S, H)
        #   .transpose(0,1).flatten(1,2) → (batch, cache_maxframes*S, H)
        # With cache_maxframes=4, ONNX shape-computation nodes create Concat ops
        # that mix dim=4 (cache_maxframes) with dim=2 (batch), causing Concat axis
        # mismatch errors in ORT. cache_maxframes=1 is valid (within TRT profile
        # min range) and avoids the conflict. FP8 only needs valid activation ranges.
        calib_cache_maxframes = 1
        kvo_cache_shapes = getattr(model_data, "kvo_cache_shapes", [])
        num_ip_layers = getattr(model_data, "num_ip_layers", 1)
        control_inputs = getattr(model_data, "control_inputs", {})
        kvo_calib_batch = effective_batch  # Must match sample batch (ONNX axis "2B")

        shape_parts = []
        try:
            input_names = model_data.get_input_names()
        except Exception:
            input_names = []

        for name in input_names:
            if name == "sample":
                shape_parts.append(f"{name}:{effective_batch}x4x{latent_h}x{latent_w}")
            elif name == "timestep":
                shape_parts.append(f"{name}:{effective_batch}")
            elif name == "encoder_hidden_states":
                shape_parts.append(f"{name}:{effective_batch}x{text_maxlen}x{embedding_dim}")
            elif name == "ipadapter_scale":
                shape_parts.append(f"{name}:{num_ip_layers}")
            elif name.startswith("input_control_") and name in control_inputs:
                spec = control_inputs[name]
                shape_parts.append(
                    f"{name}:{effective_batch}x{spec['channels']}x{spec['height']}x{spec['width']}"
                )
            elif name.startswith("kvo_cache_in_"):
                idx = int(name.rsplit("_", 1)[-1])
                if idx < len(kvo_cache_shapes):
                    seq_len, hidden_dim = kvo_cache_shapes[idx]
                    shape_parts.append(
                        f"{name}:2x{calib_cache_maxframes}x{kvo_calib_batch}x{seq_len}x{hidden_dim}"
                    )

        if shape_parts:
            calibration_shapes_str = ",".join(shape_parts)
            logger.info(
                f"[FP8] calibration_shapes: {len(shape_parts)} inputs "
                f"(sample={effective_batch}x4x{latent_h}x{latent_w}, "
                f"kvo={len([p for p in shape_parts if 'kvo_cache_in' in p])} caches "
                f"calib_frames={calib_cache_maxframes})"
            )
    else:
        logger.warning(
            "[FP8] model_data not provided — RandomDataProvider will default all "
            "dynamic dims to 1. UNet Concat nodes may fail for non-trivial models."
        )

    quantize_kwargs = {
        "quantize_mode": "fp8",
        "output_path": onnx_fp8_path,
        # entropy: minimizes KL divergence to find optimal clipping point for each tensor.
        # Better than percentile=1.0 (no clipping) which allows outliers to stretch the
        # quantization range, reducing precision for the bulk of activations.
        "calibration_method": "entropy",
        "alpha": alpha,
        "use_external_data_format": True,
        # override_shapes replaces dynamic dims in the ONNX model itself with static
        # values BEFORE any ORT sessions (MHA analysis or calibration) are created.
        # Without this, ORT's internal shape inference with dynamic dims causes
        # Concat failures (e.g. KVO cache dims vs sample batch dims).
        # calibration_shapes additionally tells RandomDataProvider what shapes to
        # generate for the calibration data.
        "override_shapes": calibration_shapes_str,
        "calibration_shapes": calibration_shapes_str,
        # Use default EPs ["cpu","cuda:0","trt"] — CPU-only would fail on this FP16 SDXL
        # model because ORT's mandatory CastFloat16Transformer inserts Cast nodes that
        # conflict with existing Cast nodes in the upsampler conv.
        # disable_mha_qdq=True: skip MHA pattern analysis (avoids 3-hour ORT inference
        # pass over the full model graph). Non-MHA ops (Conv, Gemm, MatMul outside MHA)
        # still get FP8 Q/DQ nodes via the normal KGEN/CASK path.
        "disable_mha_qdq": not quantize_mha,
        # calibrate_per_node: calibrate one node at a time to reduce peak VRAM during
        # calibration. Essential for large UNets (83 inputs, 7993 nodes) to avoid OOM.
        "calibrate_per_node": True,
    }

    try:
        modelopt_quantize(onnx_opt_path, **quantize_kwargs)
    except TypeError as e:
        # Older nvidia-modelopt versions may not support newer kwargs.
        # Strip down to base parameters and retry.
        logger.warning(f"[FP8] Retrying with reduced kwargs (TypeError: {e})")
        for _k in ("alpha", "disable_mha_qdq", "calibrate_per_node"):
            quantize_kwargs.pop(_k, None)
        modelopt_quantize(onnx_opt_path, **quantize_kwargs)
    except Exception as e:
        # MHA analysis (disable_mha_qdq=False) requires an ORT inference run that
        # fails with KVO cached attention models. Retry with disable_mha_qdq=True
        # to skip the ORT session entirely — MHA layers use FP16, rest uses FP8.
        if not quantize_kwargs.get("disable_mha_qdq", True):
            # Delete intermediate files written during the failed attempt to free
            # disk space before the retry (each set is ~23GB for SDXL-scale models).
            _base = os.path.splitext(onnx_opt_path)[0]  # strip .onnx
            for _suffix in (
                "_static.onnx", "_static.onnx_data",          # from override_shapes
                "_named.onnx", "_named.onnx_data",
                "_named_extended.onnx", "_named_extended.onnx_data",
                "_ir10.onnx", "_ir10.onnx_data",
                "_static_named.onnx", "_static_named.onnx_data",
                "_static_ir10.onnx", "_static_ir10.onnx_data",
            ):
                _f = _base + _suffix
                if os.path.exists(_f):
                    os.remove(_f)
                    logger.info(f"[FP8] Cleaned up intermediate: {os.path.basename(_f)}")
            logger.warning(
                f"[FP8] MHA analysis failed ({type(e).__name__}: {e}). "
                "Retrying with disable_mha_qdq=True (MHA layers will use FP16 precision)."
            )
            quantize_kwargs["disable_mha_qdq"] = True
            modelopt_quantize(onnx_opt_path, **quantize_kwargs)
        else:
            raise
    finally:
        _onnx.ModelProto.ByteSize = _orig_byte_size  # Restore original method
        try:
            import onnxruntime as _ort
            _ort.set_default_logger_severity(2)  # Restore to WARNING
        except Exception:
            pass

    if not os.path.exists(onnx_fp8_path):
        raise RuntimeError(
            f"[FP8] Quantization completed but output file not found: {onnx_fp8_path}"
        )

    # --- Restore dynamic axes ---
    # ModelOpt's override_shapes baked static dim_value into graph inputs for calibration.
    # TRT needs dynamic dim_param on inputs/outputs to accept optimization profiles.
    if model_data is not None:
        try:
            _restore_dynamic_axes(onnx_fp8_path, model_data)
        except Exception as restore_err:
            logger.warning(
                f"[FP8] Failed to restore dynamic axes: {restore_err}. "
                "TRT engine build may fail with static shape profile mismatch."
            )

    output_size_mb = os.path.getsize(onnx_fp8_path) / (1024 * 1024)
    ratio = output_size_mb / input_size_mb if input_size_mb > 0 else 0
    logger.info(
        f"[FP8] Quantization complete: {input_size_mb:.0f} MB → {output_size_mb:.0f} MB "
        f"(ratio: {ratio:.2f}x)"
    )
