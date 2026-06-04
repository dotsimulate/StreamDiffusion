"""
FP8 Quantization for StreamDiffusion TensorRT UNet engine.

ONNX-level approach: export a plain FP16 ONNX first, then inject native
FLOAT8E4M3FN Q/DQ nodes via modelopt.onnx.quantization.quantize with
real activation tensors captured from the diffusers pipeline.

Why this is better than the previous PyTorch nn.Module path:
- modelopt's torch path defaults trt_high_precision_dtype="Float" (FP32),
  which inserts Cast(FP16→FP32) before every Q node and stores all weight
  initializers as FP32 → 9 GB ONNX on SDXL UNet.
- The nn.Module path required generate_fp8_scales to rewrite FP8(4,3)→INT8(8)
  because torch.onnx.export's ScaledE4M3Function symbolic corrupts the graph
  for attention/embedding quantizers → INT8 kernels, not FP8 GEMMs.
- The ONNX-level path keeps weights in FP16 (high_precision_dtype="fp16") and
  emits native FLOAT8E4M3FN Q/DQ → ~2.5 GB ONNX, true FP8 tensor-core kernels.

Requirements:
    nvidia-modelopt[onnx] >= 0.19.0
    onnxruntime-gpu >= 1.17  (ORT CUDA EP for calibration)
    TensorRT >= 10.0 (FP8 E4M3 hardware support, STRONGLY_TYPED build flag)
    RTX 4090+ (Ada Lovelace, compute capability 8.9)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


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


def capture_calibration_data(
    pipe,
    prompts: List[str],
    num_inference_steps: int = 20,
    save_path: Optional[str] = None,
    batch_size: int = 1,
    guidance_scale: float = 7.5,
    onnx_path: Optional[str] = None,
    use_cached_attn: bool = False,
    use_controlnet: bool = False,
    num_ip_layers: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Capture UNet input activations from a real diffusers pipeline run.

    Registers a forward pre-hook on pipe.unet that records inputs across all
    denoising timesteps and all calibration prompts. Returns a calibration_data
    dict compatible with modelopt.onnx.quantization.quantize(calibration_data=...).

    If LoRAs are active they are baked into the captured activations, which is
    correct — quantization should see the same distribution as inference.

    Args:
        pipe: StableDiffusionPipeline or StableDiffusionXLPipeline.
        prompts: Calibration texts (32–128 recommended).
        num_inference_steps: Denoising steps per prompt. 20 for SDXL, 4 for Turbo.
        save_path: Optional path to write calib_data.npz for caching between builds.
        batch_size: Prompts per pipe() call.
        guidance_scale: CFG scale (affects conditional/unconditional stacking).

    Returns:
        Dict mapping UNet input names to np.ndarray arrays of shape [N, ...].
    """
    import torch

    _KEY_MAP = {0: "sample", 1: "timestep", 2: "encoder_hidden_states"}
    _SDXL_COND_KEYS = ["text_embeds", "time_ids"]

    # builder.py moves pipe.unet to CPU after ONNX export to free GPU during
    # optimize. Move it back to CUDA for calibration; restore on exit so the
    # next build stage starts from the same VRAM state.
    _unet_orig_device = next(pipe.unet.parameters()).device
    if _unet_orig_device.type != "cuda":
        pipe.unet.to("cuda")

    captured: Dict[str, list] = {}

    def _to_npy(t):
        # Keep dtype as-is (FP16 model → FP16 captures). modelopt's max-abs
        # calibration does not need FP32; FP32 upcast doubles transfer bandwidth.
        # atleast_1d: SDXL passes timestep as a 0-dim scalar tensor in single-
        # prompt calls; np.concatenate(axis=0) requires at least 1 axis.
        return np.atleast_1d(t.detach().cpu().numpy())

    def _hook(module, args, kwargs):
        # SDXL pipeline calls unet(sample, t, encoder_hidden_states=..., added_cond_kwargs=...)
        # — encoder_hidden_states arrives as a kwarg, not positional. Fall through to kwargs.
        for idx, key in _KEY_MAP.items():
            val = args[idx] if idx < len(args) else kwargs.get(key)
            if val is not None:
                captured.setdefault(key, []).append(_to_npy(val))
        added = kwargs.get("added_cond_kwargs") or {}
        if not added and len(args) > 3 and isinstance(args[3], dict):
            added = args[3]
        for key in _SDXL_COND_KEYS:
            if key in added and added[key] is not None:
                captured.setdefault(key, []).append(_to_npy(added[key]))

    handle = pipe.unet.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
            for i, batch in enumerate(batches):
                logger.info(f"[FP8] Capture batch {i + 1}/{len(batches)}: {batch[0][:60]}")
                try:
                    pipe(
                        prompt=batch if len(batch) > 1 else batch[0],
                        num_inference_steps=num_inference_steps,
                        output_type="latent",
                        guidance_scale=guidance_scale,
                    ).images
                except Exception as e:
                    logger.warning(f"[FP8] Capture batch {i + 1} failed ({type(e).__name__}): {e}. Skipping.")
    finally:
        handle.remove()
        if _unet_orig_device.type != "cuda":
            pipe.unet.to(_unet_orig_device)
            torch.cuda.empty_cache()

    if not captured:
        raise RuntimeError("[FP8] No UNet activations captured — check pipe.unet forward signature.")

    calib_data: Dict[str, np.ndarray] = {}
    for key, arrays in captured.items():
        stacked = np.concatenate(arrays, axis=0)
        calib_data[key] = stacked
        logger.info(f"[FP8] Captured '{key}': shape={stacked.shape}, dtype={stacked.dtype}")

    # Synthesize zero/one tensors for feature-specific ONNX inputs not captured by the
    # bare-pipe hook (kvo_cache_in_*, input_control_*, middle_control_*, ipadapter_scale).
    # These inputs feed Q/DQ-excluded layers (see _FEATURE_EXCLUDE_PATTERNS), so the
    # synthetic values only need to be shape-compatible — they never drive scale computation.
    if onnx_path and (use_cached_attn or use_controlnet or num_ip_layers):
        # modelopt's CalibrationDataProvider computes n_itr = first_input.shape[0] /
        # first_input_onnx_dim_0, then np.array_split(arr, n_itr, axis=0) for EVERY
        # input. Each chunk must satisfy the ONNX-declared dim 0 (fixed or dynamic).
        # Bound captured leading dims so n_itr stays small — synthesized KV-cache memory
        # scales as n_itr × per-layer-size × n_layers, which blows up at large n_itr.
        _MAX_CALIB_ROWS = 8
        for _k in list(calib_data.keys()):
            if calib_data[_k].shape[0] > _MAX_CALIB_ROWS:
                calib_data[_k] = calib_data[_k][:_MAX_CALIB_ROWS]

        try:
            _specs = _read_onnx_input_specs(onnx_path)

            # Reconcile captured tensors with ONNX-declared static dims. The bare-pipe
            # forward hook captures the diffusers UNet *before* feature wrappers run,
            # so dims that wrappers reshape (e.g. IPA's UnifiedExportWrapper concatenates
            # 4 image tokens onto encoder_hidden_states → seq_len 77 → 81) won't match
            # the exported ONNX. Pad with zeros if undersized, trim if oversized, on any
            # static (non-dynamic, non-leading) axis. Padding zeros is benign for max-abs
            # calibration — the zero-region contributes no signal to scale computation.
            for _name, (_, _expected_dims) in _specs.items():
                if _name not in calib_data:
                    continue
                _arr = calib_data[_name]
                _resized = False
                for _axis, _expected in enumerate(_expected_dims):
                    if _expected is None or _axis == 0 or _axis >= _arr.ndim:
                        continue
                    if _arr.shape[_axis] == _expected:
                        continue
                    if _arr.shape[_axis] < _expected:
                        _pw = [(0, 0)] * _arr.ndim
                        _pw[_axis] = (0, _expected - _arr.shape[_axis])
                        _arr = np.pad(_arr, _pw, mode="constant")
                    else:
                        _slc = [slice(None)] * _arr.ndim
                        _slc[_axis] = slice(0, _expected)
                        _arr = _arr[tuple(_slc)]
                    _resized = True
                if _resized:
                    calib_data[_name] = _arr
                    logger.info(f"[FP8] Reshaped captured '{_name}' to ONNX dims: shape={_arr.shape}")

            # Mirror CalibrationDataProvider's n_itr derivation (calib_utils.py:90):
            # first ONNX-declared input that appears in calib_data drives the count.
            _present = [n for n in _specs if n in calib_data]
            if _present:
                _first = _present[0]
                _first_d0 = max(1, (_specs[_first][1][0] or 1))
                _n_itr = max(1, calib_data[_first].shape[0] // _first_d0)
            else:
                _n_itr = 1

            for name, (dtype, dims) in _specs.items():
                if name in calib_data:
                    continue
                # Resolve symbolic dims to 1. dim 0 is the per-chunk shape ORT sees.
                resolved = [d if d is not None else 1 for d in dims]
                # ipadapter_scale: ONNX dim 0 is dynamic ("L_ip") but the exported
                # graph has hardcoded Gather(scale_vec, idx=k) for k=0..num_ip_layers-1
                # at every IPA layer. A length-1 chunk would OOB at idx≥1 during
                # modelopt's _exclude_matmuls_by_inference ORT probe. Force per-chunk
                # length to num_ip_layers so every Gather sees the expected vector.
                per_step_d0 = num_ip_layers if name == "ipadapter_scale" and num_ip_layers > 0 else resolved[0]
                # Total leading dim = n_itr × per-chunk dim 0 so every split chunk
                # has exactly the ONNX-declared leading dim (fixed Q+K=2 for kvo_cache,
                # fixed 2 for control inputs, num_ip_layers for ipadapter_scale).
                arr_shape = [_n_itr * per_step_d0] + list(resolved[1:])
                arr = (
                    np.ones(arr_shape, dtype=dtype) if name == "ipadapter_scale" else np.zeros(arr_shape, dtype=dtype)
                )
                calib_data[name] = arr
                logger.info(
                    f"[FP8] Synthesized '{name}': shape={arr.shape}, dtype={arr.dtype} "
                    f"(n_itr={_n_itr}, per-step-dim0={per_step_d0})"
                )
        except Exception as e:
            logger.warning(
                f"[FP8] Synthetic input generation failed: {e}. Missing inputs will be caught during quantization."
            )

    if save_path:
        # Uncompressed: zlib barely compresses random-ish FP16 activations and is
        # single-threaded — savings are <5 % on multi-GB calibration sets.
        # Atomic write: if the build crashes mid-save, no partial file is left
        # behind for a future run to load and corrupt calibration.
        # Save via a file handle so np.savez does not auto-append ".npz" to tmp_path.
        tmp_path = save_path + ".tmp"
        with open(tmp_path, "wb") as _f:
            np.savez(_f, **calib_data)
        os.replace(tmp_path, save_path)
        logger.info(f"[FP8] Saved calibration data: {save_path} ({os.path.getsize(save_path) / 1e6:.1f} MB)")

    return calib_data


def load_calibration_data(npz_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Load previously-saved calibration data from a .npz file.
    Returns None (and deletes the file) if loading fails.
    """
    if not os.path.exists(npz_path):
        return None
    try:
        data = dict(np.load(npz_path))
        logger.info(f"[FP8] Loaded calibration data from {npz_path} ({len(data)} tensors)")
        return data
    except Exception as e:
        logger.warning(f"[FP8] Cannot load calibration data from {npz_path}: {e}. Will recapture.")
        try:
            os.remove(npz_path)
        except OSError:
            pass
        return None


# modelopt's expand_node_names_from_patterns feeds these straight into re.match,
# so they're regex (not glob) — leading `*` would raise "nothing to repeat".
# `.*time_emb.*` already covers `time_embedding` since `time_emb` is a substring.
_DEFAULT_EXCLUDE_PATTERNS = [r".*time_emb.*", r".*add_emb.*"]

# Feature-specific Q/DQ exclusions applied only when the corresponding feature flag
# is active — keeps plain-UNet Q/DQ counts unaffected.
_FEATURE_EXCLUDE_PATTERNS = {
    "cached_attn": [r".*kvo_cache.*"],
    "feature_injection": [r".*fio_cache.*", r".*fi_strength.*", r".*fi_threshold.*"],
    "controlnet": [r".*down_block_additional_residuals.*", r".*mid_block_additional_residual.*"],
    "ipadapter": [r".*to_k_ip.*", r".*to_v_ip.*", r".*to_out_ip.*"],
}


def _read_onnx_input_specs(onnx_path: str) -> Dict[str, tuple]:
    """Return {name: (np_dtype, shape)} from ONNX graph inputs. Shape dims are int or None."""
    import onnx as _onnx
    from onnx.helper import tensor_dtype_to_np_dtype as _onnx_to_np

    m = _onnx.load(onnx_path, load_external_data=False)
    result: Dict[str, tuple] = {}
    for inp in m.graph.input:
        tt = inp.type.tensor_type
        dtype = _onnx_to_np(tt.elem_type)
        dims = []
        if tt.HasField("shape"):
            for d in tt.shape.dim:
                dims.append(d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else None)
        result[inp.name] = (dtype, dims)
    return result


def _assert_finite_qdq_scales(onnx_path: str) -> None:
    """Raise RuntimeError if any FP8 Q/DQ scale in onnx_path is non-finite.

    Root cause: modelopt/onnx/quantization/fp8.py computes
        np_fp8_scale = (np_scale * 448.0) / 127.0
    in the source dtype (FP16). An INT8 amax > ~18,500 overflows to +inf;
    the resulting Q/DQ scale produces zero output at inference.

    Checks both initializer scales AND Constant-node scales (the latter appear
    on some residual-add Q nodes injected by modelopt on SDXL UNet).
    On failure: raises RuntimeError listing the first 5 offending node names
    and directs the user to extend _DEFAULT_EXCLUDE_PATTERNS and delete the
    cached .fp8.onnx so modelopt reruns without those layers.
    """
    import onnx as _onnx
    from onnx import numpy_helper as _numpy_helper

    model = _onnx.load(onnx_path, load_external_data=True)
    graph = model.graph
    init_map = {init.name: init for init in graph.initializer}
    const_map: dict = {}
    for node in graph.node:
        if node.op_type == "Constant" and node.output:
            attr = {a.name: a for a in node.attribute}
            if "value" in attr:
                const_map[node.output[0]] = _numpy_helper.to_array(attr["value"].t)

    bad: list = []
    for node in graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if len(node.input) < 2:
            continue
        scale_name = node.input[1]
        if scale_name in init_map:
            arr = _numpy_helper.to_array(init_map[scale_name]).flatten().astype(np.float64)
        elif scale_name in const_map:
            arr = const_map[scale_name].flatten().astype(np.float64)
        else:
            continue
        if not np.isfinite(arr).all():
            bad.append(node.name or scale_name)

    if bad:
        names = ", ".join(bad[:5]) + ("..." if len(bad) > 5 else "")
        raise RuntimeError(
            f"[FP8] Non-finite Q/DQ scale in {len(bad)} node(s): {names}. "
            f"Add the offending layer substring(s) to _DEFAULT_EXCLUDE_PATTERNS "
            f"in fp8_quantize.py, delete the cached .fp8.onnx, and rebuild. "
            f"Diagnostic: modelopt/onnx/quantization/fp8.py overflow when INT8 amax > ~18500."
        )


def quantize_onnx_fp8(
    onnx_path: str,
    output_path: str,
    calibration_data: Dict[str, np.ndarray],
    nodes_to_exclude: Optional[List[str]] = None,
    disable_mha_qdq: bool = True,
    use_cached_attn: bool = False,
    use_feature_injection: bool = False,
    use_controlnet: bool = False,
    num_ip_layers: int = 0,
) -> None:
    """
    Inject native FLOAT8E4M3FN Q/DQ nodes into a FP16 ONNX model via ORT.

    The output ONNX feeds directly into Engine._build_fp8 (STRONGLY_TYPED path).

    Args:
        onnx_path: Input FP16 ONNX (may use external data format).
        output_path: Output path for the FP8-quantized ONNX.
        calibration_data: Dict[str, np.ndarray] from capture_calibration_data().
        nodes_to_exclude: ONNX node name patterns to skip quantization on.
                          Defaults to time/add embedding layers.
        disable_mha_qdq: Skip MHA-specific Q/DQ injection (default True for Ada).
                         General FP8 calibration still inserts Q/DQ on all attention
                         MatMuls; TRT Myelin fuses them into _gemm_mha_v2 FP8 kernels.
    """
    try:
        from modelopt.onnx.quantization import quantize as modelopt_quantize
    except ImportError as e:
        raise ImportError(
            "nvidia-modelopt[onnx] is required for ONNX-level FP8 quantization.\n"
            "Install with: pip install 'nvidia-modelopt[onnx]>=0.19.0'\n"
            "Also ensure onnxruntime-gpu >= 1.17 is installed."
        ) from e

    # ORT CUDA EP requires cuDNN DLLs — PyTorch ships cuDNN under torch/lib on Windows.
    # Best-effort: failing here just lets ORT surface its own loader error downstream.
    try:
        import torch as _torch

        _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(_torch_lib) and _torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception as e:
        logger.debug(f"[FP8] cuDNN PATH setup skipped: {e}")

    # Flush pending GPU work before ORT CUDA EP claims VRAM. A failure here usually
    # signals a wedged CUDA context — surface at debug so it's not invisible.
    try:
        import torch as _t

        if _t.cuda.is_available():
            _t.cuda.synchronize()
            _t.cuda.empty_cache()
            import gc as _gc

            _gc.collect()
    except Exception as e:
        logger.debug(f"[FP8] pre-quantize CUDA flush skipped: {e}")

    if nodes_to_exclude is None:
        nodes_to_exclude = list(_DEFAULT_EXCLUDE_PATTERNS)
        if use_cached_attn:
            nodes_to_exclude.extend(_FEATURE_EXCLUDE_PATTERNS["cached_attn"])
        if use_feature_injection:
            nodes_to_exclude.extend(_FEATURE_EXCLUDE_PATTERNS["feature_injection"])
        if use_controlnet:
            nodes_to_exclude.extend(_FEATURE_EXCLUDE_PATTERNS["controlnet"])
        if num_ip_layers > 0:
            nodes_to_exclude.extend(_FEATURE_EXCLUDE_PATTERNS["ipadapter"])

    # The optimized ONNX may expose fewer inputs than capture_calibration_data
    # records (e.g. SDXL UnifiedExportWrapper hides text_embeds/time_ids inside
    # the graph) and may declare different dtypes than the captured tensors —
    # e.g. SDXL exports `sample` as FP32 even though the unet runs FP16.
    # modelopt's CalibrationDataProvider asserts strict count match and ORT's
    # inference probe rejects dtype mismatches, so filter+cast accordingly.
    _specs = _read_onnx_input_specs(onnx_path)  # {name: (dtype, dims)}
    _onnx_inputs = {k: v[0] for k, v in _specs.items()}
    _dropped = set(calibration_data.keys()) - set(_onnx_inputs)
    if _dropped:
        logger.info(f"[FP8] Dropping calibration keys not exposed by ONNX: {sorted(_dropped)}")
        calibration_data = {k: v for k, v in calibration_data.items() if k in _onnx_inputs}
    _missing = set(_onnx_inputs) - set(calibration_data.keys())
    if _missing:
        raise RuntimeError(f"[FP8] Calibration data missing required ONNX inputs: {sorted(_missing)}")
    for _k, _expected in _onnx_inputs.items():
        if calibration_data[_k].dtype != _expected:
            logger.info(f"[FP8] Casting calibration '{_k}': {calibration_data[_k].dtype} → {_expected}")
            calibration_data[_k] = calibration_data[_k].astype(_expected)

    # Per-input tile: target rows = n_itr × resolved_dim0(name) so every input
    # splits into exactly n_itr chunks of shape (resolved_dim0, ...).
    # Mirrors modelopt CalibrationDataProvider: symbolic dims → 1, static dims kept.
    # Naïve _max_rows tile breaks kvo_cache_in_* (ONNX dim0=2 static) by pumping
    # sample to 2×_n_itr rows, causing modelopt to split kvo into (1,...) chunks.
    import math as _math

    _resolved_dim0 = {name: max(1, (_specs[name][1][0] or 1)) for name in calibration_data}
    _n_itr = max(arr.shape[0] // _resolved_dim0[name] for name, arr in calibration_data.items())
    _n_itr = max(1, _n_itr)
    for _k in list(calibration_data.keys()):
        _arr = calibration_data[_k]
        _target_rows = _n_itr * _resolved_dim0[_k]
        if _arr.shape[0] != _target_rows:
            _repeats = _math.ceil(_target_rows / max(1, _arr.shape[0]))
            calibration_data[_k] = np.tile(_arr, (_repeats,) + (1,) * (_arr.ndim - 1))[:_target_rows]
            logger.info(
                f"[FP8] Tiled '{_k}' {_arr.shape[0]} → {_target_rows} rows "
                f"(n_itr={_n_itr} × resolved_dim0={_resolved_dim0[_k]})"
            )

    import inspect as _inspect

    _params = set(_inspect.signature(modelopt_quantize).parameters.keys())

    kwargs = {
        "onnx_path": onnx_path,
        "quantize_mode": "fp8",
        "output_path": output_path,
        "calibration_method": "max",
        "calibration_eps": ["cuda:0"],
        "calibration_data": calibration_data,
        "high_precision_dtype": "fp16",
        "use_external_data_format": True,
        "calibrate_per_node": False,
        "disable_mha_qdq": disable_mha_qdq,
        "nodes_to_exclude": nodes_to_exclude,
    }
    # enable_gemv_detection_for_trt was removed in modelopt >= 0.42
    if "enable_gemv_detection_for_trt" in _params:
        kwargs["enable_gemv_detection_for_trt"] = False

    logger.info(
        f"[FP8] ONNX-level FP8 quantization: {os.path.basename(onnx_path)}"
        f" → {os.path.basename(output_path)}"
        f" ({next(iter(calibration_data.values())).shape[0]} calibration samples,"
        f" disable_mha_qdq={disable_mha_qdq})"
    )
    modelopt_quantize(**kwargs)

    if not os.path.exists(output_path):
        raise RuntimeError(f"[FP8] modelopt_quantize completed but output not found: {output_path}")

    _assert_finite_qdq_scales(output_path)

    size_mb = os.path.getsize(output_path) / (1024**2)
    logger.info(f"[FP8] FP8 ONNX written: {output_path} ({size_mb:.1f} MB)")
    if size_mb > 5000:
        logger.warning(
            f"[FP8] FP8 ONNX is unexpectedly large ({size_mb:.0f} MB > 5000 MB). "
            "FP32 Cast bloat may be active — check high_precision_dtype='fp16' is honored."
        )

    # Sentinel marker — only written after modelopt_quantize returns. The builder's
    # cache check looks for this file, so a crash mid-write leaves no false-positive.
    with open(output_path + ".ok", "w") as _f:
        _f.write("ok")
