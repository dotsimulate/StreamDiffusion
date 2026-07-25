import gc
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import *

import onnx
import torch

from .models.models import BaseModel
from .utilities import (
    BUILD_TRT_LOGGER,
    build_engine,
    export_onnx,
    optimize_onnx,
)

_build_logger = logging.getLogger(__name__)


class StageStatus:
    BUILT = "built"
    CACHED = "cached"
    FAILED = "failed"


def _onnx_cache_valid(path: str) -> bool:
    """Check a cached source ONNX is usable before reusing it, skipping export.

    A prior run killed mid-export (OOM, crash, manual kill) can leave a
    syntactically-valid-but-empty protobuf at onnx_path. os.path.exists() alone
    can't tell that apart from a real export, and a 0-node / opset-less graph
    makes polygraphy's constant folding blow up downstream with an opaque
    'NoneType' object has no attribute 'graph' (ORT symbolic shape inference
    refuses opset < 7 and returns None). Load structure-only (no external
    weight data) and reject anything that looks truncated.
    """
    try:
        if os.path.getsize(path) == 0:
            return False
        model = onnx.load(path, load_external_data=False)
        if len(model.graph.node) == 0:
            return False
        opset = max((o.version for o in model.opset_import), default=0)
        return opset >= 7
    except Exception:
        return False


def _write_build_stats(engine_path: str, stats: dict):
    """Append build stats to a JSON-lines file next to the engine directory."""
    try:
        engine_dir = Path(engine_path).parent
        # Write stats file inside the engine directory
        stats_file = engine_dir / "build_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        # Also append to the global build log in the engines root
        engines_root = engine_dir.parent
        global_log = engines_root / "build_log.jsonl"
        with open(global_log, "a") as f:
            f.write(json.dumps(stats) + "\n")
    except Exception as e:
        _build_logger.warning(f"Failed to write build stats: {e}")


def _run_fp8_stage(name: str, fn, stats: dict, allow_fallback: bool, engine_filename: str) -> bool:
    """Run an FP8 build stage with timing + fallback handling. Returns True on success."""
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        stats["stages"][name] = {"status": StageStatus.BUILT, "elapsed_s": round(elapsed, 2)}
        _build_logger.info(f"[BUILD] {name} ({engine_filename}): {elapsed:.1f}s")
        return True
    except Exception as err:
        elapsed = time.perf_counter() - t0
        stats["stages"][name] = {
            "status": StageStatus.FAILED,
            "elapsed_s": round(elapsed, 2),
            "error": str(err),
        }
        if allow_fallback:
            _build_logger.warning(f"[BUILD] {name} failed after {elapsed:.1f}s: {err}. Falling back to FP16.")
            return False
        raise RuntimeError(
            f"{name} failed: {err}.\n"
            "Set fp8_allow_fp16_fallback=True in TRT_PROFILES to silently fall back to FP16, "
            "or fix the error above."
        ) from err


def _check_fp8_disk_space(onnx_opt_path: str, allow_fallback: bool) -> bool:
    """Preflight disk-space check before FP8 ONNX quantization.

    ModelOpt keeps several full-size copies of the external-data ONNX weights on disk
    at once (opt, opt_named, opt_named_extended, fp8 output) plus calibration tensors.
    Require free space >= 5x the source ONNX's on-disk footprint (external data
    included), with a ~28 GB floor for SDXL-scale models, so we fail fast before the
    ~40 min export/optimize/calibrate pipeline instead of after it.
    """
    onnx_dir = os.path.dirname(onnx_opt_path)
    source_size = os.path.getsize(onnx_opt_path)
    weights_pb = os.path.join(onnx_dir, "weights.pb")
    if os.path.exists(weights_pb):
        source_size += os.path.getsize(weights_pb)

    required = max(5 * source_size, 28 * 1024**3)
    free = shutil.disk_usage(onnx_dir).free
    if free >= required:
        return True

    _build_logger.warning(
        f"[BUILD] Low disk space for FP8 quantization on {onnx_dir}: "
        f"{free / 1024**3:.1f} GB free, need ~{required / 1024**3:.1f} GB."
    )
    if allow_fallback:
        _build_logger.warning("[BUILD] Falling back to FP16 (fp8_allow_fp16_fallback=True).")
        return False
    raise RuntimeError(
        f"Insufficient disk space for FP8 quantization on {onnx_dir}: "
        f"{free / 1024**3:.1f} GB free, need ~{required / 1024**3:.1f} GB. "
        "Free up space, or set fp8_allow_fp16_fallback=True in TRT_PROFILES to build FP16 instead."
    )


def _cleanup_intermediates(engine_dir: str, fp8_ok: bool):
    """Delete intermediate ONNX/build artifacts, preserving .engine, .cache, calib_data.npz,
    build_stats.json, and (only when fp8_ok) the cached unet.fp8.onnx* artifact.

    Two-pass deletion handles Windows file locks (gc.collect releases Python handles).
    Runs from a `finally` block so it also fires when a build stage raises, instead of
    orphaning tens of GB of external-data ONNX copies on failure.
    """
    _keep_suffixes = (".engine", ".cache")
    _keep_exact = {"build_stats.json", "timing.cache", "calib_data.npz"}
    _to_delete = []
    for file in os.listdir(engine_dir):
        # Keep the FP8 quantized ONNX artifact only if quantization actually succeeded
        # (marked by the ".ok" sentinel) -- a partial file from a failed run must be swept.
        if fp8_ok and "fp8.onnx" in file:
            continue
        if file in _keep_exact or any(file.endswith(s) for s in _keep_suffixes):
            continue
        _to_delete.append(os.path.join(engine_dir, file))

    if not _to_delete:
        return

    _failed = []
    for fpath in _to_delete:
        try:
            os.remove(fpath)
        except OSError:
            _failed.append(fpath)

    # Release Python-held file handles (ONNX model refs), retry locked files.
    # Per-file poll with 50ms backoff instead of a single global sleep -- most
    # handles release within 1-2 retries on Windows; worst case ~0.5s same as before.
    if _failed:
        gc.collect()
        torch.cuda.empty_cache()
        _still_failed = []
        for fpath in _failed:
            _last_err = None
            for _attempt in range(10):
                try:
                    os.remove(fpath)
                    _last_err = None
                    break
                except OSError as _e:
                    _last_err = _e
                    time.sleep(0.05)
            if _last_err is not None:
                _still_failed.append(os.path.basename(fpath))
                _build_logger.warning(f"[BUILD] Could not delete temp file {os.path.basename(fpath)}: {_last_err}")
        if _still_failed:
            _build_logger.warning(
                f"[BUILD] {len(_still_failed)} intermediate files could not be cleaned. "
                f"Manual cleanup: delete all files except *.engine, calib_data.npz, unet.fp8.onnx from {engine_dir}"
            )
        cleaned = len(_to_delete) - len(_still_failed)
    else:
        cleaned = len(_to_delete)
    _build_logger.info(f"[BUILD] Cleaned {cleaned}/{len(_to_delete)} intermediate files")


def create_onnx_path(name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, name + (".opt" if opt else "") + ".onnx")


class EngineBuilder:
    def __init__(
        self,
        model: BaseModel,
        network: Any,
        device=torch.device("cuda"),
    ):
        self.device = device

        self.model = model
        self.network = network

    def build(
        self,
        onnx_path: str,
        onnx_opt_path: str,
        engine_path: str,
        opt_image_height: int = 512,
        opt_image_width: int = 512,
        opt_batch_size: Optional[int] = None,
        min_image_resolution: int = 256,
        max_image_resolution: int = 1024,
        build_enable_refit: bool = False,
        build_static_batch: bool = False,
        build_dynamic_shape: bool = True,
        onnx_opset: int = 17,
        force_engine_build: bool = False,
        force_onnx_export: bool = False,
        force_onnx_optimize: bool = False,
        fp8: bool = False,
        pipe_ref=None,
        calibration_prompts=None,
        calibration_steps: int = 20,
        fp8_guidance_scale: float = 7.5,
        fp8_allow_fp16_fallback: bool = False,
        fp8_use_cached_attn: bool = False,
        fp8_use_feature_injection: bool = False,
        fp8_use_controlnet: bool = False,
        fp8_num_ip_layers: int = 0,
        builder_optimization_level: Optional[int] = None,
        is_controlnet: bool = False,
        artifact_prefix: str = "unet",
    ):
        if opt_batch_size is None:
            raise ValueError("build() requires an explicit opt_batch_size")
        build_total_start = time.perf_counter()
        engine_name = Path(engine_path).parent.name
        engine_filename = Path(engine_path).name
        stats = {
            "engine_dir": engine_name,
            "engine_file": engine_filename,
            "build_start": datetime.now(timezone.utc).isoformat(),
            "opt_resolution": f"{opt_image_width}x{opt_image_height}",
            "dynamic_range": f"{min_image_resolution}-{max_image_resolution}" if build_dynamic_shape else "static",
            "batch_size": opt_batch_size,
            "stages": {},
        }

        # FP8 paths are resolved relative to the engine directory.
        # calib_data.npz: cached activations (survives engine rebuilds).
        # {prefix}.fp8.onnx: ONNX with native FLOAT8E4M3FN Q/DQ (also cached).
        engine_dir_early = os.path.dirname(engine_path)
        _calib_data_path = os.path.join(engine_dir_early, "calib_data.npz")
        _fp8_onnx_path = os.path.join(engine_dir_early, f"{artifact_prefix}.fp8.onnx")

        # --- ONNX Export ---
        if not force_onnx_export and os.path.exists(onnx_path) and _onnx_cache_valid(onnx_path):
            print(f"Found cached model: {onnx_path}")
            stats["stages"]["onnx_export"] = {"status": "cached"}
        else:
            if not force_onnx_export and os.path.exists(onnx_path):
                _build_logger.warning(
                    f"[BUILD] Cached ONNX at {onnx_path} is empty/corrupt (likely from an "
                    f"interrupted prior export) -- discarding and re-exporting."
                )
                os.remove(onnx_path)
            print(f"Exporting model: {onnx_path}")
            t0 = time.perf_counter()
            _export_kwargs = {
                "onnx_path": onnx_path,
                "model_data": self.model,
                "opt_image_height": opt_image_height,
                "opt_image_width": opt_image_width,
                "opt_batch_size": opt_batch_size,
                "onnx_opset": onnx_opset,
            }
            export_onnx(self.network, **_export_kwargs)
            elapsed = time.perf_counter() - t0
            stats["stages"]["onnx_export"] = {"status": "built", "elapsed_s": round(elapsed, 2)}
            _build_logger.info(f"[BUILD] ONNX export ({engine_filename}): {elapsed:.1f}s")
            self.network = self.network.to("cpu")
            del self.network
            gc.collect()
            torch.cuda.empty_cache()

        # --- ONNX Optimize ---
        if not force_onnx_optimize and os.path.exists(onnx_opt_path):
            print(f"Found cached model: {onnx_opt_path}")
            stats["stages"]["onnx_optimize"] = {"status": "cached"}
        else:
            print(f"Generating optimizing model: {onnx_opt_path}")
            t0 = time.perf_counter()
            optimize_onnx(
                onnx_path=onnx_path,
                onnx_opt_path=onnx_opt_path,
                model_data=self.model,
            )
            elapsed = time.perf_counter() - t0
            stats["stages"]["onnx_optimize"] = {"status": "built", "elapsed_s": round(elapsed, 2)}
            _build_logger.info(f"[BUILD] ONNX optimize ({engine_filename}): {elapsed:.1f}s")

        self.model.min_latent_shape = min_image_resolution // 8
        self.model.max_latent_shape = max_image_resolution // 8

        # --- Verify ONNX artifacts exist before TRT build ---
        if not os.path.exists(onnx_opt_path):
            raise RuntimeError(
                f"Optimized ONNX file missing: {onnx_opt_path}\n"
                f"This usually means the ONNX optimization step failed silently.\n"
                f"Try deleting the engine directory and rebuilding."
            )
        opt_file_size = os.path.getsize(onnx_opt_path)
        if opt_file_size == 0:
            os.remove(onnx_opt_path)
            raise RuntimeError(
                f"Optimized ONNX file is empty (0 bytes): {onnx_opt_path}\n"
                f"This usually indicates a protobuf serialization failure for >2GB models.\n"
                f"Try deleting the engine directory and rebuilding."
            )
        _build_logger.info(f"Verified ONNX opt file: {onnx_opt_path} ({opt_file_size / (1024**2):.1f} MB)")

        try:
            # --- FP8: Capture calibration tensors (once, cached in calib_data.npz) ---
            if fp8 and pipe_ref is not None:
                if os.path.exists(_calib_data_path):
                    _build_logger.info(f"[BUILD] FP8 calibration data cached: {_calib_data_path}")
                    stats["stages"]["fp8_calib_capture"] = {"status": StageStatus.CACHED}
                else:

                    def _calib_fn():
                        if is_controlnet:
                            from .fp8_quantize import capture_calibration_data_controlnet

                            _build_logger.info(
                                f"[BUILD] FP8 CN calibration: {calibration_steps} synthetic passes, "
                                f"res={opt_image_width}x{opt_image_height}"
                            )
                            capture_calibration_data_controlnet(
                                cn_model=pipe_ref,
                                n_calibration_steps=calibration_steps,
                                image_height=opt_image_height,
                                image_width=opt_image_width,
                                batch_size=opt_batch_size,
                                save_path=_calib_data_path,
                            )
                        else:
                            from .fp8_quantize import _load_calibration_prompts, capture_calibration_data

                            prompts = calibration_prompts or _load_calibration_prompts()
                            _build_logger.info(
                                f"[BUILD] FP8 activation capture: {len(prompts)} prompts × "
                                f"{calibration_steps} steps, guidance_scale={fp8_guidance_scale}"
                            )
                            capture_calibration_data(
                                pipe_ref,
                                prompts,
                                num_inference_steps=calibration_steps,
                                save_path=_calib_data_path,
                                guidance_scale=fp8_guidance_scale,
                                onnx_path=onnx_opt_path,
                                use_cached_attn=fp8_use_cached_attn,
                                use_controlnet=fp8_use_controlnet,
                                num_ip_layers=fp8_num_ip_layers,
                            )

                    if not _run_fp8_stage(
                        "fp8_calib_capture", _calib_fn, stats, fp8_allow_fp16_fallback, engine_filename
                    ):
                        fp8 = False
            elif fp8 and pipe_ref is None:
                _build_logger.warning(
                    "[BUILD] fp8=True but pipe_ref not provided — FP8 calibration skipped. "
                    "Pass pipe_ref in engine_build_options for proper activation capture."
                )
                fp8 = False

            # --- FP8: Inject native FLOAT8E4M3FN Q/DQ into the ONNX (cached in unet.fp8.onnx) ---
            if fp8:
                if os.path.exists(_fp8_onnx_path + ".ok"):
                    _build_logger.info(f"[BUILD] FP8 ONNX cached: {_fp8_onnx_path}")
                    stats["stages"]["fp8_onnx_quantize"] = {"status": StageStatus.CACHED}
                elif not _check_fp8_disk_space(onnx_opt_path, fp8_allow_fp16_fallback):
                    fp8 = False
                else:

                    def _quant_fn():
                        from .fp8_quantize import load_calibration_data, quantize_onnx_fp8

                        calib_data = load_calibration_data(_calib_data_path)
                        if calib_data is None:
                            raise RuntimeError(f"Calibration data missing after capture step: {_calib_data_path}")
                        quantize_onnx_fp8(
                            onnx_path=onnx_opt_path,
                            output_path=_fp8_onnx_path,
                            calibration_data=calib_data,
                            use_cached_attn=fp8_use_cached_attn,
                            use_feature_injection=fp8_use_feature_injection,
                            use_controlnet=fp8_use_controlnet,
                            num_ip_layers=fp8_num_ip_layers,
                        )

                    if not _run_fp8_stage(
                        "fp8_onnx_quantize", _quant_fn, stats, fp8_allow_fp16_fallback, engine_filename
                    ):
                        fp8 = False

            # Select the ONNX to feed into TRT: FP8-quantized when available, else plain opt.
            _trt_onnx_path = _fp8_onnx_path if (fp8 and os.path.exists(_fp8_onnx_path + ".ok")) else onnx_opt_path

            # --- TRT Engine Build ---
            if not force_engine_build and os.path.exists(engine_path):
                print(f"Found cached engine: {engine_path}")
                stats["stages"]["trt_build"] = {"status": "cached"}
            else:
                t0 = time.perf_counter()
                build_engine(
                    engine_path=engine_path,
                    onnx_opt_path=_trt_onnx_path,
                    model_data=self.model,
                    opt_image_height=opt_image_height,
                    opt_image_width=opt_image_width,
                    opt_batch_size=opt_batch_size,
                    build_static_batch=build_static_batch,
                    build_dynamic_shape=build_dynamic_shape,
                    build_enable_refit=build_enable_refit,
                    fp8=fp8,
                    builder_optimization_level=builder_optimization_level,
                )
                elapsed = time.perf_counter() - t0
                stats["stages"]["trt_build"] = {"status": "built", "elapsed_s": round(elapsed, 2)}
                _build_logger.info(f"[BUILD] TRT engine build ({engine_filename}): {elapsed:.1f}s")

            # --- FP8 Q/DQ layer count (sanity gate: < 500 means quantization is inactive) ---
            if fp8 and os.path.exists(engine_path):
                try:
                    import json as _json
                    import re as _re

                    import tensorrt as trt

                    _rt = trt.Runtime(BUILD_TRT_LOGGER)
                    with open(engine_path, "rb") as _f:
                        _eng = _rt.deserialize_cuda_engine(_f.read())
                    _insp = _eng.create_engine_inspector()
                    _info = _insp.get_engine_information(trt.LayerInformationFormat.JSON)
                    _qdq = _info.count("QuantizeLinear") + _info.count("DequantizeLinear")
                    stats["fp8_qdq_layers"] = _qdq
                    _build_logger.info(f"[BUILD] FP8 engine Q/DQ layer count: {_qdq}")
                    if _qdq < 500:
                        _build_logger.warning(
                            f"[BUILD] Low Q/DQ count ({_qdq} < 500) — FP8 quantization likely inactive or incomplete"
                        )

                    # Fused-MHA check: count attention layers TRT fused into a single kernel.
                    # Pattern is empirical — FLUX uses "_gemm_mha_v2"; SDXL on Ada may differ.
                    # First build logs sample names so the regex can be confirmed or tightened.
                    _MHA_RE = _re.compile(r"mha|fmha|MultiHead|FlashAttn", _re.IGNORECASE)
                    try:
                        _layers = _json.loads(_info).get("Layers", [])
                    except Exception:
                        _layers = []
                    _total = len(_layers)
                    _mha_names = [_l.get("Name", "") for _l in _layers if _MHA_RE.search(_l.get("Name", ""))]
                    _mha_count = len(_mha_names)
                    stats["mha_fused_kernels"] = _mha_count
                    stats["total_engine_layers"] = _total
                    _build_logger.info(f"[BUILD] FP8 engine fused MHA layers: {_mha_count} / {_total} total")
                    if _mha_count == 0 and _total > 0:
                        _build_logger.warning(
                            "[BUILD] No fused MHA layers detected — attention may be running decomposed "
                            "(slower). Sample layer names (first 5): "
                            + str([_l.get("Name", "") for _l in _layers[:5]])
                        )
                    else:
                        _build_logger.info(f"[BUILD] Sample fused-MHA layer names: {_mha_names[:3]}")
                except Exception as _e:
                    _build_logger.warning(f"[BUILD] FP8 inspector check skipped: {_e}")
        finally:
            _fp8_ok = fp8 and os.path.exists(_fp8_onnx_path + ".ok")
            _cleanup_intermediates(engine_dir_early, _fp8_ok)

        # Cleanup already ran in the `finally` above (also covers the failure path).
        total_elapsed = time.perf_counter() - build_total_start
        stats["total_elapsed_s"] = round(total_elapsed, 2)
        stats["build_end"] = datetime.now(timezone.utc).isoformat()

        # Engine file size
        if os.path.exists(engine_path):
            stats["engine_size_mb"] = round(os.path.getsize(engine_path) / (1024 * 1024), 1)

        _build_logger.info(f"[BUILD] {engine_filename} complete: {total_elapsed:.1f}s total")
        _write_build_stats(engine_path, stats)
