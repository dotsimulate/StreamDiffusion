import gc
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import *

import torch

from .models.models import BaseModel
from .utilities import (
    build_engine,
    export_onnx,
    optimize_onnx,
)


_build_logger = logging.getLogger(__name__)


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
        opt_batch_size: int = 1,
        min_image_resolution: int = 256,
        max_image_resolution: int = 1024,
        build_enable_refit: bool = False,
        build_static_batch: bool = False,
        build_dynamic_shape: bool = True,
        build_all_tactics: bool = False,
        onnx_opset: int = 17,
        force_engine_build: bool = False,
        force_onnx_export: bool = False,
        force_onnx_optimize: bool = False,
        fp8: bool = False,
        pipe_ref=None,
        calibration_prompts=None,
        calibration_steps: int = 20,
        amax_save_path: Optional[str] = None,
        fp8_alpha: float = 0.8,
        fp8_allow_fp16_fallback: bool = False,
    ):
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
            "build_all_tactics": build_all_tactics,
            "stages": {},
        }

        # --- FP8 Torch-Level Calibration (must run before ONNX export) ---
        # Only calibrate when the ONNX does not yet exist; if cached, Q/DQ nodes
        # are already embedded from the prior build.
        _fp8_quantized = False
        if fp8 and (force_onnx_export or not os.path.exists(onnx_path)):
            if pipe_ref is not None:
                _build_logger.info(
                    f"[BUILD] FP8 calibration: fp8=True, pipe={type(pipe_ref).__name__}, "
                    f"steps={calibration_steps}, alpha={fp8_alpha}, amax={amax_save_path}"
                )
                t0 = time.perf_counter()
                try:
                    from .fp8_quantize import (
                        _load_calibration_prompts,
                        calibrate_unet_fp8_torch,
                        load_unet_amax,
                    )

                    prompts = calibration_prompts or _load_calibration_prompts()
                    amax_loaded = False
                    if amax_save_path:
                        amax_loaded = load_unet_amax(pipe_ref.unet, amax_save_path)
                    if not amax_loaded:
                        calibrate_unet_fp8_torch(
                            pipe_ref,
                            pipe_ref.unet,
                            prompts,
                            num_inference_steps=calibration_steps,
                            alpha=fp8_alpha,
                            amax_save_path=amax_save_path,
                        )
                    _fp8_quantized = True
                    elapsed = time.perf_counter() - t0
                    stats["stages"]["fp8_calibrate"] = {"status": "built", "elapsed_s": round(elapsed, 2)}
                    _build_logger.info(f"[BUILD] FP8 calibration ({engine_filename}): {elapsed:.1f}s")
                except Exception as calib_err:
                    elapsed = time.perf_counter() - t0
                    stats["stages"]["fp8_calibrate"] = {
                        "status": "failed",
                        "elapsed_s": round(elapsed, 2),
                        "error": str(calib_err),
                    }
                    if fp8_allow_fp16_fallback:
                        _build_logger.warning(
                            f"[BUILD] FP8 calibration failed after {elapsed:.1f}s: {calib_err}. "
                            "Falling back to FP16 engine (fp8_allow_fp16_fallback=True)."
                        )
                        fp8 = False
                    else:
                        raise RuntimeError(
                            f"FP8 calibration failed: {calib_err}.\n"
                            "Set fp8_allow_fp16_fallback=True in TRT_PROFILES to silently fall "
                            "back to FP16, or fix the error above."
                        ) from calib_err
            else:
                _build_logger.warning(
                    "[BUILD] fp8=True but pipe_ref not provided — FP8 calibration skipped. "
                    "Pass pipe_ref in engine_build_options for proper torch-level calibration."
                )

        # --- ONNX Export ---
        if not force_onnx_export and os.path.exists(onnx_path):
            print(f"Found cached model: {onnx_path}")
            stats["stages"]["onnx_export"] = {"status": "cached"}
        else:
            print(f"Exporting model: {onnx_path}")
            t0 = time.perf_counter()
            _export_kwargs = dict(
                onnx_path=onnx_path,
                model_data=self.model,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                onnx_opset=onnx_opset,
            )
            if _fp8_quantized:
                from modelopt.torch.quantization.utils import export_torch_mode

                with export_torch_mode():
                    export_onnx(self.network, **_export_kwargs)
            else:
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

        # --- TRT Engine Build ---
        if not force_engine_build and os.path.exists(engine_path):
            print(f"Found cached engine: {engine_path}")
            stats["stages"]["trt_build"] = {"status": "cached"}
        else:
            t0 = time.perf_counter()
            build_engine(
                engine_path=engine_path,
                onnx_opt_path=onnx_opt_path,
                model_data=self.model,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                build_static_batch=build_static_batch,
                build_dynamic_shape=build_dynamic_shape,
                build_all_tactics=build_all_tactics,
                build_enable_refit=build_enable_refit,
                fp8=fp8,
            )
            elapsed = time.perf_counter() - t0
            stats["stages"]["trt_build"] = {"status": "built", "elapsed_s": round(elapsed, 2)}
            _build_logger.info(f"[BUILD] TRT engine build ({engine_filename}): {elapsed:.1f}s")

        # Record totals (before cleanup so build_stats.json is preserved)
        total_elapsed = time.perf_counter() - build_total_start
        stats["total_elapsed_s"] = round(total_elapsed, 2)
        stats["build_end"] = datetime.now(timezone.utc).isoformat()

        # Engine file size
        if os.path.exists(engine_path):
            stats["engine_size_mb"] = round(os.path.getsize(engine_path) / (1024 * 1024), 1)

        _build_logger.info(f"[BUILD] {engine_filename} complete: {total_elapsed:.1f}s total")
        _write_build_stats(engine_path, stats)

        # Cleanup ONNX artifacts — preserve .engine, unet_amax.pt, timing.cache, build_stats.json
        # Two-pass deletion to handle Windows file locks (gc.collect releases Python handles)
        _keep_suffixes = (".engine", ".cache")
        _keep_exact = {"build_stats.json", "timing.cache", "unet_amax.pt"}
        engine_dir = os.path.dirname(engine_path)
        _to_delete = []
        for file in os.listdir(engine_dir):
            if file in _keep_exact or any(file.endswith(s) for s in _keep_suffixes):
                continue
            _to_delete.append(os.path.join(engine_dir, file))

        if _to_delete:
            _failed = []
            for fpath in _to_delete:
                try:
                    os.remove(fpath)
                except OSError:
                    _failed.append(fpath)

            # Release Python-held file handles (ONNX model refs), retry locked files.
            # Per-file poll with 50ms backoff instead of a single global sleep — most
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
                        _build_logger.warning(
                            f"[BUILD] Could not delete temp file {os.path.basename(fpath)}: {_last_err}"
                        )
                if _still_failed:
                    _build_logger.warning(
                        f"[BUILD] {len(_still_failed)} intermediate files could not be cleaned. "
                        f"Manual cleanup: delete all files except *.engine and unet_amax.pt from {engine_dir}"
                    )
                cleaned = len(_to_delete) - len(_still_failed)
            else:
                cleaned = len(_to_delete)
            _build_logger.info(f"[BUILD] Cleaned {cleaned}/{len(_to_delete)} intermediate files")
