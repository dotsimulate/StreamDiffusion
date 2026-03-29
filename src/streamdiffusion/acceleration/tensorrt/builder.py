import gc
import json
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

import logging
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

        # --- ONNX Export ---
        if not force_onnx_export and os.path.exists(onnx_path):
            print(f"Found cached model: {onnx_path}")
            stats["stages"]["onnx_export"] = {"status": "cached"}
        else:
            print(f"Exporting model: {onnx_path}")
            t0 = time.perf_counter()
            export_onnx(
                self.network,
                onnx_path=onnx_path,
                model_data=self.model,
                opt_image_height=opt_image_height,
                opt_image_width=opt_image_width,
                opt_batch_size=opt_batch_size,
                onnx_opset=onnx_opset,
            )
            elapsed = time.perf_counter() - t0
            stats["stages"]["onnx_export"] = {"status": "built", "elapsed_s": round(elapsed, 2)}
            _build_logger.warning(f"[BUILD] ONNX export ({engine_filename}): {elapsed:.1f}s")
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
            _build_logger.warning(f"[BUILD] ONNX optimize ({engine_filename}): {elapsed:.1f}s")

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
            )
            elapsed = time.perf_counter() - t0
            stats["stages"]["trt_build"] = {"status": "built", "elapsed_s": round(elapsed, 2)}
            _build_logger.warning(f"[BUILD] TRT engine build ({engine_filename}): {elapsed:.1f}s")

        # Cleanup ONNX artifacts — tolerate Windows file-lock failures (Issue #4)
        for file in os.listdir(os.path.dirname(engine_path)):
            if file.endswith('.engine'):
                continue
            try:
                os.remove(os.path.join(os.path.dirname(engine_path), file))
            except OSError as cleanup_err:
                _build_logger.warning(f"[BUILD] Could not delete temp file {file}: {cleanup_err}")

        # Record totals
        total_elapsed = time.perf_counter() - build_total_start
        stats["total_elapsed_s"] = round(total_elapsed, 2)
        stats["build_end"] = datetime.now(timezone.utc).isoformat()

        # Engine file size
        if os.path.exists(engine_path):
            stats["engine_size_mb"] = round(os.path.getsize(engine_path) / (1024 * 1024), 1)

        _build_logger.warning(f"[BUILD] {engine_filename} complete: {total_elapsed:.1f}s total")
        _write_build_stats(engine_path, stats)

        gc.collect()
        torch.cuda.empty_cache()
