import gc
import json
import logging
import os
import re
import shutil
import subprocess
import sys
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


def _write_build_stats(engine_path: str, stats: dict, append_global: bool = True):
    """Write build stats to the per-engine JSON file, and (by default) append to
    the JSON-lines build log at the engines root.

    ``append_global=False`` is for a mid-build flush (fp8-round-13 provenance: the
    per-engine file is written once the calibration-capture stage completes, so an
    aborted run leaves a record instead of an orphan npz) -- the same ``stats``
    dict is then written again, in full, when the build finishes, and every
    successful build must only ever contribute ONE line to build_log.jsonl.
    """
    try:
        engine_dir = Path(engine_path).parent
        # Write stats file inside the engine directory
        stats_file = engine_dir / "build_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        if append_global:
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


_VRAM_RELEASE_ATTRS = ("unet", "vae", "text_encoder", "text_encoder_2")


def _release_torch_vram(builder: "EngineBuilder", pipe_ref, ipadapter_ref=None):
    """Move resident PyTorch modules to CPU for the duration of FP8 ONNX quantize.

    modelopt's ORT CUDA EP session lands in the same VRAM pool PyTorch is still
    holding: TAESD VAE, both SDXL text encoders, and — on a cached-ONNX build,
    where the fresh-export `.to("cpu")` above never ran — the full UNet too.
    None of it is needed again until the first post-build inference, so free it
    here and restore via the returned closure.

    ``ipadapter_ref`` (the ``IPAdapterModule`` instance, i.e. ``stream._ipadapter_module``)
    reaches a second holder outside the ``pipe_ref`` chain: its vendored
    ``ipadapter.image_encoder`` is a plain CLIP ``nn.Module`` (3.4+ GiB for ViT-bigG/14)
    that's dead weight for FaceID-non-plus configs but was previously invisible here
    since it hangs off ``stream``, not ``stream.pipe``. InsightFace's ONNX-runtime-backed
    face detector/recognizer sessions are a separate holder again and are not covered —
    they're skipped by the isinstance check below like any other non-torch.nn.Module.

    Best-effort by design: a module that's missing, not a torch.nn.Module (e.g.
    an ONNX-runtime-backed IPAdapter/InsightFace encoder), or already has no
    parameters is skipped rather than raising -- a build must not fail because
    VRAM release did. Returns a no-arg restore closure; call it from a `finally`
    so the pipeline is guaranteed usable again even if quantize itself raises.
    """
    moved: list = []  # (holder, attr_name, module, original_device)
    seen_ids: set = set()

    def _try_move(holder, attr_name, module):
        if module is None or id(module) in seen_ids or not isinstance(module, torch.nn.Module):
            return
        first_param = next(module.parameters(), None)
        if first_param is None:
            return
        original_device = first_param.device
        if original_device.type == "cpu":
            seen_ids.add(id(module))
            return
        try:
            module.to("cpu")
        except Exception as e:
            _build_logger.debug(f"[BUILD] VRAM release: could not move '{attr_name}' to cpu: {e}")
            return
        seen_ids.add(id(module))
        moved.append((holder, attr_name, module, original_device))

    # self.network may already be gone -- del'd right after a fresh ONNX export
    # (see the export branch above). getattr default handles that safely.
    _try_move(builder, "network", getattr(builder, "network", None))
    if pipe_ref is not None:
        for _attr in _VRAM_RELEASE_ATTRS:
            _try_move(pipe_ref, _attr, getattr(pipe_ref, _attr, None))
    if ipadapter_ref is not None:
        _ip = getattr(ipadapter_ref, "ipadapter", None)
        if _ip is not None:
            _try_move(_ip, "image_encoder", getattr(_ip, "image_encoder", None))

    if moved:
        _names = ", ".join(f"{h.__class__.__name__}.{a}" for h, a, _, _ in moved)
        _build_logger.info(f"[BUILD] VRAM release: moved to CPU before FP8 quantize: {_names}")
    gc.collect()
    torch.cuda.empty_cache()

    def _restore():
        if not moved:
            return
        for holder, attr_name, module, original_device in moved:
            try:
                module.to(original_device)
                setattr(holder, attr_name, module)
            except Exception as e:
                _build_logger.warning(
                    f"[BUILD] VRAM release: failed to restore '{attr_name}' to {original_device}: {e}"
                )
        _build_logger.info(f"[BUILD] VRAM release: restored {len(moved)} module(s) to GPU")

    return _restore


_KEEP_INTERMEDIATES_ENV = "STREAMDIFFUSION_FP8_KEEP_INTERMEDIATES"


def _cleanup_intermediates(engine_dir: str, fp8_ok: bool):
    """Delete intermediate ONNX/build artifacts, preserving .engine, .cache, calib_data.npz,
    calib_data.meta.json, build_stats.json, and (only when fp8_ok) the cached
    unet.fp8.onnx* artifact.

    Set STREAMDIFFUSION_FP8_KEEP_INTERMEDIATES=1 to additionally preserve the pre-FP8
    optimized ONNX (`*.opt.onnx`) and its external-data `weights.pb`. Without this, every
    build wipes the one fixture fp8_quantize.py's `__main__` entry needs to run the FP8
    quantize stage standalone -- opt-in rather than default so routine builds don't leave
    an extra multi-GB ONNX behind in every engine dir.

    Two-pass deletion handles Windows file locks (gc.collect releases Python handles).
    Runs from a `finally` block so it also fires when a build stage raises, instead of
    orphaning tens of GB of external-data ONNX copies on failure.
    """
    _keep_suffixes = (".engine", ".cache")
    _keep_exact = {"build_stats.json", "timing.cache", "calib_data.npz", "calib_data.meta.json"}
    if os.environ.get(_KEEP_INTERMEDIATES_ENV) == "1":
        _keep_suffixes = _keep_suffixes + (".opt.onnx",)
        _keep_exact = _keep_exact | {"weights.pb"}
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
                f"Manual cleanup: delete all files except *.engine, calib_data.npz, "
                f"calib_data.meta.json, unet.fp8.onnx from {engine_dir}"
            )
        cleaned = len(_to_delete) - len(_still_failed)
    else:
        cleaned = len(_to_delete)
    _build_logger.info(f"[BUILD] Cleaned {cleaned}/{len(_to_delete)} intermediate files")


# fp8-round-8's two-activation attention BMMs (Q@K^T, softmax@V) -- same anchor as
# fp8_quantize.py's _ATTENTION_EXCLUDE_PATTERNS (r".*/attn[12]/MatMul(_\d+)?$"), kept
# as a separate constant here rather than imported since this module must stay
# importable without pulling in fp8_quantize.py's modelopt-adjacent dependencies.
_ATTN_BMM_NAME_RE = re.compile(r"/attn[12]/MatMul(_\d+)?$")
_MYL_PARTITION_RE = re.compile(r"_myl(\d+)_")


def _count_attn_bmm_dq_fed(onnx_path: str) -> Optional[Tuple[int, int]]:
    """Count attention BMMs (Q@K^T / softmax@V, never the QKV/output projections)
    whose both direct inputs are DequantizeLinear-produced, vs the total attention-BMM
    count, by parsing the ONNX graph structure only (FP8 Round 11 evidence record).

    "Two-activation BMM" (neither input is a weight initializer) distinguishes these
    from projection MatMuls, where one input is always a weight -- same distinction
    fp8_quantize.py's _ATTENTION_EXCLUDE_PATTERNS regex targets. Graph-only load
    (load_external_data=False): the graph itself is ~13 MB even though the model's
    external weight data is multi-GB, so this is a cheap, read-only structural check,
    not a real load of the model.

    With fp8_mha_qdq (-mhaq) active, dq_fed should equal total (466/466 verified on a
    real build); with it off, dq_fed should be 0 while total is unchanged -- this is
    what settles whether mhaq's Q/DQ insertion is actually taking, independent of the
    engine inspector's mha_fused_kernels (which measures TRT's fusion decision, not
    modelopt's quantization).

    Returns (dq_fed_count, total_count), or None if onnx_path doesn't exist or fails
    to parse -- best-effort, matching the rest of the inspector block's posture.
    """
    if not onnx_path or not os.path.exists(onnx_path):
        return None
    try:
        model = onnx.load(onnx_path, load_external_data=False)
        initializer_names = {init.name for init in model.graph.initializer}
        producer = {}
        for node in model.graph.node:
            for out in node.output:
                producer[out] = node
        total = 0
        dq_fed = 0
        for node in model.graph.node:
            if node.op_type != "MatMul" or not _ATTN_BMM_NAME_RE.search(node.name):
                continue
            if any(inp in initializer_names for inp in node.input):
                continue  # a projection MatMul (one operand is a weight), not a BMM
            total += 1
            producers = [producer.get(inp) for inp in node.input[:2]]
            if all(p is not None and p.op_type == "DequantizeLinear" for p in producers):
                dq_fed += 1
        return dq_fed, total
    except Exception:
        return None


def _find_best_sibling_mha_ratio(engine_dir: str, precision: str, window: int = 5) -> Optional[float]:
    """Best (lowest) mha_kernels_per_attn_block seen among the `window` most recent
    sibling engine dirs' build_stats.json files at the same precision -- the
    same-precision regression baseline for the inspector block's warning gate
    (FP8 Round 11; direction and windowing corrected by the 2026-09-06 MHA-fusion
    investigation, see the comment at the gate's call site in `EngineBuilder.build`).

    Lower is better here. This ratio is *kernels needed per attention block*, not
    "modules fused": SDXL's UNet has `kvo_cache_count` self-attention (attn1) blocks
    and an equal number of cross-attention (attn2) blocks (`get_kvo_cache_info` in
    `models/utils.py` counts attn1 only), so TRT's fusion ceiling is one kernel per
    attention module, i.e. ratio == 2.0. A build needing *more* kernels per block than
    history means some attention sites stopped fusing into a single kernel each --
    worse fusion, not better. Originally this function returned the historical
    *maximum* and the gate warned when a build fell *below* it, which flagged a fully
    saturated 1-kernel-per-module build (ratio 2.0) as a regression against a
    worse-fused historical outlier (ratio 3.0, one build on 2026-08-22) that needed
    two kernels for some sites. Confirmed empirically via
    scripts/fp8/probe_engine_evidence.py: every sibling engine's fused-MHA layer names
    normalize (stripping the trailing `_myl<N>_<M>` suffix) to the single pattern
    `_gemm_mha_v2` -- the 3.0-ratio build never fused a second *kind* of kernel, its
    per-Myelin-partition kernel counts are uniformly 1.5x the 2.0-ratio builds', i.e.
    the same sites needing more kernels each, not more sites being covered.

    Restricted to the `window` most recent sibling builds (by `build_end`, falling
    back to `build_start`, falling back to the stats file's mtime) rather than an
    all-time extremum, so a single anomalous historical build cannot pin the gate
    forever once enough newer builds age it out of the window -- a defect independent
    of the polarity bug above: an unbounded, unfiltered all-time extremum over
    incomparable graph revisions/resolutions/opt levels is fragile regardless of which
    direction counts as "best".

    Skips sibling files with no ``precision`` key (e.g. VAE engine dirs, which never
    run this inspector) or no ``mha_kernels_per_attn_block`` key (e.g. builds from
    before this round, or ``use_cached_attn=False`` builds where the normalized
    metric can't be computed). Returns None if no comparable sibling exists -- the
    metric self-populates on this engine's first build; no seeded magic constant.
    """
    try:
        root = Path(engine_dir).parent
        candidates = []  # (sort_key, ratio), sort_key is an ISO-8601 string
        for sibling in root.iterdir():
            if not sibling.is_dir():
                continue
            stats_path = sibling / "build_stats.json"
            if not stats_path.exists():
                continue
            try:
                with open(stats_path) as f:
                    sibling_stats = json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            if sibling_stats.get("precision") != precision:
                continue
            ratio = sibling_stats.get("mha_kernels_per_attn_block")
            if ratio is None:
                continue
            sort_key = sibling_stats.get("build_end") or sibling_stats.get("build_start")
            if sort_key is None:
                try:
                    sort_key = datetime.fromtimestamp(stats_path.stat().st_mtime, tz=timezone.utc).isoformat()
                except OSError:
                    sort_key = ""
            candidates.append((sort_key, ratio))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0])
        recent_ratios = [ratio for _, ratio in candidates[-window:]]
        return min(recent_ratios)
    except Exception:
        return None


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
        ipadapter_ref=None,
        calibration_prompts=None,
        calibration_steps: int = 20,
        fp8_calibration_timesteps=None,
        fp8_calibration_scheduler_ref=None,
        fp8_guidance_scale: float = 7.5,
        fp8_allow_fp16_fallback: bool = False,
        fp8_mha_qdq: bool = False,
        # fp8-round-8: corrects modelopt's inverted INT8->FP8 scale-conversion factor
        # (fp8_quantize.py::_rescale_fp8_qdq_scales). 1.0 = full E4M3 range.
        fp8_scale_headroom: float = 1.0,
        # fp8-round-8 fallback lever: excludes the two attention BMMs from FP8 Q/DQ.
        fp8_exclude_attention: bool = False,
        # fp8-round-9 fallback lever: excludes the three IPA cross-attention activation
        # tensors from FP8 Q/DQ (see _IPADAPTER_ACTIVATION_EXCLUDE_PATTERNS).
        fp8_exclude_ipadapter: bool = False,
        fp8_use_cached_attn: bool = False,
        fp8_use_feature_injection: bool = False,
        fp8_use_controlnet: bool = False,
        fp8_num_ip_layers: int = 0,
        # fp8-round-5-handoff Step 3: deployment values for scalar/vector inputs that
        # otherwise calibrate on zeros/ones (see capture_calibration_data's Args docstring).
        # Read from config by the caller (wrapper.py) — never hardcoded here.
        fp8_fi_strength: float = 0.0,
        fp8_fi_threshold: float = 0.0,
        fp8_ipadapter_scale: float = 1.0,
        # fp8-round-9: real IP-Adapter projection tokens for calibration's
        # encoder_hidden_states reconciliation (see capture_calibration_data's
        # ipadapter_tokens docstring). None preserves the prior zero-pad behavior.
        fp8_ipadapter_tokens=None,
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
            _build_logger.info(f"Exporting model: {onnx_path}")
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
            _build_logger.info(f"Generating optimizing model: {onnx_opt_path}")
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

        _restore_vram = None
        try:
            # --- FP8: Capture calibration tensors (once, cached in calib_data.npz) ---
            if fp8 and pipe_ref is not None:
                from .fp8_quantize import load_calib_provenance

                if os.path.exists(_calib_data_path):
                    _provenance = load_calib_provenance(_calib_data_path)
                    if _provenance is not None:
                        _build_logger.info(
                            f"[BUILD] FP8 calibration data cached: {_calib_data_path} "
                            f"(captured res={_provenance.get('image_width')}x{_provenance.get('image_height')}, "
                            f"selected_calls={_provenance.get('selected_calls')})"
                        )
                        stats["stages"]["fp8_calib_capture"] = {"status": StageStatus.CACHED, **_provenance}
                    else:
                        # fp8-round-13: this is the exact silence that hid fp8-round-10's
                        # aborted ~07:14 capture -- a cache hit with no record of how the
                        # npz was made. Every engine dir built before this round has no
                        # sidecar; degrade gracefully rather than warn on every such build.
                        _build_logger.info(
                            f"[BUILD] FP8 calibration data cached: {_calib_data_path} "
                            "(no provenance sidecar -- captured before fp8-round-13)"
                        )
                        stats["stages"]["fp8_calib_capture"] = {
                            "status": StageStatus.CACHED,
                            "provenance": "unavailable (captured before provenance records)",
                        }
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
                                f"{calibration_steps} steps, guidance_scale={fp8_guidance_scale}, "
                                f"res={opt_image_width}x{opt_image_height}"
                            )
                            capture_calibration_data(
                                pipe_ref,
                                prompts,
                                num_inference_steps=calibration_steps,
                                save_path=_calib_data_path,
                                guidance_scale=fp8_guidance_scale,
                                onnx_path=onnx_opt_path,
                                use_cached_attn=fp8_use_cached_attn,
                                use_feature_injection=fp8_use_feature_injection,
                                use_controlnet=fp8_use_controlnet,
                                num_ip_layers=fp8_num_ip_layers,
                                fi_strength=fp8_fi_strength,
                                fi_threshold=fp8_fi_threshold,
                                ipadapter_scale=fp8_ipadapter_scale,
                                ipadapter_tokens=fp8_ipadapter_tokens,
                                timesteps=fp8_calibration_timesteps,
                                scheduler_ref=fp8_calibration_scheduler_ref,
                                # fp8-round-10: without these, diffusers falls back to
                                # pipe.unet.config.sample_size (512 for SDXL-Turbo) regardless
                                # of the engine's actual build resolution — the CN branch above
                                # already passes these; this restores the same symmetry here.
                                image_height=opt_image_height,
                                image_width=opt_image_width,
                            )

                    if not _run_fp8_stage(
                        "fp8_calib_capture", _calib_fn, stats, fp8_allow_fp16_fallback, engine_filename
                    ):
                        fp8 = False
                    else:
                        _provenance = load_calib_provenance(_calib_data_path)
                        if _provenance is not None:
                            stats["stages"]["fp8_calib_capture"].update(_provenance)

                # fp8-round-13: flush here, right after the capture stage resolves
                # (cached / built / failed-with-fallback), so an aborted run past this
                # point still leaves a build_stats.json recording that capture at least
                # started -- the exact failure mode that hid fp8-round-10's aborted
                # ~07:14 capture (it never reached the end-of-build _write_build_stats
                # call, so it left neither a stats file nor a build_log.jsonl line).
                # append_global=False: this is a mid-build snapshot of the same `stats`
                # dict the end-of-build call below writes in full -- appending here too
                # would double every successful build's build_log.jsonl entry.
                _write_build_stats(engine_path, stats, append_global=False)
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
                        # Runs fp8_quantize.py's __main__ entry in a subprocess rather than
                        # calling quantize_onnx_fp8() in-process. This process boundary is
                        # the actual point of Step 4: modelopt/ORT's CUDA arena is scoped to
                        # the child and is fully released on exit no matter how it fails,
                        # instead of leaking into this long-lived build process. Calibration
                        # data crosses via calib_data.npz on disk, not as a subprocess arg --
                        # it's multi-GB. Note this does NOT free the parent's own resident
                        # torch modules; _release_torch_vram (called just below) still is.
                        if not os.path.exists(_calib_data_path):
                            raise RuntimeError(f"Calibration data missing after capture step: {_calib_data_path}")

                        _quant_args = json.dumps(
                            {
                                "onnx_path": onnx_opt_path,
                                "output_path": _fp8_onnx_path,
                                "calib_data_path": _calib_data_path,
                                "disable_mha_qdq": not fp8_mha_qdq,
                                "use_cached_attn": fp8_use_cached_attn,
                                "use_feature_injection": fp8_use_feature_injection,
                                "use_controlnet": fp8_use_controlnet,
                                "num_ip_layers": fp8_num_ip_layers,
                                "fp8_scale_headroom": fp8_scale_headroom,
                                "fp8_exclude_attention": fp8_exclude_attention,
                                "fp8_exclude_ipadapter": fp8_exclude_ipadapter,
                            }
                        )
                        _cmd = [
                            sys.executable,
                            "-m",
                            "streamdiffusion.acceleration.tensorrt.fp8_quantize",
                            _quant_args,
                        ]
                        _build_logger.info(f"[BUILD] fp8_onnx_quantize: launching subprocess: {_cmd[0]} -m ...")
                        # env=os.environ.copy() (not the subprocess default of inheriting
                        # implicitly) so PYTHONPATH / CUDA_VISIBLE_DEVICES / HF cache vars
                        # are carried into the child explicitly rather than by accident of
                        # not overriding env=.
                        _proc = subprocess.Popen(
                            _cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            env=os.environ.copy(),
                        )
                        for _line in _proc.stdout:
                            _build_logger.info(f"[FP8-subprocess] {_line.rstrip()}")
                        _ret = _proc.wait()
                        if _ret != 0:
                            raise RuntimeError(f"fp8_onnx_quantize subprocess exited with code {_ret}")

                    # Capture (above) needs pipe_ref/ipadapter_ref resident on GPU; quantize
                    # doesn't -- release here, right before the stage that actually spikes
                    # VRAM. Restored in the `finally` below regardless of outcome.
                    _restore_vram = _release_torch_vram(self, pipe_ref, ipadapter_ref)
                    if not _run_fp8_stage(
                        "fp8_onnx_quantize", _quant_fn, stats, fp8_allow_fp16_fallback, engine_filename
                    ):
                        fp8 = False

            # fp8-round-8: read back the rescale report from the .ok sentinel, written by
            # _rescale_fp8_qdq_scales via quantize_onnx_fp8 (fp8_quantize.py). Covers both
            # the fresh-build path (sentinel just written above) and the cache-hit path
            # (sentinel from a prior run) — either way, by this point the file exists iff
            # fp8 quantization succeeded. A bare "ok" string (older, pre-round-8 sentinel)
            # or a missing/corrupt file just means this metric is unknown, not fatal.
            if fp8 and os.path.exists(_fp8_onnx_path + ".ok"):
                try:
                    with open(_fp8_onnx_path + ".ok") as _f:
                        _ok_payload = json.load(_f)
                    _rescale = _ok_payload.get("rescale") or {}
                    if "uncalibrated_count" in _rescale:
                        stats["fp8_uncalibrated_scales"] = _rescale["uncalibrated_count"]
                    if _rescale.get("realized_peak_after") is not None:
                        stats["fp8_scale_realized_peak"] = _rescale["realized_peak_after"]
                except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                    pass

            # Select the ONNX to feed into TRT: FP8-quantized when available, else plain opt.
            _trt_onnx_path = _fp8_onnx_path if (fp8 and os.path.exists(_fp8_onnx_path + ".ok")) else onnx_opt_path

            if fp8:
                # Captured unconditionally on this build's own parameters (not on
                # fp8_onnx_quantize's cache status -- that stage's _quant_args closure
                # only runs on a cache miss) so build_stats.json always reflects what
                # this build *requested*, not just what the quantize subprocess happened
                # to run this time (FP8 Round 11 -- closes the gap that left the
                # 7104-vs-6899 qdq delta unattributable).
                stats["fp8_quantize_args"] = {
                    "disable_mha_qdq": not fp8_mha_qdq,
                    "fp8_scale_headroom": fp8_scale_headroom,
                    "fp8_exclude_attention": fp8_exclude_attention,
                    "fp8_exclude_ipadapter": fp8_exclude_ipadapter,
                }

            # --- TRT Engine Build ---
            _engine = None
            if not force_engine_build and os.path.exists(engine_path):
                print(f"Found cached engine: {engine_path}")
                _build_logger.info(f"Found cached engine: {engine_path}")
                stats["stages"]["trt_build"] = {"status": "cached"}
            else:
                t0 = time.perf_counter()
                _l2tc_before = BUILD_TRT_LOGGER.l2tc_validate_fail_count
                _engine = build_engine(
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
                # Tiling decision + TRT's own l2tc verdict (FP8 Round 11 evidence record --
                # previously existed only as INFO log lines, never persisted).
                stats["l2tc_validate_fail_count"] = BUILD_TRT_LOGGER.l2tc_validate_fail_count - _l2tc_before
                _tiling_info = getattr(_engine, "last_tiling_info", {}) or {}
                if _tiling_info:
                    stats["dynamic_shapes"] = _tiling_info.get("dynamic_shapes")
                    stats["tiling_optimization_level"] = _tiling_info.get("tiling_optimization_level")
                    stats["l2_limit_for_tiling_mib"] = _tiling_info.get("l2_limit_for_tiling_mib")
                    # Effective builder_optimization_level (2026-09-06 MHA-fusion
                    # investigation) -- previously a build parameter that was forwarded
                    # all the way to build_engine() but never persisted anywhere, which
                    # made it impossible to tell whether a historical build's fusion
                    # outcome was influenced by a different optimization level without a
                    # full rebuild. Sourced from _apply_gpu_profile_to_config's return
                    # value, which already reflects the requested override (if any).
                    stats["builder_optimization_level"] = _tiling_info.get("builder_optimization_level")

            # --- Engine inspector: Q/DQ + fused-MHA + attn-BMM evidence record (FP8 Round 11) ---
            # Was `if fp8 and os.path.exists(engine_path)` -- FP16 builds never ran this, so
            # mha_fused_kernels/total_engine_layers had no FP16 baseline to compare against
            # (Round 10 parked Item 1: "mha_fused_kernels: 140 constant across every FP8
            # build" was never capable of evidencing anything since it was FP8-vs-FP8 only).
            # Now runs for every build that produced an engine, regardless of precision.
            stats["precision"] = "fp8" if fp8 else "fp16"
            if os.path.exists(engine_path):
                try:
                    import json as _json

                    import tensorrt as trt

                    # VRAM safety: a genuine FP8 build already released torch VRAM before the
                    # quantize stage (_restore_vram set above at the FP8-quantize call site)
                    # and it stays released through here. This check exists for FP16 builds,
                    # which never call _release_torch_vram at all -- their engine (5.58 GB
                    # for a current UNet) previously deserialized here with torch modules
                    # still resident, and the blanket `except Exception` below would have
                    # swallowed an OOM and silently lost exactly the evidence this block
                    # exists to capture. Do not overwrite an already-set _restore_vram --
                    # that would orphan the earlier release's restore closure.
                    if _restore_vram is None:
                        _engine_size = os.path.getsize(engine_path)
                        _free_vram, _ = torch.cuda.mem_get_info()
                        _margin = 2 * (1024**3)
                        if _free_vram < _engine_size + _margin:
                            _build_logger.warning(
                                f"[BUILD] Low VRAM before engine inspector ({_free_vram / 1024**3:.1f} GiB free, "
                                f"need ~{(_engine_size + _margin) / 1024**3:.1f} GiB) -- releasing resident "
                                "torch modules before deserializing for inspection."
                            )
                            _restore_vram = _release_torch_vram(self, pipe_ref, ipadapter_ref)

                    _rt = trt.Runtime(BUILD_TRT_LOGGER)
                    with open(engine_path, "rb") as _f:
                        _eng = _rt.deserialize_cuda_engine(_f.read())
                    _insp = _eng.create_engine_inspector()
                    _info = _insp.get_engine_information(trt.LayerInformationFormat.JSON)
                    # Free the deserialized engine as soon as its JSON info is extracted --
                    # everything below is CPU-only string/JSON analysis. Previously `_eng`
                    # lived until build() returned.
                    del _eng, _insp
                    gc.collect()
                    torch.cuda.empty_cache()

                    if fp8:
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
                    _MHA_RE = re.compile(r"mha|fmha|MultiHead|FlashAttn", re.IGNORECASE)
                    try:
                        _layers = _json.loads(_info).get("Layers", [])
                    except Exception:
                        _layers = []
                    _total = len(_layers)
                    _mha_names = [_l.get("Name", "") for _l in _layers if _MHA_RE.search(_l.get("Name", ""))]
                    _mha_count = len(_mha_names)
                    stats["mha_fused_kernels"] = _mha_count
                    stats["total_engine_layers"] = _total
                    _build_logger.info(
                        f"[BUILD] {stats['precision']} engine fused MHA layers: {_mha_count} / {_total} total"
                    )
                    if _mha_count == 0 and _total > 0:
                        _build_logger.warning(
                            "[BUILD] No fused MHA layers detected — attention may be running decomposed "
                            "(slower). Sample layer names (first 5): "
                            + str([_l.get("Name", "") for _l in _layers[:5]])
                        )
                    else:
                        _build_logger.info(f"[BUILD] Sample fused-MHA layer names: {_mha_names[:3]}")

                    # Myelin partitions: distinct _myl<N>_ groups among the MHA-matching layer
                    # names -- a stronger fusion signal than the raw count alone (FP8 Round 11:
                    # 210 kernels in 11 partitions at FP16 vs 140 in a single myl0 at FP8 --
                    # the raw count alone hid that split).
                    _partitions = {m.group(1) for _n in _mha_names for m in [_MYL_PARTITION_RE.search(_n)] if m}
                    stats["mha_myelin_partitions"] = len(_partitions)

                    # Same-precision regression gate (replaces the old `_mha_count == 0`
                    # guard, which -- back when this block was FP8-gated -- could only ever
                    # compare FP8 against FP8 and so never caught the FP8-vs-FP16 fusion
                    # delta this round found). Normalized per attn block so engines with
                    # different kvo_cache_count remain comparable. kvo_cache_count only
                    # exists on the model when use_cached_attn is set (models.py), hence
                    # getattr with a 0 default -- 0 skips the normalized metric entirely.
                    #
                    # Direction (corrected 2026-09-06 MHA-fusion investigation): LOWER is
                    # better. mha_kernels_per_attn_block is kernels needed per block, and
                    # SDXL's fusion ceiling is exactly 2.0 (one kernel per attn1 + one per
                    # attn2, per kvo_cache_count blocks) -- a build needing *more* kernels
                    # per block than its best-known sibling means some attention sites
                    # stopped fusing into a single kernel each. The original comparison
                    # (`ratio < best`, where `best` was the historical *maximum*) flagged
                    # full 1-kernel-per-module fusion as a regression against a worse-fused
                    # historical outlier that needed two kernels for some sites -- see
                    # `_find_best_sibling_mha_ratio`'s docstring for the empirical evidence.
                    _kvo_count = getattr(self.model, "kvo_cache_count", 0)
                    if _kvo_count > 0:
                        _ratio = _mha_count / _kvo_count
                        stats["mha_kernels_per_attn_block"] = _ratio
                        _best_same = _find_best_sibling_mha_ratio(engine_dir_early, stats["precision"])
                        if _best_same is not None and _ratio > _best_same:
                            _build_logger.warning(
                                f"[BUILD] MHA fusion regression vs best (lowest) same-precision sibling: "
                                f"{_ratio:.3f} kernels/block (this build) > {_best_same:.3f} (best sibling)."
                            )
                        _other_precision = "fp16" if fp8 else "fp8"
                        _best_other = _find_best_sibling_mha_ratio(engine_dir_early, _other_precision)
                        if _best_other is not None:
                            _build_logger.info(
                                f"[BUILD] MHA fusion, cross-precision: {_ratio:.3f} kernels/block "
                                f"({stats['precision']}) vs {_best_other:.3f} ({_other_precision})."
                            )

                    # attn_bmm_dq_fed: only meaningful for FP8 -- checks that mhaq's Q/DQ
                    # insertion actually reached the two attention BMMs, independent of
                    # whether TRT went on to fuse them into a kernel. Reads the cached
                    # {prefix}.fp8.onnx graph, not the engine.
                    if fp8:
                        _bmm_result = _count_attn_bmm_dq_fed(_fp8_onnx_path)
                        if _bmm_result is not None:
                            stats["attn_bmm_dq_fed"], stats["attn_bmm_total"] = _bmm_result
                            _build_logger.info(
                                f"[BUILD] Attention BMM Q/DQ coverage: {_bmm_result[0]}/{_bmm_result[1]} DQ-fed"
                            )
                except Exception as _e:
                    _build_logger.warning(f"[BUILD] Engine inspector check skipped: {_e}")
        finally:
            # Restore before cleanup so a restore failure (logged, not raised) still
            # leaves temp-file sweep to run, and so this fires on the failure path too.
            if _restore_vram is not None:
                _restore_vram()
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
