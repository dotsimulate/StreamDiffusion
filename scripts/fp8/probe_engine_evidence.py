"""
FP8 Round 11 — permanent, read-only engine evidence probe.

Context: FP8 Round 10 parked three items in its S8 for the same stated reason
— no evidence either way. A read-only diagnosis pass over artifacts already on
disk (no build, no GPU allocation beyond a one-engine-at-a-time deserialize)
found that two of the three premises were wrong, and that all three shared one
root cause: the build produced no durable, machine-checkable record of what it
did. Change A (builder.py/utilities.py) fixed that going forward by writing
the evidence into build_stats.json at build time. This script makes the
*diagnosis loop itself* permanent and repeatable — re-derive the same evidence
straight from the engine/onnx/npz files on disk, independent of whether a
given engine was built before or after Change A landed, and cross-check
against whatever build_stats.json happened to persist.

What it reports per `unet.engine` found under --engines-root:
  - total engine layers, fused-MHA kernel count (verbatim copy of builder.py's
    _MHA_RE — see the note below on why it's copied, not imported), and
    distinct Myelin (_myl<N>_) partitions among the MHA-matching layers.
  - IO signature: total IO tensor count, kvo_cache_in_*/fio_cache_in_* counts.
  - attn_bmm_dq_fed / attn_bmm_total on the sibling unet.fp8.onnx, if present
    (graph-only ONNX load — reuses builder.py's _count_attn_bmm_dq_fed
    verbatim, so this script and a real build can never silently drift apart
    on what "DQ-fed" means).
  - calib_data.npz row count and distinct-timestep count, if present.
  - a cross-check against the engine's own build_stats.json (when it has the
    relevant keys — pre-Round-11 build_stats.json files simply won't, and
    that absence is reported as such, not treated as a mismatch).

Profiling-track step 1 (folded in, free, no build, no profiler): when both an
FP8 and an FP16 unet.engine exist at the same resolution, the two are diffed
to attribute FP8's layer-count surplus over FP16. The source plan's framing
for this was "Q/DQ reformat nodes vs decomposed attention (gemm + softmax
where FP16 has _gemm_mha_v2)" — checked directly against a real engine during
this script's development, and it does NOT hold on this TRT 10.16.1.11 /
Myelin-heavy build: individually-named QuantizeLinear/DequantizeLinear/
Reformat/Softmax nodes do not survive into the engine's layer names at all —
Myelin fusion absorbs them into unnamed "kgen"/"gemm"/"correlation" mega-
kernels (confirmed: zero layer names contain "quant" or "reformat" on a real
FP8 engine, vs FP16's inspector JSON exposing "Reformat"/"CaskConvolution"/
"wait"/"signal" as distinct LayerType buckets that don't exist on the FP8
side at all). So the categorization implemented here is a LayerType histogram
diff instead — what the inspector JSON actually exposes — not the
name-substring breakdown the source plan assumed before this was checked.

VRAM discipline: engines are deserialized and freed (del + gc.collect() +
torch.cuda.empty_cache()) one at a time, mirroring builder.py's own inspector
block — peak usage stays bounded to the single largest engine on disk (the
FP16 UNet, ~5.6 GiB), not the sum of every engine under --engines-root.

Usage (from the StreamDiffusion/ directory):
    venv/Scripts/python.exe scripts/fp8/probe_engine_evidence.py
    venv/Scripts/python.exe scripts/fp8/probe_engine_evidence.py --engines-root engines/td/stabilityai
"""

import argparse
import collections
import gc
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Repo root on sys.path so `from streamdiffusion` works without install
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import tensorrt as trt
import torch

# _MYL_PARTITION_RE and _count_attn_bmm_dq_fed are module-level in builder.py
# and imported verbatim so this probe and a real build can never drift on
# what "DQ-fed" or "a Myelin partition" means.
from streamdiffusion.acceleration.tensorrt.builder import (
    _MYL_PARTITION_RE,
    _count_attn_bmm_dq_fed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("probe_engine_evidence")

# Verbatim copy of builder.py's fused-MHA regex (inside EngineBuilder.build(),
# FP8 Round 11). Not importable: builder.py defines it as a *local* variable
# inside the build() method, not a module-level name. Kept in sync by hand —
# if builder.py's pattern ever changes, update this one too.
_MHA_RE = re.compile(r"mha|fmha|MultiHead|FlashAttn", re.IGNORECASE)

_DEFAULT_ENGINES_ROOT = _REPO_ROOT / "engines" / "td" / "stabilityai"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engines-root", type=Path, default=_DEFAULT_ENGINES_ROOT)
    p.add_argument(
        "--engine-filename",
        default="unet.engine",
        help="Only UNet engines carry MHA/attention evidence; VAE engines "
        "(vae_encoder.engine/vae_decoder.engine) are skipped by not matching this name.",
    )
    return p.parse_args()


def _find_unet_engines(root: Path, filename: str):
    if not root.exists():
        return []
    return sorted(root.rglob(filename))


def _inspect_engine(engine_path: Path) -> dict:
    """Deserialize one engine, extract its layer-info JSON and IO signature,
    then free it immediately — mirrors builder.py's own del + gc.collect() +
    torch.cuda.empty_cache() sequence in the inspector block (FP8 Round 11)."""
    trt_logger = trt.Logger(trt.Logger.WARNING)
    rt = trt.Runtime(trt_logger)
    with open(engine_path, "rb") as f:
        data = f.read()
    eng = rt.deserialize_cuda_engine(data)
    if eng is None:
        return {"error": "deserialize_cuda_engine returned None"}

    insp = eng.create_engine_inspector()
    info_json = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    io_names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    del insp, eng
    gc.collect()
    torch.cuda.empty_cache()

    try:
        layers = json.loads(info_json).get("Layers", [])
    except Exception as e:
        logger.warning(f"[probe] {engine_path}: failed to parse layer-info JSON ({e})")
        layers = []

    mha_names = [layer.get("Name", "") for layer in layers if _MHA_RE.search(layer.get("Name", ""))]
    partitions = {m.group(1) for n in mha_names for m in [_MYL_PARTITION_RE.search(n)] if m}
    layer_type_hist = collections.Counter(layer.get("LayerType", "?") for layer in layers)

    return {
        "total_layers": len(layers),
        "mha_fused_kernels": len(mha_names),
        "mha_myelin_partitions": len(partitions),
        "layer_type_histogram": dict(layer_type_hist),
        "io_tensor_count": len(io_names),
        "kvo_cache_in_count": sum(1 for n in io_names if n.startswith("kvo_cache_in_")),
        "fio_cache_in_count": sum(1 for n in io_names if n.startswith("fio_cache_in_")),
    }


def _inspect_calib_npz(npz_path: Path) -> dict:
    if not npz_path.exists():
        return {}
    try:
        data = np.load(npz_path)
    except Exception as e:
        logger.warning(f"[probe] {npz_path}: failed to load ({e})")
        return {}
    if "timestep" not in data:
        return {"row_count": None, "distinct_timesteps": None}
    ts = data["timestep"].reshape(-1)
    return {"row_count": int(ts.shape[0]), "distinct_timesteps": int(len(set(ts.tolist())))}


def _read_build_stats(engine_dir: Path) -> dict:
    p = engine_dir / "build_stats.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        logger.warning(f"[probe] {p}: failed to parse ({e})")
        return {}


def _infer_precision(engine_dir_name: str, stats: dict) -> str:
    """build_stats.json only carries an explicit "precision" key from FP8
    Round 11 (Change A) onward. For older engines, fall back to the presence
    of fp8-only stats keys, then to the directory naming convention
    (`--fp8vN-mhaq--` vs plain). Safe only within the unet.engine universe
    this script targets — VAE engine dirs never reach here since they don't
    contain a file named "unet.engine"."""
    if stats.get("precision"):
        return stats["precision"]
    if "fp8_qdq_layers" in stats or "fp8" in engine_dir_name.lower():
        return "fp8"
    return "fp16"


def _cross_check(inspected: dict, stats: dict) -> list:
    """Compare freshly re-inspected values against whatever build_stats.json
    persisted. A missing key is silently skipped (older build_stats.json
    predates the key), not reported as a mismatch."""
    mismatches = []
    field_map = {
        "total_engine_layers": "total_layers",
        "mha_fused_kernels": "mha_fused_kernels",
        "mha_myelin_partitions": "mha_myelin_partitions",
    }
    for stats_key, inspected_key in field_map.items():
        if stats_key not in stats:
            continue
        if stats[stats_key] != inspected[inspected_key]:
            mismatches.append(
                f"{stats_key}: build_stats.json={stats[stats_key]!r} vs re-inspected={inspected[inspected_key]!r}"
            )
    return mismatches


def _res_key(engine_dir_name: str) -> str:
    """Grouping key so only resolution-matched engines get diffed against
    each other (a 512x512 FP8 engine vs a 1024x1024 FP16 engine would produce
    a meaningless layer-count delta)."""
    m = re.search(r"--res-(\d+x\d+)", engine_dir_name)
    return m.group(1) if m else "unknown-res"


def _report_precision_deltas(results: dict):
    """Profiling-track step 1 fold-in — see module docstring for why this is
    a LayerType histogram diff, not a Quantize/Reformat name-substring
    breakdown. Grouped by resolution only; batch-size mismatches between the
    compared engines are flagged, not silently ignored (the plan's own
    unresolved "batch confound" — see Follow-ups)."""
    by_res = collections.defaultdict(lambda: {"fp8": [], "fp16": []})
    for name, r in results.items():
        if r.get("precision") in ("fp8", "fp16"):
            by_res[_res_key(name)][r["precision"]].append((name, r))

    for res, precisions in sorted(by_res.items()):
        fp16_list, fp8_list = precisions["fp16"], precisions["fp8"]
        if not fp16_list or not fp8_list:
            continue
        fp16_name, fp16_r = fp16_list[0]
        if len(fp16_list) > 1:
            logger.warning(
                f"[probe] Multiple fp16 unet engines at {res}; using {fp16_name} as the baseline (first found)."
            )

        for fp8_name, fp8_r in fp8_list:
            print()
            print("=" * 78)
            print(f"Precision delta @ {res}: {fp8_name} (fp8) vs {fp16_name} (fp16)")
            print("=" * 78)

            fp8_batch = fp8_r.get("build_stats", {}).get("batch_size")
            fp16_batch = fp16_r.get("build_stats", {}).get("batch_size")
            if fp8_batch is not None and fp16_batch is not None and fp8_batch != fp16_batch:
                logger.warning(
                    f"[probe] Batch-size mismatch at {res}: fp8={fp8_batch} vs fp16={fp16_batch} — "
                    "the delta below is confounded by batch size, not precision alone (unresolved per the plan; "
                    "needs a same-batch build of the missing precision to settle)."
                )
            print(f"  batch_size: fp8={fp8_batch}  fp16={fp16_batch}")

            surplus = fp8_r["total_layers"] - fp16_r["total_layers"]
            print(f"  total_layers: fp8={fp8_r['total_layers']}  fp16={fp16_r['total_layers']}  surplus={surplus:+d}")
            print(f"  mha_fused_kernels: fp8={fp8_r['mha_fused_kernels']}  fp16={fp16_r['mha_fused_kernels']}")
            print(
                f"  mha_myelin_partitions: fp8={fp8_r['mha_myelin_partitions']}  fp16={fp16_r['mha_myelin_partitions']}"
            )

            fp8_hist, fp16_hist = fp8_r["layer_type_histogram"], fp16_r["layer_type_histogram"]
            all_types = sorted(
                set(fp8_hist) | set(fp16_hist),
                key=lambda t: -(fp8_hist.get(t, 0) - fp16_hist.get(t, 0)),
            )
            print("  LayerType delta (fp8 - fp16), most-fp8-heavy first (only where they differ):")
            for t in all_types:
                a, b = fp8_hist.get(t, 0), fp16_hist.get(t, 0)
                if a == b:
                    continue
                print(f"    {t}: fp8={a}  fp16={b}  delta={a - b:+d}")


def main() -> None:
    args = _parse_args()
    engine_paths = _find_unet_engines(args.engines_root, args.engine_filename)
    if not engine_paths:
        logger.warning(f"[probe] No '{args.engine_filename}' found under {args.engines_root}")
        return

    logger.info(f"[probe] Found {len(engine_paths)} '{args.engine_filename}' file(s) under {args.engines_root}")

    results = {}
    for engine_path in engine_paths:
        engine_dir = engine_path.parent
        size_gib = engine_path.stat().st_size / 1024**3
        logger.info(f"[probe] Inspecting {engine_dir.name} ({size_gib:.2f} GiB)...")

        inspected = _inspect_engine(engine_path)
        if "error" in inspected:
            logger.warning(f"[probe] {engine_dir.name}: {inspected['error']}")
            continue

        stats = _read_build_stats(engine_dir)
        precision = _infer_precision(engine_dir.name, stats)
        calib = _inspect_calib_npz(engine_dir / "calib_data.npz")

        fp8_onnx = engine_dir / "unet.fp8.onnx"
        bmm_result = _count_attn_bmm_dq_fed(str(fp8_onnx)) if fp8_onnx.exists() else None

        mismatches = _cross_check(inspected, stats)

        results[engine_dir.name] = {
            "precision": precision,
            "build_stats": stats,
            **inspected,
            "calib": calib,
            "attn_bmm_dq_fed": bmm_result[0] if bmm_result else None,
            "attn_bmm_total": bmm_result[1] if bmm_result else None,
        }

        print()
        print("=" * 78)
        print(f"{engine_dir.name}  [{precision}]")
        print("=" * 78)
        print(
            f"  total_layers={inspected['total_layers']}  "
            f"mha_fused_kernels={inspected['mha_fused_kernels']}  "
            f"mha_myelin_partitions={inspected['mha_myelin_partitions']}"
        )
        print(
            f"  io_tensors={inspected['io_tensor_count']}  "
            f"kvo_cache_in={inspected['kvo_cache_in_count']}  "
            f"fio_cache_in={inspected['fio_cache_in_count']}"
        )
        top_types = sorted(inspected["layer_type_histogram"].items(), key=lambda kv: -kv[1])
        print(f"  layer_type_histogram: {dict(top_types)}")
        if bmm_result is not None:
            print(f"  attn_bmm_dq_fed: {bmm_result[0]}/{bmm_result[1]} DQ-fed")
        elif fp8_onnx.exists():
            print("  attn_bmm_dq_fed: <onnx present but parse failed>")
        if calib:
            print(
                f"  calib_data.npz: {calib.get('row_count')} rows, {calib.get('distinct_timesteps')} distinct timesteps"
            )
        if mismatches:
            for m in mismatches:
                logger.warning(f"[probe] {engine_dir.name}: build_stats.json mismatch -- {m}")
        elif stats:
            print(f"  build_stats.json: cross-checked, no mismatches ({len(stats)} top-level keys)")
        else:
            print("  build_stats.json: not found")

    _report_precision_deltas(results)

    print()
    print("=" * 78)
    print(f"Done: inspected {len(results)}/{len(engine_paths)} engine(s).")
    print("=" * 78)


if __name__ == "__main__":
    main()
