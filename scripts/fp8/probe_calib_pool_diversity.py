"""
FP8 Round 14 — permanent, read-only K/V-cache calibration diversity probe.

Context: the calv7 provenance sidecar (fp8-round-13's calib_data.meta.json) exposed
`"pool_missed_calls": [36, 45, 54, 63]` on both calv7 engines -- half of every
staggered stagger-fix selection (fp8-round-11) was silently dropped by the K/V/FI
recorder before _pool_for_layer ever saw it, because `_MAX_CALIB_RECORD_BYTES`
(4 GiB) trips before the widened `_MAX_CALIB_RECORD_CALLS` (64) at ~124 MiB/call on
a real SDXL-Turbo UNet. `_pool_for_layer` does not zero-fill a missed call -- it
skips it, shrinking the pool, and `_fill_kvo_from_pool`'s frame-shift-by-one cyclic
tiling (`src = (m + 1) % pool_len`) reuses the survivors, silently halving
calibration diversity rather than raising or zero-filling. capture_calibration_data
now predicts (before the capture loop runs, `_predict_selected_calls`) exactly
which calls will be selected and gates the recorder on membership in that set
instead of a call-count prefix -- the fix this probe verifies.

This is the ad-hoc measurement that surfaced the defect, made permanent and
repeatable -- same convention as the other scripts/fp8/ probes:
  - probe_cache_saturation.py (round-6 Step 0): settle a rebuild decision before
    paying for a ~50 minute build.
  - measure_fp8_calib_amax.py (round-9.1 §10): real-vs-surrogate token magnitude.
  - probe_engine_evidence.py (round-11): "the diagnosis loop itself permanent".

What it reports per `calib_data.npz` found under --engines-root:
  - Ground truth, direct from disk: for every `kvo_cache_in_i` array actually
    saved into the npz, split it into its n_itr (K, V) chunk pairs (mirroring
    `_fill_kvo_from_pool`'s `arr[m*2+0]=K, arr[m*2+1]=V` layout) and count how
    many are byte-distinct via SHA-1 hash. This is exactly the method used to
    produce the 8/4/4 distinct-K-source table in the FP8 Round 14 plan and
    SESSION_LOG entry -- no production code involved, just the artifact itself.
  - Cross-check via the REAL production pooling/tiling code (not reimplemented):
    given the sidecar's own `selected_calls`/`pool_missed_calls`, the "kept"
    subset (selected minus missed) is fed through the actual `_pool_for_layer`
    + `_fill_kvo_from_pool` using single-value markers (each recorded call's
    synthetic activation IS its own call index), and the resulting distinct-
    source count is compared against the ground-truth hash count above. A
    mismatch would mean either the sidecar's own miss-list is stale relative to
    what's actually in the npz, or a future change to the frame-shift/tiling
    logic broke the simple "N selected - M missed = N-M distinct" invariant --
    either way, this catches it without hand-deriving the tiling math.
  - Predicted vs. recorded calls: re-derives `_predict_selected_calls`'s output
    from build_stats.json (batch_size) + the sidecar (num_inference_steps,
    prompt_count) and checks it against the sidecar's `selected_calls` --
    reusing the real function so this probe can never silently drift from what
    a real capture predicts.

Usage (from the StreamDiffusion/ directory):
    venv/Scripts/python.exe scripts/fp8/probe_calib_pool_diversity.py
    venv/Scripts/python.exe scripts/fp8/probe_calib_pool_diversity.py --engines-root engines/td/stabilityai
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Repo root on sys.path so `from streamdiffusion` works without install
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# Imported, not reimplemented (FP8 Round 14 convention -- see probe_engine_evidence.py's
# _count_attn_bmm_dq_fed reuse for the precedent): this probe's "expected distinct
# sources" number can never silently drift from what a real capture would actually
# produce, even if the frame-shift/tiling math in fp8_quantize.py changes later.
from streamdiffusion.acceleration.tensorrt.fp8_quantize import (
    _MAX_CALIB_ROWS,
    _fill_kvo_from_pool,
    _pool_for_layer,
    _predict_selected_calls,
    load_calib_provenance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("probe_calib_pool_diversity")

_DEFAULT_ENGINES_ROOT = _REPO_ROOT / "engines" / "td" / "stabilityai"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engines-root", type=Path, default=_DEFAULT_ENGINES_ROOT)
    return p.parse_args()


def _find_calib_npz(root: Path):
    if not root.exists():
        return []
    return sorted(root.glob("*/calib_data.npz"))


def _read_build_stats(engine_dir: Path) -> dict:
    p = engine_dir / "build_stats.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        logger.warning(f"[probe] {p}: failed to parse ({e})")
        return {}


def _fmt_bytes(n) -> str:
    return f"{n / 1024**3:.2f} GiB" if isinstance(n, (int, float)) else str(n)


def _distinct_kv_sources(npz_data: dict) -> dict:
    """Ground-truth measurement, direct from the saved npz: for every
    kvo_cache_in_i array, split into n_itr (K, V) chunk pairs and count how
    many are byte-distinct. See module docstring."""
    result = {}
    for name in sorted(npz_data):
        if not name.startswith("kvo_cache_in_"):
            continue
        arr = npz_data[name]
        n_itr = arr.shape[0] // 2
        if n_itr == 0:
            continue
        k_hashes = {hashlib.sha1(np.ascontiguousarray(arr[m * 2 + 0]).tobytes()).hexdigest() for m in range(n_itr)}
        v_hashes = {hashlib.sha1(np.ascontiguousarray(arr[m * 2 + 1]).tobytes()).hexdigest() for m in range(n_itr)}
        result[name] = {"n_itr": n_itr, "distinct_k": len(k_hashes), "distinct_v": len(v_hashes)}
    return result


def _expected_distinct_kvo(selected_calls, kept_calls, n_itr: int):
    """Run the REAL production _pool_for_layer + _fill_kvo_from_pool (not a
    reimplementation) on synthetic single-value markers -- each "recorded"
    call's fake activation IS its own call index -- so the resulting distinct-
    source count after pooling/tiling can be read straight back out, and this
    probe's expectation can never drift from the actual frame-shift/cyclic-
    reuse logic even if that logic changes. Returns (distinct_k, missed_calls).
    """
    missed: set = set()
    # (1 sub-row, seq=1, hidden=1): _pool_for_layer indexes _rec[_r] for _r in
    # range(_rec.shape[0]) to unpack possibly-multiple captured (seq, hidden)
    # rows per call -- the leading dim of 1 here is that "1 sub-row", not the
    # (seq, hidden) shape itself.
    marker_shape = (1, 1, 1)
    k_records = {int(c): np.full(marker_shape, float(c), dtype=np.float32) for c in kept_calls}
    v_records = {int(c): np.full(marker_shape, float(c), dtype=np.float32) for c in kept_calls}
    k_pool = _pool_for_layer(k_records, selected_calls, missed)
    v_pool = _pool_for_layer(v_records, selected_calls, missed)
    arr_shape = [2 * n_itr, 1, 1]
    arr = _fill_kvo_from_pool(k_pool, v_pool, arr_shape, np.float32)
    k_values = {float(arr[m * 2 + 0].flat[0]) for m in range(n_itr)}
    return len(k_values), sorted(missed)


def _rederive_predicted_calls(stats: dict, provenance: dict):
    """Re-derive what _predict_selected_calls would output for this capture,
    from build_stats.json (batch_size) + the sidecar (num_inference_steps,
    prompt_count) alone -- reuses the real function, see module docstring.

    Approximate (not exact) for an explicit-schedule capture (timesteps=.../
    scheduler_ref=..., fp8-round-5-handoff Step 2d): the sidecar's
    num_inference_steps is the value passed into capture_calibration_data, not
    necessarily len(timesteps) when those differ. Exact for every ordinary
    num_inference_steps capture -- every production case observed so far.
    Returns None when the inputs needed aren't available (e.g. a pre-fp8-
    round-13 engine with no sidecar at all).
    """
    capture_stats = (stats or {}).get("stages", {}).get("fp8_calib_capture", {})
    num_inference_steps = (provenance or {}).get("num_inference_steps") or capture_stats.get("num_inference_steps")
    prompt_count = (provenance or {}).get("prompt_count") or capture_stats.get("prompt_count")
    batch_size = (stats or {}).get("batch_size")
    if not num_inference_steps or not prompt_count or not batch_size:
        return None
    num_batches = -(-int(prompt_count) // int(batch_size))  # ceil division
    return {int(c) for c in _predict_selected_calls(int(num_inference_steps), num_batches, _MAX_CALIB_ROWS)}


def main() -> None:
    args = _parse_args()
    npz_paths = _find_calib_npz(args.engines_root)
    if not npz_paths:
        logger.warning(f"[probe] No calib_data.npz found under {args.engines_root}")
        return
    logger.info(f"[probe] Found {len(npz_paths)} calib_data.npz under {args.engines_root}")

    for npz_path in npz_paths:
        engine_dir = npz_path.parent
        try:
            with np.load(npz_path) as _npz:
                data = {k: _npz[k] for k in _npz.files}
        except Exception as e:
            logger.warning(f"[probe] {npz_path}: failed to load ({e})")
            continue

        stats = _read_build_stats(engine_dir)
        provenance = load_calib_provenance(str(npz_path))
        diversity = _distinct_kv_sources(data)

        print()
        print("=" * 78)
        print(f"{engine_dir.name}")
        print("=" * 78)

        if provenance is None:
            print("  calib_data.meta.json: not found (pre-fp8-round-13 engine)")
        else:
            print(f"  sidecar schema_version={provenance.get('schema_version')}")
            print(f"  selected_calls={provenance.get('selected_calls')}")
            print(f"  pool_missed_calls={provenance.get('pool_missed_calls')}")
            if provenance.get("schema_version", 1) >= 2:
                print(f"  predicted_calls (recorded, schema v2)={provenance.get('predicted_calls')}")
                print(
                    f"  record_calls_kept={provenance.get('record_calls_kept')}  "
                    f"record_bytes={_fmt_bytes(provenance.get('record_bytes'))}  "
                    f"kvo_pool_len={provenance.get('kvo_pool_len')}"
                )

        predicted = _rederive_predicted_calls(stats, provenance)
        if predicted is not None:
            print(f"  re-derived predicted_calls (via _predict_selected_calls)={sorted(predicted)}")
            selected = {int(c) for c in (provenance or {}).get("selected_calls") or []}
            if selected and not selected <= predicted:
                logger.warning(
                    f"[probe] {engine_dir.name}: selected_calls "
                    f"{sorted(selected - predicted)} not covered by the re-derived prediction -- "
                    "the superset guarantee (_predict_selected_calls docstring) appears violated, investigate."
                )
        else:
            print("  re-derived predicted_calls: unavailable (missing num_inference_steps/prompt_count/batch_size)")

        if not diversity:
            print("  kvo_cache_in_*: none found in this npz (use_cached_attn was False for this capture)")
        else:
            for name, d in sorted(diversity.items()):
                print(f"  {name}: n_itr={d['n_itr']}  distinct_k={d['distinct_k']}  distinct_v={d['distinct_v']}")
            headline = diversity.get("kvo_cache_in_0")
            if headline:
                print(f"  headline (kvo_cache_in_0): {headline['distinct_k']}/{headline['n_itr']} distinct K sources")

            if provenance is not None and provenance.get("selected_calls") and headline:
                selected_calls = provenance["selected_calls"]
                missed = set(provenance.get("pool_missed_calls") or [])
                kept_calls = [c for c in selected_calls if c not in missed]
                expected_k, re_missed = _expected_distinct_kvo(selected_calls, kept_calls, headline["n_itr"])
                print(
                    f"  cross-check via real _pool_for_layer/_fill_kvo_from_pool: expected_distinct_k={expected_k}"
                    + (f"  (re-derived missed={sorted(re_missed)})" if re_missed else "")
                )
                if expected_k != headline["distinct_k"]:
                    logger.warning(
                        f"[probe] {engine_dir.name}: ground-truth distinct_k ({headline['distinct_k']}) != "
                        f"cross-check expected_distinct_k ({expected_k}) derived from the sidecar's own "
                        "selected_calls/pool_missed_calls -- the sidecar may be stale relative to this npz, "
                        "or investigate."
                    )

    print()
    print("=" * 78)
    print(f"Done: inspected {len(npz_paths)} calib_data.npz file(s).")
    print("=" * 78)


if __name__ == "__main__":
    main()
