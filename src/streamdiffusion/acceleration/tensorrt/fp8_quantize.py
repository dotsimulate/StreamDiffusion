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

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_BUNDLED_PROMPTS_PATH = Path(__file__).parent / "calibration_prompts_sdxl.txt"

# fp8-round-9.1: extensions recognized by both wrapper.py's
# _load_fp8_calibration_style_images (loads the files) and engine_manager.py's
# _calibration_image_signature (hashes them for the --ci<hash> cache-key
# fork). Both import this constant rather than each maintaining their own
# list — a mismatch would let the cache key disagree with what was actually
# loaded (e.g. tag hashes a file the loader silently skips).
_CALIBRATION_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _list_calibration_images(path: Optional[str]) -> List[Path]:
    """Resolve `fp8_calibration_style_image` (a file or a directory) to the
    ordered list of image files it denotes (fp8-round-9.1 / round-9.1 §10).

    Single source of truth for "which files count as calibration images" —
    wrapper.py's loader, wrapper.py's mode-folder resolver, and
    engine_manager.py's `--ci<hash>` signature all call this instead of each
    re-implementing the same `iterdir()` + sort + extension filter, so they
    cannot silently drift apart (e.g. the cache key hashing a file the loader
    skips, or vice versa).

    Sorted by filename for a deterministic, reproducible order. Deliberately
    **non-recursive** — a directory is `iterdir()`'d one level only, so a
    nested subdirectory (e.g. a `general/`/`faces/` mode split sitting a
    level below `path`) is invisible here unless `path` itself points at it.
    A `path` that names a single file is returned as that one-element list
    unconditionally (matching the pre-existing loader/hasher behaviour) —
    the extension filter only applies when listing a directory. Returns `[]`
    for a path that is `None`, doesn't exist, or (as a directory) contains
    nothing with a recognized extension.
    """
    if not path:
        return []
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in _CALIBRATION_IMAGE_EXTENSIONS)
    if p.is_file():
        return [p]
    return []


# Row budget shared with wrapper.py's build_calibration_t_indices() call (fp8-round-5-handoff
# Step 2b/2c) so the derived calibration timestep schedule and the row cap that consumes it
# never drift apart. See the n_itr comment below for why this must stay small.
_MAX_CALIB_ROWS = 8

# Upper bound on how many *captured calls* (not final selected rows) the K/V/FI
# recorder in capture_calibration_data keeps in memory (fp8-round-5-handoff Step 2).
# FP8 Round 11 raised this from 4*_MAX_CALIB_ROWS (32) to _MAX_CALIB_ROWS**2 (64) on
# the theory that a full stagger pass needs a window that wide -- true in principle,
# but _MAX_CALIB_RECORD_BYTES below (4 GiB) trips first on a real SDXL-Turbo UNet, at
# ~33 of the needed 64 calls (~124 MiB/call). Calls 36/45/54/63 (the second half of
# the stagger) were silently dropped every calv7 build, and _pool_for_layer reusing
# the surviving 4 recorded sources twice each halved kvo/fio calibration diversity
# (measured: 8 distinct K/V sources pre-Round-11 -> 4 under calv7, both resolutions
# -- see the calv8 note in engine_manager.py). FP8 Round 14 fixed this at the root:
# capture_calibration_data now predicts the ~8 calls _select_calibration_calls will
# actually select (_predict_selected_calls, above _select_calibration_rows) *before*
# recording starts, and the hooks admit only those -- so this call-count cap is now a
# hard ceiling behind predictive selection, not a window predictive selection has to
# fit inside, and normal-path peak memory drops to ~1 GiB. It still matters as the
# fallback cap for the rare case a mid-capture batch failure invalidates the
# prediction (call indices shift) -- see _predict_valid in capture_calibration_data.
# Override: SDTD_FP8_CALIB_RECORD_CALLS=<int>
_MAX_CALIB_RECORD_CALLS = int(os.environ.get("SDTD_FP8_CALIB_RECORD_CALLS", _MAX_CALIB_ROWS**2))

# Byte-budget companion to the call cap above (FP8 Round 11 decision: a flat cap
# increase alone is unsafe on this host -- 31.6 GB RAM, 16.8 GB free measured this
# round, with the final aligned calib_data.npz at 1024 prompts already 17.8 GiB.
# The K/V/FI recorder pool is live in memory throughout capture, alongside the
# not-yet-trimmed captured-activations dict, so doubling the call cap with no
# ceiling risks an OOM mid-capture instead of a graceful degrade. This is an
# estimate, not a measured figure -- raise via the override below for hosts with
# more headroom or larger per-layer tensors. Self-limiting: hooks stop recording
# new calls once either this or _MAX_CALIB_RECORD_CALLS is hit, whichever comes
# first, and log how many calls were actually kept. FP8 Round 14: under normal
# (non-fallback) operation the recorder only ever admits the ~8 predicted calls, so
# this cap sits at ~1 GiB of real usage and is not expected to trip -- it remains a
# hard OOM backstop for the fallback path and for future larger resolutions/tensors.
# Override: SDTD_FP8_CALIB_RECORD_BYTES=<bytes>
_MAX_CALIB_RECORD_BYTES = int(os.environ.get("SDTD_FP8_CALIB_RECORD_BYTES", 4 * 1024**3))


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
    with open(path, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    logger.info(f"[FP8] Loaded {len(prompts)} calibration prompts from {path.name}")
    return prompts


def _select_calibration_calls(calib_data: Dict[str, np.ndarray], max_rows: int) -> np.ndarray:
    """Pick which captured *calls* (keyed on ``timestep``, 1 row/call) survive
    an ``max_rows``-row budget, stratified round-robin over the distinct
    timestep values actually visited (descending, so every calibrated
    timestep is represented before any is doubled up).

    Extracted out of ``_select_calibration_rows`` (fp8-round-5-handoff Step 2)
    so the K/V/FI cache recorder in ``capture_calibration_data`` can select
    the exact same calls ``_select_calibration_rows`` will keep, without
    ``_select_calibration_rows`` itself needing to expose call indices — its
    return type (a row-expanded dict) is load-bearing for
    ``tests/quality/test_fp8_calib_tile.py``.

    Returns an empty array if ``timestep`` is missing or there are 0 calls;
    returns every call index (``0..num_calls-1``) if the capture already fits
    the budget.
    """
    if "timestep" not in calib_data or not calib_data:
        return np.array([], dtype=int)

    num_calls = calib_data["timestep"].shape[0]
    if num_calls <= 0:
        return np.array([], dtype=int)

    multipliers = {}
    for key, arr in calib_data.items():
        n = arr.shape[0]
        m = n // num_calls if n % num_calls == 0 else 1
        multipliers[key] = max(m, 1)
    max_multiplier = max(multipliers.values())

    # Budget *calls*, not rows, so the most CFG-doubled key still lands at
    # <= max_rows rows after expansion.
    calls_budget = max(1, max_rows // max_multiplier)
    if num_calls <= calls_budget:
        return np.arange(num_calls, dtype=int)

    timestep_values = calib_data["timestep"].reshape(num_calls, -1)[:, 0].astype(np.float64)
    groups: Dict[float, List[int]] = {}
    for call_idx, t in enumerate(timestep_values):
        groups.setdefault(float(t), []).append(call_idx)

    # Round-robin one call per distinct timestep (highest first, matching the
    # descending schedule order) until the budget is spent.
    ordered_keys = sorted(groups.keys(), reverse=True)
    selected: List[int] = []
    round_idx = 0
    while len(selected) < calls_budget:
        progressed = False
        for pos, tkey in enumerate(ordered_keys):
            bucket = groups[tkey]
            # Stagger: position `pos` within this round draws bucket[(round_idx +
            # pos) % len(bucket)], not bucket[round_idx] for every key. The old
            # per-round-shared index meant that whenever calls_budget matched the
            # distinct-timestep count (the common case: 8 timesteps, budget 8),
            # only round 0 ever ran, and round 0 = occurrence 0 of every timestep
            # = the first pipe() batch, for all 8 selected calls (FP8 Round 11
            # Item 2). Staggering by position spreads a single round across as
            # many distinct occurrences (batches) as the bucket holds, wrapping
            # via modulo instead of dropping when a bucket is shorter than the
            # round+position offset would otherwise require.
            offset = (round_idx + pos) % len(bucket)
            candidate = bucket[offset]
            if candidate in selected:
                # Already used (bucket shorter than the stagger window already
                # wrapped onto it) -- fall back to the earliest unused occurrence
                # in this bucket rather than a straight duplicate. Bucket is in
                # ascending call-index order, so unused[0] is the earliest.
                unused = [c for c in bucket if c not in selected]
                if not unused:
                    continue
                candidate = unused[0]
            selected.append(candidate)
            progressed = True
            if len(selected) >= calls_budget:
                break
        if not progressed:
            break
        round_idx += 1
    return np.array(sorted(selected), dtype=int)


def _predict_selected_calls(schedule_len: int, num_batches: int, max_rows: int) -> np.ndarray:
    """Predict, *before capture runs*, which call indices ``_select_calibration_calls``
    will keep -- so the K/V/FI recorder in ``capture_calibration_data`` can record only
    those calls instead of an unrelated call-count prefix (FP8 Round 14).

    Round 11 widened ``_MAX_CALIB_RECORD_CALLS`` to 64 so a staggered selection spanning
    up to 8 distinct prompt batches per timestep would fit inside the recorder's kept
    window -- but the companion ``_MAX_CALIB_RECORD_BYTES`` byte budget (4 GiB) trips
    first, at ~33 of the needed 64 calls on a real SDXL-Turbo UNet. Calls 36/45/54/63
    (the second half of the stagger) were silently dropped, and ``_pool_for_layer``
    reusing the surviving 4 sources twice each halved kvo/fio calibration diversity --
    see the FP8 Round 14 SESSION_LOG entry / calv8 cache-tag note in engine_manager.py
    for the measured 8-vs-4-distinct-source evidence. Recording only the ~8 calls that
    will actually be selected fixes this and drops peak recorder memory from ~4 GiB to
    ~1 GiB, well under the byte budget.

    ``_select_calibration_calls`` only needs the per-call ``timestep`` values to decide
    what to select -- not the real ``sample``/``encoder_hidden_states`` tensors -- so a
    synthetic descending-timestep schedule, tiled once per prompt batch (mirroring how
    the real capture visits ``schedule_len`` timesteps per batch, ``num_batches`` times),
    reproduces the exact same call indices. Delegates to ``_select_calibration_calls``
    rather than re-deriving the stagger independently -- this module already carries a
    scar from two copies of comparable per-input-shape logic drifting apart (see
    ``_resolve_per_step_dim0``'s "caused an n_itr=560 blowup" comment) and this is the
    same failure mode waiting to happen again.

    Superset guarantee: call with ``max_rows=_MAX_CALIB_ROWS`` (the largest possible
    row budget). The real post-hoc call inside ``capture_calibration_data`` passes the
    full captured ``calib_data``, whose CFG-doubled keys can shrink
    ``_select_calibration_calls``'s effective ``calls_budget`` below ``max_rows`` (see
    its own docstring) -- which only *truncates* round 0's stagger sequence to a prefix
    of what a full-budget prediction returns. Predicting at the maximum budget is
    therefore always a superset of what will actually be selected, never a miss.
    """
    if schedule_len <= 0 or num_batches <= 0:
        return np.array([], dtype=int)
    stub = {"timestep": np.tile(np.arange(schedule_len, dtype=np.float32)[::-1], num_batches)}
    return _select_calibration_calls(stub, max_rows)


def _recorder_admits_call(call_idx: int, predict_valid_ok: bool, predicted_calls, max_calls: int) -> bool:
    """Whether the K/V/FI recorder should admit ``call_idx`` -- the gating
    decision shared by ``_kv_hook``/``_fio_hook`` inside
    ``capture_calibration_data``, extracted to module level (mirroring
    ``_pool_for_layer``) so the FP8 Round 14 predictive-recording behaviour,
    including its batch-failure fallback, can be pinned without a real
    diffusers pipeline.

    While ``predict_valid_ok`` (the ``_predicted_calls`` set computed before
    the capture loop is still trustworthy), admit only calls in
    ``predicted_calls``. A mid-capture batch failure shifts every subsequent
    call index and invalidates the prediction -- once that happens the caller
    flips ``predict_valid_ok`` to ``False`` and this falls back to the
    pre-Round-14 call-count-prefix rule: admit any call below ``max_calls``.

    The byte-budget cap (``_MAX_CALIB_RECORD_BYTES``) is a separate,
    unconditional check the caller applies after this returns ``True`` -- it
    is not part of this decision.
    """
    if predict_valid_ok:
        return call_idx in predicted_calls
    return call_idx < max_calls


def _select_calibration_rows(calib_data: Dict[str, np.ndarray], max_rows: int) -> Dict[str, np.ndarray]:
    """Downselect captured calibration rows, stratified over the distinct UNet
    timestep values actually visited (fp8-round-5-handoff Step 2e).

    Why not a naive per-key ``linspace(0, n-1, max_rows)``: ``timestep`` is
    captured once per ``unet()`` forward call, but CFG-doubled keys (``sample``,
    ``encoder_hidden_states``, and the SDXL ``added_cond_kwargs`` keys) are
    captured with 2 rows per call whenever ``guidance_scale > 1`` stacks
    cond+uncond into one batched forward. Computing each key's stride from its
    own (different) row count silently misaligns which forward call each key's
    row N came from — row 3 of ``sample`` no longer corresponds to row 3 of
    ``timestep``. Worse, an even stride across many prompts x few timesteps can
    alias onto the same handful of timestep phases and miss others entirely —
    the original round-5 defect this plan's Step 1 isolated.

    Instead this selects whole calls via ``_select_calibration_calls``, then
    expands each selected call into the matching rows for every other key using
    that key's own rows-per-call multiplier. The call budget is sized so the
    most CFG-doubled key still lands at <= max_rows total rows — this preserves
    the exact memory footprint the old per-key cap gave modelopt's n_itr
    derivation downstream, it just aligns the rows correctly across keys.
    """
    if "timestep" not in calib_data or not calib_data:
        # No timestep key captured (shouldn't happen for a real UNet hook) --
        # fall back to the old independent-linspace behavior per key.
        out = dict(calib_data)
        for key, arr in out.items():
            n = arr.shape[0]
            if n > max_rows:
                idx = np.linspace(0, n - 1, max_rows).astype(int)
                out[key] = arr[idx]
        return out

    num_calls = calib_data["timestep"].shape[0]
    if num_calls <= 0:
        return dict(calib_data)

    selected_calls = _select_calibration_calls(calib_data, max_rows)
    if len(selected_calls) >= num_calls:
        return dict(calib_data)

    multipliers = {}
    for key, arr in calib_data.items():
        n = arr.shape[0]
        m = n // num_calls if n % num_calls == 0 else 1
        multipliers[key] = max(m, 1)

    out: Dict[str, np.ndarray] = {}
    for key, arr in calib_data.items():
        n = arr.shape[0]
        multiplier = multipliers[key]
        row_idx = np.concatenate([np.arange(c * multiplier, c * multiplier + multiplier) for c in selected_calls])
        row_idx = row_idx[row_idx < n]
        out[key] = arr[row_idx]
    return out


def _walk_attn1_modules(unet) -> List[Any]:
    """Return ``attn1`` (self-attention) modules in down -> mid -> up walk
    order — identical to ``get_kvo_cache_info`` (``models/utils.py``) and
    ``_collect_fi_processors`` (``export_wrappers/unet_unified_export.py``),
    so index ``i`` lines up 1:1 with ``kvo_cache_in_{i}`` (fp8-round-5-handoff
    Step 2). Those two helpers only return shapes/processor instances, not the
    raw ``attn1`` module objects a K/V recorder hook needs, hence this local
    re-walk instead of reusing them directly.
    """
    modules: List[Any] = []
    for block in unet.down_blocks:
        if getattr(block, "attentions", None) is not None:
            for attn_block in block.attentions:
                for transformer in attn_block.transformer_blocks:
                    modules.append(transformer.attn1)
    if getattr(unet.mid_block, "attentions", None) is not None:
        for attn_block in unet.mid_block.attentions:
            for transformer in attn_block.transformer_blocks:
                modules.append(transformer.attn1)
    for block in unet.up_blocks:
        if getattr(block, "attentions", None) is not None:
            for attn_block in block.attentions:
                for transformer in attn_block.transformer_blocks:
                    modules.append(transformer.attn1)
    return modules


def _reconcile_2d(row: np.ndarray, seq_t: int, hidden_t: int) -> np.ndarray:
    """Pad (zeros) / trim a captured ``(seq, hidden)`` activation row to an
    ONNX-declared ``(seq_t, hidden_t)``, for the K/V/FI rows recorded off the
    bare (pre-export-wrapper) UNet (fp8-round-5-handoff Step 2).

    This always zero-pads, unlike ``_reconcile_calib_to_onnx_dims`` below —
    zero-pad is benign here because K/V/FI rows are never *exclusively* fed by
    the padded region the way ``encoder_hidden_states``' IP-Adapter token slice
    is (fp8-round-9). Do not reuse this for ``encoder_hidden_states``."""
    if row.shape[0] != seq_t:
        row = (
            np.pad(row, [(0, seq_t - row.shape[0]), (0, 0)], mode="constant") if row.shape[0] < seq_t else row[:seq_t]
        )
    if row.shape[1] != hidden_t:
        row = (
            np.pad(row, [(0, 0), (0, hidden_t - row.shape[1])], mode="constant")
            if row.shape[1] < hidden_t
            else row[:, :hidden_t]
        )
    return row


def _reconcile_calib_to_onnx_dims(
    calib_data: Dict[str, np.ndarray],
    specs: Dict[str, Any],
    ipadapter_tokens: Optional[np.ndarray] = None,
    num_ip_layers: int = 0,
) -> Dict[str, np.ndarray]:
    """Reconcile captured tensors with ONNX-declared static dims.

    The bare-pipe forward hook in ``capture_calibration_data`` captures the
    diffusers UNet *before* feature wrappers run, so dims that wrappers
    reshape (e.g. IPA's UnifiedExportWrapper concatenates ``num_image_tokens``
    image tokens onto ``encoder_hidden_states`` → seq_len 77 → 81) won't match
    the exported ONNX. Trim if oversized, on any static (non-dynamic,
    non-leading) axis.

    When undersized, ``encoder_hidden_states`` is special-cased: if
    ``ipadapter_tokens`` is given, the missing width is filled by
    concatenating those real IP-Adapter projection tokens instead of
    zero-padding. ``to_k_ip``/``to_v_ip`` are bias-free ``nn.Linear`` layers,
    so a zero-padded region produces identically-zero activations and starves
    that branch's FP8 calibration entirely — the fp8-round-9 defect (measured:
    140 blocks × 3 tensors × 2 scale initializers = 420 uncalibrated
    initializers landing at ORT's default scale). Every other undersized
    tensor, and ``encoder_hidden_states`` itself when tokens are unavailable,
    still falls back to zero-pad — benign for those, since nothing downstream
    is exclusively fed by the padded region the way the IPA cross-attention
    branch is.

    ``ipadapter_tokens``'s leading axis (N) need not equal the calibration
    row count (``_arr.shape[0]``): N==1 (the zeros-surrogate, a real cached
    embedding, or a single calibration image) broadcasts across every row
    unchanged; N>1 (fp8-round-9.1: multiple ``fp8_calibration_style_image``
    files) tiles round-robin and trims to the row count, so e.g. 2 images
    across 8 rows land as ``[img0, img1, img0, img1, ...]`` — every row still
    gets a real, non-degenerate token, and both images' activation ranges
    land in the same amax statistic instead of only the first.

    Returns a new dict (``calib_data`` is not mutated in place); logs
    ``logger.warning`` when the appended/padded ``encoder_hidden_states``
    region ends up identically zero while ``num_ip_layers > 0`` — the exact
    precondition that produced the original defect.
    """
    calib_data = dict(calib_data)
    for _name, (_, _expected_dims) in specs.items():
        if _name not in calib_data:
            continue
        _arr = calib_data[_name]
        _resized = False
        for _axis, _expected in enumerate(_expected_dims):
            if _expected is None or _axis == 0 or _axis >= _arr.ndim:
                continue
            if _arr.shape[_axis] == _expected:
                continue
            _is_ipa_token_axis = _name == "encoder_hidden_states" and _axis == 1
            if _arr.shape[_axis] < _expected:
                _missing = _expected - _arr.shape[_axis]
                if _is_ipa_token_axis and ipadapter_tokens is not None:
                    _tokens = np.asarray(ipadapter_tokens)
                    if _tokens.ndim == _arr.ndim - 1:
                        _tokens = _tokens[None, ...]
                    if _tokens.shape[0] != _arr.shape[0]:
                        if _tokens.shape[0] <= 1:
                            _tokens = np.broadcast_to(_tokens, (_arr.shape[0],) + _tokens.shape[1:])
                        else:
                            # fp8-round-9.1: N>1 style images. The old broadcast_to here
                            # required N==1 or N==_arr.shape[0] and raised ValueError on
                            # any other N — unreachable while every token source was N==1,
                            # reachable the moment fp8_calibration_style_image points at a
                            # multi-image folder. Tile round-robin instead so every
                            # calibration row still gets a real token and no image is
                            # dropped just because N doesn't divide the row count evenly.
                            _reps = -(-_arr.shape[0] // _tokens.shape[0])  # ceil division
                            _tokens = np.tile(_tokens, (_reps,) + (1,) * (_tokens.ndim - 1))[: _arr.shape[0]]
                    if _tokens.shape[1] > _missing:
                        _tokens = _tokens[:, :_missing]
                    elif _tokens.shape[1] < _missing:
                        _tokens = np.pad(
                            _tokens,
                            [(0, 0), (0, _missing - _tokens.shape[1])] + [(0, 0)] * (_tokens.ndim - 2),
                            mode="edge",
                        )
                    _fill = np.ascontiguousarray(_tokens).astype(_arr.dtype)
                else:
                    _fill_shape = list(_arr.shape)
                    _fill_shape[_axis] = _missing
                    _fill = np.zeros(_fill_shape, dtype=_arr.dtype)
                if _is_ipa_token_axis and num_ip_layers > 0:
                    # fp8-round-9.1: the magnitude on its own turns "did real tokens
                    # arrive" into a grep instead of a post-hoc ONNX census — the
                    # fp8-round-9 plan asked for this and only the zero-warning below
                    # landed.
                    _fill_absmax = float(np.abs(_fill).max())
                    logger.info(
                        f"[FP8] IP-Adapter token region for '{_name}': appended-fill |x|max={_fill_absmax:.6g}"
                    )
                    if _fill_absmax == 0.0:
                        logger.warning(
                            f"[FP8] IP-Adapter token region for '{_name}' is identically zero after "
                            "calibration reconciliation — the IPA cross-attention branch will "
                            "calibrate on zero signal again (fp8-round-9 defect: 420 uncalibrated "
                            "scale initializers). Check the ipadapter_tokens source."
                        )
                _arr = np.concatenate([_arr, _fill], axis=_axis)
            else:
                _slc = [slice(None)] * _arr.ndim
                _slc[_axis] = slice(0, _expected)
                _arr = _arr[tuple(_slc)]
            _resized = True
        if _resized:
            calib_data[_name] = _arr
            logger.info(f"[FP8] Reshaped captured '{_name}' to ONNX dims: shape={_arr.shape}")
    return calib_data


def _pool_for_layer(
    records: Dict[int, np.ndarray], selected_calls, missed_calls: Optional[set] = None
) -> List[np.ndarray]:
    """Row-major (call, subrow) pool of real 2-D (seq, hidden) activations for
    one layer, restricted to ``selected_calls`` (the calls
    ``_select_calibration_rows`` is keeping), in call-ascending order.

    Extracted to module level (FP8 Round 11, mirroring the
    ``_build_capture_call_kwargs`` precedent) so this can be pinned directly
    without a real diffusers pipeline — it used to be a closure inside
    ``capture_calibration_data`` over ``_selected_calls``/``_pool_missed_calls``.

    A selected call missing from ``records`` — outside the K/V/FI recorder's
    kept window (see ``_MAX_CALIB_RECORD_CALLS`` / ``_MAX_CALIB_RECORD_BYTES``)
    — is skipped, not raised, and its index is added to ``missed_calls`` (if
    given) so the caller can log it once as a summary rather than per-miss.
    """
    pool: List[np.ndarray] = []
    for _c in selected_calls:
        _rec = records.get(int(_c))
        if _rec is None:
            if missed_calls is not None:
                missed_calls.add(int(_c))
            continue
        for _r in range(_rec.shape[0]):
            pool.append(_rec[_r])
    return pool


def _fill_kvo_from_pool(k_pool: List[np.ndarray], v_pool: List[np.ndarray], arr_shape: List[int], dtype) -> np.ndarray:
    """Build a ``kvo_cache_in_i`` array (leading axis = n_itr chunks of the
    fixed K+V pair, per ``_resolve_per_step_dim0``) from real recorded K/V
    rows instead of zeros (fp8-round-5-handoff Step 2 / Defect B).

    Each n_itr chunk is filled from the *next* pool position — a frame-shift
    by one, so the "cache" holds a neighbouring call's K/V rather than the
    current call's own (matching deployment: the cache always holds a
    previous frame's K/V, never the current one — see
    ``attention_processors.py``'s ``cached_key``/``cached_value`` concat) —
    tiled across every remaining axis (maxframes, batch). Falls back to an
    all-zeros array (the prior behaviour) when no recording exists for this
    layer.
    """
    arr = np.zeros(arr_shape, dtype=dtype)
    pool_len = min(len(k_pool), len(v_pool))
    if pool_len == 0:
        return arr
    tail_shape = tuple(arr_shape[1:])
    seq_t, hidden_t = tail_shape[-2], tail_shape[-1]
    n_itr = arr_shape[0] // 2
    for m in range(n_itr):
        src = (m + 1) % pool_len
        k_tile = np.broadcast_to(_reconcile_2d(k_pool[src], seq_t, hidden_t), tail_shape)
        v_tile = np.broadcast_to(_reconcile_2d(v_pool[src], seq_t, hidden_t), tail_shape)
        arr[m * 2 + 0] = k_tile
        arr[m * 2 + 1] = v_tile
    return arr


def _fill_fio_from_pool(pool: List[np.ndarray], arr_shape: List[int], dtype, chunk: int) -> np.ndarray:
    """Build a ``fio_cache_in_i`` array from real recorded ``attn1`` block
    outputs instead of zeros (fp8-round-5-handoff Step 2 / Defect B), using
    the same frame-shift-by-one-pool-position scheme as
    ``_fill_kvo_from_pool``. ``chunk`` is ``per_step_d0`` — the ONNX-declared
    per-call leading-axis size for this input, which folds the maxframes axis
    into axis 0 when it is pinned static (see ``_resolve_per_step_dim0``).
    Falls back to zeros when no recording exists for this layer.
    """
    arr = np.zeros(arr_shape, dtype=dtype)
    pool_len = len(pool)
    if pool_len == 0:
        return arr
    tail_shape = tuple(arr_shape[1:])
    seq_t, hidden_t = tail_shape[-2], tail_shape[-1]
    n_itr = arr_shape[0] // chunk
    for m in range(n_itr):
        src = (m + 1) % pool_len
        tile = np.broadcast_to(_reconcile_2d(pool[src], seq_t, hidden_t), tail_shape)
        arr[m * chunk : (m + 1) * chunk] = tile
    return arr


def _build_capture_call_kwargs(
    batch: List[str],
    guidance_scale: float,
    use_explicit_schedule: bool,
    timesteps: Optional[List[int]],
    num_inference_steps: int,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
) -> Dict[str, Any]:
    """Pure builder for the per-batch ``pipe()`` call kwargs used by
    ``capture_calibration_data``'s capture loop (fp8-round-10). Extracted so a
    unit test can pin its behavior without CUDA/a real diffusers pipeline.

    ``image_height``/``image_width`` are appended only when **both** are
    given — a lone one is ambiguous, mirroring the existing
    ``timesteps``/``scheduler_ref`` one-of-two handling in
    ``capture_calibration_data``. Omitting them (the default, and the only
    behavior every pre-fp8-round-10 caller used) reproduces the exact prior
    kwargs, so 512-native builds are byte-for-byte unaffected by this change.
    """
    call_kwargs: Dict[str, Any] = {
        "prompt": batch if len(batch) > 1 else batch[0],
        "output_type": "latent",
        "guidance_scale": guidance_scale,
    }
    if use_explicit_schedule:
        call_kwargs["timesteps"] = list(timesteps)
    else:
        call_kwargs["num_inference_steps"] = num_inference_steps
    if image_height is not None and image_width is not None:
        call_kwargs["height"] = image_height
        call_kwargs["width"] = image_width
    return call_kwargs


# fp8-round-13: schema version for calib_data.meta.json, bumped whenever a key is
# added/removed/renamed so a future reader can tell which fields to expect.
# v2 (FP8 Round 14): added predicted_calls/record_calls_kept/record_bytes/
# record_max_calls/record_max_bytes/kvo_pool_len -- see _build_calib_provenance.
# load_calib_provenance reads the sidecar as plain JSON with no schema
# validation, so v1 sidecars (missing these keys) still load without raising.
_CALIB_PROVENANCE_SCHEMA_VERSION = 2


def _build_calib_provenance(
    calib_data: Dict[str, np.ndarray],
    *,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
    prompt_count: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    ipadapter_tokens_real: Optional[bool] = None,
    selected_calls: Optional[np.ndarray] = None,
    pool_missed_calls: Optional[set] = None,
    predicted_calls: Optional[Any] = None,
    record_calls_kept: Optional[int] = None,
    record_bytes: Optional[int] = None,
    record_max_calls: Optional[int] = None,
    record_max_bytes: Optional[int] = None,
    kvo_pool_len: Optional[int] = None,
) -> Dict[str, Any]:
    """Pure builder for the ``calib_data.meta.json`` sidecar payload (fp8-round-13
    calibration provenance). Kept separate from the atomic-write wrapper below so
    it can be unit tested without touching disk.

    Answers "how was this cached calib_data.npz actually made?" -- before this,
    a cache hit at the top of ``capture_calibration_data``'s caller
    (``builder.py``) logged only the path, and an aborted run that crashed before
    reaching ``_write_build_stats`` left literally zero trace of having run at
    all (fp8-round-10's ~07:14 aborted capture, found only by the npz's own
    mtime predating the engine it fed). This is metadata-only -- it does not
    change what gets calibrated or how, so it never forks the ``calv*`` cache tag.

    Deliberately NOT written as extra npz keys: ``load_calibration_data`` (below)
    returns every key in the npz verbatim via ``{k: _npz[k] for k in
    _npz.files}``, and the quantize-side row/spec alignment
    (``_reconcile_calib_to_onnx_dims`` et al.) indexes ONNX input specs by name
    for every key present in the loaded dict -- an extra non-tensor key would
    raise ``KeyError`` the first time a cached capture is loaded back for
    quantization. The sidecar is a parallel file for this reason, not stylistic
    preference.

    ``predicted_calls``/``record_calls_kept``/``record_bytes``/``record_max_calls``/
    ``record_max_bytes``/``kvo_pool_len`` (FP8 Round 14, schema v2) record the K/V/FI
    recorder's own state: which calls ``_predict_selected_calls`` predicted, how many
    calls and bytes it actually kept, the caps it was bounded by (both env-overridable
    via ``SDTD_FP8_CALIB_RECORD_CALLS``/``SDTD_FP8_CALIB_RECORD_BYTES`` and therefore
    *not* part of the ``calv*`` cache tag), and the observed kvo pool length after
    filling. All ``None`` on the ControlNet capture path (``capture_calibration_data_
    controlnet``), which has no K/V recorder at all -- expected, not a defect.
    """
    captured_timesteps: List[float] = []
    if "timestep" in calib_data and calib_data["timestep"].size:
        captured_timesteps = sorted({float(v) for v in calib_data["timestep"].reshape(-1)}, reverse=True)

    row_count = None
    for _arr in calib_data.values():
        row_count = int(_arr.shape[0])
        break

    return {
        "schema_version": _CALIB_PROVENANCE_SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_height": image_height,
        "image_width": image_width,
        "prompt_count": prompt_count,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "ipadapter_tokens_real": ipadapter_tokens_real,
        # calib_data here is the final, post-selection dict that was saved to
        # the npz, so "captured" and "selected" coincide -- both are recorded
        # under distinct keys anyway so a future caller that passes the
        # pre-selection dict does not silently mislabel one as the other.
        "captured_timesteps": captured_timesteps,
        "selected_timesteps": captured_timesteps,
        "selected_calls": (sorted(int(c) for c in selected_calls) if selected_calls is not None else None),
        "pool_missed_calls": sorted(int(c) for c in pool_missed_calls) if pool_missed_calls else [],
        "row_count": row_count,
        # FP8 Round 14 / schema v2:
        # `is not None` + `len()`, not a bare truthiness check -- predicted_calls is
        # typed Optional[Any] (the real caller passes a frozenset, but a future caller
        # could pass a numpy array) and `if predicted_calls` raises ValueError ("truth
        # value of an array... is ambiguous") for any multi-element ndarray.
        "predicted_calls": (
            sorted(int(c) for c in predicted_calls) if predicted_calls is not None and len(predicted_calls) else []
        ),
        "record_calls_kept": record_calls_kept,
        "record_bytes": record_bytes,
        "record_max_calls": record_max_calls,
        "record_max_bytes": record_max_bytes,
        "kvo_pool_len": kvo_pool_len,
    }


def _write_calib_provenance_sidecar(save_path: str, provenance: Dict[str, Any]) -> None:
    """Atomically write the provenance sidecar beside ``save_path`` (i.e.
    ``calib_data.npz`` -> ``calib_data.meta.json``, same directory).

    Wrapped in try/except and never raises -- a metadata write must not abort a
    capture that just spent up to ~47 minutes producing the npz it describes.
    """
    meta_path = str(Path(save_path).with_name(Path(save_path).stem + ".meta.json"))
    try:
        tmp_path = meta_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as _f:
            json.dump(provenance, _f, indent=2)
        os.replace(tmp_path, meta_path)
        logger.info(f"[FP8] Saved calibration provenance: {meta_path}")
    except Exception as e:
        logger.warning(f"[FP8] Failed to write calibration provenance sidecar {meta_path}: {e}")


def load_calib_provenance(npz_path: str) -> Optional[Dict[str, Any]]:
    """Load the ``calib_data.meta.json`` sidecar beside ``npz_path``, if present.

    Returns ``None`` (not an error) for every pre-fp8-round-13 engine dir, which
    has a ``calib_data.npz`` but no sidecar -- callers should degrade to a
    "provenance unavailable" message rather than warn.
    """
    meta_path = str(Path(npz_path).with_name(Path(npz_path).stem + ".meta.json"))
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as _f:
            return json.load(_f)
    except Exception as e:
        logger.warning(f"[FP8] Cannot load calibration provenance {meta_path}: {e}")
        return None


def capture_calibration_data(
    pipe,
    prompts: List[str],
    num_inference_steps: int = 20,
    save_path: Optional[str] = None,
    batch_size: int = 1,
    guidance_scale: float = 7.5,
    onnx_path: Optional[str] = None,
    use_cached_attn: bool = False,
    use_feature_injection: bool = False,
    use_controlnet: bool = False,
    num_ip_layers: int = 0,
    max_fi_up_blocks: int = 2,
    fi_strength: float = 0.0,
    fi_threshold: float = 0.0,
    ipadapter_scale: float = 1.0,
    ipadapter_tokens: Optional[np.ndarray] = None,
    timesteps: Optional[List[int]] = None,
    scheduler_ref: Any = None,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
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
            Ignored when ``timesteps``/``scheduler_ref`` are both given.
        save_path: Optional path to write calib_data.npz for caching between builds.
        batch_size: Prompts per pipe() call.
        guidance_scale: CFG scale (affects conditional/unconditional stacking).
        timesteps: Explicit descending raw UNet timestep list to calibrate
            against (fp8-round-5-handoff Step 2d), e.g. the output of
            param_schema.compute_sub_timesteps(build_calibration_t_indices(...)).
            When given together with ``scheduler_ref``, drives
            ``pipe(timesteps=timesteps)`` instead of ``pipe(num_inference_steps=...)``
            — SDXL's ``retrieve_timesteps`` dispatches an explicit ``timesteps=``
            straight to ``scheduler.set_timesteps(timesteps=...)``, bypassing
            ``num_inference_steps`` entirely. This is what lets calibration sample
            the exact region deployment's ``t_index_list`` visits instead of a
            generic turbo/base step sweep.
        scheduler_ref: The deployment scheduler instance (``stream.scheduler``)
            that produced ``timesteps``. ``pipe.scheduler`` is temporarily swapped
            to this for the capture and restored afterward — the checkpoint's
            default scheduler is not identity in ``scale_model_input`` the way the
            deployment LCM scheduler is, so calibrating on the wrong one produces
            mismatched ``sample`` activation magnitudes even with the right
            timestep values.
        use_cached_attn: Whether the deployment engine has StreamV2V cached
            self-attention (``kvo_cache_in_*``) inputs. When True, this also
            records real ``attn1.to_k``/``attn1.to_v`` outputs during the same
            capture pass, so those inputs calibrate on realistic K/V instead
            of zeros (fp8-round-5-handoff Step 2 / Defect B).
        use_feature_injection: Whether the deployment engine also has Feature
            Injection (``fio_cache_in_*``) inputs. Only meaningful together
            with ``use_cached_attn=True``. When True, also records real
            ``attn1`` block outputs for FI-eligible layers during capture.
        max_fi_up_blocks: Must match the value the engine is built with
            (default 2, the same default used everywhere else in this
            codebase — see ``models/utils.get_fi_eligible_mask``). Used to
            recompute which layers are FI-eligible for the recorder above.
        fi_strength: Deployment value of the ``fi_strength`` scalar input
            (fp8-round-5-handoff Step 3). Read from config by the caller —
            defaults to 0.0 (the prior always-zero synthesis) when omitted.
        fi_threshold: Deployment value of the ``fi_threshold`` scalar input.
            Same sourcing/default rationale as ``fi_strength``.
        ipadapter_scale: Deployment value of the ``ipadapter_scale`` input
            (broadcast across all IP-Adapter layers). Defaults to 1.0 (the
            prior always-one synthesis) when omitted.
        ipadapter_tokens: Real IP-Adapter projection tokens, shape
            ``[num_image_tokens, hidden_dim]`` or ``[1, num_image_tokens,
            hidden_dim]``, used by ``_reconcile_calib_to_onnx_dims`` to fill
            the ``encoder_hidden_states`` region the export wrapper appends
            for IPA cross-attention, instead of zero-padding it
            (fp8-round-9). ``None`` (the default) preserves the prior
            zero-pad behavior.
        image_height: Deployment build height in pixels. When given together
            with ``image_width``, passed straight through to ``pipe(height=...,
            width=...)`` so the capture runs at the engine's actual build
            resolution instead of silently falling back to
            ``pipe.unet.config.sample_size`` (512 for SDXL-Turbo) — the
            fp8-round-10 defect. This matters because ``sample``'s H/W ONNX
            axes are exported dynamic, so nothing downstream can recover the
            right shape once the capture itself ran at the wrong one: the
            statically-declared ``kvo_cache_in_*``/``fio_cache_in_*`` inputs
            get zero-padded to the ONNX-declared size by ``_reconcile_2d``,
            and the real-vs-padded fraction is ``(512 / R)²`` — 25% real at
            1024, 11% at 1536. ``None`` (the default) preserves the prior
            resolution-blind behavior.
        image_width: Companion to ``image_height``; see above.

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

    _use_explicit_schedule = timesteps is not None and scheduler_ref is not None
    _pipe_orig_scheduler = pipe.scheduler if _use_explicit_schedule else None
    if _use_explicit_schedule:
        pipe.scheduler = scheduler_ref
        logger.info(
            f"[FP8] Calibrating on explicit deployment schedule: {len(timesteps)} timesteps "
            f"{list(timesteps)} via {type(scheduler_ref).__name__} (overrides num_inference_steps)"
        )
    elif timesteps is not None or scheduler_ref is not None:
        logger.warning(
            "[FP8] capture_calibration_data got only one of timesteps/scheduler_ref "
            f"(timesteps={'set' if timesteps is not None else None}, "
            f"scheduler_ref={'set' if scheduler_ref is not None else None}) — both are "
            "required to drive the explicit schedule; falling back to num_inference_steps."
        )

    if image_height is not None and image_width is not None:
        # width x height, matching builder.py's "[BUILD] FP8 activation capture: ...
        # res={width}x{height}" line -- the two used to disagree in axis order (height x
        # width here), which is invisible at square 512x512 builds but reads as a
        # transposition bug on a non-square build (e.g. 640x384 logged as 640x384 vs
        # 384x640 back to back).
        logger.info(f"[FP8] Calibrating at build resolution {image_width}x{image_height}")
    elif image_height is not None or image_width is not None:
        logger.warning(
            "[FP8] capture_calibration_data got only one of image_height/image_width "
            f"(image_height={image_height}, image_width={image_width}) — both are "
            "required to set the capture resolution; falling back to the checkpoint's "
            "default (pipe.unet.config.sample_size), which is the exact fp8-round-10 defect."
        )

    captured: Dict[str, list] = {}

    # Computed here (rather than inside the capture loop below, where it lived
    # before FP8 Round 14) because the K/V/FI recorder's predicted-calls gate,
    # a few lines down, needs len(batches) before the use_cached_attn block runs.
    batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

    def _to_npy(t):
        # Keep dtype as-is (FP16 model → FP16 captures). modelopt's max-abs
        # calibration does not need FP32; FP32 upcast doubles transfer bandwidth.
        # atleast_1d: SDXL passes timestep as a 0-dim scalar tensor in single-
        # prompt calls; np.concatenate(axis=0) requires at least 1 axis.
        return np.atleast_1d(t.detach().cpu().numpy())

    # fp8-round-5-handoff Step 2 (Defect B): record real per-layer K/V (and, for
    # FI-eligible layers, real attn1 block output) during this same capture pass,
    # keyed by call index, so the kvo_cache_in_*/fio_cache_in_* synthesis below can
    # fill from real deployment-shaped activations instead of zeros. Call index is
    # 0-based and shared with `captured["timestep"]`'s row order (both advance once
    # per pipe.unet() forward call).
    _call_counter = {"n": -1}
    _kv_records: Dict[int, Dict[str, Dict[int, np.ndarray]]] = {}  # layer_idx -> {"k"/"v": {call_idx: arr}}
    _fio_records: Dict[int, Dict[int, np.ndarray]] = {}  # fi_local_idx -> {call_idx: arr}
    _fi_mask: List[bool] = []
    _kv_hook_handles: List[Any] = []
    # Shared mutable byte counter (self-limiting recorder cap, FP8 Round 11) --
    # both hooks below stop recording once this crosses _MAX_CALIB_RECORD_BYTES,
    # independent of the call-count cap. Dict instead of a plain int so the
    # closures below can mutate it without a `nonlocal` declaration per hook.
    _record_bytes = {"n": 0}
    _record_calls_kept: set = set()
    # FP8 Round 14: whether the predicted-calls set below (`_predicted_calls`) is
    # still trustworthy. A mid-capture batch failure (the `except` in the capture
    # loop) shifts every subsequent call index, invalidating the prediction --
    # flipped to False there, at which point the hooks fall back to the pre-Round-14
    # call-count-prefix behavior (still bounded by _MAX_CALIB_RECORD_CALLS/_BYTES).
    _predict_valid = {"ok": True}
    # Initialized ahead of the use_cached_attn branch below (mirroring the
    # _selected_calls/_pool_missed_calls pre-init further down) so the provenance
    # sidecar writer can reference it without a NameError when use_cached_attn is
    # False or K/V recorder setup raises before reassigning it.
    _predicted_calls: frozenset = frozenset()

    def _hook(module, args, kwargs):
        # SDXL pipeline calls unet(sample, t, encoder_hidden_states=..., added_cond_kwargs=...)
        # — encoder_hidden_states arrives as a kwarg, not positional. Fall through to kwargs.
        _call_counter["n"] += 1
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

    if use_cached_attn:
        try:
            from .models.utils import get_fi_eligible_mask

            _attn1_modules = _walk_attn1_modules(pipe.unet)
            # height/width don't change get_fi_eligible_mask's output (eligibility is
            # purely structural — down/mid/up-block position), but threading them
            # keeps this call consistent with capture resolution rather than the
            # function's 512 default (fp8-round-10).
            _fi_mask = (
                list(
                    get_fi_eligible_mask(
                        pipe.unet,
                        height=image_height or 512,
                        width=image_width or 512,
                        max_fi_up_blocks=max_fi_up_blocks,
                    )
                )
                if use_feature_injection
                else [False] * len(_attn1_modules)
            )
            _fi_local_of_global = {gi: li for li, gi in enumerate(gi for gi, e in enumerate(_fi_mask) if e)}

            # FP8 Round 14: predict, before the capture loop runs, exactly which call
            # indices _select_calibration_calls will keep -- see _predict_selected_calls
            # for why this is safe (it's a superset of what post-hoc selection picks) and
            # the module-preamble comment above _MAX_CALIB_RECORD_CALLS for the defect
            # this replaces (byte cap defeating the call cap, halving kvo/fio diversity).
            _schedule_len = len(timesteps) if _use_explicit_schedule else num_inference_steps
            _predicted_calls = frozenset(_predict_selected_calls(_schedule_len, len(batches), _MAX_CALIB_ROWS))

            def _make_kv_hook(_layer_idx, _which):
                def _kv_hook(_module, _inputs, _output):
                    if not _recorder_admits_call(
                        _call_counter["n"], _predict_valid["ok"], _predicted_calls, _MAX_CALIB_RECORD_CALLS
                    ):
                        return
                    if _record_bytes["n"] >= _MAX_CALIB_RECORD_BYTES:
                        return
                    _arr = _to_npy(_output)
                    _kv_records.setdefault(_layer_idx, {}).setdefault(_which, {})[_call_counter["n"]] = _arr
                    _record_bytes["n"] += _arr.nbytes
                    _record_calls_kept.add(_call_counter["n"])

                return _kv_hook

            def _make_fio_hook(_layer_idx):
                _fi_local = _fi_local_of_global[_layer_idx]

                def _fio_hook(_module, _inputs, _output):
                    if not _recorder_admits_call(
                        _call_counter["n"], _predict_valid["ok"], _predicted_calls, _MAX_CALIB_RECORD_CALLS
                    ):
                        return
                    if _record_bytes["n"] >= _MAX_CALIB_RECORD_BYTES:
                        return
                    _out = _output[0] if isinstance(_output, tuple) else _output
                    _arr = _to_npy(_out)
                    _fio_records.setdefault(_fi_local, {})[_call_counter["n"]] = _arr
                    _record_bytes["n"] += _arr.nbytes
                    _record_calls_kept.add(_call_counter["n"])

                return _fio_hook

            for _gi, _attn1 in enumerate(_attn1_modules):
                _kv_hook_handles.append(_attn1.to_k.register_forward_hook(_make_kv_hook(_gi, "k")))
                _kv_hook_handles.append(_attn1.to_v.register_forward_hook(_make_kv_hook(_gi, "v")))
                if _gi < len(_fi_mask) and _fi_mask[_gi]:
                    _kv_hook_handles.append(_attn1.register_forward_hook(_make_fio_hook(_gi)))
            logger.info(
                f"[FP8] K/V recorder armed: {len(_attn1_modules)} attn1 layers"
                + (f", {sum(_fi_mask)} FI-eligible" if use_feature_injection else "")
            )
        except Exception as e:
            logger.warning(f"[FP8] K/V recorder setup failed: {e}. kvo/fio calibration data will fall back to zeros.")
            for _h in _kv_hook_handles:
                _h.remove()
            _kv_hook_handles = []
            _kv_records, _fio_records = {}, {}

    handle = pipe.unet.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            _res_suffix = (
                f" @ {image_width}x{image_height}" if (image_height is not None and image_width is not None) else ""
            )
            for i, batch in enumerate(batches):
                logger.info(f"[FP8] Capture batch {i + 1}/{len(batches)}{_res_suffix}: {batch[0][:60]}")
                _call_kwargs = _build_capture_call_kwargs(
                    batch,
                    guidance_scale,
                    _use_explicit_schedule,
                    timesteps,
                    num_inference_steps,
                    image_height=image_height,
                    image_width=image_width,
                )
                try:
                    _ = pipe(**_call_kwargs).images
                except Exception as e:
                    logger.warning(f"[FP8] Capture batch {i + 1} failed ({type(e).__name__}): {e}. Skipping.")
                    # A skipped batch shifts every subsequent call index, invalidating
                    # the predicted-calls set computed above _make_kv_hook -- fall back
                    # to the pre-Round-14 call-count-prefix admission rule (FP8 Round 14).
                    _predict_valid["ok"] = False
    finally:
        handle.remove()
        for _h in _kv_hook_handles:
            _h.remove()
        if _use_explicit_schedule:
            pipe.scheduler = _pipe_orig_scheduler
        if _unet_orig_device.type != "cuda":
            pipe.unet.to(_unet_orig_device)
            torch.cuda.empty_cache()

    if not captured:
        raise RuntimeError("[FP8] No UNet activations captured — check pipe.unet forward signature.")

    if use_cached_attn and _record_calls_kept:
        logger.info(
            f"[FP8] K/V/FI recorder kept {len(_record_calls_kept)} calls "
            f"({_record_bytes['n'] / 1024**3:.2f} GiB) — capped at "
            f"{_MAX_CALIB_RECORD_CALLS} calls / {_MAX_CALIB_RECORD_BYTES / 1024**3:.1f} GiB."
        )

    calib_data: Dict[str, np.ndarray] = {}
    for key, arrays in captured.items():
        stacked = np.concatenate(arrays, axis=0)
        calib_data[key] = stacked
        logger.info(f"[FP8] Captured '{key}': shape={stacked.shape}, dtype={stacked.dtype}")

    # kvo_cache_in_*/fio_cache_in_* (and ipadapter_scale/control inputs) are not
    # produced by the bare-pipe forward hook above -- they only exist once the
    # export wrappers run. Measured via ONNX graph reachability from these
    # synthetic inputs (fp8-round-5-handoff Defect B): 1782 of 1955 quantized
    # activation tensors (91.2%) are transitively fed by them, so filling them
    # with zeros/ones starves 91.2% of FP8 scales of real signal. Real per-layer
    # K/V (and, for FI, per-layer block output) recorded above during this same
    # capture pass is used instead wherever a recording exists (see
    # _fill_kvo_from_pool / _fill_fio_from_pool); np.zeros is kept only as the
    # fallback for layers with no recording, and ipadapter_scale/control inputs
    # (never recorded here) keep the prior ones/zeros synthesis.
    # fp8-round-13: initialized ahead of the branch below (which only runs when
    # onnx_path + one of use_cached_attn/use_controlnet/num_ip_layers is set) so
    # the provenance sidecar writer at the save block can reference these
    # without a NameError when that branch doesn't run.
    _selected_calls: Optional[np.ndarray] = None
    _pool_missed_calls: set = set()
    # FP8 Round 14 provenance field: observed length of the K/V pool actually used
    # to fill kvo_cache_in_* (post _pool_for_layer, so it reflects any misses).
    # Every kvo layer draws from the same _selected_calls against recorder state
    # gated by the same shared _predicted_calls/_call_counter, so pool length is
    # expected to be identical across layers -- tracked as a max defensively.
    _kvo_pool_len = 0
    if onnx_path and (use_cached_attn or use_controlnet or num_ip_layers):
        # modelopt's CalibrationDataProvider computes n_itr = first_input.shape[0] /
        # first_input_onnx_dim_0, then np.array_split(arr, n_itr, axis=0) for EVERY
        # input. Each chunk must satisfy the ONNX-declared dim 0 (fixed or dynamic).
        # Bound captured leading dims so n_itr stays small — synthesized KV-cache memory
        # scales as n_itr × per-layer-size × n_layers, which blows up at large n_itr.
        # Stratified-over-timesteps + call-aligned selection (fp8-round-5-handoff
        # Step 2e) — see _select_calibration_rows for why a naive per-key linspace
        # stride is unsafe once CFG doubles some keys but not `timestep`.
        # _select_calibration_calls is re-derived here (same calib_data, same
        # budget) purely to learn *which* calls _select_calibration_rows is about
        # to keep, so the K/V/FI pools below stay aligned with the rows that
        # actually survive -- deterministic given the same inputs, so this never
        # drifts from the trim on the next line without also changing
        # _select_calibration_rows's own behavior.
        _selected_calls = _select_calibration_calls(calib_data, _MAX_CALIB_ROWS)
        calib_data = _select_calibration_rows(calib_data, _MAX_CALIB_ROWS)
        # Aggregated across every _pool_for_layer call below (FP8 Round 11) --
        # previously a selected call missing from a layer's records was a silent
        # `continue`, and the layer just zero-fell-back with no visibility into
        # *why*. Logged once as a summary after the fill loop, not per-miss, since
        # a single out-of-window call can recur across many layers.
        _pool_missed_calls: set = set()

        try:
            _specs = _read_onnx_input_specs(onnx_path)

            # See _reconcile_calib_to_onnx_dims (beside _reconcile_2d, above) for the
            # IP-Adapter real-token vs. zero-pad reconciliation logic (fp8-round-9).
            calib_data = _reconcile_calib_to_onnx_dims(
                calib_data, _specs, ipadapter_tokens=ipadapter_tokens, num_ip_layers=num_ip_layers
            )

            # Mirror CalibrationDataProvider's n_itr derivation (calib_utils.py:90):
            # first ONNX-declared input that appears in calib_data drives the count.
            _present = [n for n in _specs if n in calib_data]
            if _present:
                _first = _present[0]
                # Shared with the quantize-side row alignment — see
                # _resolve_per_step_dim0 for the ipadapter_scale exception. (Also
                # guards a rank-0-scalar IndexError the old [0] index had.)
                _first_d0 = _resolve_per_step_dim0(_first, _specs[_first][1], num_ip_layers)
                _n_itr = max(1, calib_data[_first].shape[0] // _first_d0)
            else:
                _n_itr = 1

            _kvo_real, _fio_real, _kvo_zero, _fio_zero = 0, 0, 0, 0
            # fp8-round-10: this failure was silent (the h92807e04ef45/hb290e7269028
            # comparison found 75% zero-padded kvo/fio rows with no warning anywhere in
            # the build log). Track the worst real-captured-seq vs ONNX-target-seq
            # fraction across every real-sourced kvo/fio row and warn once, loudly, if
            # it drops far below 1.0 -- the signature of a capture resolution mismatch.
            _worst_real_seq_fraction = 1.0
            _worst_real_seq_detail = None
            for name, (dtype, dims) in _specs.items():
                if name in calib_data:
                    continue
                # Resolve symbolic dims to 1. dim 0 is the per-chunk shape ORT sees.
                resolved = [d if d is not None else 1 for d in dims]
                # Shared with the quantize-side row alignment — see
                # _resolve_per_step_dim0 for the ipadapter_scale / "L_ip" exception.
                # Keeping this in one place is load-bearing: the two copies
                # drifting apart caused an n_itr=560 blowup (~290 GiB tile attempt).
                per_step_d0 = _resolve_per_step_dim0(name, dims, num_ip_layers)
                # Total leading dim = n_itr × per-chunk dim 0 so every split chunk
                # has exactly the ONNX-declared leading dim (fixed Q+K=2 for kvo_cache,
                # fixed 2 for control inputs, num_ip_layers for ipadapter_scale).
                arr_shape = [_n_itr * per_step_d0] + list(resolved[1:])

                if name.startswith("kvo_cache_in_") and _kv_records:
                    _layer_idx = int(name.rsplit("_", 1)[-1])
                    _rec = _kv_records.get(_layer_idx, {})
                    _k_pool = _pool_for_layer(_rec.get("k", {}), _selected_calls, _pool_missed_calls)
                    _v_pool = _pool_for_layer(_rec.get("v", {}), _selected_calls, _pool_missed_calls)
                    arr = _fill_kvo_from_pool(_k_pool, _v_pool, arr_shape, dtype)
                    if min(len(_k_pool), len(_v_pool)) > 0:
                        _kvo_real += 1
                        _kvo_pool_len = max(_kvo_pool_len, min(len(_k_pool), len(_v_pool)))
                        _source = "recorded K/V activations"
                        _target_seq = arr_shape[-2]
                        if _target_seq:
                            _frac = min(1.0, _k_pool[0].shape[0] / _target_seq)
                            if _frac < _worst_real_seq_fraction:
                                _worst_real_seq_fraction = _frac
                                _worst_real_seq_detail = (
                                    f"'{name}': captured seq={_k_pool[0].shape[0]}, ONNX target seq={_target_seq}"
                                )
                    else:
                        _kvo_zero += 1
                        _source = "zero fallback (no recording)"
                elif name.startswith("fio_cache_in_") and _fio_records:
                    _fi_local = int(name.rsplit("_", 1)[-1])
                    _pool = _pool_for_layer(_fio_records.get(_fi_local, {}), _selected_calls, _pool_missed_calls)
                    arr = _fill_fio_from_pool(_pool, arr_shape, dtype, per_step_d0)
                    if _pool:
                        _fio_real += 1
                        _source = "recorded FI activations"
                        _target_seq = arr_shape[-2]
                        if _target_seq:
                            _frac = min(1.0, _pool[0].shape[0] / _target_seq)
                            if _frac < _worst_real_seq_fraction:
                                _worst_real_seq_fraction = _frac
                                _worst_real_seq_detail = (
                                    f"'{name}': captured seq={_pool[0].shape[0]}, ONNX target seq={_target_seq}"
                                )
                    else:
                        _fio_zero += 1
                        _source = "zero fallback (no recording)"
                elif name == "ipadapter_scale":
                    # fp8-round-5-handoff Step 3: deployment scale (config-derived via the
                    # caller), not a hardcoded 1.0 placeholder.
                    arr = np.full(arr_shape, ipadapter_scale, dtype=dtype)
                    _source = "deployment config value"
                elif name == "fi_strength":
                    arr = np.full(arr_shape, fi_strength, dtype=dtype)
                    _source = "deployment config value"
                elif name == "fi_threshold":
                    arr = np.full(arr_shape, fi_threshold, dtype=dtype)
                    _source = "deployment config value"
                else:
                    arr = np.zeros(arr_shape, dtype=dtype)
                    _source = "synthetic zeros"

                calib_data[name] = arr
                logger.info(
                    f"[FP8] Filled '{name}' from {_source}: shape={arr.shape}, dtype={arr.dtype} "
                    f"(n_itr={_n_itr}, per-step-dim0={per_step_d0})"
                )
            if _kvo_real or _fio_real or _kvo_zero or _fio_zero:
                logger.info(
                    f"[FP8] K/V cache recorder: kvo real={_kvo_real} zero-fallback={_kvo_zero}, "
                    f"fio real={_fio_real} zero-fallback={_fio_zero}"
                )
            if _pool_missed_calls:
                # FP8 Round 14: this is not a zero-fallback -- _pool_for_layer skips a
                # missed call rather than inserting a zero row, shrinking pool_len, and
                # _fill_kvo_from_pool's src = (m + 1) % pool_len then cyclically re-uses
                # the surviving recorded calls to fill the gap. The cost is calibration
                # *diversity*, not zeros: fewer distinct real activations get reused
                # across more n_itr chunks (measured: 8 distinct K/V sources dropped to
                # 4 under the pre-Round-14 recorder — see the calv8 note in
                # engine_manager.py). Under normal (non-fallback) operation the
                # predictive recorder above should make this list empty; a non-empty
                # list here means _predict_valid["ok"] went False mid-capture (a batch
                # failed) and the legacy call-count-prefix admission missed some of the
                # calls actually selected.
                logger.warning(
                    f"[FP8] {len(_pool_missed_calls)} selected calibration call(s) "
                    f"{sorted(_pool_missed_calls)} were not recorded by the K/V/FI recorder "
                    f"(_MAX_CALIB_RECORD_CALLS={_MAX_CALIB_RECORD_CALLS}, "
                    f"_MAX_CALIB_RECORD_BYTES={_MAX_CALIB_RECORD_BYTES / 1024**3:.1f} GiB) — "
                    "affected layers reuse other recorded calls for those positions instead, "
                    "reducing calibration diversity (not a zero-fill)."
                )
            if _worst_real_seq_detail is not None and _worst_real_seq_fraction < 0.9:
                # fp8-round-10: a low fraction here means capture ran at a lower
                # resolution than the engine's build target -- pass image_height/
                # image_width so pipe() captures at the real shape, or the zero-padded
                # remainder starves FP8 scales of real signal (measured: 91.2% of
                # quantized activation tensors are transitively fed by these inputs).
                logger.warning(
                    f"[FP8] K/V/FI cache calibration is under-resolution: worst real-vs-target "
                    f"sequence-length fraction is {_worst_real_seq_fraction:.1%} ({_worst_real_seq_detail}). "
                    "This usually means capture_calibration_data ran at a lower resolution than "
                    "the engine is being built for — pass image_height/image_width."
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
        _write_calib_provenance_sidecar(
            save_path,
            _build_calib_provenance(
                calib_data,
                image_height=image_height,
                image_width=image_width,
                prompt_count=len(prompts),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                ipadapter_tokens_real=ipadapter_tokens is not None,
                selected_calls=_selected_calls,
                pool_missed_calls=_pool_missed_calls,
                predicted_calls=_predicted_calls,
                record_calls_kept=len(_record_calls_kept),
                record_bytes=_record_bytes["n"],
                record_max_calls=_MAX_CALIB_RECORD_CALLS,
                record_max_bytes=_MAX_CALIB_RECORD_BYTES,
                kvo_pool_len=_kvo_pool_len,
            ),
        )

    return calib_data


def capture_calibration_data_controlnet(
    cn_model,
    n_calibration_steps: int = 32,
    image_height: int = 704,
    image_width: int = 704,
    batch_size: int = 1,
    save_path: Optional[str] = None,
    embedding_dim: int = 2048,
) -> Dict[str, np.ndarray]:
    """
    Capture ControlNet input activations for FP8 calibration using synthetic inputs.

    Generates synthetic SDXL-compatible inputs (random latents, zero edge maps,
    varying timesteps) and runs them through cn_model to obtain real per-layer
    activation statistics for modelopt's max-abs calibration.

    Args:
        cn_model: Diffusers ControlNetModel (may be on CPU; moved to CUDA internally).
        n_calibration_steps: Number of forward passes to capture (default 32).
        image_height, image_width: Spatial resolution for synthetic inputs.
        batch_size: Batch size per forward pass.
        save_path: Optional path to save calib_data.npz.
        embedding_dim: Encoder hidden state depth (2048 for SDXL).

    Returns:
        Dict mapping SDXL ControlNet ONNX input names to np.ndarray arrays.
    """
    import torch

    _orig_device = next(cn_model.parameters()).device
    if _orig_device.type != "cuda":
        cn_model.to("cuda")

    latent_h, latent_w = image_height // 8, image_width // 8
    # Cover the full scheduler noise range so every activation bucket is exercised.
    _timesteps = [999, 899, 799, 699, 599, 499, 399, 299, 199, 99, 50, 20]

    captured: Dict[str, list] = {}

    def _to_npy(t):
        return np.atleast_1d(t.detach().cpu().numpy())

    def _hook(module, args, kwargs):
        _key_map = {0: "sample", 1: "timestep", 2: "encoder_hidden_states"}
        for idx, key in _key_map.items():
            val = args[idx] if idx < len(args) else kwargs.get(key)
            if val is not None:
                captured.setdefault(key, []).append(_to_npy(val))
        ctrl_cond = args[3] if len(args) > 3 else kwargs.get("controlnet_cond")
        if ctrl_cond is not None:
            captured.setdefault("controlnet_cond", []).append(_to_npy(ctrl_cond))
        added = kwargs.get("added_cond_kwargs") or {}
        for key in ("text_embeds", "time_ids"):
            if key in added and added[key] is not None:
                captured.setdefault(key, []).append(_to_npy(added[key]))

    handle = cn_model.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(n_calibration_steps):
                t_val = _timesteps[i % len(_timesteps)]
                sample = torch.randn(batch_size, 4, latent_h, latent_w, dtype=torch.float16, device="cuda")
                timestep = torch.tensor([t_val] * batch_size, dtype=torch.float32, device="cuda")
                enc_hidden = torch.randn(batch_size, 77, embedding_dim, dtype=torch.float16, device="cuda")
                ctrl_cond = torch.rand(batch_size, 3, image_height, image_width, dtype=torch.float16, device="cuda")
                text_embeds = torch.randn(batch_size, 1280, dtype=torch.float16, device="cuda")
                time_ids = torch.tensor(
                    [[image_height, image_width, 0, 0, image_height, image_width]] * batch_size,
                    dtype=torch.float32,
                    device="cuda",
                )
                logger.info(f"[FP8-CN] Calibration step {i + 1}/{n_calibration_steps} (t={t_val})")
                try:
                    cn_model(
                        sample,
                        timestep,
                        enc_hidden,
                        ctrl_cond,
                        conditioning_scale=1.0,
                        added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                        return_dict=False,
                    )
                except Exception as e:
                    logger.warning(f"[FP8-CN] Calibration step {i + 1} failed: {e}. Skipping.")
    finally:
        handle.remove()
        if _orig_device.type != "cuda":
            cn_model.to(_orig_device)
            torch.cuda.empty_cache()

    if not captured:
        raise RuntimeError("[FP8-CN] No ControlNet activations captured — check model forward signature.")

    calib_data: Dict[str, np.ndarray] = {}
    for key, arrays in captured.items():
        stacked = np.concatenate(arrays, axis=0)
        calib_data[key] = stacked
        logger.info(f"[FP8-CN] Captured '{key}': shape={stacked.shape}, dtype={stacked.dtype}")

    # conditioning_scale is a Python float in diffusers forward but exported as a
    # rank-1 (1,) ONNX tensor input. np.ones(n_total) already has shape (n_total,),
    # which splits cleanly into n_total × (1,) slices for modelopt CalibrationDataProvider.
    n_total = list(calib_data.values())[0].shape[0]
    calib_data["conditioning_scale"] = np.ones(n_total, dtype=np.float32)
    logger.info(f"[FP8-CN] Synthesized 'conditioning_scale': shape=({n_total},), dtype=float32")

    if save_path:
        tmp_path = save_path + ".tmp"
        with open(tmp_path, "wb") as _f:
            np.savez(_f, **calib_data)
        os.replace(tmp_path, save_path)
        logger.info(f"[FP8-CN] Saved calibration data: {save_path} ({os.path.getsize(save_path) / 1e6:.1f} MB)")
        _write_calib_provenance_sidecar(
            save_path,
            _build_calib_provenance(
                calib_data,
                image_height=image_height,
                image_width=image_width,
                num_inference_steps=n_calibration_steps,
            ),
        )

    return calib_data


def load_calibration_data(npz_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Load previously-saved calibration data from a .npz file.
    Returns None (and deletes the file) if loading fails.
    """
    if not os.path.exists(npz_path):
        return None
    try:
        # Explicit `with` (not dict(np.load(...))) so the NpzFile's underlying
        # zip handle closes deterministically here rather than whenever GC gets
        # around to the finalizer — this file is 4+ GB, don't leave it pinned.
        with np.load(npz_path) as _npz:
            data = {k: _npz[k] for k in _npz.files}
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

# fp8-round-8: the two attention BMMs (Q@K^T and softmax@V), never the QKV/output
# *projections*. Node names verified against a built graph:
#   catches  .../attn1/MatMul, .../attn1/MatMul_1, .../attn2/MatMul, _1, _2, _3
#   spares   .../attn1/to_q/MatMul, .../attn2/to_out.0/MatMul, .../attn2/to_k_ip/MatMul, ...
# The trailing `(_\d+)?$` anchor is load-bearing — `.*attn1.*` would also catch every
# projection MatMul under that block, which are correctly calibrated and must stay FP8.
_ATTENTION_EXCLUDE_PATTERNS = [r".*/attn[12]/MatMul(_\d+)?$"]

# fp8-round-9: the three IP-Adapter cross-attention *activation* tensors that fed
# fp8_exclude_ipadapter's motivating defect (Mul_4 = k_ip after RoPE-less transpose,
# Transpose_4 = v_ip, Mul_5 = scale·ip_attn_out — see fp8-round-9 handoff). Deliberately
# NOT the two IPA BMMs (attn2/MatMul_2, attn2/MatMul_3) — those already match
# _ATTENTION_EXCLUDE_PATTERNS above (r".*/attn[12]/MatMul(_\d+)?$"), per
# tests/unit/test_fp8_rescale.py. Duplicating them here would make the two flags
# silently overlap. Net effect of each flag alone:
#   fp8_exclude_ipadapter alone  -> IPA BMMs stay quantized, these 3 activations don't
#   fp8_exclude_attention alone  -> these 3 activations stay quantized, IPA BMMs don't
#   both together                -> the whole IPA cross-attention path is FP16
# The trailing `$` is load-bearing — modelopt's expand_node_names_from_patterns uses
# re.match (anchored at the start only, graph_utils.py:1018), so an unanchored tail
# would also swallow `Mul_5_output_0`-style consumer names and any future `Mul_40+`.
_IPADAPTER_ACTIVATION_EXCLUDE_PATTERNS = [r".*/attn2/(Mul_4|Mul_5|Transpose_4)$"]


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


def _resolve_per_step_dim0(name: str, dims: List[Optional[int]], num_ip_layers: int = 0) -> int:
    """Per-chunk leading dim for `name` — the dim 0 ORT sees after modelopt's
    CalibrationDataProvider does np.array_split(arr, n_itr, axis=0).

    Static ONNX dim 0 → itself; symbolic (None) → 1; empty dims (rank-0 scalar) → 1.
    Sole exception is ipadapter_scale: its ONNX dim 0 is the dynamic symbol "L_ip"
    (models/models.py:716), but the traced graph Gathers scale_vec[0..num_ip_layers-1]
    at every IPA layer, so a length-1 chunk goes out of bounds during modelopt's
    _exclude_matmuls_by_inference ORT probe.

    MUST remain the single source of truth for both capture_calibration_data's
    synthesis and _align_calibration_rows. The two drifting apart previously let
    ipadapter_scale contribute 560 rows / resolved_dim0=1 = 560 to the elected
    n_itr instead of 8, tiling every kvo/fio cache tensor 70× (~290 GiB, OOM).
    """
    if name == "ipadapter_scale" and num_ip_layers > 0:
        return num_ip_layers
    if not dims:
        return 1
    return max(1, dims[0] or 1)


# Upper bound on the row-aligned calibration set's in-memory footprint. Sized well
# above a real SDXL + cached-attn + FI + IPA capture (~4.1 GiB at n_itr=8) and far
# below the ~290 GiB an n_itr drift demands, so it fires long before a numpy
# MemoryError would. Override for exotic configs: SDTD_FP8_CALIB_MAX_BYTES=<bytes>
_MAX_CALIB_TOTAL_BYTES = 24 * 1024**3


def _align_calibration_rows(
    calibration_data: Dict[str, np.ndarray],
    specs: Dict[str, tuple],
    num_ip_layers: int = 0,
) -> Dict[str, np.ndarray]:
    """Tile/trim every calibration tensor to n_itr × per-step-dim0 rows.

    modelopt's CalibrationDataProvider does np.array_split(arr, n_itr, axis=0) for
    EVERY input, so each input needs exactly n_itr × its own per-chunk dim 0 rows —
    a single global row count breaks kvo_cache_in_* (ONNX dim0=2 static) by handing
    modelopt (1, ...) chunks. n_itr is the max over inputs so no input is silently
    down-sampled; because the loop re-tiles every input — including the first ONNX
    graph input — modelopt recomputes that same n_itr from first_input.shape[0] //
    dim0, so max() stays self-consistent with what modelopt does downstream.

    Pure numpy (no ORT/modelopt/torch): unit-testable by passing a hand-built
    `specs` dict in _read_onnx_input_specs' format. Returns a new dict; entries
    that already match their target are returned by identity (no copy).
    """
    _dim0 = {name: _resolve_per_step_dim0(name, specs[name][1], num_ip_layers) for name in calibration_data}
    _per_key_itr = {name: arr.shape[0] // _dim0[name] for name, arr in calibration_data.items()}
    _n_itr = max(1, max(_per_key_itr.values(), default=1))

    # Projected-footprint guard. A per-step-dim0 that disagrees with what capture
    # wrote inflates n_itr by that input's true chunk length, which then multiplies
    # into every other calibration tensor. Fail loudly and actionably here instead
    # of letting numpy raise a bare MemoryError partway through the tile loop.
    # a[:1] is a view, so .nbytes gives the per-row size with no allocation.
    _projected = sum(_n_itr * _dim0[n] * a[:1].nbytes for n, a in calibration_data.items())
    _cap = int(os.environ.get("SDTD_FP8_CALIB_MAX_BYTES", _MAX_CALIB_TOTAL_BYTES))
    if _projected > _cap:
        _sorted_itr = sorted(_per_key_itr.values())
        _median = _sorted_itr[len(_sorted_itr) // 2]
        _drivers = sorted(n for n, v in _per_key_itr.items() if v == _n_itr)
        raise RuntimeError(
            f"[FP8] Calibration row alignment would need {_projected / 1024**3:.1f} GiB "
            f"(cap {_cap / 1024**3:.1f} GiB) at n_itr={_n_itr}. "
            f"n_itr is driven by {_drivers[:5]}"
            f"{f' (+{len(_drivers) - 5} more)' if len(_drivers) > 5 else ''}, "
            f"while the median input implies n_itr={_median}. "
            f"That gap means a calibration input's row count disagrees with its "
            f"ONNX-declared dim 0 — delete calib_data.npz to recapture, or raise "
            f"SDTD_FP8_CALIB_MAX_BYTES if this row count is genuinely intended."
        )

    _aligned: Dict[str, np.ndarray] = {}
    for _k, _arr in calibration_data.items():
        _target_rows = _n_itr * _dim0[_k]
        if _arr.shape[0] == _target_rows:
            _aligned[_k] = _arr
            continue
        if _arr.shape[0] == 0:
            raise RuntimeError(f"[FP8] Calibration input '{_k}' has 0 rows — delete calib_data.npz and recapture.")
        # Modulo fancy-index instead of np.tile(...)[:target]: allocates only the
        # output. np.tile first materializes ceil(target/rows)×rows rows, and the
        # trailing slice is a *view*, so that oversized base stays alive for the
        # lifetime of the dict. Semantics are identical in both directions: up-tile
        # gives arr[j % n] for j < target (what np.tile(...)[:target] produces);
        # down-trim (target < n) gives arr[:target] in both.
        _aligned[_k] = _arr[np.arange(_target_rows) % _arr.shape[0]]
        logger.info(
            f"[FP8] Tiled '{_k}' {_arr.shape[0]} → {_target_rows} rows (n_itr={_n_itr} × per-step-dim0={_dim0[_k]})"
        )
    return _aligned


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

    # modelopt's save_onnx (utils.py:654-687) externalizes any tensor >= 1024 bytes
    # once the model as a whole exceeds 2 GB — and it passes per_channel=True to
    # quantize_static (fp8.py:302), so wide layers' per-channel weight scales are
    # exactly the kind of tensor that legitimately ends up in external data, not
    # just the multi-GB weights. A previous version of this check loaded with
    # load_external_data=False to avoid that read and silently skipped any scale
    # stored externally — host RAM was never actually the constraint here, and
    # skipping meant those scales were never checked for non-finite values at all.
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


def _rescale_fp8_qdq_scales(onnx_path: str, headroom: float = 1.0, dry_run: bool = False) -> Dict[str, Any]:
    """Correct modelopt's inverted INT8->FP8 scale-conversion factor, in place.

    modelopt/onnx/quantization/fp8.py's _int8_scale_to_fp8_scale computes

        np_fp8_scale = (np_scale * 448.0) / 127.0

    where np_scale is the *INT8* QuantizeLinear scale (amax / 127). The correct FP8
    scale is amax / 448 = np_scale * 127 / 448 -- modelopt applies the reciprocal
    factor instead, so every calibrated amax lands at 127**2/448 = 36.0 in E4M3 space
    (full range is 448.0), a uniform 12.4437x under-utilization. Measured directly on
    a shipped engine: every sampled weight peaked at exactly 36.04. See the fp8-round-8
    plan for the full derivation and why this specifically shows up as blur in
    attention (softmax outputs are bounded in [0,1] with typical values far below 1,
    the tensor class most likely to fall below E4M3's narrowed normal-range floor).

    TensorRT's STRONGLY_TYPED FP8 builder (utilities.py::_build_fp8) parses this ONNX
    file directly and constant-folds each weight's QuantizeLinear at *engine build*
    time -- there is no pre-baked FP8 payload sitting in this file for either weights
    or activations, only the FP16 value plus the scale that will quantize it later.
    Rescaling the scale tensors here, before TRT ever sees this file, is therefore
    sufficient: TRT re-derives every quantized value from the corrected scale.

    Called once, directly on modelopt_quantize's raw output, before
    _assert_finite_qdq_scales and before the .ok sentinel is written. Not idempotent:
    running this twice on the same file compounds the correction.

    Args:
        onnx_path: Path to the modelopt-quantized ONNX (mutated in place unless
            dry_run). External-data scales are patched in the sibling *_data file at
            their existing offset/length -- byte length is unchanged (fp16 in, fp16
            out), so nothing else in the model shifts.
        headroom: Extra multiplier past the corrected amax->448 mapping. >1.0 leaves
            outlier margin for activations that exceed the calibrated max, at the
            cost of resolution. 1.0 uses the full E4M3 range.
        dry_run: Compute and return the report without writing anything -- used to
            verify this function against an already-shipped (uncorrected) engine
            without mutating it.

    Returns:
        Report dict: scales_seen, scales_corrected, realized_peak_before,
        realized_peak_after, min_scale_after, underflow_count, uncalibrated_count,
        uncalibrated_samples.
    """
    import onnx as _onnx
    from onnx import TensorProto as _TensorProto
    from onnx import numpy_helper as _numpy_helper
    from onnx.external_data_helper import load_external_data_for_tensor as _load_ext_tensor

    # modelopt stores  s_mo  = amax * 448 / 127**2      (fp8.py:70-74, inverted factor)
    # we want          s_tgt = amax * headroom / 448
    # so               k     = headroom * (127 / 448) ** 2      # 0.080362 at headroom=1.0
    k = headroom * (127.0 / 448.0) ** 2
    _UNCALIBRATED = float(np.float16(448.0 / 127.0))  # INT8 amax == 127.0 exactly -> never calibrated
    _UNCALIBRATED_TOL = 1e-3
    _FP16_MIN_NORMAL = 6.103515625e-05
    _SAMPLE_N = 5

    model = _onnx.load(onnx_path, load_external_data=False)
    graph = model.graph
    base_dir = os.path.dirname(os.path.abspath(onnx_path))
    init_map = {init.name: init for init in graph.initializer}

    const_scale_nodes: Dict[str, Any] = {}
    for node in graph.node:
        if node.op_type == "Constant" and node.output:
            attr = {a.name: a for a in node.attribute}
            if "value" in attr:
                const_scale_nodes[node.output[0]] = node

    # Dedupe by initializer/constant-output name: a QuantizeLinear/DequantizeLinear
    # pair for the same tensor share one scale source (modelopt's own
    # `processed_tensor` guard, fp8.py:85-96). A set is load-bearing here -- applying
    # k twice to the same tensor would be a silent 12.4x error in the other direction.
    inline_names: set = set()
    const_names: set = set()
    for node in graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear") or len(node.input) < 2:
            continue
        sname = node.input[1]
        if sname in init_map:
            inline_names.add(sname)
        elif sname in const_scale_nodes:
            const_names.add(sname)

    def _materialize(t) -> np.ndarray:
        tc = _onnx.TensorProto()
        tc.CopyFrom(t)
        if tc.data_location == _TensorProto.EXTERNAL:
            _load_ext_tensor(tc, base_dir)
            tc.data_location = _TensorProto.DEFAULT
            del tc.external_data[:]
        return _numpy_helper.to_array(tc)

    def _set_raw(t, arr: np.ndarray) -> None:
        t.raw_data = arr.tobytes()
        if t.int32_data:
            del t.int32_data[:]
        if t.float_data:
            del t.float_data[:]
        t.data_location = _TensorProto.DEFAULT

    # --- realized-peak sample, taken BEFORE any mutation: max|w_fp16| / scale over a
    # handful of static-weight QuantizeLinear nodes. This is what makes the fix
    # self-verifying in the build log (36.04 -> 448.00). ---
    peak_before = None
    sampled = 0
    for node in graph.node:
        if node.op_type != "QuantizeLinear" or sampled >= _SAMPLE_N:
            continue
        w, s = init_map.get(node.input[0]), init_map.get(node.input[1])
        if w is None or s is None:
            continue
        try:
            wa = _materialize(w).astype(np.float32)
            sa = _materialize(s).astype(np.float32).reshape(-1)
        except Exception:
            continue
        axis = next((a.i for a in node.attribute if a.name == "axis"), None)
        if sa.size > 1 and axis is not None:
            mv = np.moveaxis(np.abs(wa), axis, 0).reshape(sa.size, -1).max(axis=1)
            peak = float((mv / sa).max())
        else:
            peak = float(np.abs(wa).max() / sa[0])
        peak_before = peak if peak_before is None else max(peak_before, peak)
        sampled += 1

    report: Dict[str, Any] = {
        "scales_seen": len(inline_names) + len(const_names),
        "scales_corrected": 0,
        "underflow_count": 0,
        "uncalibrated_count": 0,
        "uncalibrated_samples": [],
        "min_scale_after": None,
        "realized_peak_before": peak_before,
        "realized_peak_after": (peak_before / k) if peak_before is not None else None,
    }
    min_after_holder = [None]

    def _correct(arr16: np.ndarray, name: str) -> np.ndarray:
        arr32 = arr16.astype(np.float32)
        if np.all(np.abs(arr32 - _UNCALIBRATED) < _UNCALIBRATED_TOL):
            report["uncalibrated_count"] += 1
            if len(report["uncalibrated_samples"]) < 10:
                report["uncalibrated_samples"].append(name)
        corrected = (arr32 * np.float32(k)).astype(np.float16)
        if np.any(corrected == 0):
            raise RuntimeError(
                f"[FP8] Rescale produced an exact-zero FP16 scale for '{name}' in "
                f"{onnx_path} -- its DequantizeLinear output would be zeroed. This "
                f"scale was far below anything measured while deriving fp8_scale_headroom; "
                f"investigate before lowering headroom further."
            )
        mag = np.abs(corrected.astype(np.float32))
        if mag.size:
            m = float(mag.min())
            if min_after_holder[0] is None or m < min_after_holder[0]:
                min_after_holder[0] = m
        report["underflow_count"] += int(np.count_nonzero(mag < _FP16_MIN_NORMAL))
        report["scales_corrected"] += 1
        return corrected

    # --- inline (protobuf) initializer scales ---
    for name in inline_names:
        t = init_map[name]
        if t.data_location == _TensorProto.EXTERNAL:
            continue  # handled in the external-data pass below
        arr16 = _numpy_helper.to_array(t).reshape(-1).astype(np.float16)
        corrected = _correct(arr16, name)
        if not dry_run:
            _set_raw(t, corrected)

    # --- scales produced by a Constant node (fp8.py:86-95's other storage form; none
    # observed in this build, handled defensively for future modelopt/graph changes) ---
    for name in const_names:
        node = const_scale_nodes[name]
        attr = {a.name: a for a in node.attribute}
        t = attr["value"].t
        arr16 = _numpy_helper.to_array(t).reshape(-1).astype(np.float16)
        corrected = _correct(arr16, name)
        if not dry_run:
            _set_raw(t, corrected)

    # --- external-data scales: patch bytes in place at their existing offset/length.
    # Never load the multi-GB weight payload alongside them; group by backing file so
    # one handle serves every tensor stored in it. ---
    by_location: Dict[str, list] = {}
    for name in inline_names:
        t = init_map[name]
        if t.data_location != _TensorProto.EXTERNAL:
            continue
        ed = {kv.key: kv.value for kv in t.external_data}
        by_location.setdefault(ed["location"], []).append((name, int(ed["offset"]), int(ed.get("length", 0))))

    for location, entries in by_location.items():
        data_path = os.path.join(base_dir, location)
        mode = "r+b" if not dry_run else "rb"
        with open(data_path, mode) as f:
            for name, offset, length in entries:
                f.seek(offset)
                raw = f.read(length)
                arr16 = np.frombuffer(raw, dtype=np.float16)
                corrected = _correct(arr16, name)
                if not dry_run:
                    f.seek(offset)
                    f.write(corrected.tobytes())

    report["min_scale_after"] = min_after_holder[0]

    if not dry_run:
        _onnx.save_model(model, onnx_path)

    peak_before_s = f"{report['realized_peak_before']:.2f}" if report["realized_peak_before"] is not None else "n/a"
    peak_after_s = f"{report['realized_peak_after']:.2f}" if report["realized_peak_after"] is not None else "n/a"
    min_after_s = f"{report['min_scale_after']:.3e}" if report["min_scale_after"] is not None else "n/a"
    logger.info(
        f"[FP8] Rescale: {report['scales_corrected']}/{report['scales_seen']} scales corrected "
        f"(k={k:.6f}, headroom={headroom}); realized peak {peak_before_s} -> {peak_after_s} "
        f"(target {448.0 * headroom:.1f}); {report['uncalibrated_count']} uncalibrated "
        f"(amax==127.0 exactly); {report['underflow_count']} FP16-subnormal after correction "
        f"(min {min_after_s})."
    )
    return report


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
    fp8_scale_headroom: float = 1.0,
    fp8_exclude_attention: bool = False,
    fp8_exclude_ipadapter: bool = False,
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
        disable_mha_qdq: Skip MHA-specific Q/DQ injection (default True for Ada). When
                         True, modelopt excludes the attention MatMuls from
                         quantization entirely (graph_utils.find_nodes_from_mha_to_exclude
                         -> fp8.py's nodes_to_exclude filter) — no Q/DQ is inserted on
                         them, so TRT cannot fuse a _gemm_mha_v2 FP8 kernel there; those
                         MatMuls run at their original (FP16) precision instead.
        fp8_scale_headroom: Multiplier applied on top of the corrected amax->448 mapping
                         (see _rescale_fp8_qdq_scales). 1.0 uses the full E4M3 range;
                         >1.0 leaves outlier headroom for activations that exceed the
                         calibrated max, at the cost of resolution.
        fp8_exclude_attention: When True, adds _ATTENTION_EXCLUDE_PATTERNS to
                         nodes_to_exclude so the two attention BMMs (QK^T and
                         softmax@V) run FP16 instead of FP8. Fallback lever for the
                         12.4x scale defect fp8-round-8 fixes directly — should not be
                         needed once _rescale_fp8_qdq_scales is in place, but forks the
                         engine cache tag (--fp8v4-noattn) so it stays a clean A/B.
        fp8_exclude_ipadapter: When True, adds _IPADAPTER_ACTIVATION_EXCLUDE_PATTERNS
                         to nodes_to_exclude so the three IP-Adapter cross-attention
                         activations (Mul_4/Mul_5/Transpose_4) run FP16 instead of FP8.
                         Orthogonal to fp8_exclude_attention — see the pattern's own
                         comment for the non-overlap with the IPA BMMs. Fallback lever
                         for fp8-round-9's calibration fix; not needed once real
                         IP-Adapter tokens feed calibration, but forks the engine cache
                         tag (--fp8v4-noip) so it stays a clean A/B.
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
        if fp8_exclude_attention:
            nodes_to_exclude.extend(_ATTENTION_EXCLUDE_PATTERNS)
        if fp8_exclude_ipadapter:
            nodes_to_exclude.extend(_IPADAPTER_ACTIVATION_EXCLUDE_PATTERNS)

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

    # Per-input row alignment: target rows = n_itr × per-step-dim0(name) so every
    # input splits into exactly n_itr chunks of shape (per-step-dim0, ...). Shares
    # _resolve_per_step_dim0 with the capture-side synthesis so the two cannot
    # drift — see _align_calibration_rows for the full rationale and the
    # projected-memory guard that replaces an opaque numpy MemoryError.
    calibration_data = _align_calibration_rows(calibration_data, _specs, num_ip_layers)

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
    # enable_gemv_detection_for_trt moved from a named parameter into **kwargs in
    # modelopt 0.42 — it was NOT removed. The old `in _params` test therefore went
    # permanently False and fp8.py's `kwargs.get(..., True)` default silently
    # re-enabled the probe: it promotes every MatMul/Gemm output to a graph output
    # (ORT can never free a graph output) and runs the full model once, pinning
    # multiple GB of activations in VRAM and writing a full extra copy of the model
    # to disk. Detect **kwargs acceptance too so this can't silently regress again.
    _takes_var_kw = any(
        p.kind is _inspect.Parameter.VAR_KEYWORD for p in _inspect.signature(modelopt_quantize).parameters.values()
    )
    _gemv_param_supported = "enable_gemv_detection_for_trt" in _params or _takes_var_kw
    # --- ITERATION 1 DIAGNOSTIC (round-5 quality regression, see the
    # fp8-round-5-handoff plan) --- The capability check above is correct: it
    # detects that modelopt 0.42+ moved this kwarg into **kwargs rather than
    # dropping it, fixing a check that used to be permanently False. But round
    # 5 then used that corrected detection to force
    # enable_gemv_detection_for_trt=False, which silently changed *what gets
    # quantized*: HEAD's broken check always left the kwarg unset, so
    # modelopt's own default (True, fp8.py:218) applied and excluded every
    # GEMV-shaped MatMul/Gemm (m or n == 1, can't use TensorCores) from FP8.
    # Force it explicitly True here to reproduce that effective behaviour,
    # isolated from the calibration-stride fix above. DO NOT SHIP this if it
    # doesn't restore render sharpness — re-enabling the probe reintroduces
    # the full-model GEMV probe (every MatMul/Gemm output promoted to a graph
    # output, a full extra ~5.5 GB model copy on disk, a large VRAM spike, and
    # extra build time) that round 5 removed for a reason.
    if _gemv_param_supported:
        kwargs["enable_gemv_detection_for_trt"] = True
    else:
        logger.warning(
            "[FP8] modelopt build accepts no enable_gemv_detection_for_trt — "
            "expect a full-model GEMV probe and a large VRAM/disk spike."
        )
    _gemv_enabled = kwargs.get("enable_gemv_detection_for_trt", True)

    logger.info(
        f"[FP8] ONNX-level FP8 quantization: {os.path.basename(onnx_path)}"
        f" → {os.path.basename(output_path)}"
        f" ({next(iter(calibration_data.values())).shape[0]} calibration samples,"
        f" disable_mha_qdq={disable_mha_qdq}, enable_gemv_detection_for_trt={_gemv_enabled})"
    )
    # The dtype-cast loop above and _align_calibration_rows can each leave a
    # transient multi-GB copy of the calibration set as garbage — collect before
    # handing off to modelopt so ORT's own (already large) allocations don't
    # stack on top of copies Python hasn't reclaimed yet.
    import gc as _gc

    _gc.collect()

    # Works around an nvidia-modelopt bug that makes ORT's CUDA arena degenerate during static
    # calibration (arena_extend_strategy hardcoded to exact-size-only) — see
    # _patches/modelopt_arena_patch.py for the full writeup, including why a related
    # arena-shrinkage fix was tried and abandoned as unreachable on this ORT/CUDA EP stack.
    # Scoped to this call only: applied immediately before modelopt_quantize and always
    # reverted, so nothing leaks into any other caller of modelopt in this process.
    from streamdiffusion._patches import modelopt_arena_patch

    modelopt_arena_patch.apply()
    try:
        modelopt_quantize(**kwargs)
    finally:
        modelopt_arena_patch.revert()

    if not os.path.exists(output_path):
        raise RuntimeError(f"[FP8] modelopt_quantize completed but output not found: {output_path}")

    # Round 8: modelopt's int8->fp8 scale conversion is inverted (fp8.py:70-74),
    # mapping every calibrated amax to 36.0 in E4M3 space instead of 448.0. Must run
    # before _assert_finite_qdq_scales so the finite check validates the shipped
    # values, and before the .ok sentinel so a crash mid-rescale leaves no
    # false-positive cache hit — the next run re-quantizes from scratch instead of
    # double-applying the correction.
    rescale_report = _rescale_fp8_qdq_scales(output_path, headroom=fp8_scale_headroom)

    _assert_finite_qdq_scales(output_path)

    size_mb = os.path.getsize(output_path) / (1024**2)
    logger.info(f"[FP8] FP8 ONNX written: {output_path} ({size_mb:.1f} MB)")
    if size_mb > 5000:
        logger.warning(
            f"[FP8] FP8 ONNX is unexpectedly large ({size_mb:.0f} MB > 5000 MB). "
            "FP32 Cast bloat may be active — check high_precision_dtype='fp16' is honored."
        )

    # Sentinel marker — only written after modelopt_quantize returns. The builder's
    # cache check only tests os.path.exists() on this file (never its content), so
    # widening it from a plain "ok" string to JSON carrying the rescale report is
    # safe for the cache-hit path; the report gives the parent process (builder.py)
    # the uncalibrated-scale count without re-opening the multi-GB ONNX itself.
    import json as _json

    with open(output_path + ".ok", "w") as _f:
        _json.dump({"ok": True, "rescale": rescale_report}, _f)


def _main(argv: List[str]) -> int:
    """Standalone entry point for the FP8 ONNX quantize stage.

    Lets this one stage run in isolation — as its own process (builder.py's
    _quant_fn launches it via `subprocess.run`) or invoked directly for local
    iteration against a cached unet.engine.opt.onnx + calib_data.npz, without
    paying for a full TouchDesigner + model-load cycle per attempt.

    Usage: python -m streamdiffusion.acceleration.tensorrt.fp8_quantize '<json>'
    JSON keys mirror quantize_onnx_fp8's args, plus calib_data_path (loaded here
    via load_calibration_data so the multi-GB calibration set never has to cross
    a process boundary as a subprocess argument):
        onnx_path, output_path, calib_data_path, disable_mha_qdq,
        use_cached_attn, use_feature_injection, use_controlnet, num_ip_layers,
        fp8_scale_headroom, fp8_exclude_attention, fp8_exclude_ipadapter.
    """
    import json as _json

    if len(argv) != 2:
        print(
            "Usage: python -m streamdiffusion.acceleration.tensorrt.fp8_quantize '<json args>'",
            file=sys.stderr,
        )
        return 2

    args = _json.loads(argv[1])
    calib_data = load_calibration_data(args["calib_data_path"])
    if calib_data is None:
        print(f"[FP8] Calibration data missing: {args['calib_data_path']}", file=sys.stderr)
        return 1

    try:
        quantize_onnx_fp8(
            onnx_path=args["onnx_path"],
            output_path=args["output_path"],
            calibration_data=calib_data,
            disable_mha_qdq=args.get("disable_mha_qdq", True),
            use_cached_attn=args.get("use_cached_attn", False),
            use_feature_injection=args.get("use_feature_injection", False),
            use_controlnet=args.get("use_controlnet", False),
            num_ip_layers=args.get("num_ip_layers", 0),
            fp8_scale_headroom=args.get("fp8_scale_headroom", 1.0),
            fp8_exclude_attention=args.get("fp8_exclude_attention", False),
            fp8_exclude_ipadapter=args.get("fp8_exclude_ipadapter", False),
        )
    except Exception:
        logger.exception("[FP8] quantize_onnx_fp8 failed")
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sys.exit(_main(sys.argv))
