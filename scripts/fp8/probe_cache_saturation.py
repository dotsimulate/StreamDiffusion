"""
FP8 round-6 Step 0 diagnostic probe — cache-saturation hypothesis check.

Context: fp8-round-5-handoff plan, "Defect B" — capture_calibration_data hooks the
raw diffusers UNet, which runs *before* the export wrappers exist, so every input
that only exists on the wrapped/quantized graph (kvo_cache_in_*, fio_cache_in_*,
fi_strength, fi_threshold, ipadapter_scale) was synthesized as zeros (ones for
ipadapter_scale) rather than captured from real activity. Graph-reachability
analysis found 91.2% of quantized activation tensors are transitively fed by
those zero-filled inputs. This script checks whether that actually moves
activation magnitudes enough to explain the reported "less blurry but not as
sharp" softness, BEFORE paying for a ~50 minute engine rebuild.

Method: load the plain (pre-export-wrapper) SDXL-Turbo UNet, install
CachedSTAttnProcessor2_0 on every self-attention (attn1) layer exactly as
wrapper.py does for deployment (down -> mid -> up walk, cache_maxframes=4),
and drive ONE real row (sample/timestep/encoder_hidden_states/text_embeds/
time_ids) from the existing round-5 calib_data.npz through it twice:

  (a) "calibration condition" — kvo_cache = zeros, fi_strength = 0,
      fi_threshold = 0. This reproduces exactly what capture_calibration_data
      fed the quantizer before fp8-round-5-handoff Step 2/3 existed.
  (b) "deployment condition" — kvo_cache tiled from THIS SAME forward's own
      curr_key/curr_value (harvested via attn1.to_k/to_v hooks during pass
      (a)), fio_cache tiled from pass (a)'s own attn1 block outputs
      (CachedSTAttnProcessor2_0._fi_cache_out), fi_strength/fi_threshold at
      the deployment config's values.

Every nn.Linear submodule's output |amax| is recorded for both passes (Q/DQ
nodes in the exported ONNX graph flank exactly these boundaries). The report
is the per-tensor distribution of amax_b / amax_a.

Decision rule (per the plan): median ratio >= ~2 confirms Defect B is a
material contributor to the softness -> proceed with the full fix. Median
~= 1 -> Defect B is not the driver; the band-math fix (Defect A) should ship
alone and the softness re-diagnosed.

Scope note: ipadapter_scale is NOT modeled here. IP-Adapter's cross-attention
(attn2) image-token pathway is a structurally separate mechanism from the
attn1 K/V-cache + Feature-Injection blend this script probes, and standing up
a live IP-Adapter forward pass (image encoder, projection, attn2 processor
swap) is much heavier machinery than this quick diagnostic warrants. The two
axes this script *does* drive (kvo_cache: 70 inputs, always active regardless
of fi_strength; fio_cache/fi_strength/fi_threshold: 46+2 inputs, active only
in the deployment condition) account for 118 of the defect's zero-filled
synthetic inputs -- ipadapter_scale is filled with 1.0 (not zero) and is
called out as a separate row in the plan's defect table.

Usage (from the StreamDiffusion/ directory):
    venv/Scripts/python.exe scripts/fp8/probe_cache_saturation.py
    venv/Scripts/python.exe scripts/fp8/probe_cache_saturation.py \
        --fi-strength 0.75 --fi-threshold 0.988 --cache-maxframes 4
"""

import argparse
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

import torch

# Importing streamdiffusion applies the kvo_cache monkeypatch to
# UNet2DConditionModel.forward (src/streamdiffusion/_patches/__init__.py) —
# required before any unet(..., kvo_cache=...) call below.
import streamdiffusion  # noqa: F401
from streamdiffusion.acceleration.tensorrt.fp8_quantize import _walk_attn1_modules
from streamdiffusion.acceleration.tensorrt.models.attention_processors import (
    CachedSTAttnProcessor2_0,
)
from streamdiffusion.acceleration.tensorrt.models.utils import (
    convert_list_to_structure,
    create_fi_cache,
    create_kvo_cache,
    get_fi_eligible_mask,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("probe_cache_saturation")


def _newest_calib_npz() -> "Path | None":
    """Resolve the newest calib_data.npz under engines/td/stabilityai, rather than
    a single hardcoded engine-dir name (FP8 Round 14). The calv* cache tag forks
    on every capture-affecting fix (calv4 -> calv5 -> ... -> calv8 and counting),
    so a fixed default goes stale the day any of them lands -- confirmed: the
    calv3 dir this constant used to hardcode, h6048ff5c55a8, no longer exists on
    disk as of the calv8 fork, and the script could not run at its own default."""
    engines_root = _REPO_ROOT / "engines" / "td" / "stabilityai"
    candidates = sorted(engines_root.glob("*/calib_data.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


_DEFAULT_CALIB_NPZ = _newest_calib_npz()
_SDXL_COND_KEYS = ["text_embeds", "time_ids"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="stabilityai/sdxl-turbo")
    p.add_argument(
        "--calib-npz",
        type=Path,
        default=_DEFAULT_CALIB_NPZ,
        required=_DEFAULT_CALIB_NPZ is None,
        help="Existing calib_data.npz to source real row-sets from. Defaults to the "
        "newest one found under engines/td/stabilityai (resolved dynamically -- see "
        "_newest_calib_npz).",
    )
    p.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help=(
            "Row indices inside calib_data.npz to probe (results pooled across all of them). "
            "Default [1, 2, 3] = the three real deployment timesteps (799/599/479 for the "
            "round-5 schedule); row 0 (and its duplicate, row 4) is t=999 — the spurious "
            "all-noise index Defect A introduces and never visits at inference, and a "
            "misleading row to test Defect B on since cache content barely matters at pure "
            "noise. Pass --rows 0 explicitly if you want that row anyway."
        ),
    )
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--cache-maxframes", type=int, default=4, help="Matches the use_cached_attn default in wrapper.py.")
    p.add_argument("--max-fi-up-blocks", type=int, default=2)
    # Deployment values from the plan's Step 0 spec / the current config.
    p.add_argument("--fi-strength", type=float, default=0.75)
    p.add_argument("--fi-threshold", type=float, default=0.988)
    p.add_argument("--ratio-threshold", type=float, default=1.5, help="Count tensors with amax_b/amax_a above this.")
    return p.parse_args()


def _load_row(npz_path: Path, row: int, device: str):
    data = np.load(npz_path)
    sample = torch.from_numpy(data["sample"][row : row + 1]).to(device)
    timestep = torch.tensor(int(data["timestep"][row]), device=device)
    encoder_hidden_states = torch.from_numpy(data["encoder_hidden_states"][row : row + 1]).to(device)
    added_cond_kwargs = {key: torch.from_numpy(data[key][row : row + 1]).to(device) for key in _SDXL_COND_KEYS}
    logger.info(
        f"[probe] Loaded row {row} from {npz_path.name}: sample={tuple(sample.shape)} "
        f"timestep={int(timestep)} encoder_hidden_states={tuple(encoder_hidden_states.shape)}"
    )
    return sample, timestep, encoder_hidden_states, added_cond_kwargs


def _install_cached_attn(unet, fi_mask):
    """Mirror wrapper.py's cached-attn install block (_load_model, ~:2431-2463):
    walk down -> mid -> up and install CachedSTAttnProcessor2_0 on every attn1,
    in the same order get_kvo_cache_info/_walk_attn1_modules use so index i
    lines up 1:1 with fi_mask[i]."""
    attn1_modules = _walk_attn1_modules(unet)
    assert len(attn1_modules) == len(fi_mask), (
        f"attn1 module count {len(attn1_modules)} != fi_eligible_mask length {len(fi_mask)}"
    )
    for i, attn1 in enumerate(attn1_modules):
        attn1.set_processor(CachedSTAttnProcessor2_0(fi_eligible=bool(fi_mask[i])))
    return attn1_modules


def _register_amax_hooks(unet, amax_store: dict):
    """Record output |amax| for every nn.Linear submodule — the ONNX-exported
    graph's Q/DQ nodes flank exactly these boundaries, per the plan's
    reachability analysis (1955 quantized activation tensors)."""
    handles = []

    def _make_hook(name):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            amax_store[name] = float(out.detach().abs().max().item())

        return hook

    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(_make_hook(name)))
    return handles


def _register_kv_capture_hooks(attn1_modules):
    """Capture attn1.to_k / attn1.to_v outputs (= curr_key/curr_value, the
    same tensors CachedSTAttnProcessor2_0 stacks into kvo_cache_out) so pass
    (a)'s own forward can supply pass (b)'s deployment-condition cache."""
    kv_capture: dict = {}
    handles = []

    def _make_hook(layer_idx, which):
        def hook(_module, _inputs, output):
            kv_capture.setdefault(layer_idx, {})[which] = output.detach().clone()

        return hook

    for i, attn1 in enumerate(attn1_modules):
        handles.append(attn1.to_k.register_forward_hook(_make_hook(i, "k")))
        handles.append(attn1.to_v.register_forward_hook(_make_hook(i, "v")))
    return kv_capture, handles


def _tile_maxframes(t: torch.Tensor, cache_maxframes: int) -> torch.Tensor:
    """(batch, seq, hidden) -> (cache_maxframes, batch, seq, hidden), each
    frame slot filled with the same real tensor (self-tiled — we only have
    one real row-set available, per the plan's "tiled from the same forward's
    own curr_key/curr_value")."""
    return t.unsqueeze(0).expand(cache_maxframes, *t.shape).contiguous()


def _percentiles(values: np.ndarray) -> dict:
    pcts = [10, 25, 50, 75, 90, 95]
    return {f"p{p}": float(np.percentile(values, p)) for p in pcts}


def main() -> None:
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.warning("[probe] CUDA not available — running on CPU (will be slow, but still correct).")

    from diffusers import UNet2DConditionModel

    logger.info(f"[probe] Loading UNet from {args.model_id} (fp16, {device})...")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16)
    unet = unet.to(device)
    unet.eval()

    fi_mask = get_fi_eligible_mask(unet, height=args.height, width=args.width, max_fi_up_blocks=args.max_fi_up_blocks)
    attn1_modules = _install_cached_attn(unet, fi_mask)
    logger.info(
        f"[probe] Installed CachedSTAttnProcessor2_0 on {len(attn1_modules)} attn1 layers ({sum(fi_mask)} FI-eligible)."
    )

    per_layer_views, kvo_structure, _kvo_buckets, _kvo_out_by_bucket = create_kvo_cache(
        unet,
        batch_size=1,
        cache_maxframes=args.cache_maxframes,
        height=args.height,
        width=args.width,
        device=device,
        dtype=torch.float16,
    )
    per_fi_layer_views, fi_layer_indices, _fi_buckets, _fi_out_by_bucket = create_fi_cache(
        unet,
        batch_size=1,
        cache_maxframes=args.cache_maxframes,
        fi_eligible_mask=fi_mask,
        height=args.height,
        width=args.width,
        max_fi_up_blocks=args.max_fi_up_blocks,
        device=device,
        dtype=torch.float16,
    )
    kvo_cache_nested = convert_list_to_structure(per_layer_views, kvo_structure)

    amax_store: dict = {}
    linear_handles = _register_amax_hooks(unet, amax_store)
    kv_capture, kv_handles = _register_kv_capture_hooks(attn1_modules)

    per_row_ratios: dict = {}
    pooled_ratios: list = []

    for row in args.rows:
        sample, timestep, encoder_hidden_states, added_cond_kwargs = _load_row(args.calib_npz, row, device)

        # --- condition (a) setup: zero the cache/scalars fresh for this row
        # (a prior row's condition (b) pass leaves real data in these buffers) ---
        for views in per_layer_views:
            views.zero_()
        for views in per_fi_layer_views:
            views.zero_()
        for fi_local, global_idx in enumerate(fi_layer_indices):
            proc = attn1_modules[global_idx].processor
            proc._fi_cache = per_fi_layer_views[fi_local]  # zeros — mutated in place for (b) below
            proc._fi_strength = torch.zeros(1, dtype=torch.float32, device=device)
            proc._fi_threshold = torch.zeros(1, dtype=torch.float32, device=device)

        amax_store.clear()
        kv_capture.clear()
        logger.info(
            f"[probe] row {row} (t={int(timestep)}) — pass (a): calibration condition (kvo_cache=zeros, fi_strength=0)..."
        )
        with torch.inference_mode():
            unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                kvo_cache=kvo_cache_nested,
                return_dict=False,
            )
        amax_a = dict(amax_store)
        amax_store.clear()

        missing_kv = [
            i
            for i in range(len(attn1_modules))
            if i not in kv_capture or "k" not in kv_capture[i] or "v" not in kv_capture[i]
        ]
        if missing_kv:
            raise RuntimeError(f"[probe] to_k/to_v capture missing for layer indices {missing_kv} — hook wiring bug.")

        # --- condition (b) setup: real self-derived K/V + real FI output + deployment scalars ---
        for i, views in enumerate(per_layer_views):
            k_tile = _tile_maxframes(kv_capture[i]["k"], args.cache_maxframes)
            v_tile = _tile_maxframes(kv_capture[i]["v"], args.cache_maxframes)
            views.copy_(torch.stack([k_tile, v_tile], dim=0))

        for fi_local, global_idx in enumerate(fi_layer_indices):
            proc = attn1_modules[global_idx].processor
            fio_raw = proc._fi_cache_out  # (1, batch, seq, hidden), captured during pass (a)
            if fio_raw is None:
                raise RuntimeError(f"[probe] _fi_cache_out unset for FI-eligible layer {global_idx} after pass (a).")
            fio_tile = _tile_maxframes(fio_raw.squeeze(0), args.cache_maxframes)
            per_fi_layer_views[fi_local].copy_(fio_tile)
            proc._fi_strength = torch.tensor([args.fi_strength], dtype=torch.float32, device=device)
            proc._fi_threshold = torch.tensor([args.fi_threshold], dtype=torch.float32, device=device)

        logger.info(
            f"[probe] row {row} (t={int(timestep)}) — pass (b): deployment condition (kvo_cache=self-derived K/V, "
            f"fi_strength={args.fi_strength}, fi_threshold={args.fi_threshold})..."
        )
        with torch.inference_mode():
            unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                kvo_cache=kvo_cache_nested,
                return_dict=False,
            )
        amax_b = dict(amax_store)

        common_names = sorted(set(amax_a) & set(amax_b))
        dropped = sorted(set(amax_a) ^ set(amax_b))
        if dropped:
            logger.warning(
                f"[probe] row {row}: {len(dropped)} Linear module(s) fired in only one pass (dead code path?): "
                f"{dropped[:10]}{'...' if len(dropped) > 10 else ''}"
            )

        eps = 1e-8
        row_ratios = np.array([amax_b[n] / max(amax_a[n], eps) for n in common_names], dtype=np.float64)
        per_row_ratios[row] = row_ratios
        pooled_ratios.append(row_ratios)

    for h in linear_handles + kv_handles:
        h.remove()

    ratios = np.concatenate(pooled_ratios)
    over_thresh = int((ratios > args.ratio_threshold).sum())

    pct = _percentiles(ratios)
    print()
    print("=" * 70)
    print(
        f"FP8 round-6 Step 0 probe — rows {args.rows} pooled, {len(ratios)} tensor comparisons ({len(ratios) // len(args.rows)} nn.Linear tensors/row)"
    )
    print("=" * 70)
    for row, row_ratios in per_row_ratios.items():
        print(
            f"  row {row}: median={np.percentile(row_ratios, 50):.3f}  p90={np.percentile(row_ratios, 90):.3f}  max={row_ratios.max():.3f}"
        )
    print("-" * 70)
    print("pooled amax_b / amax_a percentiles: " + ", ".join(f"{k}={v:.3f}" for k, v in pct.items()))
    print(f"min={ratios.min():.3f}  max={ratios.max():.3f}  mean={ratios.mean():.3f}")
    print(
        f"tensor-comparisons with ratio > {args.ratio_threshold}: {over_thresh}/{len(ratios)} ({100.0 * over_thresh / len(ratios):.1f}%)"
    )
    print("-" * 70)
    median = pct["p50"]
    if median >= 2.0:
        print(f"DECISION: pooled median ratio {median:.2f} >= ~2 -> Defect B confirmed as a material")
        print("contributor. Proceed with the full plan (Steps 1-4 already implemented).")
    elif median <= 1.2:
        print(f"DECISION: pooled median ratio {median:.2f} ~= 1 -> Defect B is NOT the dominant softness")
        print("driver. Ship Defect A (band-math fix) alone and re-diagnose after that build.")
    else:
        print(f"DECISION: pooled median ratio {median:.2f} is inconclusive (between ~1 and ~2).")
        print("Inspect the full percentile spread above before deciding whether to proceed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
