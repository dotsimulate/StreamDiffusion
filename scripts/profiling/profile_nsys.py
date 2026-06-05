"""
StreamDiffusion Nsight Systems profiling launcher.

Supports two targets:
  --target benchmark  In-process wrapper loop (clean, deterministic, no TouchDesigner needed)
  --target td_main    Subprocess td_main.py with deferred-capture env vars (real production path)

── benchmark target ───────────────────────────────────────────────────────────
Run standalone (torch.profiler only, no nsys):
    .venv/Scripts/python scripts/profiling/profile_nsys.py --target benchmark

Run under nsys for GPU timeline:
    set NSYS="C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe"
    %NSYS% profile --trace=cuda,nvtx,cublas --cuda-memory-usage=true ^
        -o profiles/sdtd_benchmark --force-overwrite true ^
        .venv/Scripts/python scripts/profiling/profile_nsys.py --target benchmark

── td_main target ─────────────────────────────────────────────────────────────
Launches StreamDiffusionTD/td_main.py with deferred-capture env vars so
cudaProfilerStart fires after --warmup frames (default 20) and stop + exit
fires after warmup + --frames (default 500).  The launcher script itself does
NOT need to run under nsys; td_main.py runs nsys-attached:

    set NSYS="C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe"
    %NSYS% profile --trace=cuda,nvtx,cublas --cuda-memory-usage=true ^
        --capture-range cudaProfilerApi ^
        -o profiles/sdtd_td_main --force-overwrite true ^
        .venv/Scripts/python StreamDiffusionTD/td_main.py

    # Or let this script manage the subprocess (SDTD_NSYS_CAPTURE=1 deferred handshake):
    .venv/Scripts/python scripts/profiling/profile_nsys.py --target td_main --warmup 20 --frames 500

── Post-capture analysis ──────────────────────────────────────────────────────
    %NSYS% stats --report cuda_kern_exec_trace profiles/sdtd_benchmark.nsys-rep > kernel_trace.txt
    %NSYS% stats --report nvtx_pushpop_trace   profiles/sdtd_benchmark.nsys-rep > nvtx_trace.txt
    "C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/host-windows-x64/nsys-ui.exe" profiles/sdtd_benchmark.nsys-rep

── CUDA graph + NVTX note ─────────────────────────────────────────────────────
    NVTX push/pop calls embedded in a CUDA graph fire only at capture time
    (3-warmup passes), not on each replay step.  For events-only mode (graph-safe)
    set GPU_PROFILER_NVTX=0.  The CUDA-event timings in profiler_logs/*.json are
    always accurate because they use CUDA events, not NVTX.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time


# ── CLI args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="StreamDiffusion Nsight Systems profiling launcher")
parser.add_argument(
    "--target",
    default="benchmark",
    choices=["benchmark", "td_main"],
    help="Profiling target: 'benchmark' (in-process) or 'td_main' (subprocess)",
)
parser.add_argument(
    "--frames",
    type=int,
    default=500,
    help="[td_main] Frames to capture after warmup (default: 500)",
)
parser.add_argument(
    "--warmup",
    type=int,
    default=20,
    help="[td_main] Frames to skip before cudaProfilerStart (default: 20)",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print subprocess commands without executing (td_main target only)",
)
parser.add_argument(
    "--cn-scale",
    type=float,
    default=0.0,
    metavar="SCALE",
    help="[benchmark] ControlNet conditioning scale (0 = disabled, default). "
    "When > 0, activates the first registered ControlNet at this scale using a dummy "
    "gray control image. Lets you measure CN per-frame cost alongside the UNet baseline.",
)
parser.add_argument(
    "--cn-cache-interval",
    type=int,
    default=1,
    metavar="N",
    help="[benchmark] ControlNet residual cache interval (default 1 = disabled). "
    "N>1: CN forward runs once every N frames; residuals reused between. "
    "Requires --cn-scale > 0.",
)
parser.add_argument(
    "--config",
    default="",
    metavar="PATH",
    help="[benchmark] YAML/JSON config file (e.g. StreamDiffusionTD/td_config.yaml). "
    "Loads wrapper kwargs from file so an existing cached engine is reused. "
    "When empty, uses inline defaults (KBlueLeaf/kohaku-v2.1).",
)
args = parser.parse_args()

# ── Common paths ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PROFILES_DIR = os.path.join(_PROJECT_ROOT, "profiles")
_PROFILER_LOGS_DIR = os.path.join(_PROJECT_ROOT, "profiler_logs")
_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
_PYTHON = sys.executable

os.makedirs(_PROFILES_DIR, exist_ok=True)
os.makedirs(_PROFILER_LOGS_DIR, exist_ok=True)

# ── Locate nsys ────────────────────────────────────────────────────────────────
_NSYS_CANDIDATES = [
    os.environ.get("NSYS", ""),
    "nsys",
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.2.1\target-windows-x64\nsys.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe",
]
_NSYS = next((p for p in _NSYS_CANDIDATES if p and shutil.which(p)), None)
if _NSYS:
    print(f"[profile] nsys: {_NSYS}")
else:
    print("[profile] WARNING: nsys not found — torch.profiler only (no GPU kernel timeline).")
    print("  Set NSYS=<path-to-nsys.exe> or add nsys to PATH to enable GPU tracing.")

# ═══════════════════════════════════════════════════════════════════════════════
# td_main target: deferred-capture subprocess
# ═══════════════════════════════════════════════════════════════════════════════
if args.target == "td_main":
    profile_base = os.path.join(_PROFILES_DIR, f"sdtd_td_main_{_TIMESTAMP}")
    json_path = os.path.join(_PROFILER_LOGS_DIR, f"sdtd_td_main_{_TIMESTAMP}_stats.json")
    md_path = os.path.join(_PROFILER_LOGS_DIR, f"sdtd_td_main_{_TIMESTAMP}_report.md")

    td_main_script = os.path.join(_PROJECT_ROOT, "StreamDiffusionTD", "td_main.py")
    if not os.path.exists(td_main_script):
        sys.exit(f"[profile] ERROR: td_main.py not found at {td_main_script}")

    # Build the subprocess environment — deferred-capture handshake + profiler activation
    proc_env = dict(os.environ)
    proc_env["GPU_PROFILER"] = "1"
    proc_env.setdefault("GPU_PROFILER_NVTX", "1")
    proc_env["GPU_PROFILER_EVENTS"] = "1"
    proc_env["SDTD_NSYS_CAPTURE"] = "1"
    proc_env["SDTD_NSYS_WARMUP_FRAMES"] = str(args.warmup)
    proc_env["SDTD_NSYS_CAPTURE_FRAMES"] = str(args.frames)
    proc_env["SDTD_PROFILE_JSON"] = json_path

    td_cmd = [_PYTHON, td_main_script]

    print("\n[profile] td_main deferred-capture run")
    print(f"  warmup:      {args.warmup} frames  capture: {args.frames} frames")
    print(f"  profile out: {profile_base}.nsys-rep  (wrap with nsys manually)")
    print(f"  stats json:  {json_path}")
    print(f"\n  Command: {' '.join(td_cmd)}")
    print("\n  Tip: wrap td_main.py directly with nsys for GPU kernel capture:")
    print("    nsys profile --trace=cuda,nvtx,cublas --capture-range cudaProfilerApi \\")
    print(f"        -o {profile_base} .venv/Scripts/python StreamDiffusionTD/td_main.py")

    if args.dry_run:
        print("\n[profile] --dry-run: exiting without launching.")
        sys.exit(0)

    print("\n[profile] Launching td_main.py ...")
    proc = subprocess.Popen(td_cmd, env=proc_env, cwd=_PROJECT_ROOT)
    print(f"[profile] Waiting for {args.frames} capture frames + {args.warmup} warmup frames ...")
    proc.wait()
    print(f"[profile] td_main.py exited (code {proc.returncode})")

    # ── Render Markdown report ─────────────────────────────────────────────────
    if os.path.exists(json_path):
        with open(json_path) as fh:
            data = json.load(fh)
        regions = sorted(data.get("regions", []), key=lambda r: r["total_ms"], reverse=True)
        if regions:
            with open(md_path, "w") as rpt:
                rpt.write(f"# TD Main Profile — {_TIMESTAMP}\n\n")
                rpt.write(f"**Target**: `td_main`  **Warmup**: {args.warmup}  **Frames**: {args.frames}\n\n")
                rpt.write("## Per-region timing (sorted by total ms)\n\n")
                rpt.write("| Region | Count | Mean ms | P50 ms | P95 ms | P99 ms | Min ms | Max ms | Total ms |\n")
                rpt.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                for r in regions:
                    rpt.write(
                        f"| `{r['name']}` | {r['count']} "
                        f"| {r['mean_ms']:.3f} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} "
                        f"| {r['p99_ms']:.3f} | {r['min_ms']:.3f} | {r['max_ms']:.3f} "
                        f"| {r['total_ms']:.1f} |\n"
                    )
            print(f"\n[profile] Report -> {md_path}")
            print(f"\n{'Region':<30} {'Mean ms':>8}  {'P50 ms':>8}  {'P95 ms':>8}  {'Total ms':>10}")
            print("-" * 70)
            for r in regions[:15]:
                print(
                    f"  {r['name']:<28} {r['mean_ms']:>8.3f}  {r['p50_ms']:>8.3f}  {r['p95_ms']:>8.3f}  {r['total_ms']:>10.1f}"
                )
    else:
        print(f"[profile] WARNING: stats JSON not found at {json_path} — td_main may have crashed.")

    print("\n[profile] Complete.")
    sys.exit(0)

# ═══════════════════════════════════════════════════════════════════════════════
# benchmark target: in-process wrapper loop
# ═══════════════════════════════════════════════════════════════════════════════
import torch
from torch.profiler import ProfilerActivity, profile, schedule

from streamdiffusion.tools.gpu_profiler import profiler


os.environ.setdefault("GPU_PROFILER", "1")  # wrapper.__init__ reads this to activate

WARMUP_RUNS = 3  # extra warmup before torch.profiler + nsys capture window
PROFILE_RUNS = 10  # inferences captured by nsys / torch.profiler

sys.path.insert(0, _PROJECT_ROOT)

# ── Load pipeline ──────────────────────────────────────────────────────────────
if args.config:
    # Config-file path: identical wrapper kwargs to td_main.py → cache hit, no rebuild.
    from streamdiffusion import create_wrapper_from_config, load_config

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_PROJECT_ROOT, args.config)
    print(f"[profile] benchmark target — loading wrapper from {cfg_path}")
    cfg = load_config(cfg_path)
    _WIDTH = cfg.get("width", 512)
    _HEIGHT = cfg.get("height", 512)
    t0 = time.perf_counter()
    stream = create_wrapper_from_config(cfg)  # also calls .prepare()
else:
    # Inline-defaults path: useful for experiments; may trigger an engine build.
    from streamdiffusion import StreamDiffusionWrapper

    print("[profile] benchmark target — loading StreamDiffusionWrapper (inline defaults) ...")
    print("  Tip: pass --config StreamDiffusionTD/td_config.yaml to hit an existing cached engine.")
    _MODEL_ID = os.environ.get("SDTD_MODEL_ID", "KBlueLeaf/kohaku-v2.1")
    _ACCELERATION = os.environ.get("SDTD_ACCELERATION", "tensorrt")
    _WIDTH = int(os.environ.get("SDTD_WIDTH", "512"))
    _HEIGHT = int(os.environ.get("SDTD_HEIGHT", "512"))
    t0 = time.perf_counter()
    stream = StreamDiffusionWrapper(
        model_id_or_path=_MODEL_ID,
        t_index_list=[32, 45],
        mode="img2img",
        frame_buffer_size=1,
        width=_WIDTH,
        height=_HEIGHT,
        warmup=WARMUP_RUNS,
        acceleration=_ACCELERATION,
        use_lcm_lora=True,
        use_tiny_vae=True,
        use_denoising_batch=True,
        cfg_type="initialize",
        seed=42,
    )
    stream.prepare(
        prompt="abstract flowing colorful pattern",
        negative_prompt="bad quality",
        num_inference_steps=50,
        guidance_scale=1.4,
        delta=0.5,
    )
print(f"[profile] Pipeline ready in {time.perf_counter() - t0:.1f}s\n")

# ── Dummy input image ──────────────────────────────────────────────────────────
import PIL.Image


dummy_img = PIL.Image.new("RGB", (_WIDTH, _HEIGHT), (128, 128, 128))

# ── ControlNet activation (--cn-scale > 0) ────────────────────────────────────
if args.cn_scale > 0.0:
    try:
        cn_mod = getattr(stream.stream, "_controlnet_module", None)
        if cn_mod is None:
            raise RuntimeError("No _controlnet_module found — ensure config has a ControlNet")
        # Set scale
        cn_mod.update_controlnet_scale(0, args.cn_scale)
        # update_control_image_efficient bails if _preprocessing_orchestrator is None (offline mode).
        # Bypass it: directly inject a dummy control tensor ([1,3,H,W] fp16 on GPU) so the hook's
        # 'img is not None' gate passes.  prepare_frame_tensors will expand it to the right batch.
        with cn_mod._collections_lock:
            if len(cn_mod.controlnet_images) > 0:
                dummy_cn = torch.ones(1, 3, _HEIGHT, _WIDTH, dtype=torch.float16, device="cuda") * 0.5
                cn_mod.controlnet_images[0] = dummy_cn
                cn_mod._prepared_tensors = []
                cn_mod._images_version += 1
            else:
                raise RuntimeError("ControlNet registered but controlnet_images list is empty")
        print(f"[profile] ControlNet[0] enabled: scale={args.cn_scale}, image=dummy gray tensor {_WIDTH}x{_HEIGHT}")
        if args.cn_cache_interval > 1:
            cn_mod.set_cn_cache_interval(args.cn_cache_interval)
            print(f"[profile] ControlNet residual cache: interval={args.cn_cache_interval} (CN forward every {args.cn_cache_interval} frames)")
    except Exception as _cn_err:
        print(f"[profile] WARNING: Could not activate ControlNet — {_cn_err}")
        print("  Make sure the config includes a ControlNet and its engine is built.")

# ── Preprocess once ────────────────────────────────────────────────────────────
image_tensor = stream.preprocess_image(dummy_img)


def run_inference(label: str = ""):
    """One inference step with NVTX frame label."""
    torch.cuda.nvtx.range_push(f"frame{label}")
    result = stream(image=image_tensor)
    torch.cuda.nvtx.range_pop()
    return result


# ── Extra warmup (not captured) ────────────────────────────────────────────────
print(f"[profile] Extra warmup ({WARMUP_RUNS} runs)...")
for i in range(WARMUP_RUNS):
    run_inference(f"_warmup{i}")
torch.cuda.synchronize()
print("[profile] Warmup done.\n")

# ── torch.profiler capture ─────────────────────────────────────────────────────
TRACE_PATH = os.path.join(_PROFILER_LOGS_DIR, f"sdtd_benchmark_{_TIMESTAMP}_trace.json")
TOTAL_STEPS = 1 + PROFILE_RUNS

print(f"[profile] torch.profiler capture ({PROFILE_RUNS} active steps)...")
# CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED is benign when running under nsys;
# both register CUPTI subscribers but CUDA-event timings in *_stats.json are unaffected.
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=0, warmup=1, active=PROFILE_RUNS, repeat=1),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step in range(TOTAL_STEPS):
        run_inference(f"_prof{step}")
        prof.step()

torch.cuda.synchronize()
print("\n" + "=" * 80)
print("TORCH.PROFILER [benchmark] — Top 30 ops by CUDA time")
print("=" * 80)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
prof.export_chrome_trace(TRACE_PATH)
print(f"\n[profile] Chrome trace -> {TRACE_PATH}")

# ── nsys-gated capture window (cudaProfilerStart / Stop) ──────────────────────
print(f"\n[profile] nsys-gated capture ({PROFILE_RUNS} inferences)...")
torch.cuda.cudart().cudaProfilerStart()

for i in range(PROFILE_RUNS):
    t0 = time.perf_counter()
    run_inference(f"_nsys{i}")
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    print(f"  Step {i}: {ms:.1f} ms  ({1000 / ms:.2f} FPS)")

torch.cuda.cudart().cudaProfilerStop()

# ── CUDA-event stats report ────────────────────────────────────────────────────
profiler.report()
_STATS_PATH = os.path.join(_PROFILER_LOGS_DIR, f"sdtd_benchmark_{_TIMESTAMP}_stats.json")
profiler.export_stats(_STATS_PATH)

print("\n[profile] Complete.")
print("If running under nsys, analyze with:")
print("  nsys stats --report cuda_kern_exec_trace profiles/sdtd_benchmark.nsys-rep > kernel_trace.txt")
print("  nsys stats --report nvtx_pushpop_trace   profiles/sdtd_benchmark.nsys-rep > nvtx_trace.txt")
