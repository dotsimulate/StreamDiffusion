# cuda-link is an external pip dependency, not vendored

Status: accepted

## Context & Decision

StreamDiffusion previously vendored the `cuda-link` CUDA-IPC library into
`src/streamdiffusion/_compat/` as two mirror trees (`cuda_ipc/` Python runtime +
`td_exporter/` TouchDesigner DAT source). Keeping the mirrors in lockstep was a recurring
maintenance trap (the "RE-VENDORING TRAP" documented in `VENDORED_VERSION.txt`: 5
relative-import patches re-applied on every re-vendor) and the dependency seam was
dishonest — `wrapper.py` hard-imported `cuda_link` while `setup.py` never declared it.

We now **depend solely on the pip-installed `cuda-link`** (declared in `setup.py` as
`cuda-link @ git+https://github.com/forkni/cuda-link@v1.9.0`, exposed via the `cuda_ipc`
optional extra). The TouchDesigner side consumes the same installed package through
`CUDALinkBootstrap`'s **library mode** (`CUDALINK_LIB_PATH` injects the venv onto TD's
`sys.path` and aliases the 14 bare module names used by TD DATs), so the TD DAT mirror
inside the repo is no longer needed either.

## Considered Options

- **Re-vendor on each release** — rejected: the 5-patch RE-VENDORING TRAP makes every
  upstream update a multi-step, error-prone manual process with no test coverage.
- **Keep mirrors as read-only reference copies** — rejected: they silently diverge from the
  installed version, giving a false sense of documentation accuracy.
- **Depend on pip + library mode (chosen)** — single source of truth; honest dependency
  seam; TD and Python both consume the same binary; upgrade is `pip install -U cuda-link`.

## Consequences

- The load-bearing diffusers monkey-patch (`diffusers_kvo_patch.py`) was relocated from
  `_compat/` to `src/streamdiffusion/_patches/` — it was misfiled there (nothing to do
  with cuda-link). See `_patches/__init__.py` for the import-time side effect.
- IPC is an **optional feature**: `pip install -e .[cuda_ipc]`. The `wrapper.py` imports
  are lazy/in-method so the core package installs and runs without cuda-link.
- The `github.com/forkni/cuda-link` remote **must carry the referenced tag** or clean installs
  fail. Current pin in `setup.py`: **`v1.10.1`** (as of 2026-06-10; tag pushed, SHA `1e07a62`).
  **Runtime venv is on 1.11.0** (installed 2026-06-12 from local wheel; `setup.py` pin stays at
  `v1.10.1` until `v1.11.0` is tagged and pushed on the cuda-link remote). 1.11.0 is on branch
  `feat/r2-doorbell`; once merged and tagged, bump `setup.py` line 55 to `@v1.11.0`.
- **1.10.x history and the CUDA 719 incident (2026-06-10):** cuda-link 1.10.0 introduced
  async-by-default `export()` (no per-frame `cudaStreamSynchronize`) and opt-in
  `CUDALINK_D2H_PIPELINED` for overlapped D2H copy. However, **1.10.0 had a producer-side
  source-buffer lifetime race specific to the TD Sender**: `export_frame()` reads directly
  from TD's cook-scoped TOP texture (`cm.ptr`), which TD recycles the instant the cook
  returns. Under a loaded GPU consumer (e.g. SD's TRT inference), the async IPC-stream memcpy
  queued by 1.10.0 executed *after* TD recycled the source → illegal GPU memory read → sticky
  `cudaErrorLaunchFailure` (719) cascading through the consumer's CUDA context. Root cause
  confirmed by Loop A (`CUDALINK_EXPORT_SYNC=1` eliminated the crash; Hypothesis #2 —
  producer-side async source-buffer lifetime race). Note: `record_source_sync` /
  `_arm_same_stream_ordering` is a *pre-copy ordering* primitive only — it does not guarantee
  the source buffer outlives the queued async read. **Fixed in 1.10.1**: the TD Sender now
  blocks by default (`TDSenderEngine._resolve_export_sync`: unset/`None` → blocking; explicit
  `CUDALINK_EXPORT_SYNC=0` opts back into async for callers with a stable persistent source
  buffer). A second 1.10.1 fix adds `stream_synchronize(ipc_stream)` before teardown in
  `Exporter._do_cleanup` (latent 719 on geometry-change reopen / explicit `close()`). Skip
  1.10.0 entirely — it carries the 719 regression. The 1.10.x new features (D2H pipelining,
  sender stats log) are present in 1.10.1.
- **1.10.2 / 1.10.3 fixes (folded into 1.11.0 venv install):** receiver idle-cook skip,
  import hot-path allocation reduction, non-Windows `Exporter.open()` fix, pipelined-D2H teardown
  drain + stale-frame reprime, TD-sender FPS telemetry fix.
- **1.11.0 opt-in perf features (2026-06-12):** `CUDALINK_TORCH_GPU_WAIT_ADAPTIVE=1` replaces the
  consumer's CPU-side event poll with `cudaStreamWaitEvent` on the torch path — auto-latches only
  when real sleep-blocking is detected (~20% of frames at 30 fps; correctly stays in cpu-spin at
  60 fps). Enabled by default in `td_manager.py` `start_streaming()` via `os.environ.setdefault`.
  `CUDALINK_DOORBELL=1` (Win32 named-event kernel wake) is available but intentionally left dormant
  for SD's topology; enable only if a `wait_for_doorbell()` loop is added to `_streaming_loop`.
- The `CUDALINK_LIB_PATH` env var must be set in the TD launch environment to point at the
  venv site-packages path; without it, `CUDALinkBootstrap` falls back to classic Text-DAT
  module discovery (still works, but requires the mirror DATs to be present in the COMP).
- Do **not** re-suggest re-vendoring in future architecture reviews — this is a deliberate
  reversal of the previous approach, made after confirming that the mirror trees had zero
  runtime consumers in the Python import graph.
