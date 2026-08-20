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
`cuda-link @ git+https://github.com/forkni/cuda-link@v1.12.2`, exposed via the `cuda_ipc`
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
  fail. Current pin in `setup.py`: **`v1.12.2`** (tagged 2026-08-11; published release
  `v1.12.2` on the cuda-link remote — https://github.com/forkni/cuda-link/releases/tag/v1.12.2).
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
  `CUDALINK_DOORBELL=1` (Win32 named-event kernel wake) is now persisted by the installer's
  `phase4c_cuda_link_env` (see the 1.12.1 note below) — SD's topology is bidirectional (TD Sender
  and SD Exporter are each a producer on their own IPC leg), and `Importer.open()` in
  `td_manager.py` opens the doorbell speculatively under the default `wait_backend="auto"`
  regardless, so no `wait_for_doorbell()` loop needs to be added to `_streaming_loop` for the
  consumer side to benefit.
- **1.12.0 migration (2026-07-07 pin bump; no consumer API change — all 1.11.0 code works
  unmodified):**
  - **Prebuilt wheels; no MSVC required on install.** Per
    [cuda-link ADR-0013](https://github.com/forkni/cuda-link/blob/v1.12.0/docs/adr/0013-prebuilt-wheel-distribution.md),
    CI now publishes both a native `cp311-cp311-win_amd64` wheel (compiled `_native_waiter`
    accelerator) and a compiler-free `py3-none-any` fallback as GitHub Release assets.
    `scripts/install_td_library.py` (moved from repo-root `install_td_library.py`) resolves a
    wheel **per install target's own Python version**: `--wheel <path>` override → tag-matched
    wheel already in `dist/` → auto-download the matching Release asset → only with the new
    **`--build`** flag, compile locally via `utils\build_wheel.cmd` (dev-only; no longer the
    silent default). Mode 5 (installing into TD's own bundled Python) is now deprecated in the
    installer menu — steer to mode 2 (venv) or mode 4 (system Python), both of which resolve a
    native wheel automatically.
  - **CUDA 13 runtime support**: `CUDARuntimeAPI._load_cuda_runtime()` also probes
    `cudart64_13*.dll` alongside the 12.x candidates. Transparent — no action needed on a
    CUDA 12.x box.
  - **Opt-in native wait backend** (`ImportPolicy.wait_backend`: `"auto"` default | `"python"` |
    `"native"`; env `CUDALINK_WAIT_BACKEND`) replaces the pure-Python spin/sleep poll in
    `Importer._wait_for_slot` with a native `cudaEventQuery` + doorbell-blocking call.
    Consumer/importer-side config — but SD is *also* an importer/consumer, not solely the
    producer/exporter side: `td_manager.py`'s `Importer.open(ImportSpec(...))` calls (input +
    ControlNet channels) pass no policy, so `ImportPolicy.from_env()` applies to the SD process
    too. `CUDALINK_WAIT_BACKEND` is left unset (its `"auto"` default already resolves identically
    to `"native"`), but the doorbell/native fast path it enables is gated by the *producer's*
    `CUDALINK_DOORBELL` — see the 1.12.1 note below for why that must be installer-persisted.
  - **Torn-frame race fix in the native waiter's block phase** (`native_waiter.cpp`): the block
    phase now treats `cudaEventQuery` as the only valid "ready" signal instead of accepting a
    `write_idx` advance as a proxy. Per the cuda-link 1.12.0 release notes this bug was **"not
    reachable via the TD Sender (blocking export since v1.10.1 guarantees the copy is done
    before publish); reachable via the Python `Exporter`'s async default under GPU load."** SD's
    exporter already forces blocking export (see the "force blocking IPC export" fix above), so
    this fix is defense-in-depth on the consumer side, not a gap SD was exposed to.
- **1.12.1 migration (2026-07-12 pin bump; bugfix-only, no consumer API change):** fixes two
  reference-count leaks that only surface on repeated failure/retry paths, not on the happy
  path — **`nvml_observer.py`**: a failed `start()` after `acquire()` used to leave `nvmlInit()`
  unpaired with `nvmlShutdown()`, compounding across retries; and **`TDReceiverEngine
  .initialize_receiver()`**: a raise after the validation guards but before
  `connection_committed` used to leak the locally-opened `shm_handle` on every failed attempt
  and backoff retry. Neither leak is reachable via SD's normal (non-retrying,
  non-NVML-observing) exporter/importer paths; upgrade is precautionary hardening.
  - **Installer `phase4c` now also persists `CUDALINK_DOORBELL=1`** (previously only
    `CUDALINK_LIB_PATH`). The doorbell/native-wait fast path only engages when the *producer* on
    an IPC leg has `CUDALINK_DOORBELL=1` (`exporter.py` only creates the named Win32 event under
    that policy); SD's topology is bidirectional, so TD's Sender is a producer too, and it runs in
    TD's own bundled-Python **process** — a runtime `os.environ.setdefault` (the pattern used for
    `CUDALINK_TORCH_GPU_WAIT_ADAPTIVE`) can't reach a separate process, so this var must be
    persisted via `setx`, same as `CUDALINK_LIB_PATH`. Without it, the doorbell fast path silently
    degrades to poll-sleep on a fresh install — this gap was found via a clean-slate
    reproducibility test (removing `cuda_link` + all cuda-link env vars, then re-running
    `phase4b`/`phase4c` and confirming the vars they declare are sufficient on their own).
    `CUDALINK_WAIT_BACKEND` is intentionally left unset — its default `"auto"` already selects the
    native path.
- **1.12.2 migration (2026-08-11 pin bump; bugfix-only, no consumer API change):** per the
  [cuda-link 1.12.2 release notes](https://github.com/forkni/cuda-link/releases/tag/v1.12.2):
  - Fixes an undersized `cudaPointerAttributes` ctypes binding (`cuda_runtime_types.py`) — CUDA
    13.x's `driver_types.h` grew a `reserved[8]` field the 12.x-era struct didn't account for,
    and the loader probes 13.x cudart candidates before 12.x ones, so a 13.x runtime on the host
    could receive an undersized out-parameter on the per-frame `pointer_get_attributes()` path.
  - Fixes `cudaDevAttrAsyncEngineCount` being mislabeled as attribute index `4` (the real
    `cudaDevAttrMaxBlockDimZ`) in `cuda_ipc_wrapper.py` — per the release notes this was latent,
    with no production caller querying it yet.
  - Fixes the new `check_ipc_capability()` IPC probe (added earlier in this release) hard-failing
    `Exporter.open()` on a CUDA 11.x runtime instead of degrading gracefully — relevant here
    because the release notes call out that TouchDesigner ships `cudart64_110.dll`, which the
    loader also probes.
  - Adds `check_ipc_capability()` itself to the `CudaPort` protocol/adapters, called once from
    `Exporter.open()` before minting an IPC handle; logs an informational note on a driver-support
    gap rather than hard-failing, raising only if the driver reports IPC flatly unsupported.
  - This entry summarizes the published release notes; unlike the 1.10.x–1.12.1 entries above, it
    was not re-verified against SD's own exporter/importer code paths with a live reproduction
    test this pass — worth a follow-up pass if any of these prove reachable in practice.
- The `CUDALINK_LIB_PATH` env var enables `CUDALinkBootstrap`'s library mode (`sys.path`
  injection of the installed `cuda_link` package + the 14 bare-name DAT aliases); without it,
  `CUDALinkBootstrap` falls back to classic Text-DAT module discovery (still works, but requires
  the mirror DATs to be present in the COMP). As of the installer's `phase4c_cuda_link_env`
  step, `CUDALINK_LIB_PATH` and `CUDALINK_DOORBELL` are both set automatically
  (`setx CUDALINK_LIB_PATH <venv>\Lib\site-packages`, `setx CUDALINK_DOORBELL 1`) at the end
  of a fresh install — no manual step required. TouchDesigner must be (re)started after
  installation for the persisted variables to take effect.
- Do **not** re-suggest re-vendoring in future architecture reviews — this is a deliberate
  reversal of the previous approach, made after confirming that the mirror trees had zero
  runtime consumers in the Python import graph.
