# SDTD 0.4.1 development staging

The `sdtd041_dev` branch is for paired testing with the SDTD 0.4.1 development TOX.
It has not been promoted to `SDTD_040_stable`.

This integrates the reviewed changes from PRs #61–70: ORT DLL loading, dependency
documentation/export alignment, ControlNet engine reset, Canny smoothing, pose
rendering and engine building, FP8 diagnostics, HED/scribble, error reporting,
and skip-diffusion normalization. HED's new five-scale export uses a separate
`.hed5scale.engine` cache path and preserves older engines.

| Change | Existing 0.4.0 TOX after a future stable core update |
|---|---|
| Canny smoothing and cuDNN guard | Uses existing parameters; no new TOX needed |
| HED/scribble corrections | Core implementation; first use builds the new cache |
| Pose canvas/letterboxing | Applies to existing usable pose engines |
| Pose automatic engine-build integration | Requires the matching TD backend in the new TOX |
| ControlNet reset and bypass normalization | Core fixes, no new TOX needed |
| Richer TD diagnostics, IPC preview and input-buffer ownership | Require the new TOX's backend DATs |
| TOX updater and CUDA-Link installation lookup | TD-side changes; require the new TOX |

Targeted validation: 77 core tests passed using torch 2.8.0+cu128. These include
pose geometry, Canny parameter coverage, HED/scribble helpers, cache separation,
FP8 provenance, diagnostics and bypass normalization. This is not a full engine
build/inference validation on the supported torch 2.11 stack. Validate that runtime
with the paired development TOX before promoting compatible changes to stable.

Core updates do not replace an already loaded TOX or its embedded backend code.
An existing installation must still perform its core update to receive a future
stable promotion; these changes do not arrive simply by restarting a stream.
