# StreamDiffusion Test Suite & Hardening Guide

## Test Suite Baseline & Campaign Progress

| Metric | Pre-Campaign Baseline (2026-10-10) | Current Hardened State | Status |
|--------|-----------------------------------|------------------------|--------|
| Total tests collected | 794 | 794 (766 unit, 25 quality, 3 gpu) | ✅ Maintained |
| Clean-run status | 765 passed, 1 skipped, 0 failed | 765 passed, 1 skipped, 0 failed | ✅ 100% Green |
| Fast-loop time (`tests/unit`) | 26.12 s | **18.81 s** | 🚀 28% speedup (< 30 s budget) |
| Full-suite time | ~86 s | ~75 s | ⚡ Faster execution |
| Branch coverage (clean run) | 30.00% | **30.20%** | 📈 Upward ratchet |
| Current `fail_under` | None configured | **30%** (in `pyproject.toml`) | 🛡️ Gated & Monitored |
| Functions over CRAP ceiling (30) | 169 / 983 scored functions | 169 / 983 functions tracked | 📊 Monitored via `crap4py` |
| Standing `ruff` `C901` violations (> 10) | 71 functions | 71 functions (worst: `_load_model` comp=270) | 📊 Tracked |
| Order-independent? (10× loop) | Yes | **Yes — zero ordering flakes** | ✅ `pytest-randomly` + autouse reset |
| Test-smell scan total (`scan_test_smells.py`) | 54 smells | **24 smells** | 📉 **55% smell reduction** |
| └ `no-assert` (shallow promises) | 12 | **0** | ✅ 100% resolved |
| └ `early-exit` (conditional skips) | 4 | **0** | ✅ 100% resolved |
| └ `bitwise-assert` (bitmask asserts) | 1 | **0** | ✅ 100% resolved |
| └ `parametrize-no-ids` | 6 | **0** | ✅ 100% resolved |
| └ `overprotective` | 7 | **0** | ✅ 100% resolved |
| └ `abs-path` | 24 | 24 (grandfathered in baseline) | 🛡️ Ratcheted |

---

## Architecture & Test-Value Triage (Khorikov Quadrant)

```text
                            High Complexity / Significance
                                       │
            Domain Model / Algorithms  │  Overcomplicated Code
            [UNIT-TEST HARD]           │  [REFACTOR FIRST]
            - param_schema.py (94%)    │  - wrapper.py (comp 270, CRAP 41k)
            - nan_guard.py (98%)       │  - pipeline.py (unet_step comp 59)
            - diagnostics.py (88%)     │  - stream_parameter_updater.py
            - image_filter.py (84%)    │  - builder.py / fp8_quantize.py
            - faceid_compat.py (89%)   │
  Few ─────────────────────────────────┼───────────────────────────────── Many
  Collaborators                        │                                  Collaborators
            Trivial Code               │  Controllers / Orchestrators
            [DON'T TEST]               │  [INTEGRATION-TEST BRIEFLY]
            - __init__.py modules      │  - image_processing_module.py
            - constants & dataclasses  │  - latent_processing_module.py
            - pure passthroughs        │  - orchestrator_user.py
                                       │  - utils/reporting.py
                                       │
                            Low Complexity / Significance
```

---

## Hardening Campaign Execution

### Phase 0: Test Hierarchy & Scaffolding

- Added [tests/conftest.py](file:///D:/dev/SDTD_040_Beta/StreamDiffusion/tests/conftest.py) with automated test collection hooks:
  - `tests/unit/`: auto-marked with `@pytest.mark.unit`
  - `tests/gpu/`: auto-marked with `@pytest.mark.slow` and `@pytest.mark.gpu`
  - `tests/quality/`: auto-marked with `@pytest.mark.quality`
- Added `reset_global_state` autouse fixture to reset GPU cache between tests and prevent state leakage.

### Phase 0.5: Prune & Rewrite Low-Value Tests & Smell Debt

- **12 `no-assert` tests remediated**: Replaced empty test bodies and unasserted calls with concrete contract assertions:
  - Added return callable and execution assertions in `test_release_torch_vram.py`.
  - Added `warmup_faceid` return assertions in `test_faceid_bgr_patch.py`.
  - Added tensor finiteness and shape assertions in `test_nan_guard_latch_recovery.py` and `test_delta_range_stability.py`.
  - Added graph preservation assertions in `test_onnx_bytesize_free.py`.
  - Added VAE config validation return assertions in `test_vae_class_dispatch.py`.
- **4 `early-exit` tests remediated**: Converted body `pytest.skip()` calls into declarative `@pytest.mark.skipif` conditions and active registration assertions in `test_cn_preprocessor_residency.py`.
- **1 `bitwise-assert` remediated**: Replaced set bitwise `&` operator with `.isdisjoint()` in `test_cn_cache_decay.py`.
- **6 `parametrize-no-ids` remediated**: Added explicit `ids=[...]` to all test parameter matrices across `test_fp8_calibration_capture_resolution.py`, `test_load_model_network_error_reporting.py`, `test_param_updater_binding.py`, `test_pose_tensorrt_canvas.py`, and `test_wrapper_exception_hygiene.py`.
- **7 `overprotective` tests remediated**: Removed redundant `assert x is not None` guards before accessing `x.exists()` or `x.shape`.
- Established [.smell-baseline.json](file:///D:/dev/SDTD_040_Beta/StreamDiffusion/.smell-baseline.json) ratcheted at 24.

### Phase 1: Configuration Foundation

- Migrated test configuration into [pyproject.toml](file:///D:/dev/SDTD_040_Beta/StreamDiffusion/pyproject.toml) under `[tool.pytest.ini_options]`:
  - Enforced `--import-mode=importlib`.
  - Configured `pythonpath = ["src", "tests", "tests/unit"]`.
  - Registered strict markers (`unit`, `integration`, `slow`, `gpu`, `quality`).
  - Added filterwarnings to cleanly silence external third-party deprecations.

### Phase 2: Order Randomization & Isolation Hardening

- Integrated `pytest-randomly` into test toolchain.
- Auto-state reset via `conftest.py` ensures clean isolation between test executions.
- Verified 100% pass rate under random execution order.

### Phase 3: Coverage Ratchet & Safety Floor

- Configured `[tool.coverage.run]` and `[tool.coverage.report]` in [pyproject.toml](file:///D:/dev/SDTD_040_Beta/StreamDiffusion/pyproject.toml).
- Ratcheted `fail_under = 30` (clean baseline: 30.20%).
- Upward ratchet rule: `fail_under` only moves upward.

### Phase 3.5: Complexity & CRAP Gate

- Enforced `[tool.ruff.lint.mccabe] max-complexity = 10` in [pyproject.toml](file:///D:/dev/SDTD_040_Beta/StreamDiffusion/pyproject.toml).
- Audited CRAP scores across 983 functions in `src/streamdiffusion` with `crap4py`.
- Identified 169 functions over adoption ceiling (CRAP > 30), highlighting `wrapper.py::_load_model` (comp=270, CRAP=41420.1) and `fp8_quantize.py::capture_calibration_data` (comp=72, CRAP=5256.0) as prime candidates for future Humble Object refactoring.

---

## Verification Commands

```powershell
# 1. Fast unit loop (< 20s)
venv/Scripts/pytest.exe -m unit -q

# 2. Branch coverage with ratchet verification
venv/Scripts/pytest.exe -m unit --cov=src/streamdiffusion --cov-branch --cov-report=term-missing:skip-covered

# 3. Test smell scanner against baseline ratchet
venv/Scripts/python.exe F:\RD_PROJECTS\COMPONENTS\claude-dotfiles\skills\harden-test-suite\scripts\scan_test_smells.py tests/ --baseline .smell-baseline.json

# 4. Complexity check
venv/Scripts/ruff.exe check --select C901 src/

# 5. Full test discovery verification
venv/Scripts/pytest.exe --collect-only -q
```
