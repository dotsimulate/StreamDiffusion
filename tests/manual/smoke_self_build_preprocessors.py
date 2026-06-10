"""
Manual GPU smoke test for the self-building TRT preprocessors (HED / Scribble / NormalBae).

Purpose:
    Exercise the paths that depth_tensorrt (static, constant-res) did NOT cover:
      - SelfBuildingTRTPreprocessor._build_tensorrt_engine (FP8→FP16 fallback log,
        opt-level threading, dynamic TRT profile)
      - TensorRTEngine.allocate_buffers resolving -1 dims from input_shape
      - TensorRTEngine.infer dynamic-shape RECONCILE (384 after 512) — the alignment
        guarantee that commit 850e8eb added and depth never exercised
      - TensorRTEngine._first_output postprocess path
      - NormalBaeTensorrtPreprocessor.__new__ fallback (Finding A) on real GPU

Prerequisites:
    - Run from the repo root inside the project venv:
          python tests/manual/smoke_self_build_preprocessors.py
    - TensorRT must be installed and a CUDA GPU must be present.
    - controlnet_aux must be installed (for NormalBae).

Committed as a manual (run-only) GPU smoke test; output engines are written to a tempdir and discarded.
"""

import logging
import sys
import tempfile
from pathlib import Path

import torch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("smoke_self_build")

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Part A — HED and Scribble self-build + dual-resolution reconcile + dtype guard
# ---------------------------------------------------------------------------


def _smoke_self_build(name: str, cls, tmpdir: Path, *, build_fp8: bool) -> None:
    """Run the self-build + dual-resolution + dtype-guard assertions for one preprocessor."""
    engine_path = str(tmpdir / f"{name}.engine")
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Smoke: {name}  fp8={build_fp8}  engine={engine_path}")
    logger.info(f"{'=' * 60}")

    # --- 1. Instantiate (does NOT build yet — lazy) ---
    proc = cls(
        engine_path=engine_path,
        build_fp8=build_fp8,
        builder_optimization_level=4,
        device="cuda",
    )

    # --- 2. Access .engine → triggers _ensure_engine → _build_tensorrt_engine ---
    logger.info(f"[{name}] Triggering engine build via .engine access…")
    eng = proc.engine
    _assert(eng is not None, f"{name}: .engine returned None after build")
    _assert(Path(engine_path).exists(), f"{name}: engine file not written to disk")
    logger.info(f"{PASS} [{name}] engine built and saved: {engine_path}")

    # The FP8→FP16 one-time log is emitted by _build_tensorrt_engine internally when
    # build_fp8=True but the GPU/TRT combination does not support STRONGLY_TYPED at the
    # requested opt level.  We can't assert it here without capturing logs, but it was
    # previously validated by trt_base unit tests and is only meaningful for finding regressions
    # during live builds.

    # --- 3. process_tensor at 512 (dynamic allocate_buffers -1 resolution path) ---
    t512 = torch.rand(1, 3, 512, 512, device="cuda", dtype=torch.float16)
    out512 = proc.process_tensor(t512)
    _assert(
        tuple(out512.shape) == (3, 512, 512),
        f"{name}: 512 output shape {tuple(out512.shape)} != (3,512,512)",
    )
    logger.info(f"{PASS} [{name}] process_tensor 512→{tuple(out512.shape)}")

    # --- 4. process_tensor at 384 (dynamic reconcile — the key unexercised branch) ---
    t384 = torch.rand(1, 3, 384, 384, device="cuda", dtype=torch.float16)
    out384 = proc.process_tensor(t384)
    _assert(
        tuple(out384.shape) == (3, 384, 384),
        f"{name}: 384 output shape {tuple(out384.shape)} != (3,384,384) — dynamic reconcile broken",
    )
    logger.info(f"{PASS} [{name}] process_tensor 384→{tuple(out384.shape)} (dynamic reconcile OK)")

    # --- 5. dtype mismatch guard: float32 into a float16 engine must raise ValueError ---
    t_f32 = torch.rand(1, 3, 512, 512, device="cuda", dtype=torch.float32)
    try:
        proc.process_tensor(t_f32)
        raise AssertionError(f"{name}: float32 input did NOT raise ValueError — dtype guard missing")
    except ValueError as exc:
        _assert("dtype mismatch" in str(exc), f"{name}: ValueError does not say 'dtype mismatch': {exc}")
        logger.info(f"{PASS} [{name}] float32 input raised ValueError('dtype mismatch') as expected")


def run_hed_scribble(tmpdir: Path) -> None:
    """Run HED and Scribble smoke tests under the Performance knobs (build_fp8=True)."""
    try:
        from streamdiffusion.preprocessing.processors.hed_tensorrt import HEDTensorrtPreprocessor
        from streamdiffusion.preprocessing.processors.scribble_tensorrt import ScribbleTensorrtPreprocessor
    except ImportError as e:
        logger.warning(f"{SKIP} Could not import HED/Scribble preprocessors: {e}")
        return

    for name, cls in [("hed_tensorrt", HEDTensorrtPreprocessor), ("scribble_tensorrt", ScribbleTensorrtPreprocessor)]:
        try:
            _smoke_self_build(name, cls, tmpdir, build_fp8=True)
        except AssertionError as e:
            logger.error(f"{FAIL} [{name}] {e}")
            raise


# ---------------------------------------------------------------------------
# Part A — NormalBae __new__ fallback (Finding A) on real GPU
# ---------------------------------------------------------------------------


def run_normalbae_fallback(tmpdir: Path) -> None:
    """Verify the NormalBae fallback path initializes correctly on a real GPU."""
    try:
        import streamdiffusion.preprocessing.processors.normal_bae_tensorrt as nmod
        from streamdiffusion.preprocessing.processors.normal_bae_tensorrt import NormalBaeTensorrtPreprocessor
    except ImportError as e:
        logger.warning(f"{SKIP} Could not import NormalBae preprocessor: {e}")
        return

    try:
        from controlnet_aux import NormalBaeDetector  # noqa: F401  — presence check only
    except ImportError:
        logger.warning(f"{SKIP} controlnet_aux not installed — NormalBae fallback test skipped")
        return

    logger.info(f"\n{'=' * 60}")
    logger.info("Smoke: NormalBae fallback (__new__ Finding A) on real GPU")
    logger.info(f"{'=' * 60}")

    # Reset the probe cache so patching takes effect.
    nmod._TRT_STRATEGY_AVAILABLE = None

    # Force the fallback path: pretend ONNX export probing returned False.
    from unittest.mock import patch

    with patch.object(nmod, "_probe_normal_bae_onnx_export", return_value=False):
        with patch.object(nmod, "TENSORRT_AVAILABLE", False):
            obj = NormalBaeTensorrtPreprocessor(device="cuda", detect_resolution=512)

    # Verify the object is usable (Finding A: before the fix, it had no _detector).
    _assert(hasattr(obj, "_detector"), "fallback object missing '_detector' attribute")
    _assert(obj._detector is not None, "fallback object's '_detector' is None")
    _assert(hasattr(obj, "device"), "fallback object missing 'device' attribute")
    logger.info(f"{PASS} [normal_bae_tensorrt] fallback __new__ produces fully-initialized object")

    # Run one frame (GPU) through the fallback detector.
    t = torch.rand(1, 3, 512, 512, device="cuda")
    try:
        out = obj.process_tensor(t)
        _assert(out is not None, "fallback process_tensor returned None")
        logger.info(
            f"{PASS} [normal_bae_tensorrt] fallback process_tensor ran without AttributeError: shape={tuple(out.shape)}"
        )
    except AttributeError as exc:
        raise AssertionError(f"fallback AttributeError still present — Finding A not fixed: {exc}") from exc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not torch.cuda.is_available():
        print(f"{FAIL} No CUDA GPU available — cannot run GPU smoke tests")
        sys.exit(1)

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("TensorRT: ", end="")
    try:
        import tensorrt as trt  # noqa: F401

        logger.info(f"{trt.__version__}")
    except ImportError:
        print(f"{FAIL} TensorRT not installed")
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="smoke_self_build_") as tmpdir:
        tmp = Path(tmpdir)
        logger.info(f"Temp engine dir: {tmp}")

        # Part A — HED + Scribble
        run_hed_scribble(tmp)

        # Part A — NormalBae fallback
        run_normalbae_fallback(tmp)

    logger.info("\n" + "=" * 60)
    logger.info("All smoke assertions passed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Add repo src to path when run directly.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    main()
