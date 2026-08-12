"""
Regression test for the l2tc `dynamic_shapes` fix (Phase 5, Commit B —
docs/perf_bestpractices_audit_2026-07-10.md follow-up).

Root cause: `build_engine()` passed `dynamic_shapes=build_dynamic_shape` into
`Engine.build()`, which tracks only dynamic *resolution*. Batch-dynamic /
resolution-static engines (e.g. the default "Flexible" UNet preset,
`build_static_batch=False, build_dynamic_shape=False`) got `dynamic_shapes=False`,
so `_apply_gpu_profile_to_config`'s tiling branch ran against a graph that still
has a symbolic batch dim, and TRT emitted a benign but noisy
"[l2tc] VALIDATE FAIL - Graph contains symbolic shape" warning for every
applicable layer.

Fix: `dynamic_shapes=build_dynamic_shape or not build_static_batch` — True
whenever *any* dim (resolution or batch) is symbolic, matching
`_apply_gpu_profile_to_config`'s actual requirement (l2tc needs ALL dims
concrete). Fully-static engines (e.g. ControlNet: `build_static_batch=True,
build_dynamic_shape=False`) are unaffected.

FP8 Round 11: the expression is actually three terms —
`build_dynamic_shape or not build_static_batch or getattr(model_data,
"has_symbolic_cache_dims", False)` — extracted into `_compute_dynamic_shapes`
(utilities.py). The third term covers KVO/FI cache-frames axis models
(`use_cached_attn`) whose exported ONNX still has symbolic dims even when
`build_static_batch=True` and `build_dynamic_shape=False`, which every current
UNet build actually is. That term previously had zero test coverage here even
though it decides tiling for those builds; the two-term description above is
kept for the original bug's history, not as the current full behavior — see
the third-term tests below.

These tests exercise only `build_engine`'s pure-Python wiring: GPU detection,
the memory query, and the real `Engine.build()` TRT call are all monkeypatched
out, so no CUDA device or TensorRT context is required.
"""

from unittest.mock import MagicMock

import pytest

try:
    from streamdiffusion.acceleration.tensorrt import utilities as trt_utilities

    IMPORT_OK = True
except ImportError:
    IMPORT_OK = False

pytestmark = pytest.mark.skipif(
    not IMPORT_OK,
    reason="acceleration.tensorrt.utilities not importable (TensorRT/onnx/polygraphy missing)",
)


class _FakeModelData:
    """Stand-in for a BaseModel subclass — build_engine only calls get_input_profile().
    Deliberately does NOT define has_symbolic_cache_dims, so _compute_dynamic_shapes's
    getattr(..., False) default is what most of these tests exercise for that term."""

    def get_input_profile(self, *args, **kwargs):
        return {}


class _FakeModelDataWithSymbolicCache(_FakeModelData):
    """Stand-in for a use_cached_attn model whose exported ONNX keeps a symbolic
    cache-frames axis (has_symbolic_cache_dims=True) — the third term."""

    has_symbolic_cache_dims = True


def _computed_dynamic_shapes(monkeypatch, *, build_static_batch, build_dynamic_shape, model_data=None):
    """Call build_engine() with GPU detection / memory query / the real TRT build
    all monkeypatched out, and return the `dynamic_shapes` kwarg it actually
    passed to Engine.build()."""
    monkeypatch.setattr(trt_utilities.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(trt_utilities, "detect_gpu_profile", lambda device: trt_utilities._fallback_profile())
    monkeypatch.setattr(trt_utilities.cudart, "cudaMemGetInfo", lambda: (0, 8 * 2**30, 16 * 2**30))

    captured_build = MagicMock(return_value=None)
    monkeypatch.setattr(trt_utilities.Engine, "build", captured_build)

    trt_utilities.build_engine(
        engine_path="fake.engine",
        onnx_opt_path="fake.onnx",
        model_data=model_data if model_data is not None else _FakeModelData(),
        opt_image_height=512,
        opt_image_width=512,
        opt_batch_size=1,
        build_static_batch=build_static_batch,
        build_dynamic_shape=build_dynamic_shape,
    )

    return captured_build.call_args.kwargs["dynamic_shapes"]


class TestL2tcDynamicShapesFix:
    def test_batch_dynamic_resolution_static_is_dynamic(self, monkeypatch):
        """Default 'Flexible' UNet preset: static resolution, dynamic batch —
        this is the bug case; must compute True (previously computed False)."""
        result = _computed_dynamic_shapes(monkeypatch, build_static_batch=False, build_dynamic_shape=False)
        assert result is True

    def test_fully_static_engine_is_not_dynamic(self, monkeypatch):
        """ControlNet-style fully-static engine: unaffected, still computes False."""
        result = _computed_dynamic_shapes(monkeypatch, build_static_batch=True, build_dynamic_shape=False)
        assert result is False

    def test_resolution_dynamic_is_dynamic_regardless_of_batch(self, monkeypatch):
        """build_dynamic_shape=True alone was already sufficient before this fix."""
        result = _computed_dynamic_shapes(monkeypatch, build_static_batch=True, build_dynamic_shape=True)
        assert result is True

    def test_symbolic_cache_dims_is_dynamic_even_when_fully_static(self, monkeypatch):
        """FP8 Round 11: has_symbolic_cache_dims=True must force dynamic_shapes=True
        on its own, even though build_static_batch=True and build_dynamic_shape=False
        together compute False (see test_fully_static_engine_is_not_dynamic above).
        This is the case every current pin_cache_frames UNet build actually hits."""
        result = _computed_dynamic_shapes(
            monkeypatch,
            build_static_batch=True,
            build_dynamic_shape=False,
            model_data=_FakeModelDataWithSymbolicCache(),
        )
        assert result is True

    def test_missing_has_symbolic_cache_dims_defaults_to_false_contribution(self, monkeypatch):
        """_FakeModelData (no has_symbolic_cache_dims attribute at all) must not
        crash and must not itself force dynamic_shapes=True — getattr(...,
        False) is the required default, not attribute access, per
        tests/unit/test_l2tc_dynamic_shapes.py's own _FakeModelData contract."""
        result = _computed_dynamic_shapes(
            monkeypatch,
            build_static_batch=True,
            build_dynamic_shape=False,
            model_data=_FakeModelData(),
        )
        assert result is False
