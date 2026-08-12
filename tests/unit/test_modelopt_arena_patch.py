"""
Regression test for the FP8 quantization VRAM-spill fix
(``streamdiffusion._patches.modelopt_arena_patch``).

nvidia-modelopt's ONNX static calibration hardcodes ``arena_extend_strategy=
"kSameAsRequested"`` on CPU/CUDA providers, which makes ORT's CUDA BFC arena degenerate (see
the module docstring in ``modelopt_arena_patch.py`` for the full mechanism, including why an
earlier round's attempt to fix a *second*, related bug — a discarded arena-shrinkage
``RunOptions`` — turned out to be unfixable: ORT rejects the shrink list outright on this
stack). The fix replaces one function in modelopt's ``ort_patching`` module for the duration
of one quantize call.

This is a copied third-party internal, not a call through a public API — the riskiest kind of
patch in this codebase — so what actually needs testing is the *patch mechanism itself*:
does it replace the right thing, is it idempotent, does revert() restore the original exactly,
and — the part that matters most — does a version mismatch or a missing/renamed attribute
produce a loud warning and *no* changes, rather than a silent no-op or a half-apply. None of
this requires modelopt's real CUDA calibration path, a GPU, or the 5+ GB UNet ONNX: a minimal
fake ``modelopt.onnx.quantization.ort_patching`` module, built the same way
``test_faceid_bgr_patch.py`` fakes ``diffusers_ipadapter``, is enough to exercise it.

Run with: pytest tests/unit/test_modelopt_arena_patch.py -v
"""

import logging
import sys
import types

import pytest


def _install_fake_modelopt(version: str = "0.43.0"):
    """Build a minimal fake ``modelopt.onnx.quantization.ort_patching`` module tree and
    register it in sys.modules, so ``modelopt_arena_patch`` can ``import`` it normally.

    Unlike ``test_faceid_bgr_patch.py``'s fixture, the parent->child attributes
    (``modelopt.onnx``, ``onnx.quantization``, ``quantization.ort_patching``) are wired
    explicitly rather than relying on ``sys.modules`` alone: a bare ``from modelopt.onnx.
    quantization import ort_patching`` against modules that were never loaded through a real
    ``import`` statement can raise ``AttributeError`` inside CPython's fromlist handling,
    because sys.modules being pre-populated short-circuits the step that normally sets each
    submodule as an attribute of its parent package.

    ``_collect_data_minmax_calibrator`` is included as an *unpatched* attribute — the patch
    no longer touches it (see the module docstring: that fix turned out to be unfixable, ORT
    rejects the arena shrink list outright on this stack) — so tests can assert it is left
    alone, which is what makes ``test_missing_attribute_warns_and_skips_without_half_apply``
    meaningful with only one real patch target.
    """
    modelopt_mod = types.ModuleType("modelopt")
    modelopt_mod.__version__ = version
    onnx_mod = types.ModuleType("modelopt.onnx")
    quantization_mod = types.ModuleType("modelopt.onnx.quantization")
    ort_patching_mod = types.ModuleType("modelopt.onnx.quantization.ort_patching")

    def _orig_collect(*args, **kwargs):
        return "orig_collect_data_minmax_calibrator"

    def _orig_create_session(*args, **kwargs):
        return "orig_create_inference_session_with_ep_config"

    ort_patching_mod._collect_data_minmax_calibrator = _orig_collect
    ort_patching_mod._create_inference_session_with_ep_config = _orig_create_session

    modelopt_mod.onnx = onnx_mod
    onnx_mod.quantization = quantization_mod
    quantization_mod.ort_patching = ort_patching_mod

    sys.modules["modelopt"] = modelopt_mod
    sys.modules["modelopt.onnx"] = onnx_mod
    sys.modules["modelopt.onnx.quantization"] = quantization_mod
    sys.modules["modelopt.onnx.quantization.ort_patching"] = ort_patching_mod

    return modelopt_mod, ort_patching_mod, (_orig_collect, _orig_create_session)


@pytest.fixture
def fake_modelopt(monkeypatch):
    """Install a fresh fake modelopt tree for the duration of one test and always clean up
    sys.modules afterward, so the real (or a different test's fake) modelopt is unaffected.
    """
    _MODELOPT_MODULE_NAMES = (
        "modelopt",
        "modelopt.onnx",
        "modelopt.onnx.quantization",
        "modelopt.onnx.quantization.ort_patching",
    )
    for name in _MODELOPT_MODULE_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)

    def _make(version: str = "0.43.0"):
        return _install_fake_modelopt(version)

    yield _make

    for name in _MODELOPT_MODULE_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.fixture(autouse=True)
def _reset_arena_patch_state():
    """Safety net: modelopt_arena_patch's apply/revert state is module-global. If a test
    fails mid-assert without reaching its own revert() call, force it back to unpatched so
    later tests never start from unexpected state.
    """
    yield
    from streamdiffusion._patches import modelopt_arena_patch as arena_patch

    arena_patch._PATCHED = False
    arena_patch._ORIGINALS.clear()


class TestModeloptArenaPatch:
    def test_apply_replaces_the_attribute_on_matching_version(self, fake_modelopt):
        from streamdiffusion._patches import modelopt_arena_patch as arena_patch

        _modelopt_mod, ort_patching_mod, (orig_collect, orig_create) = fake_modelopt()
        assert not arena_patch._PATCHED

        arena_patch.apply()

        assert arena_patch._PATCHED
        assert (
            ort_patching_mod._create_inference_session_with_ep_config
            is arena_patch._fixed_create_inference_session_with_ep_config
        )
        assert ort_patching_mod._create_inference_session_with_ep_config is not orig_create
        # Not a patch target — must be left exactly as installed.
        assert ort_patching_mod._collect_data_minmax_calibrator is orig_collect

        arena_patch.revert()

    def test_apply_is_idempotent(self, fake_modelopt):
        from streamdiffusion._patches import modelopt_arena_patch as arena_patch

        _modelopt_mod, ort_patching_mod, _originals = fake_modelopt()

        arena_patch.apply()
        once_create = ort_patching_mod._create_inference_session_with_ep_config

        arena_patch.apply()  # second call must be a no-op, not a double-wrap
        twice_create = ort_patching_mod._create_inference_session_with_ep_config

        assert once_create is twice_create

        arena_patch.revert()

    def test_revert_restores_originals_exactly(self, fake_modelopt):
        from streamdiffusion._patches import modelopt_arena_patch as arena_patch

        _modelopt_mod, ort_patching_mod, (orig_collect, orig_create) = fake_modelopt()

        arena_patch.apply()
        arena_patch.revert()

        assert ort_patching_mod._create_inference_session_with_ep_config is orig_create
        assert not arena_patch._PATCHED

    def test_revert_without_apply_is_a_no_op(self, fake_modelopt):
        from streamdiffusion._patches import modelopt_arena_patch as arena_patch

        _modelopt_mod, ort_patching_mod, (orig_collect, orig_create) = fake_modelopt()

        assert not arena_patch._PATCHED
        arena_patch.revert()  # must not raise, must not touch anything
        assert ort_patching_mod._create_inference_session_with_ep_config is orig_create

    def test_mismatched_version_warns_and_skips(self, fake_modelopt, caplog):
        from streamdiffusion._patches import modelopt_arena_patch as arena_patch

        _modelopt_mod, ort_patching_mod, (orig_collect, orig_create) = fake_modelopt(version="0.99.0")

        with caplog.at_level(logging.WARNING):
            arena_patch.apply()

        assert not arena_patch._PATCHED
        assert ort_patching_mod._collect_data_minmax_calibrator is orig_collect
        assert ort_patching_mod._create_inference_session_with_ep_config is orig_create
        assert "not one of the versions" in caplog.text

    def test_missing_attribute_warns_and_skips_without_half_apply(self, fake_modelopt, caplog):
        """A renamed/removed target attribute must skip the patch entirely — no half-apply,
        ever. ``_collect_data_minmax_calibrator`` (present but not a patch target) must be
        left alone too, proving apply() didn't touch anything else on the module.
        """
        from streamdiffusion._patches import modelopt_arena_patch as arena_patch

        _modelopt_mod, ort_patching_mod, (orig_collect, orig_create) = fake_modelopt()
        del ort_patching_mod._create_inference_session_with_ep_config

        with caplog.at_level(logging.WARNING):
            arena_patch.apply()

        assert not arena_patch._PATCHED
        # The attribute that DOES still exist on the fake module must be untouched too.
        assert ort_patching_mod._collect_data_minmax_calibrator is orig_collect
        assert not hasattr(ort_patching_mod, "_create_inference_session_with_ep_config")
        assert "not found on modelopt" in caplog.text

    def test_fixed_function_sets_group_qdq_tensors(self, monkeypatch):
        """Regression test for a real bug: an earlier version of the patched copy stopped
        right after building ``calibrator.infer_session`` and silently dropped upstream's
        trailing ``calibrator.group_qdq_tensors = kwargs.get(...)`` assignment (see
        ort_patching.py:350-353). Every ``MinMaxCalibrater`` then reached ``compute_data()``
        without the attribute at all, raising ``AttributeError`` three minutes into a real
        calibration run -- far too late for the fake-module tests above (which only check
        *which* attribute gets swapped, never what the replacement body actually does) to
        catch it. This test calls the replacement directly and asserts the attribute lands,
        both when modelopt passes a truthy ``group_qdq_tensors`` kwarg and when it doesn't.
        """
        from streamdiffusion._patches import modelopt_arena_patch as arena_patch

        class _FakeCalibrator:
            pass

        class _FakeSessionOptions:
            def __init__(self):
                self.graph_optimization_level = None
                self.enable_cpu_mem_arena = None

            def add_session_config_entry(self, *args, **kwargs):
                pass

        class _FakeInferenceSession:
            def __init__(self, *args, **kwargs):
                pass

        fake_ort = types.SimpleNamespace(
            SessionOptions=_FakeSessionOptions,
            GraphOptimizationLevel=types.SimpleNamespace(ORT_DISABLE_ALL=0),
            InferenceSession=_FakeInferenceSession,
            get_available_providers=lambda: [],
        )
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

        # Wire the modelopt/modelopt.onnx parent packages explicitly, not just the leaf
        # module -- see _install_fake_modelopt's docstring above for why pre-populating
        # sys.modules alone (without setting each submodule as an attribute of its parent)
        # can make CPython's fromlist handling raise AttributeError instead of resolving.
        fake_modelopt_pkg = types.ModuleType("modelopt")
        fake_modelopt_onnx_pkg = types.ModuleType("modelopt.onnx")
        fake_logging_config = types.ModuleType("modelopt.onnx.logging_config")
        fake_logging_config.logger = logging.getLogger("fake_modelopt_logger")
        fake_modelopt_pkg.onnx = fake_modelopt_onnx_pkg
        fake_modelopt_onnx_pkg.logging_config = fake_logging_config
        monkeypatch.setitem(sys.modules, "modelopt", fake_modelopt_pkg)
        monkeypatch.setitem(sys.modules, "modelopt.onnx", fake_modelopt_onnx_pkg)
        monkeypatch.setitem(sys.modules, "modelopt.onnx.logging_config", fake_logging_config)

        # kwarg present and truthy -> attribute must be set to that value.
        calibrator_with = _FakeCalibrator()
        arena_patch._fixed_create_inference_session_with_ep_config(
            calibrator_with,
            model_path="fake.onnx",
            execution_providers=[],
            group_qdq_tensors={"a": ["b"]},
        )
        assert calibrator_with.group_qdq_tensors == {"a": ["b"]}

        # kwarg absent -> attribute must still exist, as None (the common case: fp8.py only
        # includes group_qdq_tensors in trt_guided_options when it's truthy).
        calibrator_without = _FakeCalibrator()
        arena_patch._fixed_create_inference_session_with_ep_config(
            calibrator_without,
            model_path="fake.onnx",
            execution_providers=[],
        )
        assert hasattr(calibrator_without, "group_qdq_tensors")
        assert calibrator_without.group_qdq_tensors is None

    def test_modelopt_not_importable_skips_quietly(self, monkeypatch):
        """No modelopt installed at all (e.g. FP8 extras not present) must not raise —
        quantize_onnx_fp8's own ImportError guard runs first in practice, but apply() must
        be safe standalone too.
        """
        monkeypatch.delitem(sys.modules, "modelopt", raising=False)
        monkeypatch.setitem(sys.modules, "modelopt", None)  # forces ImportError on `import modelopt`

        from streamdiffusion._patches import modelopt_arena_patch as arena_patch

        arena_patch.apply()  # must not raise
        assert not arena_patch._PATCHED
