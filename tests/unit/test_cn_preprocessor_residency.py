"""
GPU-Residency guard for ControlNet-coupled preprocessors.

Verification test for plan cozy-booping-wilkinson.md, Verification step 1:
"new test imports the resolver, iterates every CN-coupled type, resolves the
preprocessor name, instantiates via get_preprocessor, and asserts gpu_native
is True. Fails today for hed/scribble/normal; passes after the port."

Run with: pytest tests/unit/test_cn_preprocessor_residency.py -v
"""

import pytest


# ---------------------------------------------------------------------------
# CN-coupled type → expected preprocessor mappings
# (matches the plan table and the updated CN_MODEL_REGISTRY 'preprocessor' fields)
# ---------------------------------------------------------------------------

CN_COUPLED_PREPROCESSORS = [
    # (cn_type_label, preprocessor_name, expect_gpu_native)
    ("canny", "canny", True),
    ("soft_edge", "soft_edge", True),
    ("lineart", "standard_lineart", True),
    ("tile", "feedback", True),
    ("color", "passthrough", True),
    ("depth", "depth_tensorrt", True),
    ("openpose", "pose_tensorrt", True),
    ("hed", "hed_tensorrt", True),
    ("scribble", "scribble_tensorrt", True),
    ("normalbae", "normal_bae_tensorrt", True),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_gpu_native(preprocessor_name: str) -> bool:
    """
    Return True if the named preprocessor class has gpu_native = True.
    Returns False if the class is not registered or has gpu_native = False.
    """
    from streamdiffusion.preprocessing.processors import get_preprocessor_class

    try:
        cls = get_preprocessor_class(preprocessor_name)
        return getattr(cls, "gpu_native", False)
    except (ValueError, Exception):
        return False


def _is_registered(preprocessor_name: str) -> bool:
    from streamdiffusion.preprocessing.processors import list_preprocessors

    return preprocessor_name in list_preprocessors()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cn_type,preprocessor_name,should_be_gpu_native", CN_COUPLED_PREPROCESSORS)
def test_cn_preprocessor_is_registered(cn_type, preprocessor_name, should_be_gpu_native):
    """Every CN-coupled preprocessor must be registered (no dangling names)."""
    assert _is_registered(preprocessor_name), (
        f"Preprocessor '{preprocessor_name}' for CN type '{cn_type}' is NOT registered. "
        "This would crash get_preprocessor_class with 'Unknown preprocessor'."
    )


@pytest.mark.parametrize("cn_type,preprocessor_name,should_be_gpu_native", CN_COUPLED_PREPROCESSORS)
def test_cn_preprocessor_gpu_native_flag(cn_type, preprocessor_name, should_be_gpu_native):
    """Every CN-coupled preprocessor class must declare gpu_native = True."""
    if not _is_registered(preprocessor_name):
        pytest.skip(f"'{preprocessor_name}' not registered (see test above)")

    actual = _check_gpu_native(preprocessor_name)
    assert actual == should_be_gpu_native, (
        f"Preprocessor '{preprocessor_name}' (for CN type '{cn_type}') "
        f"has gpu_native={actual!r}, expected {should_be_gpu_native!r}.\n"
        "If gpu_native=True, the class runs _process_tensor_core on GPU with no PIL round-trip.\n"
        "If you see False, either:\n"
        "  (a) the class still uses the base-class PIL fallback — override _process_tensor_core\n"
        "  (b) you forgot to set gpu_native=True on the class"
    )


class TestAutopreprocessResolver:
    """Tests for D9 — registry-driven + heuristic + passthrough fallback."""

    def test_registry_lookup_sd15_canny(self):
        """Exact registry match returns the 'preprocessor' field directly."""
        # Inline import mirrors model_utils__td.py location in the Scripts directory.
        import importlib.util
        import os

        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Scripts")
        spec = importlib.util.spec_from_file_location(
            "model_utils_td",
            os.path.join(scripts_dir, "StreamDiffusionTD__Text__model_utils__td.py"),
        )
        if spec is None or spec.loader is None:
            pytest.skip("model_utils__td.py not found — run from repo root")

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # SD1.5 canny → depth_tensorrt is NOT the fallback; canny is correct
        result = mod.get_preprocessor_for_controlnet("lllyasviel/control_v11p_sd15_canny", "Local")
        assert result == "canny", f"Expected 'canny', got '{result}'"

    def test_registry_lookup_sd15_normalbae(self):
        import importlib.util
        import os

        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Scripts")
        spec = importlib.util.spec_from_file_location(
            "model_utils_td",
            os.path.join(scripts_dir, "StreamDiffusionTD__Text__model_utils__td.py"),
        )
        if spec is None or spec.loader is None:
            pytest.skip("model_utils__td.py not found")

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        result = mod.get_preprocessor_for_controlnet("lllyasviel/control_v11p_sd15_normalbae", "Local")
        assert result == "normal_bae_tensorrt", (
            f"Expected 'normal_bae_tensorrt', got '{result}' — dangling 'normal_bae' reference not fixed"
        )

    def test_fallback_to_passthrough_for_unknown_model(self):
        import importlib.util
        import os

        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Scripts")
        spec = importlib.util.spec_from_file_location(
            "model_utils_td",
            os.path.join(scripts_dir, "StreamDiffusionTD__Text__model_utils__td.py"),
        )
        if spec is None or spec.loader is None:
            pytest.skip("model_utils__td.py not found")

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Completely unknown ID — should fall through to 'passthrough', not crash
        result = mod.get_preprocessor_for_controlnet("some-custom/totally-unknown-controlnet-v1", "Local")
        assert result == "passthrough", f"Expected 'passthrough' fallback for unknown model, got '{result}'"
