"""
Regression test for the UNet TensorRT engine path exceeding Windows MAX_PATH (260).

``EngineManager.get_engine_path`` (acceleration/tensorrt/engine_manager.py) encodes every
UNet build flag into the on-disk directory name. With a realistic ``engine_dir`` and a
config using fp8 + static batch + pin_cache_frames + optlvl + resolution, the generated
directory was 246 chars; the derived ``unet.engine.onnx`` path was 263 and
``unet.engine.opt.onnx`` was 267 — both over Windows' 260-char MAX_PATH. Because the
*directory* fit but the *file* didn't, ``mkdir`` silently succeeded while
``torch.onnx.export`` -> ``open(onnx_path, "wb")`` raised ``FileNotFoundError``, which
wrapper.py's OOM-only fallback does not catch — see
"Acceleration has failed: [Errno 2] No such file or directory: ...unet.engine.onnx".

This test constructs ``EngineManager`` via ``__new__`` (bypassing ``__init__``'s compile-fn
imports, which need TensorRT/onnx/polygraphy installed) so it only exercises the pure-path
logic in ``get_engine_path``.

Run with: pytest tests/unit/test_engine_path_length.py -v
"""

from pathlib import Path

import pytest

try:
    from streamdiffusion.acceleration.tensorrt.engine_manager import EngineManager, EngineType

    IMPORT_OK = True
except ImportError:
    IMPORT_OK = False

pytestmark = pytest.mark.skipif(
    not IMPORT_OK,
    reason="acceleration.tensorrt.engine_manager not importable",
)

# Mirrors the crash report: a nested Documents path, matching the real-world depth
# that pushed the UNet engine path over MAX_PATH.
_CRASH_ENGINE_DIR = r"C:\Users\deswh\Documents\sdtd040\StreamDiffusion\engines\td"

# The exact flag combination from the crash log's UNet directory name.
_CRASH_UNET_KWARGS = {
    "engine_type": EngineType.UNET,
    "model_id_or_path": "stabilityai/sd-turbo",
    "max_batch_size": 4,
    "min_batch_size": 1,
    "mode": "img2img",
    "use_tiny_vae": True,
    "fp8": True,
    "use_cached_attn": False,
    "use_feature_injection": False,
    "build_static_batch": True,
    "static_batch_size": 2,
    "pin_cache_frames": True,
    "cache_maxframes": 4,
    "builder_optimization_level": 4,
    "resolution": (512, 512),
}


def _make_engine_manager(engine_dir: str) -> EngineManager:
    """Build an EngineManager without running __init__'s heavy compile-fn imports."""
    em = EngineManager.__new__(EngineManager)
    em.engine_dir = Path(engine_dir)
    em._configs = {EngineType.UNET: {"filename": "unet.engine"}}
    return em


class TestUnetEnginePathLength:
    def test_onnx_export_paths_stay_under_max_path(self):
        """RED today (267 > 260 for .opt.onnx); GREEN once the dir name is hashed."""
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        engine_path = em.get_engine_path(**_CRASH_UNET_KWARGS)

        onnx_path = str(engine_path) + ".onnx"
        opt_onnx_path = str(engine_path) + ".opt.onnx"

        assert len(onnx_path) < 260, f"onnx path is {len(onnx_path)} chars (MAX_PATH=260): {onnx_path}"
        assert len(opt_onnx_path) < 260, f"opt.onnx path is {len(opt_onnx_path)} chars (MAX_PATH=260): {opt_onnx_path}"

    def test_engine_path_is_deterministic(self):
        """Same config -> same path, so a previously-built engine is still found on rebuild."""
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        path_a = em.get_engine_path(**_CRASH_UNET_KWARGS)
        path_b = em.get_engine_path(**_CRASH_UNET_KWARGS)

        assert path_a == path_b

    def test_distinct_configs_do_not_collide(self):
        """A differing flag (static_batch_size) must still produce a distinct directory."""
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        kwargs_b = dict(_CRASH_UNET_KWARGS, static_batch_size=4)

        path_a = em.get_engine_path(**_CRASH_UNET_KWARGS)
        path_b = em.get_engine_path(**kwargs_b)

        assert path_a != path_b
        assert path_a.parent != path_b.parent
