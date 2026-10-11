"""EngineManager pins the shared TRT timing cache to <engine_dir>/_timing_cache.

build_engine's path-derived fallback assumed engine_path was <root>/<prefix>/<file>, but the
UNet/VAE prefix keeps the "/" of a HF model id (engines/td/stabilityai/...), which split the
UNet cache from ControlNet's.
"""

from pathlib import Path

import pytest

pytest.importorskip("tensorrt", reason="EngineManager imports the TRT package")

from streamdiffusion.acceleration.tensorrt.engine_manager import EngineManager


def _manager(tmp_path):
    mgr = EngineManager.__new__(EngineManager)
    mgr.engine_dir = Path(tmp_path)
    return mgr


def _run(mgr, tmp_path, kwargs):
    captured = {}

    def compile_fn(model, cfg, onnx, opt_onnx, engine, **kw):
        captured.update(kw)

    mgr._execute_compilation(
        compile_fn, Path(tmp_path) / "stabilityai" / "sdxl-turbo--x" / "unet.engine", None, None, 2, kwargs
    )
    return captured


def test_timing_cache_dir_defaults_to_engine_dir_root(tmp_path):
    got = _run(_manager(tmp_path), tmp_path, {})
    assert got["engine_build_options"]["timing_cache_dir"] == str(Path(tmp_path) / "_timing_cache")
    assert got["opt_batch_size"] == 2


def test_caller_supplied_build_options_are_preserved(tmp_path):
    opts = {"builder_optimization_level": 4}
    got = _run(_manager(tmp_path), tmp_path, {"engine_build_options": opts})
    assert got["engine_build_options"]["builder_optimization_level"] == 4
    assert "timing_cache_dir" in got["engine_build_options"]
    assert "timing_cache_dir" not in opts  # caller's dict is not mutated


def test_explicit_timing_cache_dir_wins(tmp_path):
    got = _run(_manager(tmp_path), tmp_path, {"engine_build_options": {"timing_cache_dir": "custom"}})
    assert got["engine_build_options"]["timing_cache_dir"] == "custom"


# --- seeding the pinned cache from the legacy per-org location ---

NAME = "timing_trt10.cache"


def _legacy_cache(tmp_path, org="stabilityai", payload=b"warm"):
    legacy = Path(tmp_path) / org / "_timing_cache"
    legacy.mkdir(parents=True)
    (legacy / NAME).write_bytes(payload)
    return legacy / NAME


def test_seed_copies_legacy_org_cache_into_pinned_dir(tmp_path):
    from streamdiffusion.acceleration.tensorrt.utilities import _seed_timing_cache

    src = _legacy_cache(tmp_path)
    pinned = Path(tmp_path) / "_timing_cache"

    assert _seed_timing_cache(str(pinned), NAME) == str(src)
    assert (pinned / NAME).read_bytes() == b"warm"
    assert src.exists()  # copied, not moved


def test_seed_does_not_overwrite_existing_pinned_cache(tmp_path):
    from streamdiffusion.acceleration.tensorrt.utilities import _seed_timing_cache

    _legacy_cache(tmp_path)
    pinned = Path(tmp_path) / "_timing_cache"
    pinned.mkdir()
    (pinned / NAME).write_bytes(b"current")

    assert _seed_timing_cache(str(pinned), NAME) is None
    assert (pinned / NAME).read_bytes() == b"current"


def test_seed_is_a_noop_without_a_legacy_cache(tmp_path):
    from streamdiffusion.acceleration.tensorrt.utilities import _seed_timing_cache

    pinned = Path(tmp_path) / "_timing_cache"

    assert _seed_timing_cache(str(pinned), NAME) is None
    assert not pinned.exists()
