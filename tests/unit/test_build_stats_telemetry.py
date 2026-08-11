"""
Regression tests for fp8-round-13's builder.py telemetry helpers
(``_cleanup_intermediates`` sidecar preservation and ``_write_build_stats``'s
``append_global`` flag).

Split out of ``test_fp8_calib_provenance.py``: these two classes exercise
``builder.py`` only (no dependency on ``fp8_quantize.py``'s provenance-record
helpers), so they belong with this PR's cherry-pick of the ``builder.py``
build-evidence/telemetry commit rather than with the calibration-provenance
sidecar PR, whose cherry-pick does not touch ``builder.py``.

Run with: pytest tests/unit/test_build_stats_telemetry.py -v
"""

import json

from streamdiffusion.acceleration.tensorrt.builder import _cleanup_intermediates, _write_build_stats


class TestCleanupIntermediatesPreservesSidecar:
    """The regression that would silently gut this feature: `_cleanup_intermediates`
    runs from a `finally` block at the end of every build and deletes everything not
    on its `_keep_exact`/`_keep_suffixes` allowlist -- a sidecar not on that list
    would be destroyed at the end of the very build that wrote it."""

    def test_calib_data_meta_json_survives_cleanup(self, tmp_path):
        engine_dir = tmp_path / "engine"
        engine_dir.mkdir()
        (engine_dir / "calib_data.meta.json").write_text("{}")
        (engine_dir / "calib_data.npz").write_bytes(b"")
        (engine_dir / "build_stats.json").write_text("{}")
        (engine_dir / "unet.engine.onnx").write_bytes(b"")  # intermediate, must be deleted
        (engine_dir / "unet.engine.opt.onnx").write_bytes(b"")  # intermediate, must be deleted

        _cleanup_intermediates(str(engine_dir), fp8_ok=False)

        assert (engine_dir / "calib_data.meta.json").exists()
        assert (engine_dir / "calib_data.npz").exists()
        assert (engine_dir / "build_stats.json").exists()
        assert not (engine_dir / "unet.engine.onnx").exists()
        assert not (engine_dir / "unet.engine.opt.onnx").exists()


class TestWriteBuildStatsAppendGlobal:
    """fp8-round-13's mid-build flush (builder.py, right after the capture stage
    resolves) must not duplicate the build_log.jsonl line every successful build
    already gets from the end-of-build _write_build_stats call."""

    def _make_engine_path(self, tmp_path):
        engines_root = tmp_path / "engines" / "td" / "stabilityai"
        engine_dir = engines_root / "sdxl-turbo--fp8v4-mhaq--htest--res-384x640"
        engine_dir.mkdir(parents=True)
        return str(engine_dir / "unet.engine"), engines_root

    def test_append_global_false_writes_stats_but_not_jsonl(self, tmp_path):
        engine_path, engines_root = self._make_engine_path(tmp_path)
        stats = {"engine_dir": "test", "stages": {"fp8_calib_capture": {"status": "built"}}}

        _write_build_stats(engine_path, stats, append_global=False)

        stats_file = engines_root / "sdxl-turbo--fp8v4-mhaq--htest--res-384x640" / "build_stats.json"
        assert stats_file.exists()
        assert json.loads(stats_file.read_text()) == stats
        assert not (engines_root / "build_log.jsonl").exists()

    def test_append_global_true_still_writes_jsonl(self, tmp_path):
        """Default behavior (the pre-fp8-round-13 signature) is unchanged -- the
        end-of-build call relies on this."""
        engine_path, engines_root = self._make_engine_path(tmp_path)
        stats = {"engine_dir": "test", "stages": {}}

        _write_build_stats(engine_path, stats)

        jsonl_path = engines_root / "build_log.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == stats

    def test_mid_build_flush_then_final_write_appends_exactly_once(self, tmp_path):
        """Simulates the real sequence: a mid-build flush (append_global=False)
        followed by the end-of-build write (append_global=True, the default) --
        build_log.jsonl must gain exactly one line, not two."""
        engine_path, engines_root = self._make_engine_path(tmp_path)
        stats = {"engine_dir": "test", "stages": {"fp8_calib_capture": {"status": "built"}}}

        _write_build_stats(engine_path, stats, append_global=False)
        stats["total_elapsed_s"] = 123.4
        _write_build_stats(engine_path, stats)

        jsonl_path = engines_root / "build_log.jsonl"
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["total_elapsed_s"] == 123.4
