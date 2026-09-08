"""
Regression tests for fp8-round-13's calibration provenance record.

Before this round, a cached ``calib_data.npz`` hit
(``builder.py``'s ``if os.path.exists(_calib_data_path):`` short-circuit) logged only
the path -- nothing recorded how the npz was actually captured (resolution, prompt
count, which calibration calls survived selection). Worse, a run that crashed
mid-capture -- after ``capture_calibration_data`` finished writing the npz but before
``builder.py`` reached the end-of-build ``_write_build_stats`` call -- left literally
zero trace of having run at all: a correct, complete artifact with zero provenance
(the fp8-round-10 ~07:14 aborted capture, found only by the npz's own mtime predating
the engine it fed by roughly an hour).

The fix is metadata-only (fp8_quantize.py's ``_build_calib_provenance`` /
``_write_calib_provenance_sidecar`` / ``load_calib_provenance``, plus builder.py
echoing the sidecar into ``build_stats.json`` on both the cached and built paths and
flushing stats right after the capture stage resolves) -- it does not change what gets
calibrated, so it never forks the ``calv*`` cache tag.

Run with: pytest tests/unit/test_fp8_calib_provenance.py -v
"""

import json

import numpy as np

from streamdiffusion.acceleration.tensorrt.builder import (
    _cleanup_intermediates,
    _find_best_sibling_mha_ratio,
    _write_build_stats,
)
from streamdiffusion.acceleration.tensorrt.fp8_quantize import (
    _build_calib_provenance,
    _write_calib_provenance_sidecar,
    load_calib_provenance,
)


def _fake_calib_data():
    return {
        "sample": np.zeros((8, 4, 64, 64), dtype=np.float16),
        "timestep": np.array([919, 779, 639, 559, 499, 359, 219, 79], dtype=np.float32),
    }


class TestBuildCalibProvenance:
    """`_build_calib_provenance` is a pure function -- no disk I/O -- so these
    exercise its payload shape directly."""

    def test_payload_contains_expected_keys(self):
        provenance = _build_calib_provenance(
            _fake_calib_data(),
            image_height=384,
            image_width=640,
            prompt_count=32,
            num_inference_steps=4,
            guidance_scale=1.0,
            ipadapter_tokens_real=True,
            selected_calls=np.array([3, 1, 7, 0]),
            pool_missed_calls={5, 2},
        )
        expected_keys = {
            "schema_version",
            "captured_at_utc",
            "image_height",
            "image_width",
            "prompt_count",
            "num_inference_steps",
            "guidance_scale",
            "ipadapter_tokens_real",
            "captured_timesteps",
            "selected_timesteps",
            "selected_calls",
            "pool_missed_calls",
            "row_count",
        }
        assert expected_keys <= set(provenance.keys())
        assert provenance["image_height"] == 384
        assert provenance["image_width"] == 640
        assert provenance["row_count"] == 8
        # selected_calls/pool_missed_calls are sorted for a deterministic, diffable file.
        assert provenance["selected_calls"] == [0, 1, 3, 7]
        assert provenance["pool_missed_calls"] == [2, 5]
        assert provenance["captured_timesteps"] == sorted(provenance["captured_timesteps"], reverse=True)

    def test_selected_calls_none_does_not_raise(self):
        """`_selected_calls` is only assigned inside capture_calibration_data's
        onnx_path/(use_cached_attn or use_controlnet or num_ip_layers) branch --
        a plain UNet capture with none of those set never runs it, so the caller
        passes None here. Must not raise (e.g. sorting a None)."""
        provenance = _build_calib_provenance(
            _fake_calib_data(),
            image_height=512,
            image_width=512,
            selected_calls=None,
            pool_missed_calls=None,
        )
        assert provenance["selected_calls"] is None
        assert provenance["pool_missed_calls"] == []

    def test_round_trips_through_json(self):
        """Every value must be JSON-serializable as-is (no bare numpy scalars/arrays
        leaking through) -- this is what _write_calib_provenance_sidecar's
        json.dump would otherwise choke on."""
        provenance = _build_calib_provenance(
            _fake_calib_data(),
            image_height=384,
            image_width=640,
            selected_calls=np.array([2, 0]),
            pool_missed_calls={1},
        )
        round_tripped = json.loads(json.dumps(provenance))
        assert round_tripped == provenance

    def test_schema_v2_recorder_fields_present(self):
        """FP8 Round 14 / schema v2: the K/V/FI recorder's own state (what was
        predicted, what was actually kept, the caps it was bounded by, and the
        resulting kvo pool length) must ride along in the sidecar so two builds
        carrying an identical calv8 tag but different env-overridden recorder
        caps (SDTD_FP8_CALIB_RECORD_CALLS/_BYTES, not part of the cache key)
        have something on disk to tell them apart."""
        provenance = _build_calib_provenance(
            _fake_calib_data(),
            image_height=384,
            image_width=640,
            selected_calls=np.array([0, 9, 18, 27, 36, 45, 54, 63]),
            pool_missed_calls=set(),
            predicted_calls=np.array([0, 9, 18, 27, 36, 45, 54, 63]),
            record_calls_kept=8,
            record_bytes=1_073_741_824,
            record_max_calls=64,
            record_max_bytes=4 * 1024**3,
            kvo_pool_len=8,
        )

        assert provenance["schema_version"] == 2
        assert provenance["predicted_calls"] == [0, 9, 18, 27, 36, 45, 54, 63]
        assert provenance["record_calls_kept"] == 8
        assert provenance["record_bytes"] == 1_073_741_824
        assert provenance["record_max_calls"] == 64
        assert provenance["record_max_bytes"] == 4 * 1024**3
        assert provenance["kvo_pool_len"] == 8

    def test_schema_v2_recorder_fields_default_empty_or_none(self):
        """The ControlNet capture path (capture_calibration_data_controlnet) has
        no K/V recorder at all, so its _build_calib_provenance call site omits
        every v2 recorder kwarg -- must default to None/[] rather than raise or
        require the caller to pass placeholders."""
        provenance = _build_calib_provenance(
            _fake_calib_data(),
            image_height=512,
            image_width=512,
        )

        assert provenance["predicted_calls"] == []
        assert provenance["record_calls_kept"] is None
        assert provenance["record_bytes"] is None
        assert provenance["record_max_calls"] is None
        assert provenance["record_max_bytes"] is None
        assert provenance["kvo_pool_len"] is None


class TestWriteCalibProvenanceSidecar:
    def test_write_then_load_round_trips(self, tmp_path):
        npz_path = str(tmp_path / "calib_data.npz")
        # capture_calibration_data always writes the npz itself before calling the
        # sidecar writer -- the sidecar naming (with_name(stem + ".meta.json")) does
        # not depend on the npz actually existing, but write it anyway for realism.
        np.savez(npz_path, sample=np.zeros((1,)))
        provenance = _build_calib_provenance(_fake_calib_data(), image_height=512, image_width=512)

        _write_calib_provenance_sidecar(npz_path, provenance)

        meta_path = tmp_path / "calib_data.meta.json"
        assert meta_path.exists()
        loaded = load_calib_provenance(npz_path)
        assert loaded == provenance

    def test_missing_sidecar_returns_none_not_error(self, tmp_path):
        """Every pre-fp8-round-13 engine dir has a calib_data.npz but no sidecar --
        this must degrade to None, not raise, so builder.py's cached-branch can
        fall back to the "provenance unavailable" message."""
        npz_path = str(tmp_path / "calib_data.npz")
        np.savez(npz_path, sample=np.zeros((1,)))
        assert load_calib_provenance(npz_path) is None

    def test_v1_sidecar_loads_without_raising(self, tmp_path):
        """FP8 Round 14 bumped schema_version 1 -> 2 and added six recorder
        fields. The 640x384 verification engine (hfb5ce64fea8d) already carries
        a real v1 sidecar on disk -- load_calib_provenance reads the file as
        plain JSON with no schema validation, so a v1 payload missing every new
        key must still load, not raise or silently drop to None."""
        npz_path = str(tmp_path / "calib_data.npz")
        np.savez(npz_path, sample=np.zeros((1,)))
        v1_payload = {
            "schema_version": 1,
            "captured_at_utc": "2026-08-11T13:51:53.480065+00:00",
            "image_height": 384,
            "image_width": 640,
            "prompt_count": 32,
            "num_inference_steps": 8,
            "guidance_scale": 0.0,
            "ipadapter_tokens_real": True,
            "captured_timesteps": [919.0, 779.0, 699.0, 639.0, 499.0, 359.0, 219.0, 79.0],
            "selected_timesteps": [919.0, 779.0, 699.0, 639.0, 499.0, 359.0, 219.0, 79.0],
            "selected_calls": [0, 9, 18, 27, 36, 45, 54, 63],
            "pool_missed_calls": [36, 45, 54, 63],
            "row_count": 8,
            # deliberately no predicted_calls/record_calls_kept/record_bytes/
            # record_max_calls/record_max_bytes/kvo_pool_len -- those are v2-only.
        }
        meta_path = tmp_path / "calib_data.meta.json"
        meta_path.write_text(json.dumps(v1_payload), encoding="utf-8")

        loaded = load_calib_provenance(npz_path)

        assert loaded == v1_payload
        assert loaded["schema_version"] == 1
        assert "predicted_calls" not in loaded

    def test_write_failure_is_swallowed(self, tmp_path):
        """A metadata write must never abort a capture that just spent up to
        ~47 minutes producing the npz it describes -- point save_path at a
        directory that does not exist so the write fails, and assert no
        exception propagates."""
        bad_path = str(tmp_path / "does_not_exist" / "calib_data.npz")
        provenance = _build_calib_provenance(_fake_calib_data(), image_height=512, image_width=512)

        _write_calib_provenance_sidecar(bad_path, provenance)  # must not raise

        assert not (tmp_path / "does_not_exist").exists()


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


class TestFindBestSiblingMhaRatio:
    """Regression tests for the 2026-09-06 MHA-fusion-regression investigation.

    The original `_find_best_sibling_mha_ratio` returned the historical *maximum*
    ratio and the gate warned when a build fell *below* it -- backwards, since lower
    mha_kernels_per_attn_block means more complete fusion (see the function's own
    docstring for the full empirical argument: every sibling engine's fused-MHA layer
    names normalize to the single kernel pattern `_gemm_mha_v2`, so a higher ratio
    means the same attention sites needed *more* kernels each, not that more sites got
    fused). `test_returns_lowest_ratio_not_highest` is the test that would have caught
    that bug directly. The all-time-extremum baseline was also unbounded (`_ratio >
    best` could never clear once a single anomalous build existed anywhere in
    history), fixed here by the `window` parameter.
    """

    def _make_current_and_root(self, tmp_path):
        """Mirrors the real call site: builder.py passes the *current* build's own
        (not-yet-written) engine dir -- the function only ever reads `.parent`."""
        engines_root = tmp_path / "engines" / "td" / "stabilityai"
        engines_root.mkdir(parents=True)
        current_engine_dir = engines_root / "sdxl-turbo--current--res-384x640"
        return str(current_engine_dir), engines_root

    def _write_sibling(self, engines_root, name, stats):
        sibling_dir = engines_root / name
        sibling_dir.mkdir(parents=True)
        (sibling_dir / "build_stats.json").write_text(json.dumps(stats))

    def test_none_when_no_sibling_dirs(self, tmp_path):
        current_engine_dir, _engines_root = self._make_current_and_root(tmp_path)
        assert _find_best_sibling_mha_ratio(current_engine_dir, "fp16") is None

    def test_none_when_siblings_have_no_stats_file(self, tmp_path):
        current_engine_dir, engines_root = self._make_current_and_root(tmp_path)
        (engines_root / "sibling-no-stats").mkdir(parents=True)
        assert _find_best_sibling_mha_ratio(current_engine_dir, "fp16") is None

    def test_skips_sibling_missing_precision_key(self, tmp_path):
        current_engine_dir, engines_root = self._make_current_and_root(tmp_path)
        self._write_sibling(engines_root, "vae", {"mha_kernels_per_attn_block": 2.0})
        assert _find_best_sibling_mha_ratio(current_engine_dir, "fp16") is None

    def test_skips_sibling_with_different_precision(self, tmp_path):
        current_engine_dir, engines_root = self._make_current_and_root(tmp_path)
        self._write_sibling(engines_root, "fp8-sibling", {"precision": "fp8", "mha_kernels_per_attn_block": 1.0})
        assert _find_best_sibling_mha_ratio(current_engine_dir, "fp16") is None

    def test_skips_sibling_missing_ratio_key(self, tmp_path):
        current_engine_dir, engines_root = self._make_current_and_root(tmp_path)
        self._write_sibling(engines_root, "pre-round11", {"precision": "fp16"})
        assert _find_best_sibling_mha_ratio(current_engine_dir, "fp16") is None

    def test_returns_lowest_ratio_not_highest(self, tmp_path):
        """The polarity regression test: a fully-fused sibling (2.0, one kernel per
        attention module) and a worse-fused sibling (3.0, some modules needing two
        kernels) coexist -- "best" must be the 2.0, not the 3.0."""
        current_engine_dir, engines_root = self._make_current_and_root(tmp_path)
        self._write_sibling(
            engines_root,
            "well-fused",
            {"precision": "fp16", "mha_kernels_per_attn_block": 2.0, "build_end": "2026-08-16T00:00:00+00:00"},
        )
        self._write_sibling(
            engines_root,
            "worse-fused-outlier",
            {"precision": "fp16", "mha_kernels_per_attn_block": 3.0, "build_end": "2026-08-22T00:00:00+00:00"},
        )

        best = _find_best_sibling_mha_ratio(current_engine_dir, "fp16")

        assert best == 2.0

    def test_window_ages_out_old_outlier(self, tmp_path):
        """With window=2 and three siblings ordered oldest -> newest (outlier, then
        two well-fused builds), only the two most recent are considered -- the
        all-time-worst outlier must not pin the gate once enough newer builds exist."""
        current_engine_dir, engines_root = self._make_current_and_root(tmp_path)
        self._write_sibling(
            engines_root,
            "oldest-outlier",
            {"precision": "fp16", "mha_kernels_per_attn_block": 3.0, "build_end": "2026-08-22T00:00:00+00:00"},
        )
        self._write_sibling(
            engines_root,
            "recent-a",
            {"precision": "fp16", "mha_kernels_per_attn_block": 2.5, "build_end": "2026-09-04T00:00:00+00:00"},
        )
        self._write_sibling(
            engines_root,
            "recent-b",
            {"precision": "fp16", "mha_kernels_per_attn_block": 2.0, "build_end": "2026-09-06T00:00:00+00:00"},
        )

        best_windowed = _find_best_sibling_mha_ratio(current_engine_dir, "fp16", window=2)
        best_unwindowed = _find_best_sibling_mha_ratio(current_engine_dir, "fp16", window=3)

        assert best_windowed == 2.0  # min(2.5, 2.0) -- the 3.0 outlier aged out
        assert best_unwindowed == 2.0  # min(3.0, 2.5, 2.0) -- still 2.0, but for a different reason

    def test_falls_back_to_build_start_when_build_end_absent(self, tmp_path):
        """A build that crashed before reaching the end-of-build write only has
        `build_start` (fp8-round-13's mid-build flush) -- ordering must still work."""
        current_engine_dir, engines_root = self._make_current_and_root(tmp_path)
        self._write_sibling(
            engines_root,
            "crashed-mid-build",
            {"precision": "fp16", "mha_kernels_per_attn_block": 3.0, "build_start": "2026-08-22T00:00:00+00:00"},
        )
        self._write_sibling(
            engines_root,
            "completed",
            {"precision": "fp16", "mha_kernels_per_attn_block": 2.0, "build_start": "2026-09-06T00:00:00+00:00"},
        )

        best = _find_best_sibling_mha_ratio(current_engine_dir, "fp16", window=1)

        assert best == 2.0  # only "completed" (the newer build_start) is in the window
