"""
Unit tests for streamdiffusion.utils.diagnostics (schema v1 error-report dump).

Covers the three exported functions: collect_diagnostics(), format_report_text(),
and write_error_report(). All tests are CPU-only and do not require a live model.
GPU-dependent branches are exercised by monkeypatching torch.cuda directly (the
repo convention of patching the real imported module object) rather than gating
on hardware or adding a GPU pytest marker -- see test_safety_checker.py.
"""

import logging
from pathlib import Path

import torch

from streamdiffusion.utils import diagnostics

# ---------------------------------------------------------------------------
# format_report_text
# ---------------------------------------------------------------------------


class TestFormatReportText:
    def test_all_schema_v1_sections_present(self):
        diag = {
            "stage": "inference",
            "error": "RuntimeError: boom",
            "context_note": "streaming_loop",
            "traceback": "Traceback (most recent call last):\n  ...\nRuntimeError: boom",
            "system": {"os": "Windows-11", "gpu_name": "RTX 4090"},
            "versions": {"torch": "2.8.0", "numpy": "1.26.4"},
            "config": {"width": 512, "height": 512, "where": "streaming_loop"},
            "stream_config": {"prompt": "a photo of a cat", "num_inference_steps": 4},
            "env": {"SDTD_BASE_FOLDER_PATH": "D:/repo"},
            "log_tail": ["INFO some log line", "ERROR another line"],
        }
        text = diagnostics.format_report_text(diag)

        for section in ("SUMMARY", "TRACEBACK", "SYSTEM", "VERSIONS", "CONFIG", "STREAM CONFIG", "ENV", "LOG TAIL"):
            assert f"== {section} ==" in text

        assert "schema v1" in text
        assert "Stage: inference" in text
        assert "Error: RuntimeError: boom" in text
        assert "Context: streaming_loop" in text
        assert "RuntimeError: boom" in text  # traceback body
        assert "gpu_name: RTX 4090" in text
        assert "torch: 2.8.0" in text
        assert "width: 512" in text
        assert "prompt: a photo of a cat" in text  # stream_config, rendered via yaml.safe_dump
        assert "SDTD_BASE_FOLDER_PATH=D:/repo" in text
        assert "ERROR another line" in text

    def test_empty_sections_render_as_none_not_crash(self):
        text = diagnostics.format_report_text({})
        assert "== SUMMARY ==" in text
        assert "== STREAM CONFIG ==" in text
        assert "Error: unknown" in text
        assert "(no traceback available)" in text
        assert "(none)" in text  # empty SYSTEM/VERSIONS/CONFIG/STREAM CONFIG/ENV/LOG TAIL sections


# ---------------------------------------------------------------------------
# collect_diagnostics
# ---------------------------------------------------------------------------


class TestCollectDiagnostics:
    def test_returns_all_four_sections(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        diag = diagnostics.collect_diagnostics()
        assert set(diag.keys()) == {"system", "versions", "config", "env"}
        assert isinstance(diag["system"], dict)
        assert isinstance(diag["versions"], dict)
        assert isinstance(diag["config"], dict)
        assert isinstance(diag["env"], dict)

    def test_no_cuda_device_falls_back_gracefully(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        diag = diagnostics.collect_diagnostics()
        assert diag["system"]["gpu_name"] == "no CUDA device available"

    def test_cuda_device_reports_name_and_vram(self, monkeypatch):
        class _FakeProps:
            name = "Fake GPU"
            major = 8
            minor = 9

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: _FakeProps())
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda idx: (1_000_000_000, 2_000_000_000))

        diag = diagnostics.collect_diagnostics()

        assert diag["system"]["gpu_name"] == "Fake GPU"
        assert diag["system"]["compute_capability"] == "8.9"
        assert diag["system"]["vram_total_mb"] == 2_000_000_000 // (1024 * 1024)
        assert diag["system"]["vram_free_mb"] == 1_000_000_000 // (1024 * 1024)

    def test_cuda_device_index_resolved_from_wrapper_device_string(self, monkeypatch):
        """wrapper.device is a plain str (e.g. "cuda:1"), not a torch.device -- the resolved
        index must come from torch.device(wrapper.device).index, not str.index (a bound
        method that getattr(wrapper.device, "index", None) would otherwise return)."""

        class _FakeProps:
            name = "Fake GPU 1"
            major = 8
            minor = 9

        class _FakeWrapper:
            device = "cuda:1"

        seen_indices = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda idx: (seen_indices.append(idx), _FakeProps())[1]
        )
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda idx: (seen_indices.append(idx), (1_000_000_000, 2_000_000_000))[1]
        )

        diag = diagnostics.collect_diagnostics(wrapper=_FakeWrapper())

        assert seen_indices == [1, 1]
        assert diag["system"]["gpu_name"] == "Fake GPU 1"

    def test_cuda_device_index_falls_back_when_wrapper_device_is_cpu(self, monkeypatch):
        class _FakeWrapper:
            device = "cpu"

        seen_indices = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda idx: seen_indices.append(idx) or type("P", (), {"name": "GPU0", "major": 8, "minor": 9})(),
        )
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda idx: seen_indices.append(idx) or (1_000_000_000, 2_000_000_000)
        )

        diagnostics.collect_diagnostics(wrapper=_FakeWrapper())

        assert seen_indices == [0, 0]

    def test_torch_failure_is_caught_not_raised(self, monkeypatch):
        def _boom():
            raise RuntimeError("no driver")

        monkeypatch.setattr(torch.cuda, "is_available", _boom)
        diag = diagnostics.collect_diagnostics()
        assert "torch_error" in diag["system"]

    def test_wrapper_config_attrs_collected(self):
        class _FakeWrapper:
            width = 512
            height = 512
            batch_size = 1
            fp8 = False
            static_shapes = True
            _acceleration = "tensorrt"
            _engine_dir = "engines/"
            use_controlnet = False
            use_ipadapter = False

        diag = diagnostics.collect_diagnostics(wrapper=_FakeWrapper())
        assert diag["config"]["width"] == 512
        assert diag["config"]["acceleration"] == "tensorrt"  # leading underscore stripped
        assert diag["config"]["engine_dir"] == "engines/"

    def test_extra_merged_into_config(self):
        diag = diagnostics.collect_diagnostics(extra={"where": "streaming_loop"})
        assert diag["config"]["where"] == "streaming_loop"

    def test_env_allowlist_only(self, monkeypatch):
        # CUDALINK_LIB_PATH is retired (cuda_link resolution no longer uses an env var --
        # see cuda_link_bootstrap.py's layered lookup); CUDALINK_DOORBELL is still a live,
        # installer-persisted var, so it exercises the same CUDALINK_ prefix allowlist path.
        monkeypatch.setenv("CUDALINK_DOORBELL", "1")
        monkeypatch.setenv("HF_HOME", "C:/hf")
        monkeypatch.setenv("SECRET_TOKEN", "should-not-appear")
        diag = diagnostics.collect_diagnostics()
        assert diag["env"].get("CUDALINK_DOORBELL") == "1"
        assert diag["env"].get("HF_HOME") == "C:/hf"
        assert "SECRET_TOKEN" not in diag["env"]

    def test_env_denylist_blocks_secret_named_vars_even_if_prefix_allowlisted(self, monkeypatch):
        """A prefix match alone isn't enough -- HF_TOKEN matches the HF_ allowlist prefix but
        must never be dumped, since its name also matches the TOKEN denylist substring."""
        monkeypatch.setenv("HF_TOKEN", "should-not-appear")
        monkeypatch.setenv("HF_HOME", "C:/hf")
        monkeypatch.setenv("SDTD_BASE_FOLDER_PATH", "D:/repo")
        diag = diagnostics.collect_diagnostics()
        assert "HF_TOKEN" not in diag["env"]
        assert diag["env"].get("HF_HOME") == "C:/hf"
        assert diag["env"].get("SDTD_BASE_FOLDER_PATH") == "D:/repo"

    def test_env_denylist_blocks_auth_and_session_vars(self, monkeypatch):
        """Denylist substrings extend beyond TOKEN/KEY/SECRET to AUTH/SESSION/COOKIE, since a
        var like HF_AUTH or SD_SESSION would otherwise pass the prefix allowlist unredacted."""
        monkeypatch.setenv("HF_AUTH", "should-not-appear")
        monkeypatch.setenv("SD_SESSION", "should-not-appear")
        monkeypatch.setenv("SD_COOKIE", "should-not-appear")
        monkeypatch.setenv("HF_HOME", "C:/hf")
        diag = diagnostics.collect_diagnostics()
        assert "HF_AUTH" not in diag["env"]
        assert "SD_SESSION" not in diag["env"]
        assert "SD_COOKIE" not in diag["env"]
        assert diag["env"].get("HF_HOME") == "C:/hf"


# ---------------------------------------------------------------------------
# write_error_report
# ---------------------------------------------------------------------------


class TestWriteErrorReport:
    def test_writes_txt_with_traceback_and_all_sections(self, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        def _raise():
            raise ValueError("synthetic failure")

        try:
            _raise()
        except ValueError as exc:
            report_path = diagnostics.write_error_report(exc, stage="inference", out_dir=tmp_path)

        assert report_path is not None
        assert report_path.exists()
        assert report_path.parent == tmp_path
        assert report_path.name.startswith("inference_error_report_")

        text = report_path.read_text(encoding="utf-8")
        for section in ("SUMMARY", "TRACEBACK", "SYSTEM", "VERSIONS", "CONFIG", "STREAM CONFIG", "ENV", "LOG TAIL"):
            assert f"== {section} ==" in text
        assert "ValueError: synthetic failure" in text
        assert "_raise" in text  # traceback frame reference
        assert "Stage: inference" in text

    def test_creates_out_dir_if_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        target = tmp_path / "nested" / "error_reports"
        report_path = diagnostics.write_error_report(RuntimeError("boom"), stage="inference", out_dir=target)
        assert report_path is not None
        assert target.exists()

    def test_respects_sdtd_base_folder_path_env_var(self, tmp_path, monkeypatch):
        """SDTD_BASE_FOLDER_PATH is pinned at install time (setx, mirroring CUDALINK_*) so the
        out-of-process TD Python can locate error_reports/ without a manual env-var step."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setenv("SDTD_BASE_FOLDER_PATH", str(tmp_path))
        report_path = diagnostics.write_error_report(RuntimeError("boom"), stage="inference")
        assert report_path is not None
        assert report_path.parent == tmp_path / "error_reports"

    def test_falls_back_to_module_relative_repo_root_when_env_var_unset(self, tmp_path, monkeypatch):
        """Without SDTD_BASE_FOLDER_PATH set and no explicit out_dir, resolution falls back to
        this module's own __file__-relative repo root (reliable under the editable install).
        Fakes __file__ under tmp_path so this doesn't litter the real repo root."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.delenv("SDTD_BASE_FOLDER_PATH", raising=False)
        fake_file = tmp_path / "src" / "streamdiffusion" / "utils" / "diagnostics.py"
        monkeypatch.setattr(diagnostics, "__file__", str(fake_file))
        report_path = diagnostics.write_error_report(RuntimeError("boom"), stage="inference")
        assert report_path is not None
        assert report_path.parent == tmp_path / "error_reports"

    def test_stream_config_dumped_when_provided(self, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        report_path = diagnostics.write_error_report(
            RuntimeError("boom"),
            stage="inference",
            out_dir=tmp_path,
            config={"prompt": "a photo of a cat", "num_inference_steps": 4},
        )
        text = report_path.read_text(encoding="utf-8")
        assert "== STREAM CONFIG ==" in text
        assert "prompt: a photo of a cat" in text
        assert "num_inference_steps: 4" in text

    def test_stream_config_secrets_redacted(self, tmp_path, monkeypatch):
        """config is caller-supplied and arbitrary (unlike the wrapper-attr `config` section,
        which only ever holds a fixed allowlist of attrs) -- nested secret-looking keys must
        be masked before the report is written, including inside nested dicts/lists."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        report_path = diagnostics.write_error_report(
            RuntimeError("boom"),
            stage="inference",
            out_dir=tmp_path,
            config={
                "prompt": "a photo of a cat",
                "hf_token": "hf_secretvalue",
                "connection": {"api_key": "sk-secretvalue", "endpoint": "https://example.com"},
                "extra_models": [{"password": "hunter2"}, {"name": "model-a"}],
            },
        )
        text = report_path.read_text(encoding="utf-8")
        assert "prompt: a photo of a cat" in text
        assert "https://example.com" in text
        assert "model-a" in text
        assert "hf_secretvalue" not in text
        assert "sk-secretvalue" not in text
        assert "hunter2" not in text
        assert "***REDACTED***" in text

    def test_back_to_back_calls_do_not_overwrite_each_other(self, tmp_path, monkeypatch):
        """Filenames use microsecond precision, not just seconds -- two reports written in
        quick succession (e.g. a retried streaming-loop failure) must land at distinct paths
        rather than the second truncating/overwriting the first via write_text()."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        path_a = diagnostics.write_error_report(RuntimeError("first"), stage="inference", out_dir=tmp_path)
        path_b = diagnostics.write_error_report(RuntimeError("second"), stage="inference", out_dir=tmp_path)
        assert path_a is not None and path_b is not None
        assert path_a != path_b
        assert path_a.exists() and path_b.exists()
        assert "RuntimeError: first" in path_a.read_text(encoding="utf-8")
        assert "RuntimeError: second" in path_b.read_text(encoding="utf-8")

    def test_stream_config_absent_renders_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        report_path = diagnostics.write_error_report(RuntimeError("boom"), stage="inference", out_dir=tmp_path)
        text = report_path.read_text(encoding="utf-8")
        assert "== STREAM CONFIG ==\n(none)" in text

    def test_never_raises_when_write_fails(self, monkeypatch):
        """Best-effort contract: even if report generation itself fails partway
        through, write_error_report must swallow the error and return None
        rather than propagating -- a reporting bug must never mask the real error."""

        def _boom(*args, **kwargs):
            raise RuntimeError("format failure")

        monkeypatch.setattr(diagnostics, "format_report_text", _boom)
        result = diagnostics.write_error_report(RuntimeError("boom"), stage="inference", out_dir="C:/wherever")
        assert result is None

    def test_context_where_surfaces_as_summary_context(self, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        report_path = diagnostics.write_error_report(
            RuntimeError("boom"), stage="inference", context={"where": "streaming_loop"}, out_dir=tmp_path
        )
        text = report_path.read_text(encoding="utf-8")
        assert "Context: streaming_loop" in text
        assert "where: streaming_loop" in text  # also present in CONFIG via the `extra` merge

    def test_context_secrets_redacted_in_config(self, tmp_path, monkeypatch):
        """context is merged into CONFIG via the same `extra` path as `where` -- a secret-looking
        key there must be redacted just like the stream_config path, since nothing prevents a
        future caller from passing richer context than just `where`."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        report_path = diagnostics.write_error_report(
            RuntimeError("boom"),
            stage="inference",
            context={"where": "streaming_loop", "api_key": "sk-secretvalue"},
            out_dir=tmp_path,
        )
        text = report_path.read_text(encoding="utf-8")
        assert "where: streaming_loop" in text
        assert "sk-secretvalue" not in text
        assert "***REDACTED***" in text

    def test_filename_reflects_stage(self, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        report_path = diagnostics.write_error_report(RuntimeError("boom"), stage="startup", out_dir=tmp_path)
        assert report_path.name.startswith("startup_error_report_")


# ---------------------------------------------------------------------------
# log tail buffer
# ---------------------------------------------------------------------------


class TestLogTailBuffer:
    def test_recent_log_records_are_captured(self):
        logger = logging.getLogger("streamdiffusion.utils.diagnostics.test")
        logger.setLevel(logging.INFO)
        logger.error("distinctive log tail marker 12345")
        tail = diagnostics._get_log_tail()
        assert any("distinctive log tail marker 12345" in line for line in tail)


# ---------------------------------------------------------------------------
# package re-exports
# ---------------------------------------------------------------------------


def test_reexported_from_utils_package():
    from streamdiffusion.utils import collect_diagnostics, format_report_text, write_error_report

    assert collect_diagnostics is diagnostics.collect_diagnostics
    assert format_report_text is diagnostics.format_report_text
    assert write_error_report is diagnostics.write_error_report


# ---------------------------------------------------------------------------
# StreamDiffusionWrapper.write_error_report delegation
#
# Calls the unbound method against a bare object rather than constructing a real
# StreamDiffusionWrapper (whose __init__ loads a model/GPU pipeline) -- this only
# needs to verify the wrapper=self binding and argument forwarding, not wrapper
# construction, which is out of scope for a CPU-only diagnostics test.
# ---------------------------------------------------------------------------


def test_wrapper_write_error_report_forwards_self_and_args(monkeypatch):
    from streamdiffusion import wrapper as wrapper_module

    captured = {}

    def _fake_util(exc, *, stage, context=None, wrapper=None, config=None, out_dir=None):
        captured.update(exc=exc, stage=stage, context=context, wrapper=wrapper, config=config, out_dir=out_dir)
        return Path("fake_report.txt")

    monkeypatch.setattr(wrapper_module, "_write_error_report_util", _fake_util)

    fake_self = object()
    exc = RuntimeError("boom")
    context = {"where": "streaming_loop"}
    config = {"prompt": "a photo of a cat"}

    result = wrapper_module.StreamDiffusionWrapper.write_error_report(
        fake_self, exc, context=context, config=config, out_dir="C:/reports"
    )

    assert result == Path("fake_report.txt")
    assert captured["exc"] is exc
    assert captured["stage"] == "inference"
    assert captured["context"] is context
    assert captured["wrapper"] is fake_self
    assert captured["config"] is config
    assert captured["out_dir"] == "C:/reports"


def test_wrapper_write_error_report_returns_none_on_util_failure(monkeypatch):
    from streamdiffusion import wrapper as wrapper_module

    monkeypatch.setattr(wrapper_module, "_write_error_report_util", lambda *a, **k: None)

    result = wrapper_module.StreamDiffusionWrapper.write_error_report(object(), RuntimeError("boom"))

    assert result is None
