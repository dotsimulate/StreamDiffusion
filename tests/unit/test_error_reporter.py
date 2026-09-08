"""
Unit tests for streamdiffusion.utils.diagnostics.ErrorReporter.

Covers the dedup-aware `report()` wrapper around write_error_report() + report_error():
writing/returning a path, same-signature dedup, different-`where` distinctness, the
max_reports cap, the never-raises contract when write_error_report itself blows up, and
thread-safety under concurrent calls from several threads. All tests are CPU-only, mirroring
test_diagnostics.py's convention of monkeypatching torch.cuda directly rather than requiring
real hardware, and use the same SDTD_BASE_FOLDER_PATH env-var pattern to redirect writes into
tmp_path (ErrorReporter.report() does not accept its own out_dir -- it forwards straight to
write_error_report()'s default resolution).
"""

import threading
from pathlib import Path

import torch

from streamdiffusion.utils import diagnostics


class TestErrorReporterReport:
    def test_writes_report_and_returns_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setenv("SDTD_BASE_FOLDER_PATH", str(tmp_path))
        reporter = diagnostics.ErrorReporter()

        report_path = reporter.report(RuntimeError("boom"), stage="inference", where="streaming_loop")

        assert report_path is not None
        assert report_path.exists()
        assert report_path.parent == tmp_path / "error_reports"
        assert report_path.name.startswith("inference_error_report_")
        text = report_path.read_text(encoding="utf-8")
        assert "where: streaming_loop" in text
        assert "Context: streaming_loop" in text

    def test_same_where_and_exception_deduped_to_one_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setenv("SDTD_BASE_FOLDER_PATH", str(tmp_path))
        reporter = diagnostics.ErrorReporter()

        first = reporter.report(RuntimeError("boom"), stage="inference", where="streaming_loop")
        second = reporter.report(RuntimeError("boom"), stage="inference", where="streaming_loop")

        assert first is not None
        assert second is None
        assert len(list((tmp_path / "error_reports").glob("*.txt"))) == 1

    def test_different_where_produces_two_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setenv("SDTD_BASE_FOLDER_PATH", str(tmp_path))
        reporter = diagnostics.ErrorReporter()

        first = reporter.report(RuntimeError("boom"), stage="inference", where="streaming_loop")
        second = reporter.report(RuntimeError("boom"), stage="inference", where="get_input_frame")

        assert first is not None
        assert second is not None
        assert first != second
        assert len(list((tmp_path / "error_reports").glob("*.txt"))) == 2

    def test_different_message_same_where_not_deduped(self, tmp_path, monkeypatch):
        """Signature is (where, exception type, truncated message) -- a differently-worded
        exception at the same call site must still get its own report."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setenv("SDTD_BASE_FOLDER_PATH", str(tmp_path))
        reporter = diagnostics.ErrorReporter()

        first = reporter.report(RuntimeError("boom"), stage="inference", where="streaming_loop")
        second = reporter.report(RuntimeError("a different failure"), stage="inference", where="streaming_loop")

        assert first is not None
        assert second is not None
        assert first != second

    def test_max_reports_cap_honoured(self, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setenv("SDTD_BASE_FOLDER_PATH", str(tmp_path))
        reporter = diagnostics.ErrorReporter(max_reports=2)

        results = [reporter.report(RuntimeError(f"boom {i}"), stage="inference", where=f"where_{i}") for i in range(5)]

        assert all(r is not None for r in results[:2])
        assert all(r is None for r in results[2:])
        assert len(list((tmp_path / "error_reports").glob("*.txt"))) == 2

    def test_returns_none_without_raising_when_write_error_report_fails(self, monkeypatch):
        """Best-effort contract: report() must swallow a failure inside write_error_report
        (already best-effort itself, but report() adds its own belt-and-braces try/except
        around the whole dedup+write+log sequence) and return None rather than propagating."""

        def _boom(*args, **kwargs):
            raise RuntimeError("write failure")

        monkeypatch.setattr(diagnostics, "write_error_report", _boom)
        reporter = diagnostics.ErrorReporter()

        result = reporter.report(RuntimeError("boom"), stage="inference", where="streaming_loop")

        assert result is None

    def test_forwards_wrapper_and_config_to_write_error_report(self, monkeypatch):
        captured = {}

        def _fake_write(exc, *, stage, wrapper=None, config=None, context=None, out_dir=None):
            captured.update(exc=exc, stage=stage, wrapper=wrapper, config=config, context=context)
            return Path("fake_report.txt")

        monkeypatch.setattr(diagnostics, "write_error_report", _fake_write)
        reporter = diagnostics.ErrorReporter()

        fake_wrapper = object()
        fake_config = {"width": 512}
        result = reporter.report(
            RuntimeError("boom"),
            stage="inference",
            where="streaming_loop",
            wrapper=fake_wrapper,
            config=fake_config,
            context={"frame": 42},
        )

        assert result == Path("fake_report.txt")
        assert captured["exc"] is not None
        assert captured["stage"] == "inference"
        assert captured["wrapper"] is fake_wrapper
        assert captured["config"] is fake_config
        assert captured["context"] == {"where": "streaming_loop", "frame": 42}

    def test_concurrent_calls_from_several_threads_produce_no_duplicate(self, tmp_path, monkeypatch):
        """The same (where, exc) signature fired from many threads at once must still write
        exactly one report -- the seen-signature check-and-set happens under `self._lock` as
        a single critical section, not as two separate unlocked steps that could race."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setenv("SDTD_BASE_FOLDER_PATH", str(tmp_path))
        reporter = diagnostics.ErrorReporter()
        results: list = []
        results_lock = threading.Lock()

        def _worker():
            result = reporter.report(RuntimeError("boom"), stage="inference", where="streaming_loop")
            with results_lock:
                results.append(result)

        threads = [threading.Thread(target=_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successes = [r for r in results if r is not None]
        assert len(successes) == 1
        assert len(list((tmp_path / "error_reports").glob("*.txt"))) == 1
