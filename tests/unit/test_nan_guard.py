"""Unit tests for streamdiffusion.utils.nan_guard.NanGuard.

CPU-only: NanGuard's CPU branch checks-and-sanitizes inline (no CUDA event
plumbing needed or possible), so these tests exercise the sanitize semantics
and warn-once logging directly and deterministically. A separate CUDA-path
smoke test below (skipped if no GPU) exercises the pinned-scalar +
Event.query() deferred-readback branch.
"""

import logging

import pytest
import torch

from streamdiffusion.utils.nan_guard import NanGuard


class TestNanGuardCPU:
    def test_clean_tensor_untouched_and_reports_not_bad(self):
        guard = NanGuard("test.clean")
        t = torch.tensor([1.0, -2.0, 3.5])
        was_bad = guard.sanitize_(t)
        assert not was_bad
        assert torch.equal(t, torch.tensor([1.0, -2.0, 3.5]))

    def test_nan_replaced_with_zero_in_place(self):
        guard = NanGuard("test.nan")
        t = torch.tensor([1.0, float("nan"), 3.0])
        data_ptr_before = t.data_ptr()
        was_bad = guard.sanitize_(t)
        assert was_bad
        assert t.data_ptr() == data_ptr_before, "sanitize_ must mutate in place, not reallocate"
        assert torch.equal(t, torch.tensor([1.0, 0.0, 3.0]))

    def test_inf_and_neg_inf_replaced_with_zero(self):
        guard = NanGuard("test.inf")
        t = torch.tensor([float("inf"), float("-inf"), 2.0])
        was_bad = guard.sanitize_(t)
        assert was_bad
        assert torch.equal(t, torch.tensor([0.0, 0.0, 2.0]))
        assert torch.isfinite(t).all()

    def test_all_zero_division_nan_is_caught(self):
        """Directly mirrors the reported bug's failure mode: 0/0 -> NaN."""
        guard = NanGuard("test.zero_div")
        w = torch.tensor([0.0, 0.0])
        poisoned = w / w.sum()
        assert not torch.isfinite(poisoned).all()
        was_bad = guard.sanitize_(poisoned)
        assert was_bad
        assert torch.isfinite(poisoned).all()

    def test_warns_once_then_debug(self, caplog):
        guard = NanGuard("test.warn_once")
        with caplog.at_level(logging.DEBUG, logger="streamdiffusion.utils.nan_guard"):
            guard.sanitize_(torch.tensor([float("nan")]))
            guard.sanitize_(torch.tensor([float("nan")]))
            guard.sanitize_(torch.tensor([float("nan")]))

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(warnings) == 1, "exactly one WARNING across repeated bad frames"
        assert len(debugs) == 2, "subsequent bad frames log at DEBUG, not WARNING"
        assert "test.warn_once" in warnings[0].message, "warning must name the guarded buffer"

    def test_no_log_when_never_bad(self, caplog):
        guard = NanGuard("test.silent")
        with caplog.at_level(logging.DEBUG, logger="streamdiffusion.utils.nan_guard"):
            for _ in range(5):
                guard.sanitize_(torch.tensor([1.0, 2.0]))
        assert not caplog.records


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for the deferred-readback path")
class TestNanGuardCUDA:
    def test_deferred_detection_across_frames(self):
        """The bad verdict for a CUDA tensor lags by one call, exactly like
        SimilarImageFilter.skip_prob -- verify it eventually reflects a bad
        frame without ever requiring a blocking sync from the caller."""
        guard = NanGuard("test.cuda")
        clean = torch.tensor([1.0, 2.0], device="cuda")
        poisoned = torch.tensor([float("nan"), 1.0], device="cuda")

        # First call: always sanitizes; nothing to report yet (no prior frame).
        first_bad = guard.sanitize_(clean.clone())
        assert not first_bad

        # Second call is poisoned; sanitized immediately regardless of the
        # returned (still-previous-frame) verdict.
        t = poisoned.clone()
        guard.sanitize_(t)
        torch.cuda.synchronize()  # test-only: force the async copy to land before we poll it
        assert torch.isfinite(t).all()

        # Poll subsequent clean calls until the event has landed and the bad
        # verdict surfaces -- bounded loop, no fixed sleep.
        saw_bad = False
        for _ in range(50):
            bad = guard.sanitize_(clean.clone())
            if bad:
                saw_bad = True
                break
        assert saw_bad, "the poisoned frame's verdict never surfaced on a later call"
