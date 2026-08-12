"""
Unit tests for Fix 3 / SY-1 (CUDA implementation audit) -- image_filter.py's Step 1
read of `_skip_prob_pin`.

`_skip_prob_pin` is written by an async, non_blocking D2H copy each frame and read
back one frame later (the class's documented 1-frame delay). Nothing in
StreamDiffusion.__call__ unconditionally syncs the device before that read --
pipeline.py's only host sync is gated behind both `similar_image_filter` and a
16-frame sample cadence for an unrelated inference_time_ema timer (pipeline.py
~1430-1440), not this copy. The read is instead gated on an explicit
`torch.cuda.Event` (`_skip_evt`): a non-blocking `.query()` before trusting the pinned
buffer, falling back to `_last_skip_prob` (the last confirmed-landed value) otherwise.

These tests exercise that gate directly -- forcing `.query()` to report "not landed"
and confirming the fallback holds even though the real copy_ + record() underneath
did complete, then confirming a real post-sync `.query()` lets the fresh value
through.

CUDA-only: torch.cuda.Event / pin_memory both require a CUDA device.
"""

import pytest
import torch

from streamdiffusion.image_filter import SimilarImageFilter

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SimilarImageFilter requires CUDA (pin_memory / cuda.Event)",
)


def _frame(fill=0.0, shape=(1, 3, 8, 8), device="cuda"):
    return torch.full(shape, fill, device=device, dtype=torch.float32)


class TestSimilarImageFilterSkipEvent:
    def test_first_frame_passes_through_and_allocates_buffers(self):
        f = SimilarImageFilter(threshold=0.98)
        x = _frame(0.0)
        out = f(x)
        assert out is x
        assert f._skip_prob_pin is not None
        assert f._skip_evt is not None
        assert f._last_skip_prob == 0.0

    def test_decision_stable_and_probability_stays_in_bounds(self):
        """Feed several frames and confirm every call returns a sane decision
        (pass-through tensor or a skip signalled by None) and _last_skip_prob never
        leaves [0, 1] -- the invariant the corrected Step 1 comment documents."""
        f = SimilarImageFilter(threshold=0.98)
        f(_frame(0.0))  # frame 1: allocate
        for _ in range(5):
            out = f(_frame(0.0))  # identical frames -> high skip probability
            assert out is None or torch.is_tensor(out)
            assert 0.0 <= f._last_skip_prob <= 1.0

    def test_gate_withholds_read_until_copy_lands(self, monkeypatch):
        """Force query() to always report 'not landed' and confirm the fallback
        value is used instead of the pinned buffer, even though the real copy_ +
        record() underneath actually completed.

        Frames are kept close together (small MSE, well under the 0.02 threshold)
        so each frame's computed skip probability is genuinely non-zero -- with
        skip_prob stuck at the 0.0 fallback the whole time (gate closed), Step 3
        copies x into prev_tensor every call, so consecutive-frame MSE (not
        frame-vs-frame-1 MSE) is what lands in the pinned buffer.
        """
        f = SimilarImageFilter(threshold=0.98)
        f(_frame(0.0))  # frame 1: allocates _skip_evt

        monkeypatch.setattr(f._skip_evt, "query", lambda: False)

        f(_frame(0.05))  # frame 2: Step 1 must not trust the (real) pinned buffer
        assert f._last_skip_prob == 0.0  # unchanged fallback, not the fresh value

        f(_frame(0.09))  # frame 3: gate still closed
        assert f._last_skip_prob == 0.0

        # The real copy_ + record() calls did happen underneath -- prove the pinned
        # buffer itself is no longer the zero it started as, i.e. the gate (not an
        # absence of new data) is what withheld the read.
        torch.cuda.synchronize()
        assert f._skip_prob_pin.item() != 0.0

    def test_read_lands_once_event_actually_completes(self):
        """With the real (unpatched) event, a synchronize before the next call
        guarantees the previous copy has landed, and the gate lets it through."""
        f = SimilarImageFilter(threshold=0.98)
        f(_frame(0.0))  # frame 1: allocate
        f(_frame(0.05))  # frame 2: issues copy_ + record(); small diff -> non-zero skip prob

        torch.cuda.synchronize()
        expected = f._skip_prob_pin.item()
        assert expected != 0.0  # sanity: the value being propagated is meaningful

        f(_frame(0.09))  # frame 3: Step 1 must observe the landed copy
        assert f._last_skip_prob == pytest.approx(expected)
