"""
Regression tests for the slot-latch K/V + FI cache write scheme in
``StreamDiffusion.update_kvo_cache`` (pipeline.py).

Previously the cache write was skipped entirely unless
``frame_idx % cache_interval == 0``, freezing the whole bank between writes: its
age oscillated (newest entry cycled 1..interval calls old) and the entire slot
set shifted by ``cache_interval`` at once — visible V2V judder at FPS/interval,
scaling with the interval. The fix writes on EVERY unet_step call while the
write pointer ``(frame_idx // cache_interval - 1) % cache_maxframes`` still
advances only every ``cache_interval`` calls: the latched slot always holds the
one-call-old output (smooth EA/FI anchor) and the remaining slots freeze at
~interval spacing, preserving the long style-consistency window.

Pinned properties:
  - the just-written output is present in the bank after every call;
  - the write pointer advances exactly every ``cache_interval`` calls;
  - ``cache_interval=1`` keeps the original full round-robin (bit-identical
    slot schedule to the pre-fix code, where the early return never fired);
  - writes stay within ``cache_maxframes`` when the buffer is allocated larger
    (pin_cache_frames ceiling case, see wrapper.py);
  - the fio cache is written to the same slot as kvo on every call.

Cache ranks differ (create_kvo_cache / create_fi_cache,
acceleration/tensorrt/models/utils.py) and per-call engine outputs carry an
extra leading axis that update_kvo_cache squeezes away:
  - kvo_cache layer: (2, maxframes, batch, seq, hidden); kvo_cache_out layer:
    (2, 1, batch, seq, hidden) -> squeeze(1).
  - fio_cache layer: (maxframes, batch, seq, hidden); fio_cache_out layer:
    (1, batch, seq, hidden) -> squeeze(0).

CPU-only, model-free: StreamDiffusion built via ``__new__`` with only the
attributes update_kvo_cache reads (same convention as
test_unet_call_backend_gate.py), exercising the bucket-free fallback path
(``_kvo_buckets = None``).

Run with: pytest tests/unit/test_kvo_cache_slot_latch.py -v
"""

from itertools import pairwise

import torch

from streamdiffusion.pipeline import StreamDiffusion

BATCH = 2
SEQ_LEN = 4
HIDDEN_DIM = 8
N_LAYERS = 2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_stream(cache_interval, cache_maxframes, alloc_maxframes=None, sentinel=-1.0):
    """StreamDiffusion shell with just what update_kvo_cache reads. Buffers are
    filled with ``sentinel`` so never-written slots are detectable."""
    alloc = alloc_maxframes if alloc_maxframes is not None else cache_maxframes
    sd = StreamDiffusion.__new__(StreamDiffusion)
    sd.frame_idx = 0
    sd.cache_interval = cache_interval
    sd.cache_maxframes = cache_maxframes
    sd._kvo_buckets = None  # force the per-layer fallback write path
    sd.kvo_cache = [torch.full((2, alloc, BATCH, SEQ_LEN, HIDDEN_DIM), sentinel) for _ in range(N_LAYERS)]
    sd.fio_cache = [torch.full((alloc, BATCH, SEQ_LEN, HIDDEN_DIM), sentinel) for _ in range(N_LAYERS)]
    return sd


def _call(sd, value):
    """One update_kvo_cache call with all outputs filled with ``value`` (the
    call index), so bank contents encode which call each slot holds."""
    kvo_out = [torch.full((2, 1, BATCH, SEQ_LEN, HIDDEN_DIM), float(value)) for _ in range(N_LAYERS)]
    fio_out = [torch.full((1, BATCH, SEQ_LEN, HIDDEN_DIM), float(value)) for _ in range(N_LAYERS)]
    sd.update_kvo_cache(kvo_out, fio_out)


def _slot_values(layer_5d):
    """Per-slot scalar content of a kvo layer (slots are constant-filled here)."""
    return [layer_5d[0, s, 0, 0, 0].item() for s in range(layer_5d.shape[1])]


def _fio_slot_values(layer_4d):
    return [layer_4d[s, 0, 0, 0].item() for s in range(layer_4d.shape[0])]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestAnchorFreshness:
    def test_just_written_call_present_after_every_call(self):
        """The newest output must land in the bank on every call — the smooth
        anchor property the fix exists for (was: only every interval-th call)."""
        for interval in (1, 2, 3):
            sd = _make_stream(cache_interval=interval, cache_maxframes=4)
            for f in range(1, 13):
                _call(sd, f)
                for layer in sd.kvo_cache:
                    assert float(f) in _slot_values(layer), (
                        f"interval={interval}: call {f} missing from kvo bank "
                        f"{_slot_values(layer)} — write was skipped"
                    )
                for layer in sd.fio_cache:
                    assert float(f) in _fio_slot_values(layer), f"interval={interval}: call {f} missing from fio bank"


class TestPointerLatch:
    def test_pointer_advances_exactly_every_interval_calls(self):
        interval, maxframes, calls = 3, 4, 24
        sd = _make_stream(cache_interval=interval, cache_maxframes=maxframes)
        written_slots = []
        for f in range(1, calls + 1):
            before = [_slot_values(sd.kvo_cache[0])]
            _call(sd, f)
            after = [_slot_values(sd.kvo_cache[0])]
            changed = [s for s in range(maxframes) if before[0][s] != after[0][s]]
            assert len(changed) == 1, f"call {f}: expected exactly 1 slot write, got {changed}"
            written_slots.append(changed[0])

        # The pointer moves between call f and f+1 iff f+1 crosses a multiple of
        # interval (frame_idx // interval increments there); otherwise the slot
        # latches. Asserting per-boundary pins both the advance cadence and the
        # latch-window lengths at once.
        for i, (a, b) in enumerate(pairwise(written_slots)):
            f_next = i + 2  # call number that produced b
            if f_next % interval == 0:
                assert a != b, (
                    f"slot sequence {written_slots}: pointer failed to advance "
                    f"at call {f_next} (multiple of interval={interval})"
                )
            else:
                assert a == b, (
                    f"slot sequence {written_slots}: pointer moved at call "
                    f"{f_next}, expected latch (interval={interval})"
                )
        # Over enough calls every slot in the window gets rotated through.
        assert set(written_slots) == set(range(maxframes))

    def test_interval_1_keeps_full_round_robin(self):
        """interval=1 must reproduce the pre-fix schedule exactly: one write per
        call, pointer advancing every call, wrapping over cache_maxframes."""
        maxframes = 4
        sd = _make_stream(cache_interval=1, cache_maxframes=maxframes)
        for f in range(1, 2 * maxframes + 1):
            _call(sd, f)
            expected_slot = (f - 1) % maxframes  # (frame_idx // 1 - 1) % maxframes
            assert _slot_values(sd.kvo_cache[0])[expected_slot] == float(f)
        # After 2 full cycles the bank holds the last maxframes consecutive calls.
        assert sorted(_slot_values(sd.kvo_cache[0])) == [5.0, 6.0, 7.0, 8.0]


class TestPinnedCeilingWindow:
    def test_writes_stay_within_logical_window_when_buffer_is_larger(self):
        """pin_cache_frames allocates at max_cache_maxframes; the logical window
        (cache_maxframes) must bound the write range (wrapper.py ceiling case)."""
        sd = _make_stream(cache_interval=2, cache_maxframes=2, alloc_maxframes=4, sentinel=-1.0)
        for f in range(1, 17):
            _call(sd, f)
        for layer in sd.kvo_cache:
            vals = _slot_values(layer)
            assert all(v >= 0.0 for v in vals[:2]), f"active window never written: {vals}"
            assert vals[2] == -1.0 and vals[3] == -1.0, (
                f"write escaped the logical window into pinned-ceiling slots: {vals}"
            )


class TestKvoFioSlotAlignment:
    def test_fio_written_to_same_slot_as_kvo_every_call(self):
        sd = _make_stream(cache_interval=3, cache_maxframes=4)
        for f in range(1, 19):
            kvo_before = _slot_values(sd.kvo_cache[0])
            fio_before = _fio_slot_values(sd.fio_cache[0])
            _call(sd, f)
            kvo_changed = [s for s in range(4) if _slot_values(sd.kvo_cache[0])[s] != kvo_before[s]]
            fio_changed = [s for s in range(4) if _fio_slot_values(sd.fio_cache[0])[s] != fio_before[s]]
            assert kvo_changed == fio_changed, f"call {f}: kvo wrote slot {kvo_changed}, fio wrote slot {fio_changed}"
