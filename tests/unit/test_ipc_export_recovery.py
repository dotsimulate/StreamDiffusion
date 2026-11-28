"""Tests for streamdiffusion._ipc_export (cuda-link 2.2.0 recover() / async source buffers).

ExportRecovery and PackRing run on CPU tensors with a fake Exporter. The wrapper tests use the
object.__new__ shell pattern (see test_sync_free_output_5_2.py) and a stub cuda_link module, so
no GPU and no cuda_link install are needed.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from streamdiffusion._ipc_export import (
    PERMANENT_CUDA_ERRORS,
    ExportRecovery,
    PackRing,
    RecoveryAction,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeExporter:
    """Duck-typed cuda_link.Exporter: export() outcomes, recover() codes, ring slot state."""

    def __init__(self, outcomes=(), recover_codes=(), num_slots=2):
        self.outcomes = list(outcomes)
        self.recover_codes = list(recover_codes)
        self.write_idx = 0
        self.busy_slots = set()
        self.drains = 0
        self.closed = False
        self.exports = 0
        self.num_slots = num_slots

    def export(self, frame):
        self.exports += 1
        outcome = self.outcomes.pop(0) if self.outcomes else "PUBLISHED"
        if outcome == "PUBLISHED":
            self.write_idx += 1
        return outcome

    def recover(self):
        return self.recover_codes.pop(0) if self.recover_codes else 0

    def slot_copy_done(self, slot):
        return slot not in self.busy_slots

    def drain(self):
        self.drains += 1
        self.busy_slots.clear()

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# ExportRecovery
# ---------------------------------------------------------------------------


def _recovery():
    clock = FakeClock()
    return ExportRecovery("test", clock=clock), clock


def test_recover_zero_keeps_exporter_and_exports_next_frame():
    rec, _ = _recovery()
    assert rec.failed(FakeExporter(recover_codes=[0])) is RecoveryAction.KEEP
    assert rec.ready()
    assert not rec.stopped


@pytest.mark.parametrize("code", sorted(PERMANENT_CUDA_ERRORS))
def test_permanent_error_closes_and_stops_until_reset(code):
    rec, clock = _recovery()
    assert rec.failed(FakeExporter(recover_codes=[code])) is RecoveryAction.CLOSE
    assert rec.stopped
    clock.now += 3600
    assert not rec.ready()
    rec.reset()
    assert rec.ready()


def test_transient_error_waits_one_second_then_reopens_after_three():
    rec, clock = _recovery()
    exporter = FakeExporter(recover_codes=[1, 1, 1])
    assert rec.failed(exporter) is RecoveryAction.KEEP
    assert not rec.ready()  # retry is rate-limited
    clock.now += 1.0
    assert rec.ready()
    assert rec.failed(exporter) is RecoveryAction.KEEP
    clock.now += 1.0
    assert rec.failed(exporter) is RecoveryAction.CLOSE  # third failure: open a new stream
    assert not rec.stopped
    clock.now += 1.0
    assert rec.ready()


def test_new_stream_that_fails_the_same_way_stops():
    rec, clock = _recovery()
    exporter = FakeExporter(recover_codes=[1] * 6)
    actions = []
    for _ in range(6):
        actions.append(rec.failed(exporter))
        clock.now += 1.0
    assert actions == [RecoveryAction.KEEP, RecoveryAction.KEEP, RecoveryAction.CLOSE] * 2
    assert rec.stopped
    assert rec.last_error == 1


def test_published_frame_ends_the_fault():
    rec, clock = _recovery()
    exporter = FakeExporter(recover_codes=[1] * 6)
    for _ in range(3):
        rec.failed(exporter)
        clock.now += 1.0
    rec.published()  # the new stream works
    for _ in range(3):
        last = rec.failed(exporter)
        clock.now += 1.0
    assert last is RecoveryAction.CLOSE
    assert not rec.stopped  # a new fault gets its own reopen


def test_exporter_without_recover_is_treated_as_unknown_error():
    rec, _ = _recovery()

    class OldExporter:  # cuda-link < 2.2.0
        pass

    assert rec.failed(OldExporter()) is RecoveryAction.KEEP
    assert rec.last_error == -1


# ---------------------------------------------------------------------------
# PackRing
# ---------------------------------------------------------------------------

CPU = torch.device("cpu")


def test_blocking_ring_reuses_one_buffer():
    ring = PackRing()
    exporter = FakeExporter()
    a = ring.acquire(4, 6, CPU, exporter=exporter, num_slots=2, async_export=False)
    exporter.write_idx = 1
    b = ring.acquire(4, 6, CPU, exporter=exporter, num_slots=2, async_export=False)
    assert a.data_ptr() == b.data_ptr()
    assert a.shape == (4, 6, 4)
    assert torch.all(a[..., 3] == 255)
    assert exporter.drains == 0


def test_async_ring_gives_one_buffer_per_slot():
    ring = PackRing()
    exporter = FakeExporter()
    first = ring.acquire(4, 6, CPU, exporter=None, num_slots=2, async_export=True)  # before open
    slot0 = ring.acquire(4, 6, CPU, exporter=exporter, num_slots=2, async_export=True)
    exporter.write_idx = 1
    slot1 = ring.acquire(4, 6, CPU, exporter=exporter, num_slots=2, async_export=True)
    exporter.write_idx = 2
    slot0_again = ring.acquire(4, 6, CPU, exporter=exporter, num_slots=2, async_export=True)
    assert slot0.data_ptr() != slot1.data_ptr()
    assert slot0_again.data_ptr() == slot0.data_ptr()
    assert first.shape == slot0.shape
    # Growing from one buffer to the ring released the first one: drain before that.
    assert exporter.drains == 1


def test_async_ring_waits_for_a_busy_slot_before_reuse():
    ring = PackRing()
    exporter = FakeExporter()
    ring.acquire(4, 6, CPU, exporter=exporter, num_slots=2, async_export=True)
    drains_after_alloc = exporter.drains
    exporter.busy_slots.add(0)  # the copy from slot 0 is still queued
    exporter.write_idx = 2  # slot 0 again
    ring.acquire(4, 6, CPU, exporter=exporter, num_slots=2, async_export=True)
    assert exporter.drains == drains_after_alloc + 1
    exporter.write_idx = 3  # slot 1: copy done, no wait
    ring.acquire(4, 6, CPU, exporter=exporter, num_slots=2, async_export=True)
    assert exporter.drains == drains_after_alloc + 1


def test_async_ring_drains_before_realloc_on_shape_change():
    ring = PackRing()
    exporter = FakeExporter()
    ring.acquire(4, 6, CPU, exporter=exporter, num_slots=2, async_export=True)
    before = exporter.drains
    buf = ring.acquire(8, 6, CPU, exporter=exporter, num_slots=2, async_export=True)
    assert buf.shape == (8, 6, 4)
    assert exporter.drains == before + 1


# ---------------------------------------------------------------------------
# StreamDiffusionWrapper export paths
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_cuda_link(monkeypatch):
    mod = types.ModuleType("cuda_link")

    class FrameOutcome:
        PUBLISHED = "PUBLISHED"
        FAILED = "FAILED"
        SKIPPED_BARRIER = "SKIPPED_BARRIER"

    class GpuFrame:
        def __init__(self, ptr, size, producer_stream=None):
            self.ptr, self.size, self.producer_stream = ptr, size, producer_stream

    mod.FrameOutcome = FrameOutcome
    mod.GpuFrame = GpuFrame
    monkeypatch.setitem(sys.modules, "cuda_link", mod)
    return mod


def _wrapper_shell(exporter, clock):
    from streamdiffusion.wrapper import StreamDiffusionWrapper

    w = object.__new__(StreamDiffusionWrapper)
    w.use_cuda_ipc_output = True
    w._cuda_ipc_shm_name = "test_out"
    w._cuda_ipc_cn_processed_shm_name = "test_cn"
    w.debug_mode = False
    w._ipc_consecutive_failures = 0
    w._ipc_barrier_skip_count = 0
    w._ipc_graphs_degraded = False
    w._cn_ipc_export_warned = False
    w._cuda_ipc_async = False
    w._ipc_recovery = ExportRecovery("out", clock=clock)
    w._cn_ipc_recovery = ExportRecovery("cn", clock=clock)
    w.opened = []

    def _open(*_args):
        if w._cuda_ipc_exporter is None:
            w._cuda_ipc_exporter = exporter()
            w.opened.append(w._cuda_ipc_exporter)
        return w._cuda_ipc_exporter

    def _open_cn(*_args):
        if w._cuda_ipc_cn_exporter is None:
            w._cuda_ipc_cn_exporter = exporter()
            w.opened.append(w._cuda_ipc_cn_exporter)
        return w._cuda_ipc_cn_exporter

    fake_buf = torch.zeros((2, 2, 4), dtype=torch.uint8)
    w._cuda_ipc_exporter = None
    w._cuda_ipc_cn_exporter = None
    w._lazy_init_ipc_exporter = _open
    w._lazy_init_cn_ipc_exporter = _open_cn
    w._ipc_pack_rgba = lambda _t: fake_buf
    w._ipc_pack_unit_rgba = lambda _t: fake_buf
    return w


@pytest.fixture
def no_cuda_stream(monkeypatch):
    stream = types.SimpleNamespace(cuda_stream=0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)


def test_main_export_keeps_exporter_when_recover_returns_zero(stub_cuda_link, no_cuda_stream):
    clock = FakeClock()
    w = _wrapper_shell(lambda: FakeExporter(outcomes=["FAILED", "PUBLISHED"], recover_codes=[0]), clock)
    assert w.postprocess_image(None) is None
    assert w.postprocess_image(None) is None
    assert len(w.opened) == 1
    assert w.opened[0].exports == 2
    assert not w.opened[0].closed


def test_main_export_stops_on_permanent_error_and_reports_it(stub_cuda_link, no_cuda_stream):
    clock = FakeClock()
    w = _wrapper_shell(lambda: FakeExporter(outcomes=["FAILED"], recover_codes=[719]), clock)
    w.postprocess_image(None)
    assert w.opened[0].closed
    assert w._cuda_ipc_exporter is None
    clock.now += 10
    w.postprocess_image(None)  # stopped: no new exporter, no export
    assert len(w.opened) == 1
    assert w.get_ipc_health_status() == "STOPPED(CUDA error 719)"


def test_main_export_opens_new_stream_after_three_transient_failures(stub_cuda_link, no_cuda_stream):
    clock = FakeClock()
    w = _wrapper_shell(lambda: FakeExporter(outcomes=["FAILED"] * 3, recover_codes=[1, 1, 1]), clock)
    for _ in range(3):
        w.postprocess_image(None)
        clock.now += 1.0
    assert w.opened[0].closed
    w.postprocess_image(None)
    assert len(w.opened) == 2
    assert w.opened[1].exports == 1


def test_cn_export_keeps_exporter_when_recover_returns_zero(stub_cuda_link, no_cuda_stream):
    clock = FakeClock()
    w = _wrapper_shell(lambda: FakeExporter(outcomes=["FAILED", "PUBLISHED"], recover_codes=[0]), clock)
    assert w.export_controlnet_preview_ipc(None) is False
    assert w.export_controlnet_preview_ipc(None) is True
    assert len(w.opened) == 1
    assert not w.opened[0].closed


def test_cleanup_drains_async_exporters_and_resets_recovery(stub_cuda_link, no_cuda_stream):
    clock = FakeClock()
    w = _wrapper_shell(lambda: FakeExporter(outcomes=["FAILED"], recover_codes=[719]), clock)
    w.postprocess_image(None)
    assert w._ipc_recovery.stopped
    w._cuda_ipc_async = True
    w._cuda_ipc_cn_exporter = cn = FakeExporter()
    w.cleanup_cuda_ipc()
    assert cn.drains == 1 and cn.closed
    assert w._cuda_ipc_cn_exporter is None
    assert w._ipc_recovery.ready()


def test_cn_stopped_is_reported_in_health_status(stub_cuda_link, no_cuda_stream):
    clock = FakeClock()
    w = _wrapper_shell(lambda: FakeExporter(outcomes=["FAILED"], recover_codes=[700]), clock)
    w.export_controlnet_preview_ipc(None)
    assert w._cn_ipc_recovery.stopped
    assert w.get_ipc_health_status() == "cn-STOPPED(CUDA error 700)"


def test_lazy_init_reopens_exporter_when_frame_size_changes(stub_cuda_link, monkeypatch):
    from streamdiffusion.wrapper import StreamDiffusionWrapper

    opened = []

    class FrameSpec:
        def __init__(self, shm_name, height, width, channels, dtype, num_slots):
            self.height, self.width = height, width

    class Exporter:
        @staticmethod
        def open(spec, policy=None):
            exp = FakeExporter()
            exp.data_size = spec.height * spec.width * 4
            opened.append(exp)
            return exp

    class ExportPolicy:
        export_sync = True

        @classmethod
        def from_env(cls):
            return cls()

    stub_cuda_link.Exporter = Exporter
    stub_cuda_link.FrameSpec = FrameSpec
    stub_cuda_link.ExportPolicy = ExportPolicy
    monkeypatch.setenv("CUDALINK_EXPORT_SYNC", "1")

    w = object.__new__(StreamDiffusionWrapper)
    w._cuda_ipc_shm_name = "out"
    w._cuda_ipc_cn_processed_shm_name = "cn"
    w._cuda_ipc_num_slots = 2
    w._cuda_ipc_exporter = None
    w._cuda_ipc_cn_exporter = None

    for init in (w._lazy_init_ipc_exporter, w._lazy_init_cn_ipc_exporter):
        opened.clear()
        first = init(512, 512)
        assert init(512, 512) is first  # same size: cached
        second = init(512, 768)  # e.g. another ControlNet preprocessor output size
        assert first.closed and second is not first
        assert second.data_size == 512 * 768 * 4
        assert len(opened) == 2
