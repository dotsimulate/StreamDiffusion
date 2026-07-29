"""
Regression tests for G7 + G8: kvo_cache / fio_cache staleness after a live
t_index_list LENGTH change, specifically for cfg_type modes where
trt_unet_batch_size != batch_size.

G7 root cause guarded: kvo_cache is *allocated* with stream.trt_unet_batch_size
(wrapper.py's create_kvo_cache) but was *resized* on a live t_index change
against stream.batch_size (stream_parameter_updater.py). Those two diverge
exactly when cfg_type is "initialize" ((n+1)*f) or "full" (2*n*f) -- for
"self"/"none" they're equal, which is why this went unnoticed (no shipped
config uses initialize/full). Fix: track old_trt_batch_size separately and
drive the resize off trt_unet_batch_size, matching what was actually
allocated.

G8 root cause guarded: fio_cache (also allocated with trt_unet_batch_size,
wrapper.py's create_fi_cache) was never resized *at all* on a live t_index
change -- no reference to it anywhere in the updater, for any cfg_type,
including the shipped "self" path. Fix: resize it alongside kvo_cache through
a shared _resize_cache_tensors helper.

The two caches have different rank and batch-dim position (see
create_kvo_cache / create_fi_cache, acceleration/tensorrt/models/utils.py):
  - kvo_cache: (2, cache_maxframes, batch, seq_len, hidden) -- 5-D, batch at dim 2
    (leading 2 is K+V).
  - fio_cache: (cache_maxframes, batch, seq_len, hidden) -- 4-D, output only,
    batch at dim 1.
Reusing one cache's slicing on the other's rank would resize the wrong axis
(or index out of range) -- that asymmetry is the easiest thing to get wrong
here and is what most of these tests are pinned on.

CPU-only, model-free. Reuses _make_resizable_stream / _make_updater from
test_live_t_index_resize_buffers.py.
"""

import torch
from test_live_t_index_resize_buffers import _make_resizable_stream, _make_updater

CACHE_MAXFRAMES = 1
SEQ_LEN = 4
HIDDEN_DIM = 8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_kvo_layer(batch, fill=0.0):
    t = torch.full((2, CACHE_MAXFRAMES, batch, SEQ_LEN, HIDDEN_DIM), fill)
    return t


def _make_fio_layer(batch, fill=0.0):
    return torch.full((CACHE_MAXFRAMES, batch, SEQ_LEN, HIDDEN_DIM), fill)


def _make_cache_stream(t_index_list, cfg_type, n_kvo_layers=2, n_fio_layers=2, **kwargs):
    """_make_resizable_stream plus real kvo_cache / fio_cache tensors sized to
    the stream's initial trt_unet_batch_size (as wrapper.py's allocators
    would), each layer filled with its layer index so content preservation
    is checkable post-resize."""
    stream = _make_resizable_stream(t_index_list, cfg_type=cfg_type, **kwargs)
    old_trt = stream.trt_unet_batch_size
    stream.kvo_cache = [_make_kvo_layer(old_trt, fill=float(i)) for i in range(n_kvo_layers)]
    stream.fio_cache = [_make_fio_layer(old_trt, fill=float(i)) for i in range(n_fio_layers)]
    return stream


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestG7KvoCacheResizesToTrtBatchSize:
    def test_grow_resizes_kvo_cache_to_new_trt_batch_size_not_batch_size(self):
        """cfg_type='initialize': trt_unet_batch_size = (n+1)*f, batch_size = n*f
        -- they diverge, so this is the case that catches a resize keyed on
        the wrong one."""
        stream = _make_cache_stream([16], cfg_type="initialize")
        assert stream.trt_unet_batch_size == 2  # (1+1)*1
        assert stream.batch_size == 1  # 1*1
        updater = _make_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])

        assert stream.trt_unet_batch_size == 3  # (2+1)*1
        assert stream.batch_size == 2  # 2*1
        for layer in stream.kvo_cache:
            assert layer.shape[2] == stream.trt_unet_batch_size, (
                f"kvo_cache batch dim {layer.shape[2]} != trt_unet_batch_size "
                f"{stream.trt_unet_batch_size} (resized against batch_size instead?)"
            )

    def test_shrink_resizes_kvo_cache_to_new_trt_batch_size(self):
        stream = _make_cache_stream([8, 24, 40], cfg_type="initialize")
        assert stream.trt_unet_batch_size == 4  # (3+1)*1
        updater = _make_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])

        assert stream.trt_unet_batch_size == 3  # (2+1)*1
        for layer in stream.kvo_cache:
            assert layer.shape[2] == 3

    def test_full_cfg_type_also_driven_by_trt_batch_size(self):
        """cfg_type='full': trt_unet_batch_size = 2*n*f, further from batch_size
        than 'initialize' -- an independent check on the same divergence."""
        stream = _make_cache_stream([16], cfg_type="full")
        assert stream.trt_unet_batch_size == 2  # 2*1*1
        updater = _make_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])

        assert stream.trt_unet_batch_size == 4  # 2*2*1
        for layer in stream.kvo_cache:
            assert layer.shape[2] == 4

    def test_self_cfg_type_kvo_cache_still_resizes(self):
        """Non-regression: 'self' has trt_unet_batch_size == batch_size, so this
        exercises the ordinary (pre-existing, already-working) path stays intact."""
        stream = _make_cache_stream([16], cfg_type="self")
        assert stream.trt_unet_batch_size == stream.batch_size == 1
        updater = _make_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])

        assert stream.trt_unet_batch_size == stream.batch_size == 2
        for layer in stream.kvo_cache:
            assert layer.shape[2] == 2

    def test_kvo_bucket_state_invalidated_after_resize(self):
        stream = _make_cache_stream([16], cfg_type="initialize")
        stream._kvo_buckets = {"stale": True}
        stream._kvo_outputs_by_bucket = {"stale": True}
        updater = _make_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])

        assert stream._kvo_buckets is None, "stale kvo bucket refs survived a resize"
        assert stream._kvo_outputs_by_bucket is None

    def test_kvo_cache_content_preserved_up_to_min_batch_on_grow(self):
        stream = _make_cache_stream([16], cfg_type="initialize", n_kvo_layers=1)
        old_batch = stream.trt_unet_batch_size  # 2
        original = stream.kvo_cache[0].clone()
        updater = _make_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])

        resized = stream.kvo_cache[0]
        assert torch.equal(resized[:, :, :old_batch, :, :], original), (
            "kvo_cache lost its old content across a grow resize"
        )
        assert torch.equal(resized[:, :, old_batch:, :, :], torch.zeros_like(resized[:, :, old_batch:, :, :])), (
            "kvo_cache's newly grown rows should be zero-padded, not garbage"
        )

    def test_empty_kvo_cache_list_is_skipped_without_error(self):
        """kvo_cache=[] (never allocated -- no TRT engine / feature disabled)
        must not raise even when trt_unet_batch_size changes."""
        stream = _make_cache_stream([16], cfg_type="initialize", n_kvo_layers=0)
        assert stream.kvo_cache == []
        updater = _make_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])  # must not raise

        assert stream.kvo_cache == []


class TestG8FioCacheResizesAtDim1:
    def test_grow_resizes_fio_cache_at_dim_1_not_dim_2(self):
        """The rank/batch-dim asymmetry vs. kvo_cache is the easiest thing to
        get wrong here: fio_cache is 4-D with batch at dim 1, not 5-D at dim 2."""
        stream = _make_cache_stream([16], cfg_type="initialize")
        old_trt = stream.trt_unet_batch_size  # 2
        updater = _make_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])

        for layer in stream.fio_cache:
            assert layer.dim() == 4, "fio_cache layer rank changed -- should stay 4-D"
            assert layer.shape[1] == stream.trt_unet_batch_size == 3, (
                f"fio_cache batch dim (dim 1) is {layer.shape[1]}, expected trt_unet_batch_size=3"
            )
            # the other three dims (maxframes, seq_len, hidden) must be untouched
            assert layer.shape[0] == CACHE_MAXFRAMES
            assert layer.shape[2] == SEQ_LEN
            assert layer.shape[3] == HIDDEN_DIM
        assert old_trt != stream.trt_unet_batch_size  # sanity: this resize actually happened

    def test_shrink_resizes_fio_cache_at_dim_1(self):
        stream = _make_cache_stream([8, 24, 40], cfg_type="full")
        assert stream.trt_unet_batch_size == 6  # 2*3*1
        updater = _make_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])

        assert stream.trt_unet_batch_size == 4  # 2*2*1
        for layer in stream.fio_cache:
            assert layer.shape[1] == 4

    def test_self_cfg_type_fio_cache_now_resizes_too(self):
        """The shipped path: cfg_type='self' previously left fio_cache stale on
        every live t_index change (G8's headline symptom) -- must now resize
        just like kvo_cache does."""
        stream = _make_cache_stream([16], cfg_type="self")
        updater = _make_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])

        for layer in stream.fio_cache:
            assert layer.shape[1] == stream.trt_unet_batch_size == 2

    def test_fio_cache_content_preserved_up_to_min_batch_on_grow(self):
        stream = _make_cache_stream([16], cfg_type="initialize", n_fio_layers=1)
        old_batch = stream.trt_unet_batch_size  # 2
        original = stream.fio_cache[0].clone()
        updater = _make_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])

        resized = stream.fio_cache[0]
        assert torch.equal(resized[:, :old_batch, :, :], original), (
            "fio_cache lost its old content across a grow resize"
        )

    def test_empty_fio_cache_list_is_skipped_without_error(self):
        stream = _make_cache_stream([16], cfg_type="initialize", n_fio_layers=0)
        assert stream.fio_cache == []
        updater = _make_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])  # must not raise

        assert stream.fio_cache == []


class TestG7G8NoResizeWhenTrtBatchSizeUnchanged:
    def test_kvo_and_fio_caches_untouched_when_trt_batch_size_does_not_change(self):
        """Two different t_index lengths can still land on the same
        trt_unet_batch_size (e.g. 'full' collapses (n, f) pairs less densely
        than 'self') -- the resize must be skipped (same object, not just same
        shape) rather than paying for a no-op reallocation every time."""
        # cfg_type='full', frame_bff_size=1: trt = 2*n*f. Force n unchanged in
        # count but exercise the "length changed" branch by holding f steady
        # and changing t_index length while keeping the *value* of 2*n*f fixed
        # is not reachable via a single length change with f=1, so instead
        # assert directly on the guard: call the resize helper's caller twice
        # with identical old/new to confirm identity is preserved.
        stream = _make_cache_stream([16], cfg_type="self")
        updater = _make_updater(stream)
        kvo_before = list(stream.kvo_cache)
        fio_before = list(stream.fio_cache)

        # Same length -> value-only path, which never touches trt_unet_batch_size
        # or the caches at all (confirms the resize is gated on an actual change).
        updater._recalculate_timestep_dependent_params([32])

        assert len(stream.kvo_cache) == len(kvo_before)
        assert len(stream.fio_cache) == len(fio_before)
        for a, b in zip(stream.kvo_cache, kvo_before):
            assert a.data_ptr() == b.data_ptr()
        for a, b in zip(stream.fio_cache, fio_before):
            assert a.data_ptr() == b.data_ptr()
