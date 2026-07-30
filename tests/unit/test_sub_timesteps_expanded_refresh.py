"""
Regression tests for G1: `_sub_timesteps_expanded` never rebuilt after a live
t_index_list change.

Root cause guarded: `_sub_timesteps_expanded` (pipeline.py's precomputed
per-step timestep table for the TCD / non-batched sequential loop, read by
loop position in `predict_x0_batch`) was built only in `prepare()`. A live
t_index_list change through StreamParameterUpdater never touched it:

  - LENGTH grow: the sequential loop indexes past the old (shorter) table ->
    IndexError, which is NOT a RuntimeError, so __call__'s error fallback
    (which only catches RuntimeError) never catches it — the stream dies.
  - LENGTH shrink: stale extra rows just go unused (silent, not crashing).
  - VALUE-only change (same length): the table keeps its stale per-frame
    timestep values — wrong denoising schedule, no crash.

Fix guarded here: the build is extracted into `_rebuild_sub_timesteps_expanded()`
(pipeline.py), called from `prepare()` (unchanged), from
`_update_timestep_calculations()` (stream_parameter_updater.py — the single
funnel shared by the value-only path, the length-changed path, and __call__'s
error fallback), and (redundantly but harmlessly) from `_refresh_derived_tensors()`.

CPU-only, model-free. Reuses the `_make_stream` harness from
test_rcfg_self_single_step_reseed.py and the `_make_resizable_stream` /
`_make_updater` harness from test_live_t_index_resize_buffers.py, forced onto
the non-batched (TCD-style) sequential path — `_use_seq_loop` is true whenever
`use_denoising_batch` is False, regardless of scheduler type.
"""

import torch
from test_live_t_index_resize_buffers import _make_resizable_stream, _make_updater

from streamdiffusion.pipeline import StreamDiffusion

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_sequential_stream(t_index_list, **kwargs):
    """Non-batched (TCD-style) stream: forces the sequential-loop code path
    that reads _sub_timesteps_expanded by index."""
    stream = _make_resizable_stream(t_index_list, **kwargs)
    stream.use_denoising_batch = False
    stream.trt_unet_batch_size = stream.frame_bff_size
    stream.batch_size = stream.frame_bff_size
    stream._rebuild_sub_timesteps_expanded()
    return stream


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestSubTimestepsExpandedRefresh:
    def test_initial_build_matches_t_index_length(self):
        stream = _make_sequential_stream([16])
        assert stream._sub_timesteps_expanded.shape == (1, stream.frame_bff_size)

    def test_grow_length_rebuilds_table(self):
        """RED before the fix: the table stays at length 1 after a live grow
        to length 2 -- predict_x0_batch's sequential loop then indexes
        position 1 of a length-1 tensor -> IndexError."""
        stream = _make_sequential_stream([16])
        updater = _make_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])

        assert stream._sub_timesteps_expanded is not None
        assert stream._sub_timesteps_expanded.shape == (2, stream.frame_bff_size), (
            f"_sub_timesteps_expanded not rebuilt after t_index length grow: "
            f"shape={stream._sub_timesteps_expanded.shape}"
        )

    def test_shrink_length_rebuilds_table(self):
        stream = _make_sequential_stream([8, 24, 40])
        updater = _make_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])

        assert stream._sub_timesteps_expanded.shape == (2, stream.frame_bff_size)

    def test_value_only_change_updates_table_values(self):
        """Same-length t_index VALUE change must also refresh the table's
        content, not just survive a length check -- _update_timestep_values_only
        never reaches _refresh_derived_tensors(), so this only passes once the
        rebuild is reachable from _update_timestep_calculations() itself."""
        stream = _make_sequential_stream([16, 32])
        updater = _make_updater(stream)
        old_table = stream._sub_timesteps_expanded.clone()

        updater._recalculate_timestep_dependent_params([8, 40])

        assert stream._sub_timesteps_expanded.shape == old_table.shape
        assert not torch.equal(old_table, stream._sub_timesteps_expanded), (
            "_sub_timesteps_expanded left stale after a same-length t_index value change"
        )
        expected = stream.sub_timesteps_tensor.view(-1).unsqueeze(1).expand(-1, stream.frame_bff_size)
        assert torch.equal(stream._sub_timesteps_expanded, expected)

    def test_error_fallback_rebuild_is_idempotent(self):
        """_refresh_derived_tensors() (the __call__ error-fallback rebuild
        hook) must not corrupt an already-correct table when called again."""
        stream = _make_sequential_stream([16, 32])
        before = stream._sub_timesteps_expanded.clone()

        StreamDiffusion._refresh_derived_tensors(stream)

        assert torch.equal(before, stream._sub_timesteps_expanded)

    def test_lcm_batched_path_still_collapses_to_none(self):
        """Non-regression: the batched LCM path (denoising batch + LCMScheduler)
        must keep collapsing this table to None -- only the sequential path
        (TCD / non-batched) uses it."""
        stream = _make_resizable_stream([16, 32])  # default use_denoising_batch=True, LCMScheduler
        stream._rebuild_sub_timesteps_expanded()
        assert stream._sub_timesteps_expanded is None
