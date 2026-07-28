"""
Regression tests for stale batch-sized buffers after a live t_index_list
LENGTH change (TD: TIndexBlock numBlocks changed while the stream runs).

Root cause guarded: StreamParameterUpdater._recalculate_timestep_dependent_params
(length-changed branch) rebuilds batch_size, x_t_latent_buffer, init_noise,
stock_noise and prompt_embeds — but never `_stock_noise_bufs` / `_stock_noise_pong`
/ `_combined_latent_buf` / `_cfg_latent_buf` / `_cfg_t_buf`, all allocated only in
prepare(). predict_x0_batch's n>1 path subscripts `_combined_latent_buf` and runs
the ping-pong rotation unconditionally, so the next frame after a live resize:

  - 1 -> n>1:  `None[: frame_bff_size]` TypeError (uncaught: __call__'s fallback
               only catches RuntimeError) — stream dead until restart.
  - grow n>=2: stale-shape copy_ RuntimeError; the fallback's
               _refresh_derived_tensors() rebuilt _combined_latent_buf but NOT
               _stock_noise_bufs, so it failed again every frame (permanent
               decode(randn) noise loop).
  - shrink:    copy_ broadcast-succeeds silently, then `stock_noise = _sn_dst`
               rebinds stock_noise to the stale larger tensor — shape mismatch
               downstream / silent corruption.

Fix guarded here: _refresh_derived_tensors() also rebuilds the ping-pong buffers,
and the updater's length-change branch delegates its derived-tensor rebuild to it
(prepare() parity from one shared implementation).

CPU-only, model-free. Reuses the object.__new__ harnesses from
test_rcfg_self_single_step_reseed.py (real StreamDiffusion + fake UNet) and
test_derived_tensor_sync.py (real StreamParameterUpdater shell).
"""

import torch
from test_rcfg_self_single_step_reseed import _make_stream, _zero_input

from streamdiffusion.pipeline import StreamDiffusion
from streamdiffusion.stream_parameter_updater import StreamParameterUpdater

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_resizable_stream(t_index_list, **kwargs):
    """_make_stream plus the attributes the updater's length-change branch reads
    (t_list, timesteps, kvo plumbing, instance-assigned scheduler scalings —
    the object.__new__ LCMScheduler shell has no .config for the real method)."""
    stream = _make_stream(t_index_list, **kwargs)
    stream.t_list = list(t_index_list)
    stream.timesteps = torch.linspace(999, 19, 50).long()
    stream.sub_timesteps = [int(stream.timesteps[i]) for i in t_index_list]
    stream._kvo_buckets = None
    stream._kvo_outputs_by_bucket = None

    def _scalings(timestep):
        # Same LCM boundary-condition formulas as _make_stream (sigma_data=0.5,
        # timestep_scaling=10): c_skip ~ 0, c_out ~ 1 at these timesteps
        t = float(timestep.item() if isinstance(timestep, torch.Tensor) else timestep)
        st = 10.0 * t
        return torch.tensor(0.25 / (st**2 + 0.25)), torch.tensor(st / (st**2 + 0.25) ** 0.5)

    stream.scheduler.get_scalings_for_boundary_condition_discrete = _scalings
    return stream


def _make_updater(stream):
    """Construct StreamParameterUpdater without calling __init__ (avoids deps)."""
    updater = object.__new__(StreamParameterUpdater)
    updater.stream = stream
    updater._lock = __import__("threading").Lock()
    return updater


def _assert_frames_healthy(stream, n_new, num_frames):
    """Drive predict_x0_batch and assert per-frame batch-shape consistency."""
    x_in = _zero_input(stream)
    for frame in range(num_frames):
        x0 = StreamDiffusion.predict_x0_batch(stream, x_in)
        assert torch.isfinite(x0).all(), f"frame {frame}: non-finite x0 after resize to n={n_new}"
        assert stream.stock_noise.shape[0] == n_new, (
            f"frame {frame}: stock_noise has {stream.stock_noise.shape[0]} rows, expected {n_new} — "
            "stale ping-pong buffer rebound after live resize"
        )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestLiveTIndexResize:
    """Live length change via updater._recalculate_timestep_dependent_params,
    then frames through the real predict_x0_batch — no prepare() in between,
    exactly like the OSC /t_list path in TD."""

    def test_grow_1_to_2(self):
        stream = _make_resizable_stream([16])
        updater = _make_updater(stream)
        _assert_frames_healthy(stream, 1, 2)  # pre-resize baseline works

        updater._recalculate_timestep_dependent_params([16, 32])

        # Pre-fix: TypeError — _combined_latent_buf is still None from n==1
        _assert_frames_healthy(stream, 2, 4)
        assert stream._combined_latent_buf.shape[0] == 2
        for buf in stream._stock_noise_bufs:
            assert buf.shape[0] == 2

    def test_grow_1_to_2_ping_pong_rotates(self):
        stream = _make_resizable_stream([16])
        updater = _make_updater(stream)
        updater._recalculate_timestep_dependent_params([16, 32])

        x_in = _zero_input(stream)
        expected_pong = stream._stock_noise_pong
        for frame in range(4):
            StreamDiffusion.predict_x0_batch(stream, x_in)
            expected_pong = 1 - expected_pong
            assert stream._stock_noise_pong == expected_pong, (
                f"frame {frame}: ping-pong rotation not engaged after live 1->2 resize"
            )

    def test_grow_2_to_3(self):
        stream = _make_resizable_stream([16, 32])
        updater = _make_updater(stream)
        _assert_frames_healthy(stream, 2, 2)

        updater._recalculate_timestep_dependent_params([8, 24, 40])

        # Pre-fix: RuntimeError — 2-row _combined_latent_buf vs 3-row tensors,
        # and _stock_noise_bufs stay 2-row even after _refresh_derived_tensors()
        _assert_frames_healthy(stream, 3, 4)
        assert stream._combined_latent_buf.shape[0] == 3
        for buf in stream._stock_noise_bufs:
            assert buf.shape[0] == 3

    def test_shrink_3_to_2(self):
        stream = _make_resizable_stream([8, 24, 40])
        updater = _make_updater(stream)
        _assert_frames_healthy(stream, 3, 2)

        updater._recalculate_timestep_dependent_params([16, 32])

        # Pre-fix: the stale 3-row buffers broadcast-accept the 2-row copies
        # silently, then stock_noise is rebound to 3 rows (caught per-frame in
        # _assert_frames_healthy) and unet_step raises on the shape mismatch
        _assert_frames_healthy(stream, 2, 4)
        assert stream._combined_latent_buf.shape[0] == 2
        for buf in stream._stock_noise_bufs:
            assert buf.shape[0] == 2

    def test_shrink_2_to_1(self):
        stream = _make_resizable_stream([16, 32])
        updater = _make_updater(stream)
        _assert_frames_healthy(stream, 2, 2)

        updater._recalculate_timestep_dependent_params([16])

        # n==1 path uses none of the batch buffers (functional today); the buf
        # must also be dropped to None for prepare() parity so a later grow
        # re-allocates instead of reusing a stale tensor
        x_in = _zero_input(stream)
        for frame in range(4):
            x0 = StreamDiffusion.predict_x0_batch(stream, x_in)
            assert torch.isfinite(x0).all()
            assert torch.allclose(stream.stock_noise, stream.init_noise), (
                f"frame {frame}: n==1 per-frame reseed invariant broken after live 2->1 resize"
            )
        assert stream._combined_latent_buf is None
        assert stream.x_t_latent_buffer is None


class TestRefreshDerivedTensorsSelfHeal:
    """_refresh_derived_tensors() is the error-fallback's rebuild hook — it must
    restore the ping-pong buffers too, or the fallback loops forever."""

    def test_refresh_rebuilds_ping_pong_buffers(self):
        stream = _make_resizable_stream([16, 32])
        h = stream.latent_height
        # Stale the ping-pong state the way a live resize leaves it
        stream._stock_noise_bufs = [
            torch.empty(1, 4, h, h, dtype=stream.dtype),
            torch.empty(1, 4, h, h, dtype=stream.dtype),
        ]
        stream._stock_noise_pong = 1

        StreamDiffusion._refresh_derived_tensors(stream)

        assert stream._stock_noise_pong == 0
        for buf in stream._stock_noise_bufs:
            assert buf.shape == (stream.batch_size, 4, h, h), (
                "_refresh_derived_tensors left _stock_noise_bufs stale — the __call__ "
                "error fallback can never self-heal a live batch-size change"
            )
        # The rebuilt bufs must not alias stock_noise (ping-pong precondition)
        for buf in stream._stock_noise_bufs:
            assert buf.data_ptr() != stream.stock_noise.data_ptr()
