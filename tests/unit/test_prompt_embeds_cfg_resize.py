"""
Regression tests for G2 + G3: prompt_embeds / CFG buffer staleness across a
live t_index_list resize and a live guidance_scale crossing of 1.0.

G2 root cause guarded: _recalculate_timestep_dependent_params's length-changed
branch rebuilt prompt_embeds as prompt_embeds[0].repeat(batch_size, 1, 1) with
no cfg guard. Row [0] is the *uncond* row for cfg_type in (initialize, full)
(see prepare()'s [uncond|cond] cat) — repeating it turned every row into
uncond: wrong content, and (for cfg_type == "initialize" with
denoising_steps_num > 1) also the wrong row count, since _apply_prompt_blending's
own CFG branch was *also* repeating uncond by the full batch_size instead of
frame_bff_size — only masked at denoising_steps_num == 1, where the two are
equal. Fix: the resize path re-runs _apply_prompt_blending (the one function
that already knows every cfg layout) instead of hand-rolling the shape, and
_apply_prompt_blending's "initialize" branch itself was corrected to repeat
uncond by frame_bff_size, matching prepare() and unet_step's _cfg_latent_buf.

G3 root cause guarded: a live guidance_scale update crossing 1.0 (either
direction) left _cfg_latent_buf/_cfg_t_buf (pipeline.py's CFG expansion
buffers, allocated only when guidance_scale > 1.0) stale — None if it had
never crossed above 1.0 before, or sized for the OLD cfg regime otherwise —
until some unrelated batch-size-changing call happened to reach
_refresh_derived_tensors(). Fix: update_stream_params now detects the
crossing and rebuilds them (plus re-blends prompt_embeds) in step.

CPU-only, model-free. Builds a real StreamDiffusion instance via the
_make_stream / _make_resizable_stream harnesses (test_rcfg_self_single_step_reseed.py
/ test_live_t_index_resize_buffers.py), plus a minimal stream.pipe.encode_prompt
stand-in so _apply_prompt_blending's CFG branch can run without a real text encoder.
"""

import threading
import types

import torch
from test_live_t_index_resize_buffers import _make_resizable_stream, _make_updater

UNCOND_VALUE = -1.0
COND_VALUE = 2.0
TOKENS = 77
HIDDEN = 8

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fake_encode_prompt(**kwargs):
    """Deterministic uncond embedding, distinguishable from the cached cond embed."""
    return (torch.full((1, TOKENS, HIDDEN), UNCOND_VALUE),)


def _make_cfg_stream(t_index_list, **kwargs):
    stream = _make_resizable_stream(t_index_list, **kwargs)
    stream.embedding_hooks = []
    stream.pipe = types.SimpleNamespace(encode_prompt=_fake_encode_prompt)
    return stream


def _make_cfg_updater(stream, prompt="cat"):
    """_make_updater plus a populated prompt cache (so _apply_prompt_blending
    has something to re-blend) and the lock/warn-once flags
    update_stream_params reads unconditionally."""
    updater = _make_updater(stream)
    updater._prompt_cache = {0: {"embed": torch.full((1, TOKENS, HIDDEN), COND_VALUE), "text": prompt}}
    updater._current_prompt_list = [(prompt, 1.0)]
    updater._current_negative_prompt = ""
    updater._last_prompt_interpolation_method = "linear"
    updater.normalize_prompt_weights = True
    updater._update_lock = threading.RLock()
    updater._warned_delta_above_ceiling = False
    updater._warned_delta_out_of_range = False
    return updater


# ---------------------------------------------------------------------------
# G2 tests
# ---------------------------------------------------------------------------


class TestG2PromptEmbedsCfgResize:
    def test_initialize_resize_uncond_cond_content_and_row_count(self):
        stream = _make_cfg_stream([16], cfg_type="initialize", guidance_scale=1.4)
        updater = _make_cfg_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])

        f = stream.frame_bff_size
        n = stream.batch_size
        assert stream.prompt_embeds.shape[0] == f + n, (
            f"initialize prompt_embeds row count {stream.prompt_embeds.shape[0]} != "
            f"frame_bff_size({f}) + batch_size({n}) — mismatches _cfg_latent_buf/UNet batch"
        )
        uncond_rows = stream.prompt_embeds[:f]
        cond_rows = stream.prompt_embeds[f:]
        assert torch.allclose(uncond_rows, torch.full_like(uncond_rows, UNCOND_VALUE)), (
            "uncond rows are not real uncond content after resize"
        )
        assert torch.allclose(cond_rows, torch.full_like(cond_rows, COND_VALUE)), (
            "cond rows corrupted — pre-fix bug made every row uncond (row[0].repeat)"
        )

    def test_self_resize_all_rows_are_cond(self):
        """cfg_type='self' has no uncond block in prompt_embeds at all — every
        row must be the real cond content post-resize (not silently zeroed or
        left partially stale)."""
        stream = _make_cfg_stream([16], cfg_type="self", guidance_scale=1.4)
        updater = _make_cfg_updater(stream)

        updater._recalculate_timestep_dependent_params([16, 32])

        assert stream.prompt_embeds.shape[0] == stream.batch_size
        assert torch.allclose(stream.prompt_embeds, torch.full_like(stream.prompt_embeds, COND_VALUE))

    def test_resize_falls_back_to_row0_repeat_when_no_cached_prompt(self):
        """No cfg guard needed for the fallback branch either — must survive
        the case where there's nothing cached yet to re-blend from (fresh
        construction before the first update_prompt call)."""
        stream = _make_cfg_stream([16], cfg_type="self", guidance_scale=1.4)
        updater = _make_updater(stream)  # no _prompt_cache / _current_prompt_list entries
        old_row0 = stream.prompt_embeds[0].clone()

        updater._recalculate_timestep_dependent_params([16, 32])

        assert stream.prompt_embeds.shape[0] == stream.batch_size
        assert torch.allclose(stream.prompt_embeds, old_row0.repeat(stream.batch_size, 1, 1))


# ---------------------------------------------------------------------------
# G3 tests
# ---------------------------------------------------------------------------


class TestG3GuidanceScaleCrossing:
    def test_crossing_above_1_allocates_cfg_buffers(self):
        stream = _make_cfg_stream([16, 32], cfg_type="initialize", guidance_scale=1.0)
        assert stream._cfg_latent_buf is None  # baseline: guidance <= 1.0, never allocated
        updater = _make_cfg_updater(stream)

        updater.update_stream_params(guidance_scale=1.4)

        assert stream._cfg_latent_buf is not None, (
            "_cfg_latent_buf still None after guidance_scale crossed above 1.0 — "
            "unet_step would TypeError on the next frame"
        )
        assert stream._cfg_t_buf is not None
        assert stream._cfg_latent_buf.shape[0] == stream.frame_bff_size + stream.batch_size

    def test_crossing_below_1_drops_cfg_buffers(self):
        stream = _make_cfg_stream([16, 32], cfg_type="initialize", guidance_scale=1.4)
        assert stream._cfg_latent_buf is not None
        updater = _make_cfg_updater(stream)

        updater.update_stream_params(guidance_scale=1.0)

        assert stream._cfg_latent_buf is None
        assert stream._cfg_t_buf is None

    def test_non_crossing_change_leaves_cfg_buffers_untouched(self):
        """A guidance_scale change that stays on the same side of 1.0 must not
        pay for a rebuild — same buffer object, not just an equal-shaped one."""
        stream = _make_cfg_stream([16, 32], cfg_type="initialize", guidance_scale=1.4)
        updater = _make_cfg_updater(stream)
        buf_before = stream._cfg_latent_buf

        updater.update_stream_params(guidance_scale=1.6)

        assert stream._cfg_latent_buf is buf_before

    def test_full_cfg_type_also_covered(self):
        stream = _make_cfg_stream([16, 32], cfg_type="full", guidance_scale=1.0)
        updater = _make_cfg_updater(stream)

        updater.update_stream_params(guidance_scale=1.4)

        assert stream._cfg_latent_buf is not None
        assert stream._cfg_latent_buf.shape[0] == 2 * stream.batch_size

    def test_none_and_self_cfg_types_unaffected(self):
        """'none'/'self' never allocate these buffers regardless of guidance —
        crossing must not spuriously allocate them."""
        for cfg_type in ("none", "self"):
            stream = _make_cfg_stream([16, 32], cfg_type=cfg_type, guidance_scale=1.0)
            updater = _make_cfg_updater(stream)

            updater.update_stream_params(guidance_scale=1.4)

            assert stream._cfg_latent_buf is None
            assert stream._cfg_t_buf is None
