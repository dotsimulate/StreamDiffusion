"""
Regression tests for G4 + G6: unvalidated / partially-handled cfg_type values.

G6 root cause guarded: cfg_type is Literal-hinted in StreamDiffusion.__init__'s
signature but that's a static-analysis hint only -- nothing enforced it at
runtime. A typo (or a caller passing an arbitrary string) sailed through
construction and behaved like "none" until the guidance combine in
unet_step, then raised an opaque UnboundLocalError with no indication of
what was actually wrong. Fix: validate at pipeline.py's single choke point
(the one assignment to self.cfg_type, :85-91) against VALID_CFG_TYPES.

G4 root cause guarded: prepare()'s prompt-embedding setup only assigns
uncond_prompt_embeds under `use_denoising_batch and cfg_type == "full"` (or
unconditionally for "initialize"), but consumes it unconditionally whenever
guidance_scale > 1.0 and cfg_type is "initialize"/"full". The TCD scheduler
forces use_denoising_batch=False (pipeline.py:104-109), so {scheduler: tcd,
cfg_type: full} left uncond_prompt_embeds unbound -> UnboundLocalError deep
inside torch.cat. Fix: initialize uncond_prompt_embeds = None ahead of the
branches (both the SDXL and SD1.5/2.1 copies of this logic) and raise a
ValueError naming the unsupported combination at the consumption site,
instead of letting Python raise UnboundLocalError.

CPU-only, model-free. G6 is tested through the real StreamDiffusion.__init__
with a minimal fake pipe (only .vae_scale_factor is read before the cfg_type
check fires; detect_model() -- the next thing __init__ touches -- is never
reached in the rejection case). G4 is tested through the real prepare(), via
an object.__new__ stub wired with only the attributes prepare() reads before
its CFG-embeds block, plus a sentinel scheduler whose set_timesteps() raises
a marker exception -- this bounds each test to exactly the block under test
without needing to stand up the rest of prepare()'s scheduler machinery.
"""

import types

import pytest
import torch

from streamdiffusion.param_schema import VALID_CFG_TYPES
from streamdiffusion.pipeline import StreamDiffusion

TOKENS = 77
HIDDEN = 8


# ---------------------------------------------------------------------------
# G6 helpers
# ---------------------------------------------------------------------------


def _fake_pipe_for_init():
    """Only .vae_scale_factor is read before the G6 cfg_type check (:79-80);
    detect_model(pipe.unet, pipe) is the next thing __init__ touches, and it
    isn't reached when cfg_type is rejected."""
    return types.SimpleNamespace(vae_scale_factor=8)


# ---------------------------------------------------------------------------
# G4 helpers
# ---------------------------------------------------------------------------


class _StopHere(Exception):
    """Marks 'prepare() got past the CFG-embeds block cleanly' -- raised by
    the sentinel scheduler's set_timesteps(), the next thing prepare() calls
    after the block under test."""


class _SentinelScheduler:
    def set_timesteps(self, *a, **k):
        raise _StopHere()


def _fake_sd15_encode_prompt(**kwargs):
    return (torch.zeros(1, TOKENS, HIDDEN), torch.zeros(1, TOKENS, HIDDEN))


def _fake_sdxl_encode_prompt(**kwargs):
    return (
        torch.zeros(1, TOKENS, HIDDEN),
        torch.zeros(1, TOKENS, HIDDEN),
        torch.zeros(1, HIDDEN),
        torch.zeros(1, HIDDEN),
    )


def _make_prepare_stub(
    cfg_type,
    use_denoising_batch,
    is_sdxl=False,
    guidance_scale=1.4,
    batch_size=2,
    frame_bff_size=1,
    denoising_steps_num=2,
):
    """object.__new__ stub wired with exactly what prepare() reads up to (and
    including) the CFG-embeds block -- not the full prepare() surface."""
    stream = object.__new__(StreamDiffusion)
    stream.height = 512
    stream.width = 512
    stream.dtype = torch.float32
    stream.device = "cpu"
    stream.denoising_steps_num = denoising_steps_num
    stream.frame_bff_size = frame_bff_size
    stream.latent_height = 64
    stream.latent_width = 64
    stream.cfg_type = cfg_type
    stream.is_sdxl = is_sdxl
    stream.use_denoising_batch = use_denoising_batch
    stream.batch_size = batch_size
    stream.embedding_hooks = []
    stream.scheduler = _SentinelScheduler()
    stream.pipe = types.SimpleNamespace(
        encode_prompt=_fake_sdxl_encode_prompt if is_sdxl else _fake_sd15_encode_prompt
    )
    return stream


# ---------------------------------------------------------------------------
# G6 tests
# ---------------------------------------------------------------------------


class TestG6CfgTypeValidation:
    def test_unknown_cfg_type_raises_listing_valid_values(self):
        with pytest.raises(ValueError, match="cfg_type must be one of"):
            StreamDiffusion(pipe=_fake_pipe_for_init(), t_index_list=[16], device="cpu", cfg_type="bogus")

    def test_all_valid_cfg_types_pass_the_gate(self):
        """Must not raise G6's ValueError for any real cfg_type -- construction
        is expected to fail later (AttributeError on the fake pipe's missing
        .unet, inside detect_model) for reasons unrelated to this check."""
        for cfg_type in VALID_CFG_TYPES:
            try:
                StreamDiffusion(pipe=_fake_pipe_for_init(), t_index_list=[16], device="cpu", cfg_type=cfg_type)
            except ValueError as e:
                if "cfg_type must be one of" in str(e):
                    pytest.fail(f"valid cfg_type {cfg_type!r} rejected by the G6 gate")
                raise
            except AttributeError:
                pass  # expected: detect_model(pipe.unet, ...) -- fake pipe has no .unet


# ---------------------------------------------------------------------------
# G4 tests
# ---------------------------------------------------------------------------


class TestG4UncondPromptEmbedsUnboundLocal:
    def test_tcd_full_raises_legible_value_error_sd15(self):
        """The actual regression: {scheduler: tcd (-> use_denoising_batch=False),
        cfg_type: full} at guidance_scale > 1.0 must raise a ValueError naming
        the unsupported combination, not UnboundLocalError."""
        stream = _make_prepare_stub(cfg_type="full", use_denoising_batch=False, guidance_scale=1.4)
        with pytest.raises(ValueError, match="requires use_denoising_batch=True"):
            stream.prepare("a cat")

    def test_tcd_full_raises_legible_value_error_sdxl(self):
        """Same regression, SDXL branch (pipeline.py duplicates this logic)."""
        stream = _make_prepare_stub(cfg_type="full", use_denoising_batch=False, is_sdxl=True, guidance_scale=1.4)
        with pytest.raises(ValueError, match="requires use_denoising_batch=True"):
            stream.prepare("a cat")

    def test_lcm_full_reaches_scheduler_setup_without_valueerror(self):
        """Non-regression: {scheduler: lcm (-> use_denoising_batch=True),
        cfg_type: full} is the reachable, working combination -- must NOT
        raise G4's ValueError. Reaching the sentinel scheduler proves the
        CFG-embeds block (including the torch.cat) completed successfully."""
        stream = _make_prepare_stub(cfg_type="full", use_denoising_batch=True, guidance_scale=1.4)
        with pytest.raises(_StopHere):
            stream.prepare("a cat")

    def test_tcd_initialize_does_not_raise(self):
        """cfg_type='initialize' assigns uncond_prompt_embeds unconditionally
        (pipeline.py's elif has no use_denoising_batch guard) -- G4 does not
        affect this mode even under TCD. Non-regression guard."""
        stream = _make_prepare_stub(cfg_type="initialize", use_denoising_batch=False, guidance_scale=1.4)
        with pytest.raises(_StopHere):
            stream.prepare("a cat")

    def test_none_cfg_type_never_touches_uncond_branch(self):
        """cfg_type='none' never satisfies the guidance_scale>1.0-and-cfg-mode
        gate, so uncond_prompt_embeds staying None is never consumed."""
        stream = _make_prepare_stub(cfg_type="none", use_denoising_batch=False, guidance_scale=1.4)
        with pytest.raises(_StopHere):
            stream.prepare("a cat")
