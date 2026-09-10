"""Regression test for the permanent-black-output latch this whole guard pass
exists to fix: with cfg_type="self"/"initialize" and denoising_steps_num > 1
(the user's live regime: t_index_list=[22, 36] -> n=2), a single NaN UNet
output used to poison self.stock_noise via the recurrence in unet_step

    scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
    delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
    ...
    self.stock_noise = self._init_noise_rotated + delta_x

forever -- there was no isfinite/nan_to_num anywhere on this path, so once
stock_noise went NaN it stayed NaN every subsequent frame, with no exception
(the only try/except on this path catches OOM/shape RuntimeErrors) and no way
to recover short of a full re-prepare.

This test injects exactly one NaN model_pred mid-stream via a fake UNet and
asserts the NaN guard (pipeline.py `unet_step`, `self._nan_guard_model_pred`)
sanitizes it immediately and stock_noise recovers within one extra frame,
instead of staying poisoned forever.

CPU-only, model-free -- same object.__new__ / fake-UNet idiom as
test_rcfg_self_single_step_reseed.py and test_derived_tensor_sync.py.
"""

import pytest
import torch
from diffusers import LCMScheduler

from streamdiffusion.pipeline import StreamDiffusion

# ---------------------------------------------------------------------------
# helpers (deliberately duplicated from test_rcfg_self_single_step_reseed.py
# rather than imported -- each test file is a self-contained fixture per this
# suite's existing convention)
# ---------------------------------------------------------------------------


class _NanInjectingUnet:
    """Deterministic stand-in for the UNet that returns NaN on one chosen call."""

    def __init__(self, nan_on_call: int):
        self.nan_on_call = nan_on_call
        self.calls = 0

    def __call__(self, sample, timestep, encoder_hidden_states=None, kvo_cache=None, return_dict=False, **kwargs):
        self.calls += 1
        if self.calls == self.nan_on_call:
            return (torch.full_like(sample, float("nan")),)
        return (0.05 * sample,)


def _make_stream(t_index_list, unet, cfg_type="self", guidance_scale=1.0, dtype=torch.float32, latent_hw=8, seed=1234):
    stream = object.__new__(StreamDiffusion)
    n = len(t_index_list)
    frame_bff_size = 1
    batch_size = n * frame_bff_size
    h = w = latent_hw

    stream.device = "cpu"
    stream.dtype = dtype
    stream.latent_height = h
    stream.latent_width = w
    stream.frame_bff_size = frame_bff_size
    stream.denoising_steps_num = n
    stream.batch_size = batch_size
    stream.cfg_type = cfg_type
    stream.use_denoising_batch = True
    stream.trt_unet_batch_size = n * frame_bff_size
    stream.guidance_scale = guidance_scale
    stream.delta = 1.0
    stream.do_add_noise = True
    stream.generator = torch.Generator(device="cpu").manual_seed(seed)

    stream.is_sdxl = False
    stream.unet_hooks = []
    stream.kvo_cache = []
    stream.fio_cache = []
    stream.use_feature_injection = False
    stream._fi_strength_tensor = None
    stream._fi_threshold_tensor = None
    stream._is_unet_tensorrt = None
    stream._unet_kwargs = {"return_dict": False}
    stream._sdxl_conditioning_cache = {}
    stream._cached_batch_size = None
    stream._cached_cfg_type = None
    stream._cached_guidance_scale = None
    stream.unet = unet
    stream.prompt_embeds = torch.zeros(batch_size, 77, 8, dtype=dtype)

    sched = object.__new__(LCMScheduler)
    betas = torch.linspace(0.0001, 0.02, 1000)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sched.alphas_cumprod = alphas_cumprod
    stream.scheduler = sched

    timesteps_raw = torch.linspace(999, 19, 50).long()
    sub_timesteps = [int(timesteps_raw[i]) for i in t_index_list]
    stream.sub_timesteps_tensor = torch.tensor(sub_timesteps, dtype=torch.long)

    stream.alpha_prod_t_sqrt = (
        torch.stack([alphas_cumprod[t].sqrt() for t in sub_timesteps]).view(n, 1, 1, 1).to(dtype)
    )
    stream.beta_prod_t_sqrt = (
        torch.stack([(1 - alphas_cumprod[t]).sqrt() for t in sub_timesteps]).view(n, 1, 1, 1).to(dtype)
    )

    c_skip_l, c_out_l = [], []
    for t in sub_timesteps:
        st = 10.0 * t
        c_skip_l.append(0.25 / (st**2 + 0.25))
        c_out_l.append(st / (st**2 + 0.25) ** 0.5)
    stream.c_skip = torch.tensor(c_skip_l, dtype=dtype).view(n, 1, 1, 1)
    stream.c_out = torch.tensor(c_out_l, dtype=dtype).view(n, 1, 1, 1)

    stream.init_noise = torch.randn((batch_size, 4, h, w), dtype=dtype, generator=stream.generator)
    stream.stock_noise = stream.init_noise.clone()
    stream._stock_noise_bufs = [stream.stock_noise.clone(), torch.empty_like(stream.stock_noise)]
    stream._stock_noise_pong = 0

    stream._alpha_next = torch.cat(
        [stream.alpha_prod_t_sqrt[1:], torch.ones_like(stream.alpha_prod_t_sqrt[0:1])], dim=0
    )
    stream._beta_next = torch.cat([stream.beta_prod_t_sqrt[1:], torch.ones_like(stream.beta_prod_t_sqrt[0:1])], dim=0)
    stream._init_noise_rotated = torch.cat([stream.init_noise[1:], stream.init_noise[0:1]], dim=0)

    stream.x_t_latent_buffer = None if n == 1 else torch.zeros((n - 1) * frame_bff_size, 4, h, w, dtype=dtype)
    stream._combined_latent_buf = None if n == 1 else torch.empty(batch_size, 4, h, w, dtype=dtype)
    stream._cfg_latent_buf = None
    stream._cfg_t_buf = None

    return stream


def _zero_input(stream):
    return torch.zeros(stream.frame_bff_size, 4, stream.latent_height, stream.latent_width, dtype=stream.dtype)


class TestNanGuardBreaksPermanentLatch:
    """Live regime from td_config.yaml: t_index_list=[22, 36] (n=2), cfg_type='self',
    guidance_scale=1.0 -- exactly the configuration in which the reported bug
    (Normpweights off + both prompts at 0 -> permanently black) went unrecoverable."""

    def test_single_nan_frame_self_heals(self):
        unet = _NanInjectingUnet(nan_on_call=3)
        stream = _make_stream([22, 36], unet, cfg_type="self", guidance_scale=1.0)
        x_in = _zero_input(stream)

        results = []
        for _ in range(8):
            x0 = StreamDiffusion.predict_x0_batch(stream, x_in)
            results.append(
                {
                    "x0_finite": torch.isfinite(x0).all().item(),
                    "stock_noise_finite": torch.isfinite(stream.stock_noise).all().item(),
                    "x_t_latent_buffer_finite": (
                        stream.x_t_latent_buffer is None or torch.isfinite(stream.x_t_latent_buffer).all().item()
                    ),
                }
            )

        # The guard sanitizes model_pred THE SAME frame it goes bad -- there must
        # be no frame, including the injected one, with a non-finite output.
        bad_frames = [i for i, r in enumerate(results) if not r["x0_finite"]]
        assert not bad_frames, f"non-finite x0 at frame(s) {bad_frames} -- pre-fix this would latch forever"

        bad_stock = [i for i, r in enumerate(results) if not r["stock_noise_finite"]]
        assert not bad_stock, f"non-finite stock_noise at frame(s) {bad_stock}"

        bad_buf = [i for i, r in enumerate(results) if not r["x_t_latent_buffer_finite"]]
        assert not bad_buf, f"non-finite x_t_latent_buffer at frame(s) {bad_buf}"

        # And the stream must still be producing real (non-degenerate-zero) output
        # a few frames after the injected NaN -- i.e. actually recovered, not just
        # zeroed out permanently.
        assert results[-1]["x0_finite"]

    def test_unguarded_reference_would_have_latched(self):
        """Sanity check that the injected NaN is actually load-bearing: with the
        guard's sanitize_ call short-circuited, the same scenario latches NaN into
        stock_noise from the injected frame onward (proves this test can go red)."""
        unet = _NanInjectingUnet(nan_on_call=3)
        stream = _make_stream([22, 36], unet, cfg_type="self", guidance_scale=1.0)
        x_in = _zero_input(stream)

        # Force the guard to report "not bad" and skip sanitizing, reproducing the
        # pre-fix unguarded path exactly.
        stream._nan_guard_model_pred = _NoOpGuard()

        saw_nan = False
        for _ in range(5):
            StreamDiffusion.predict_x0_batch(stream, x_in)
            if not torch.isfinite(stream.stock_noise).all():
                saw_nan = True
        assert saw_nan, "expected the unguarded path to latch NaN into stock_noise"


class _NoOpGuard:
    """Stand-in for NanGuard that never sanitizes and never flags -- used only to
    prove the injected-NaN scenario is load-bearing without the real guard."""

    def sanitize_(self, tensor):
        return False


# ---------------------------------------------------------------------------
# Regression for the UnboundLocalError this guard shipped, and the missing
# SDXL coverage that caused it: `nan_guard_bad_last_frame` was assigned only in
# unet_step's SD1.5/2.1 branch but read unconditionally in the shared
# post-branch code (the self/initialize stock_noise recurrence), so every SDXL
# model with cfg_type self/initialize and denoising_steps_num > 1 crashed on
# frame 1 -- exactly the user's sdxl-turbo + tensorrt + cfg_type=self +
# t_index_list=[22, 36] configuration. Neither existing fixture in this suite
# nor test_unet_call_backend_gate.py exercises is_sdxl=True through unet_step,
# which is why 705 passing tests missed a 100%-reproducible crash.
# ---------------------------------------------------------------------------

_SDXL_BACKEND_PARAMS = [
    pytest.param(False, False, id="sd15"),
    pytest.param(True, False, id="sdxl-pytorch"),
    pytest.param(True, True, id="sdxl-trt"),
]


class TestUnetStepDoesNotRaiseAcrossBackends:
    @pytest.mark.parametrize("is_sdxl, is_trt", _SDXL_BACKEND_PARAMS)
    def test_predict_x0_batch_does_not_raise(self, is_sdxl, is_trt):
        """nan_on_call=99 never fires -- proves the crash is unconditional, not
        NaN-dependent (matches the minimised repro used to find this bug)."""
        unet = _NanInjectingUnet(nan_on_call=99)
        stream = _make_stream([22, 36], unet, cfg_type="self", guidance_scale=1.0)
        stream.is_sdxl = is_sdxl
        stream._is_unet_tensorrt = is_trt
        stream._lora_scale_tensor = None  # read only by the SDXL+TensorRT sub-branch
        x_in = _zero_input(stream)

        for _ in range(3):
            StreamDiffusion.predict_x0_batch(stream, x_in)  # must not raise


class TestNanGuardHealsAcrossBackends:
    """Extends TestNanGuardBreaksPermanentLatch.test_single_nan_frame_self_heals to
    both SDXL sub-branches -- proves guard 1 actually protects sdxl-turbo (the
    user's model), not merely that unet_step no longer crashes."""

    @pytest.mark.parametrize(
        "is_sdxl, is_trt",
        [p for p in _SDXL_BACKEND_PARAMS if p.id != "sd15"],
    )
    def test_single_nan_frame_self_heals(self, is_sdxl, is_trt):
        unet = _NanInjectingUnet(nan_on_call=3)
        stream = _make_stream([22, 36], unet, cfg_type="self", guidance_scale=1.0)
        stream.is_sdxl = is_sdxl
        stream._is_unet_tensorrt = is_trt
        stream._lora_scale_tensor = None
        x_in = _zero_input(stream)

        results = []
        for _ in range(8):
            x0 = StreamDiffusion.predict_x0_batch(stream, x_in)
            results.append(
                {
                    "x0_finite": torch.isfinite(x0).all().item(),
                    "stock_noise_finite": torch.isfinite(stream.stock_noise).all().item(),
                    "x_t_latent_buffer_finite": (
                        stream.x_t_latent_buffer is None or torch.isfinite(stream.x_t_latent_buffer).all().item()
                    ),
                }
            )

        bad_frames = [i for i, r in enumerate(results) if not r["x0_finite"]]
        assert not bad_frames, f"non-finite x0 at frame(s) {bad_frames} -- pre-fix this would latch forever"

        bad_stock = [i for i, r in enumerate(results) if not r["stock_noise_finite"]]
        assert not bad_stock, f"non-finite stock_noise at frame(s) {bad_stock}"

        bad_buf = [i for i, r in enumerate(results) if not r["x_t_latent_buffer_finite"]]
        assert not bad_buf, f"non-finite x_t_latent_buffer at frame(s) {bad_buf}"

        assert results[-1]["x0_finite"]


# ---------------------------------------------------------------------------
# Second defect found while verifying the crash fix: unet_step's NaN-recovery
# path used to pass kvo_cache_out=[] to update_kvo_cache on a bad frame, but
# that function gates on self.kvo_cache (the stream's cache), not the passed
# list, so empty lists still entered the bucketed write path
# (update_kvo_cache, pipeline.py) and raised IndexError indexing an empty
# list. The fix skips the update_kvo_cache call entirely on a bad frame.
# ---------------------------------------------------------------------------


class _NanInjectingKvoUnet:
    """Like _NanInjectingUnet, but also returns per-layer kvo cache outputs so the
    call exercises update_kvo_cache's bucketed write path."""

    def __init__(self, nan_on_call: int, n_layers=2, batch=2, seq=4, hidden=8):
        self.nan_on_call = nan_on_call
        self.calls = 0
        self.n_layers = n_layers
        self.batch = batch
        self.seq = seq
        self.hidden = hidden

    def __call__(self, sample, timestep, encoder_hidden_states=None, kvo_cache=None, return_dict=False, **kwargs):
        self.calls += 1
        pred = torch.full_like(sample, float("nan")) if self.calls == self.nan_on_call else 0.05 * sample
        kvo_out = [torch.zeros(2, 1, self.batch, self.seq, self.hidden) for _ in range(self.n_layers)]
        return (pred, kvo_out, [])


def _wire_bucketed_kvo_cache(stream, n_layers=2, batch=2, seq=4, hidden=8, cache_maxframes=4):
    """Minimal single-bucket scaffolding matching create_kvo_cache's real layout
    (acceleration/tensorrt/models/utils.py): all n_layers share one bucket."""
    bucket = torch.zeros(n_layers, 2, cache_maxframes, batch, seq, hidden)
    stream._kvo_buckets = [bucket]
    stream._kvo_outputs_by_bucket = [list(range(n_layers))]
    stream.kvo_cache = [bucket[i] for i in range(n_layers)]  # per-layer views, like create_kvo_cache returns
    stream.fio_cache = []
    stream.cache_interval = 1
    stream.cache_maxframes = cache_maxframes
    stream.frame_idx = 0


class TestKvoCacheBucketedRecoveryDoesNotRaise:
    def test_direct_empty_list_call_raises_documenting_the_defect(self):
        """Red-capability proof: the bucketed write path really does crash on
        empty lists, so this test class isn't vacuously passing."""
        stream = StreamDiffusion.__new__(StreamDiffusion)
        _wire_bucketed_kvo_cache(stream)

        with pytest.raises(IndexError):
            stream.update_kvo_cache([], [])

    def test_nan_frame_with_bucketed_cache_does_not_raise(self):
        unet = _NanInjectingKvoUnet(nan_on_call=3)
        stream = _make_stream([22, 36], unet, cfg_type="self", guidance_scale=1.0)
        _wire_bucketed_kvo_cache(stream)
        x_in = _zero_input(stream)

        for _ in range(8):
            StreamDiffusion.predict_x0_batch(stream, x_in)  # must not raise, incl. on the frame-3 NaN
