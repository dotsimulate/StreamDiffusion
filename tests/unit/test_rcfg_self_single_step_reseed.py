"""
Regression tests for the R-CFG single-step divergence bug (black output at
guidance_scale > 1 with a 1-element t_index_list).

Root cause guarded: with cfg_type='self' and denoising_steps_num == 1, the
per-frame stock_noise reseed (predict_x0_batch ping-pong rotation) is gated
behind `denoising_steps_num > 1` and never runs, while unet_step's R-CFG
recurrence `stock_noise = _init_noise_rotated + delta_x` still executes with
degenerate coefficients (_alpha_next = _beta_next = 1.0). The resulting
cross-frame linear recurrence has growth factor |A| > 1 at typical single-step
timesteps, so stock_noise diverges geometrically (~x100/frame), overflows fp16
to Inf, and Inf - Inf = NaN in the CFG combine renders black frames.

Correct behavior per StreamDiffusion paper Eq. 5: at the first (and only)
denoising step, the Self-Negative virtual residual equals init_noise exactly,
so stock_noise must be reseeded from init_noise every frame when n == 1.

CPU-only, model-free. Builds a real StreamDiffusion instance via object.__new__
(same idiom as test_derived_tensor_sync.py) with a deterministic fake UNet and
drives the actual predict_x0_batch / unet_step code paths.
"""

import torch
from diffusers import LCMScheduler

from streamdiffusion.pipeline import StreamDiffusion

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fake_unet(sample, timestep, encoder_hidden_states=None, kvo_cache=None, return_dict=False, **kwargs):
    """Deterministic stand-in for the UNet (SD1.5 calling convention)."""
    return (0.05 * sample,)


def _make_stream(
    t_index_list,
    cfg_type="self",
    guidance_scale=1.4,
    delta=1.0,
    dtype=torch.float32,
    latent_hw=8,
    seed=1234,
):
    """Real StreamDiffusion instance (no __init__) wired with exactly the
    attributes predict_x0_batch / unet_step read, mirroring prepare()'s
    formulas. frame_bff_size fixed at 1 (the TD configuration)."""
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
    # G7/G8: _recalculate_timestep_dependent_params reads trt_unet_batch_size
    # unconditionally (it differs from batch_size for "initialize"/"full" --
    # see pipeline.py's __init__ formula, :113-118) -- the object.__new__ stub
    # must set it too, or the resize path raises AttributeError before ever
    # reaching the buffer-rebuild logic downstream tests exercise.
    if cfg_type == "initialize":
        stream.trt_unet_batch_size = (n + 1) * frame_bff_size
    elif cfg_type == "full":
        stream.trt_unet_batch_size = 2 * n * frame_bff_size
    else:
        stream.trt_unet_batch_size = n * frame_bff_size
    stream.guidance_scale = guidance_scale
    stream.delta = delta
    stream.do_add_noise = True
    stream.generator = torch.Generator(device="cpu").manual_seed(seed)

    # UNet plumbing: SD1.5 branch, hooks/caches disabled
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
    stream.unet = _fake_unet
    stream.update_kvo_cache = lambda *a, **k: None
    stream.prompt_embeds = torch.zeros(batch_size, 77, 8, dtype=dtype)

    # Scheduler shell with a real cumulative-alpha schedule (matches
    # test_derived_tensor_sync.py's mock; isinstance(LCMScheduler) must hold)
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

    # LCM boundary-condition scalings (diffusers formulas, sigma_data=0.5,
    # timestep_scaling=10): c_skip ~ 0, c_out ~ 1 at these timesteps
    c_skip_l, c_out_l = [], []
    for t in sub_timesteps:
        st = 10.0 * t
        c_skip_l.append(0.25 / (st**2 + 0.25))
        c_out_l.append(st / (st**2 + 0.25) ** 0.5)
    stream.c_skip = torch.tensor(c_skip_l, dtype=dtype).view(n, 1, 1, 1)
    stream.c_out = torch.tensor(c_out_l, dtype=dtype).view(n, 1, 1, 1)

    # Noise state + ping-pong buffers, exactly as prepare() builds them
    stream.init_noise = torch.randn((batch_size, 4, h, w), dtype=dtype, generator=stream.generator)
    stream.stock_noise = stream.init_noise.clone()
    stream._stock_noise_bufs = [stream.stock_noise.clone(), torch.empty_like(stream.stock_noise)]
    stream._stock_noise_pong = 0

    # Derived shifted tensors (degenerate ones-padding at n == 1)
    if cfg_type in ("self", "initialize"):
        stream._alpha_next = torch.cat(
            [stream.alpha_prod_t_sqrt[1:], torch.ones_like(stream.alpha_prod_t_sqrt[0:1])], dim=0
        )
        stream._beta_next = torch.cat(
            [stream.beta_prod_t_sqrt[1:], torch.ones_like(stream.beta_prod_t_sqrt[0:1])], dim=0
        )
        stream._init_noise_rotated = torch.cat([stream.init_noise[1:], stream.init_noise[0:1]], dim=0)
    else:
        stream._alpha_next = None
        stream._beta_next = None
        stream._init_noise_rotated = None

    # Latent buffers
    stream.x_t_latent_buffer = None if n == 1 else torch.zeros((n - 1) * frame_bff_size, 4, h, w, dtype=dtype)
    stream._combined_latent_buf = None if n == 1 else torch.empty(batch_size, 4, h, w, dtype=dtype)

    # CFG expansion buffers (only allocated for the cfg types that use them)
    if guidance_scale > 1.0 and cfg_type in ("initialize", "full"):
        cfg_batch = (1 + batch_size) if cfg_type == "initialize" else (2 * batch_size)
        stream._cfg_latent_buf = torch.empty(cfg_batch, 4, h, w, dtype=dtype)
        stream._cfg_t_buf = torch.empty(cfg_batch, dtype=stream.sub_timesteps_tensor.dtype)
    else:
        stream._cfg_latent_buf = None
        stream._cfg_t_buf = None

    return stream


def _zero_input(stream):
    return torch.zeros(stream.frame_bff_size, 4, stream.latent_height, stream.latent_width, dtype=stream.dtype)


def _run_frames(stream, num_frames):
    """Drive predict_x0_batch with a constant zero input latent; returns the
    per-frame x0 outputs. NOTE: runs all frames up front — use an inline loop
    instead when asserting per-frame stream state."""
    x_in = _zero_input(stream)
    return [StreamDiffusion.predict_x0_batch(stream, x_in) for _ in range(num_frames)]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestSingleStepReseedFixesDivergence:
    """n == 1, cfg_type='self', guidance > 1: stock_noise must be pinned to
    init_noise every frame and stay bounded/finite. FAILS on unpatched code
    (observed geometric blow-up ~x100/frame); PASSES once predict_x0_batch's
    n == 1 reseed and the unet_step recurrence gate land."""

    def test_stock_noise_pinned_to_init_noise_every_frame(self):
        stream = _make_stream([16], cfg_type="self", guidance_scale=1.4)
        x_in = _zero_input(stream)
        for frame in range(8):
            x0 = StreamDiffusion.predict_x0_batch(stream, x_in)
            max_abs = stream.stock_noise.abs().max().item()
            assert torch.allclose(stream.stock_noise, stream.init_noise), (
                f"frame {frame}: stock_noise diverged from init_noise (max_abs={max_abs:.3e}) — "
                "n==1 per-frame reseed missing"
            )
            # copy_, not alias: init_noise must never be corrupted through stock_noise
            assert stream.stock_noise.data_ptr() != stream.init_noise.data_ptr()
            assert max_abs < 100.0, f"frame {frame}: stock_noise unbounded (max_abs={max_abs:.3e})"
            assert torch.isfinite(x0).all(), f"frame {frame}: non-finite x0 output"

    def test_divergence_gone_even_at_guidance_1(self):
        """The recurrence ran (and silently diverged) even at guidance 1.0 —
        raising guidance later merely exposed the already-corrupt buffer."""
        stream = _make_stream([16], cfg_type="self", guidance_scale=1.0)
        x_in = _zero_input(stream)
        for frame in range(8):
            x0 = StreamDiffusion.predict_x0_batch(stream, x_in)
            max_abs = stream.stock_noise.abs().max().item()
            assert max_abs < 100.0, (
                f"frame {frame}: stock_noise diverging in background at guidance 1.0 (max_abs={max_abs:.3e})"
            )
            assert torch.isfinite(x0).all()

    def test_initialize_cfg_type_bounded(self):
        """'initialize' self-heals at n==1 (real uncond overwrites the single
        slot each frame) — must stay bounded before and after the fix."""
        stream = _make_stream([16], cfg_type="initialize", guidance_scale=1.4)
        x_in = _zero_input(stream)
        for frame in range(8):
            x0 = StreamDiffusion.predict_x0_batch(stream, x_in)
            max_abs = stream.stock_noise.abs().max().item()
            assert max_abs < 100.0, f"frame {frame}: stock_noise unbounded (max_abs={max_abs:.3e})"
            assert torch.isfinite(x0).all()

    def test_full_and_none_cfg_types_unaffected(self):
        """'full'/'none' never synthesize uncond from stock_noise — just assert
        the n==1 path runs clean and finite."""
        for cfg_type in ("full", "none"):
            stream = _make_stream([16], cfg_type=cfg_type, guidance_scale=1.4)
            for x0 in _run_frames(stream, 4):
                assert torch.isfinite(x0).all(), f"cfg_type={cfg_type}: non-finite x0"


class TestMultiStepPathUnaffectedByReseed:
    """n == 2 ping-pong path must be untouched by the n == 1 fix (the new code
    sits in an elif only reachable when denoising_steps_num == 1)."""

    def test_ping_pong_still_engages_and_stays_stable(self):
        stream = _make_stream([16, 32], cfg_type="self", guidance_scale=1.4)
        x_in = _zero_input(stream)
        expected_pong = 0
        for frame in range(6):
            x0 = StreamDiffusion.predict_x0_batch(stream, x_in)
            expected_pong = 1 - expected_pong
            assert stream._stock_noise_pong == expected_pong, (
                f"frame {frame}: ping-pong rotation did not run (n==1 branch leaked into n==2?)"
            )
            assert stream.x_t_latent_buffer is not None
            # The n==1 invariant (stock_noise pinned to init_noise) must NOT leak here
            assert not torch.allclose(stream.stock_noise, stream.init_noise)
            max_abs = stream.stock_noise.abs().max().item()
            assert max_abs < 10_000.0, f"frame {frame}: n==2 stock_noise unstable (max_abs={max_abs:.3e})"
            assert torch.isfinite(x0).all()

    def test_multi_step_outputs_deterministic_and_finite(self):
        """Seeded n==2 run is reproducible — used with capture_reference.py-style
        pre/post-fix diffing to pin the multi-step path bit-identical."""
        outs_a = _run_frames(_make_stream([16, 32], cfg_type="self", guidance_scale=1.4, seed=77), 6)
        outs_b = _run_frames(_make_stream([16, 32], cfg_type="self", guidance_scale=1.4, seed=77), 6)
        for a, b in zip(outs_a, outs_b):
            assert torch.equal(a, b), "seeded n==2 run not deterministic"
            assert torch.isfinite(a).all()
