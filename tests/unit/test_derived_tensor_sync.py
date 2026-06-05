"""
Regression tests for F2: _update_timestep_calculations must refresh pre-computed
shifted tensors (_alpha_next, _beta_next, _init_noise_rotated) after a value-only
t_index_list update.

Root cause guarded: _alpha_next / _beta_next are built only in prepare() and the
error-fallback _refresh_derived_tensors(); without F2 they stay stale when the user
changes Tindexblockstep at runtime, causing incorrect stock_noise rotation at
guidance > 1.0 (RCFG-self path, pipeline.py:979-984).

CPU-only, model-free.  Constructs a minimal stream shell via object.__new__ and
wires only the attributes StreamParameterUpdater._update_timestep_calculations reads.
"""

import torch
import pytest

from streamdiffusion.stream_parameter_updater import StreamParameterUpdater


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_mock_lcm_scheduler(num_steps=50, device="cpu"):
    """Minimal scheduler shell with alphas_cumprod and get_scalings_for_boundary_condition_discrete."""
    import types
    from diffusers import LCMScheduler

    sched = object.__new__(LCMScheduler)
    # Use a real alphas_cumprod from cosine schedule (avoids needing model weights)
    betas = torch.linspace(0.0001, 0.02, 1000)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sched.alphas_cumprod = alphas_cumprod

    # Provide get_scalings_for_boundary_condition_discrete for c_skip/c_out
    def _scalings(timestep):
        t = timestep if isinstance(timestep, torch.Tensor) else torch.tensor(float(timestep))
        sigma = ((1 - alphas_cumprod[int(t.item())]) / alphas_cumprod[int(t.item())]).sqrt()
        c_skip = 1.0 / (sigma**2 + 1.0)
        c_out = -sigma / (sigma**2 + 1.0).sqrt()
        return torch.tensor(c_skip), torch.tensor(c_out)

    sched.get_scalings_for_boundary_condition_discrete = _scalings
    return sched


def _make_stream_shell(t_index_list, device="cpu", dtype=torch.float32,
                       frame_bff_size=1, use_denoising_batch=True,
                       cfg_type="self", do_add_noise=True):
    """Minimal StreamDiffusion pipeline shell for updater testing."""
    import types
    from diffusers import LCMScheduler

    stream = types.SimpleNamespace()
    stream.device = device
    stream.dtype = dtype
    stream.frame_bff_size = frame_bff_size
    stream.use_denoising_batch = use_denoising_batch
    stream.cfg_type = cfg_type
    stream.do_add_noise = do_add_noise
    stream.batch_size = 1
    stream.latent_height = 64
    stream.latent_width = 64
    stream.generator = torch.Generator(device=device)

    num_steps = 50
    stream.scheduler = _make_mock_lcm_scheduler(num_steps)
    # Build timesteps as a linear space (same as LCM set_timesteps for num_steps=50)
    betas = torch.linspace(0.0001, 0.02, 1000)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    timesteps_raw = torch.linspace(999, 19, num_steps).long()
    stream.timesteps = timesteps_raw

    stream.t_list = t_index_list
    stream.sub_timesteps = [int(timesteps_raw[i]) for i in t_index_list]
    stream.sub_timesteps_tensor = torch.tensor(stream.sub_timesteps, dtype=torch.long)

    # Build initial alpha/beta matching the logic in _update_timestep_calculations
    a_list, b_list = [], []
    for t in stream.sub_timesteps:
        a_list.append(alphas_cumprod[t].sqrt())
        b_list.append((1 - alphas_cumprod[t]).sqrt())
    alpha_raw = torch.stack(a_list).view(len(t_index_list), 1, 1, 1).to(dtype=dtype)
    beta_raw = torch.stack(b_list).view(len(t_index_list), 1, 1, 1).to(dtype=dtype)
    stream.alpha_prod_t_sqrt = alpha_raw.repeat_interleave(frame_bff_size, dim=0)
    stream.beta_prod_t_sqrt = beta_raw.repeat_interleave(frame_bff_size, dim=0)

    stream.c_skip = torch.ones(len(t_index_list) * frame_bff_size, 1, 1, 1, dtype=dtype)
    stream.c_out = torch.ones(len(t_index_list) * frame_bff_size, 1, 1, 1, dtype=dtype)

    # init_noise / stock_noise
    h, w = stream.latent_height, stream.latent_width
    stream.init_noise = torch.randn(
        (len(t_index_list) * frame_bff_size, 4, h, w), dtype=dtype, generator=stream.generator
    )
    stream.stock_noise = stream.init_noise.clone()

    # _alpha_next / _beta_next / _init_noise_rotated — only set when denoising batch + RCFG-self
    if use_denoising_batch and (cfg_type == "self" or cfg_type == "initialize"):
        stream._alpha_next = torch.cat(
            [stream.alpha_prod_t_sqrt[1:], torch.ones_like(stream.alpha_prod_t_sqrt[0:1])], dim=0
        )
        stream._beta_next = torch.cat(
            [stream.beta_prod_t_sqrt[1:], torch.ones_like(stream.beta_prod_t_sqrt[0:1])], dim=0
        )
        stream._init_noise_rotated = torch.cat(
            [stream.init_noise[1:], stream.init_noise[0:1]], dim=0
        )
    else:
        stream._alpha_next = None
        stream._beta_next = None
        stream._init_noise_rotated = None

    return stream


def _make_updater(stream):
    """Construct StreamParameterUpdater without calling __init__ (avoids deps)."""
    updater = object.__new__(StreamParameterUpdater)
    updater.stream = stream
    updater._lock = __import__("threading").Lock()
    return updater


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestDerivedTensorSync:
    """F2: _update_timestep_calculations must keep _alpha_next/_beta_next in sync."""

    def test_alpha_next_updated_after_t_index_change(self):
        """After a same-length value-only t_index change, _alpha_next must equal
        cat([alpha_prod_t_sqrt[1:], ones]) — not the stale pre-change value."""
        stream = _make_stream_shell([14, 36])
        updater = _make_updater(stream)

        # Capture old value
        old_alpha_next = stream._alpha_next.clone()

        # Change t_index values (same length, different values)
        updater._update_timestep_values_only([14, 28])

        expected = torch.cat(
            [stream.alpha_prod_t_sqrt[1:], torch.ones_like(stream.alpha_prod_t_sqrt[0:1])], dim=0
        )
        assert not torch.allclose(old_alpha_next, stream._alpha_next), \
            "_alpha_next was not updated (stale)"
        assert torch.allclose(stream._alpha_next, expected, atol=1e-5), \
            f"_alpha_next mismatch: max_diff={( stream._alpha_next - expected).abs().max().item():.6f}"

    def test_beta_next_updated_after_t_index_change(self):
        """After a same-length value-only t_index change, _beta_next must equal
        cat([beta_prod_t_sqrt[1:], ones])."""
        stream = _make_stream_shell([14, 36])
        updater = _make_updater(stream)
        old_beta_next = stream._beta_next.clone()

        updater._update_timestep_values_only([14, 28])

        expected = torch.cat(
            [stream.beta_prod_t_sqrt[1:], torch.ones_like(stream.beta_prod_t_sqrt[0:1])], dim=0
        )
        assert not torch.allclose(old_beta_next, stream._beta_next), \
            "_beta_next was not updated (stale)"
        assert torch.allclose(stream._beta_next, expected, atol=1e-5), \
            f"_beta_next mismatch: max_diff={(stream._beta_next - expected).abs().max().item():.6f}"

    def test_init_noise_rotated_stays_consistent_after_t_index_change(self):
        """_init_noise_rotated must equal cat([init_noise[1:], init_noise[0:1]])
        after a value-only t_index update (init_noise itself is unchanged)."""
        stream = _make_stream_shell([14, 36])
        updater = _make_updater(stream)
        saved_init_noise = stream.init_noise.clone()

        updater._update_timestep_values_only([14, 28])

        # init_noise should be unchanged
        assert torch.allclose(stream.init_noise, saved_init_noise), \
            "init_noise was unexpectedly mutated by _update_timestep_values_only"

        expected_rotated = torch.cat(
            [stream.init_noise[1:], stream.init_noise[0:1]], dim=0
        )
        assert torch.allclose(stream._init_noise_rotated, expected_rotated, atol=1e-6), \
            "_init_noise_rotated out of sync with init_noise after t_index update"

    def test_no_update_when_derived_tensors_not_initialized(self):
        """When _alpha_next is None (non-batched or non-RCFG-self), updater must
        leave it None — not attempt to update."""
        stream = _make_stream_shell([14, 36], use_denoising_batch=False, cfg_type="none")
        updater = _make_updater(stream)

        assert stream._alpha_next is None
        updater._update_timestep_values_only([14, 28])
        assert stream._alpha_next is None, "_alpha_next should remain None for non-RCFG-self config"

    def test_warn_on_do_add_noise_false_high_beta(self, caplog):
        """When do_add_noise=False and inter-step beta_sqrt > 0.75, a warning must be logged."""
        import logging
        stream = _make_stream_shell([14, 28], do_add_noise=False)
        updater = _make_updater(stream)

        with caplog.at_level(logging.WARNING, logger="streamdiffusion.stream_parameter_updater"):
            updater._update_timestep_values_only([14, 28])

        assert any("do_add_noise=False" in r.message for r in caplog.records), \
            "Expected do_add_noise bleed-risk warning not emitted"

    def test_no_warn_when_do_add_noise_true(self, caplog):
        """When do_add_noise=True, no bleed-risk warning should be logged."""
        import logging
        stream = _make_stream_shell([14, 28], do_add_noise=True)
        updater = _make_updater(stream)

        with caplog.at_level(logging.WARNING, logger="streamdiffusion.stream_parameter_updater"):
            updater._update_timestep_values_only([14, 28])

        bleed_warns = [r for r in caplog.records if "do_add_noise=False" in r.message]
        assert not bleed_warns, f"Unexpected bleed-risk warning when do_add_noise=True: {bleed_warns}"
