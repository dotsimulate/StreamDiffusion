"""
Regression tests for ControlNet residual decay (cn_cache_decay).

CPU-only and model-free, following test_controlnet_residual_merge.py conventions:
fake ControlNet callables stand in for real engines so build_unet_hook()'s closure
can be exercised without CUDA/TRT.

Behavior under lock:
  - decay == 0.0 (default): byte-identical legacy path -- version-gated hold, so a
    live feed (per-frame _images_version bump) recomputes every frame and no EMA
    buffer is ever allocated.
  - decay > 0.0: the interval becomes authoritative (image-version bumps no longer
    invalidate the hold) and every frame the applied residual moves toward the
    newest computed one via applied.lerp_(target, decay).
  - Forced recompute regardless of schedule: scale-hash change, active-set change
    (enable/disable toggle, add/remove, same-index model swap -- the id(cn)-keyed
    active_key regression lock), empty cache.
  - EMA buffers are module-owned and pointer-stable; the engine's own output
    tensors (which the single-CN _result aliases on the TRT path) are never
    mutated by the EMA math.

ASCII only -- no Unicode symbols (Windows cp1252 terminal compatibility).
"""

from typing import Any, List, cast

import torch

from streamdiffusion.hooks import StepCtx, UnetKwargsDelta
from streamdiffusion.modules.controlnet_module import ControlNetModule

DOWN_SHAPES = [(1, 4, 8, 8), (1, 4, 4, 4)]
MID_SHAPE = (1, 4, 2, 2)


class _RampingCN:
    """Fake CN whose residual value ramps with each call (1.0, 2.0, ...) so EMA
    sequences are hand-checkable. Records the tensors it returned so tests can
    verify the EMA math never mutates or aliases engine outputs."""

    def __init__(self, down_shapes=None, mid_shape=MID_SHAPE):
        self.calls = 0
        self._down_shapes = down_shapes if down_shapes is not None else DOWN_SHAPES
        self._mid_shape = mid_shape
        self.returned = []  # (down_list, mid) per call

    def __call__(
        self, sample, timestep, encoder_hidden_states, controlnet_cond, conditioning_scale, return_dict=False
    ):
        self.calls += 1
        v = float(self.calls)
        down = [torch.full(shape, v, dtype=torch.float32) for shape in self._down_shapes]
        mid = torch.full(self._mid_shape, v, dtype=torch.float32)
        self.returned.append((down, mid))
        return down, mid


class _FakeStream:
    def __init__(self, text_len: int = 77, batch: int = 1):
        self.prompt_embeds = torch.randn(batch, text_len, 8)


def _make_module_with_controlnets(*cns) -> ControlNetModule:
    module = ControlNetModule(device="cpu", dtype=torch.float32)
    module._stream = _FakeStream()
    module.controlnets = list(cns)
    module.controlnet_images = [torch.randn(1, 3, 8, 8) for _ in cns]
    module.controlnet_scales = [1.0 for _ in cns]
    module.enabled_list = [True for _ in cns]
    return module


def _make_ctx(batch: int = 1) -> StepCtx:
    return StepCtx(
        x_t_latent=torch.randn(batch, 4, 8, 8),
        t_list=torch.tensor([0]),
        step_index=0,
        guidance_mode="none",
        sdxl_cond=None,
    )


def _bump_images_version(module: ControlNetModule) -> None:
    """Simulate TD pushing a new control image each frame: the real path
    (update_control_image_efficient) bumps _images_version per frame."""
    module._images_version += 1


def _down(delta: UnetKwargsDelta) -> List[torch.Tensor]:
    """Narrow UnetKwargsDelta's Optional down-residual list for the type checker."""
    down = delta.down_block_additional_residuals
    assert down is not None
    return down


def _mid(delta: UnetKwargsDelta) -> torch.Tensor:
    """Narrow UnetKwargsDelta's Optional mid residual for the type checker."""
    mid = delta.mid_block_additional_residual
    assert mid is not None
    return mid


class TestLegacyPathUnchanged:
    """decay == 0.0 must be byte-identical to the pre-decay behavior."""

    def test_live_feed_recomputes_every_frame_no_ema(self):
        cn = _RampingCN()
        module = _make_module_with_controlnets(cn)
        module.set_cn_cache_interval(2)
        hook = module.build_unet_hook()

        for frame in range(1, 6):
            _bump_images_version(module)
            result = hook(_make_ctx())
            # Newest residual every frame -- the interval never engages on a live feed.
            assert cn.calls == frame
            expected = torch.full(DOWN_SHAPES[0], float(frame))
            assert torch.allclose(_down(result)[0], expected)

        assert module._cn_ema_down is None, "legacy path must never allocate EMA buffers"
        assert module._cn_ema_mid is None
        assert module._cn_ema_shape_key is None

    def test_static_image_holds_residual_verbatim(self):
        cn = _RampingCN()
        module = _make_module_with_controlnets(cn)
        module.set_cn_cache_interval(2)
        hook = module.build_unet_hook()

        result1 = hook(_make_ctx())
        assert cn.calls == 1
        result2 = hook(_make_ctx())
        assert cn.calls == 1, "intermediate frame must reuse the cache, not recompute"
        assert result2 is result1, "held residual must be the cached delta, verbatim"
        result3 = hook(_make_ctx())
        assert cn.calls == 2, "schedule frame must recompute"
        assert torch.allclose(_down(result3)[0], torch.full(DOWN_SHAPES[0], 2.0))
        assert module._cn_ema_down is None


class TestDecayAuthoritativeInterval:
    """decay > 0: interval holds even on a live feed; applied residual is an EMA."""

    def test_live_feed_respects_interval_and_ema_sequence(self):
        cn = _RampingCN()
        module = _make_module_with_controlnets(cn)
        module.set_cn_cache_interval(2)
        module.set_cn_cache_decay(0.5)
        hook = module.build_unet_hook()

        # Hand-computed: compute frames 1/3/5 (targets 1.0/2.0/3.0), EMA init by
        # copy on frame 1, then lerp(prev, target, 0.5) every frame incl. compute
        # frames: [1.0, 1.0, 1.5, 1.75, 2.375]. All values dyadic -- exact in fp32.
        expected_values = [1.0, 1.0, 1.5, 1.75, 2.375]
        expected_calls = [1, 1, 2, 2, 3]

        for frame in range(5):
            _bump_images_version(module)
            result = hook(_make_ctx())
            assert cn.calls == expected_calls[frame], f"frame {frame + 1}: interval must be authoritative"
            expected = torch.full(DOWN_SHAPES[0], expected_values[frame])
            assert torch.allclose(_down(result)[0], expected), f"frame {frame + 1}"
            assert torch.allclose(_mid(result), torch.full(MID_SHAPE, expected_values[frame]))

    def test_interval_one_pure_smoothing(self):
        """interval=1 + decay>0: CN runs every frame, EMA still applied."""
        cn = _RampingCN()
        module = _make_module_with_controlnets(cn)
        module.set_cn_cache_decay(0.5)  # interval stays 1
        hook = module.build_unet_hook()

        # Targets 1.0/2.0/3.0 -> applied [1.0 (copy-init), 1.5, 2.25].
        for frame, expected_value in enumerate([1.0, 1.5, 2.25], start=1):
            _bump_images_version(module)
            result = hook(_make_ctx())
            assert cn.calls == frame, "interval=1 must still run CN every frame"
            assert torch.allclose(_down(result)[0], torch.full(DOWN_SHAPES[0], expected_value))


class TestEmaBufferOwnership:
    def test_returned_tensors_module_owned_and_pointer_stable(self):
        cn = _RampingCN()
        module = _make_module_with_controlnets(cn)
        module.set_cn_cache_interval(2)
        module.set_cn_cache_decay(0.5)
        hook = module.build_unet_hook()

        ptrs = []
        for _ in range(4):
            _bump_images_version(module)
            result = hook(_make_ctx())
            ptrs.append(
                (
                    tuple(t.data_ptr() for t in _down(result)),
                    _mid(result).data_ptr(),
                )
            )

        assert all(p == ptrs[0] for p in ptrs), "EMA buffers must be pointer-stable across frames"
        assert module._cn_ema_down is not None
        assert _down(result)[0] is module._cn_ema_down[0]
        assert _mid(result) is module._cn_ema_mid

        # Module-owned: never alias what the fake engine returned.
        fake_ptrs = set()
        for down, mid in cn.returned:
            fake_ptrs.update(t.data_ptr() for t in down)
            fake_ptrs.add(mid.data_ptr())
        returned_ptrs = set(ptrs[0][0]) | {ptrs[0][1]}
        assert not (returned_ptrs & fake_ptrs), "EMA buffers must not alias engine outputs"

        # Engine outputs unmutated by the EMA math (they are read-only targets).
        for call_idx, (down, mid) in enumerate(cn.returned, start=1):
            for t, shape in zip(down, DOWN_SHAPES):
                assert torch.allclose(t, torch.full(shape, float(call_idx)))
            assert torch.allclose(mid, torch.full(MID_SHAPE, float(call_idx)))

    def test_two_cn_merged_targets_use_separate_ema_buffers(self):
        cn_a = _RampingCN()
        cn_b = _RampingCN()
        module = _make_module_with_controlnets(cn_a, cn_b)
        module.set_cn_cache_interval(2)
        module.set_cn_cache_decay(0.5)
        hook = module.build_unet_hook()

        result = hook(_make_ctx())

        # Frame 1: both CNs return 1.0 -> merged target 2.0, EMA copy-inits to it.
        assert torch.allclose(_down(result)[0], torch.full(DOWN_SHAPES[0], 2.0))
        # Applied tensors are the EMA buffers, distinct from the merge buffers
        # (the merge buffers are the EMA *target* on the multi-CN path).
        assert module._cn_ema_down is not None
        assert _down(result)[0] is module._cn_ema_down[0]
        assert module._cn_merged_down is not None
        assert module._cn_merged_mid is not None
        assert _down(result)[0].data_ptr() != module._cn_merged_down[0].data_ptr()
        assert _mid(result).data_ptr() != module._cn_merged_mid.data_ptr()


class TestForcedRecompute:
    """Scale-hash and active-set changes must break the hold even off-schedule."""

    def test_scale_change_mid_hold_recomputes_and_smooths(self):
        cn = _RampingCN()
        module = _make_module_with_controlnets(cn)
        module.set_cn_cache_interval(3)
        module.set_cn_cache_decay(0.5)
        hook = module.build_unet_hook()

        _bump_images_version(module)
        hook(_make_ctx())  # frame 1: compute, EMA = 1.0
        _bump_images_version(module)
        hook(_make_ctx())  # frame 2: hold, EMA stays 1.0
        assert cn.calls == 1

        module.update_controlnet_scale(0, 0.5)
        _bump_images_version(module)
        result = hook(_make_ctx())  # frame 3: off-schedule (counter 2 % 3 != 0)

        assert cn.calls == 2, "scale change must force an off-schedule recompute"
        # EMA smooths toward the new target (2.0), no snap: lerp(1.0, 2.0, 0.5) = 1.5.
        assert torch.allclose(_down(result)[0], torch.full(DOWN_SHAPES[0], 1.5))

    def test_enable_toggle_mid_hold_recomputes(self):
        """V-D1 regression lock: update_controlnet_enabled changes neither
        scale_hash nor _images_version -- only the active-set key catches it."""
        cn_a = _RampingCN()
        cn_b = _RampingCN()
        module = _make_module_with_controlnets(cn_a, cn_b)
        module.set_cn_cache_interval(4)
        module.set_cn_cache_decay(0.5)
        hook = module.build_unet_hook()

        _bump_images_version(module)
        hook(_make_ctx())  # frame 1: both compute
        _bump_images_version(module)
        hook(_make_ctx())  # frame 2: hold
        assert cn_a.calls == 1 and cn_b.calls == 1

        module.update_controlnet_enabled(0, False)
        _bump_images_version(module)
        hook(_make_ctx())  # frame 3: off-schedule (counter 2 % 4 != 0)

        assert cn_b.calls == 2, "active-set change must force an off-schedule recompute"
        assert cn_a.calls == 1, "disabled CN must not run"

    def test_same_index_model_swap_recomputes(self):
        """V-D1 regression lock: swapping the model at an index with unchanged
        scales is invisible to scale_hash; the id(cn)-keyed active key catches it."""
        cn_old = _RampingCN()
        module = _make_module_with_controlnets(cn_old)
        module.set_cn_cache_interval(4)
        module.set_cn_cache_decay(0.5)
        hook = module.build_unet_hook()

        _bump_images_version(module)
        hook(_make_ctx())  # frame 1: compute
        _bump_images_version(module)
        hook(_make_ctx())  # frame 2: hold
        assert cn_old.calls == 1

        cn_new = _RampingCN()
        assert module.controlnets is not None
        module.controlnets[0] = cast(Any, cn_new)  # same index, same scale list
        _bump_images_version(module)
        hook(_make_ctx())  # frame 3: off-schedule

        assert cn_new.calls == 1, "model swap must force an off-schedule recompute"
        assert cn_old.calls == 1, "replaced CN must not run again"

    def test_remove_mid_hold_recomputes(self):
        """add/remove change the scale-hash length; covered here for completeness."""
        cn_a = _RampingCN()
        cn_b = _RampingCN()
        module = _make_module_with_controlnets(cn_a, cn_b)
        module.set_cn_cache_interval(4)
        module.set_cn_cache_decay(0.5)
        hook = module.build_unet_hook()

        _bump_images_version(module)
        hook(_make_ctx())
        _bump_images_version(module)
        hook(_make_ctx())
        assert cn_a.calls == 1 and cn_b.calls == 1

        module.remove_controlnet(1)
        _bump_images_version(module)
        hook(_make_ctx())

        assert cn_a.calls == 2, "remove must force an off-schedule recompute"
        assert cn_b.calls == 1


class TestSetterAndReset:
    def test_decay_setter_clamps(self):
        module = ControlNetModule(device="cpu", dtype=torch.float32)
        module.set_cn_cache_decay(-0.5)
        assert module._cn_cache_decay == 0.0
        module.set_cn_cache_decay(1.5)
        assert module._cn_cache_decay == 1.0

    def test_decay_change_resets_ema_but_keeps_schedule(self):
        cn = _RampingCN()
        module = _make_module_with_controlnets(cn)
        module.set_cn_cache_interval(2)
        module.set_cn_cache_decay(0.5)
        hook = module.build_unet_hook()

        _bump_images_version(module)
        hook(_make_ctx())
        assert module._cn_ema_down is not None
        counter_before = module._cn_frame_counter

        module.set_cn_cache_decay(0.8)

        assert module._cn_ema_down is None, "decay change must reset EMA buffers"
        assert module._cn_ema_mid is None
        assert module._cn_cached_residuals is not None, "cached residuals must survive a decay change"
        assert module._cn_frame_counter == counter_before, "schedule must not be disturbed"

    def test_shape_change_reallocates_ema_buffers(self):
        cn = _RampingCN()
        module = _make_module_with_controlnets(cn)
        module.set_cn_cache_decay(0.5)  # interval 1: compute every frame
        hook = module.build_unet_hook()

        _bump_images_version(module)
        result1 = hook(_make_ctx())
        ptr1 = _down(result1)[0].data_ptr()

        # Simulate a batch/resolution change: new residual shapes from the engine.
        new_down_shapes = [(2, 4, 8, 8), (2, 4, 4, 4)]
        new_mid_shape = (2, 4, 2, 2)
        cn._down_shapes = new_down_shapes
        cn._mid_shape = new_mid_shape

        _bump_images_version(module)
        result2 = hook(_make_ctx())

        assert _down(result2)[0].shape == new_down_shapes[0]
        assert _down(result2)[0].data_ptr() != ptr1, "shape change must reallocate"
        # Re-init is a copy from the new target (2.0), not a lerp from the old EMA (1.5).
        assert torch.allclose(_down(result2)[0], torch.full(new_down_shapes[0], 2.0))
        assert torch.allclose(_mid(result2), torch.full(new_mid_shape, 2.0))

    def test_install_resets_decay_state_but_keeps_setting(self):
        cn = _RampingCN()
        module = _make_module_with_controlnets(cn)
        module.set_cn_cache_interval(2)
        module.set_cn_cache_decay(0.5)
        module.build_unet_hook()(_make_ctx())
        assert module._cn_ema_down is not None
        assert module._cn_cache_active_key is not None

        class _MinimalStream:
            unet_hooks = []
            controlnets = None
            controlnet_scales = None
            preprocessors = None

        # attach_orchestrator requires a preprocessing orchestrator; install() only
        # touches it when _preprocessing_orchestrator is None.
        module._preprocessing_orchestrator = cast(Any, object())
        module.install(_MinimalStream())

        assert module._cn_ema_down is None
        assert module._cn_ema_mid is None
        assert module._cn_ema_shape_key is None
        assert module._cn_cache_active_key is None
        assert module._cn_cached_residuals is None
        assert module._cn_frame_counter == 0
        # User settings survive re-install.
        assert module._cn_cache_decay == 0.5
        assert module._cn_cache_interval == 2
