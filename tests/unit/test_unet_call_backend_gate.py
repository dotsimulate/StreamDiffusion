"""
Regression test for the SD1.5/2.1 UNet call path passing TensorRT-only
feature-injection kwargs to a non-TRT (eager PyTorch) UNet.

``import streamdiffusion`` monkeypatches diffusers' ``UNet2DConditionModel.forward``
(see ``_patches/diffusers_kvo_patch.py``) to add exactly one new kwarg, ``kvo_cache``,
for StreamV2V cached-attention. It does *not* add ``fio_cache`` / ``fi_strength`` /
``fi_threshold`` — feature-injection is TensorRT-engine-only
(``acceleration/tensorrt/runtime_engines/unet_engine.py``,
``UNet2DConditionModelEngine.__call__``).

``StreamDiffusion.unet_step()`` (pipeline.py) has two model-family branches. The SDXL
branch gates the FI kwargs behind ``self._check_unet_tensorrt()``; the SD1.5/2.1
branch did not, and unconditionally passed all four cache kwargs to ``self.unet(...)``.
On any non-TRT backend (``acceleration: none``/``xformers``/``sfast``) this raised:

    UNet2DConditionModel.forward() got an unexpected keyword argument 'fio_cache'

...on every frame — see the crash report from a `sd-turbo` (SD2.1) install running
``acceleration: none``. Error report path:
``.../error_reports/inference_error_report_20260718_203842_389083.txt``.

This test constructs ``StreamDiffusion`` via ``__new__`` (bypassing ``__init__``'s
heavy model-loading), sets only the attributes ``unet_step``'s SD1.5/2.1 branch reads,
and drives the real call site with a real (patched) tiny ``UNet2DConditionModel`` — no
model download, sub-second runtime.

Run with: pytest tests/unit/test_unet_call_backend_gate.py -v
"""

from __future__ import annotations

from typing import List, Optional

import pytest
import torch

try:
    # Import first: this is what applies the kvo_cache monkeypatch onto diffusers'
    # UNet2DConditionModel. Importing diffusers alone (without streamdiffusion) would
    # show a *different* symptom (kvo_cache itself rejected) than the real bug.
    from diffusers import UNet2DConditionModel

    import streamdiffusion  # noqa: F401  (import side effect: applies the patch)
    from streamdiffusion.pipeline import StreamDiffusion

    IMPORT_OK = True
except ImportError:
    IMPORT_OK = False

pytestmark = pytest.mark.skipif(
    not IMPORT_OK,
    reason="streamdiffusion / diffusers not importable",
)


def _make_tiny_unet() -> UNet2DConditionModel:
    """Smallest UNet2DConditionModel that still exercises a CrossAttnDownBlock2D /
    CrossAttnUpBlock2D pair (the blocks the kvo_cache patch threads cache through)."""
    return UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16),
        norm_num_groups=8,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=8,
        attention_head_dim=4,
    )


class _RecordingUnet:
    """Wraps a real (patched) UNet2DConditionModel and records the kwargs it's called
    with, so tests can assert exactly which cache kwargs reached the UNet."""

    def __init__(self, unet: UNet2DConditionModel) -> None:
        self._unet = unet
        self.last_kwargs: dict = {}

    def __call__(self, sample, timestep, **kwargs):
        self.last_kwargs = dict(kwargs)
        return self._unet(sample, timestep, **kwargs)


class _FakeTrtEngine:
    """Minimal stand-in for UNet2DConditionModelEngine: exposes .engine/.stream so
    _check_unet_tensorrt() reports True, and accepts all four cache kwargs."""

    def __init__(self) -> None:
        self.engine = object()
        self.stream = object()
        self.last_kwargs: dict = {}

    def __call__(self, sample, timestep, **kwargs):
        self.last_kwargs = dict(kwargs)
        # Mimic the TRT engine's 3-tuple return: (pred, kvo_cache_out, fio_cache_out)
        return (sample, [], [])


def _make_pipeline(unet, *, prompt_tokens: int = 4) -> StreamDiffusion:
    """Build a StreamDiffusion instance via __new__, setting only the attributes that
    unet_step()'s SD1.5/2.1 branch (and the profiler/hook plumbing around it) reads."""
    sd = StreamDiffusion.__new__(StreamDiffusion)

    sd.unet = unet
    sd.is_sdxl = False
    sd.unet_hooks: list = []
    sd._unet_kwargs: dict = {}
    sd._is_unet_tensorrt: Optional[bool] = None  # lazy cache read by _check_unet_tensorrt()

    sd.guidance_scale = 1.0  # skip CFG latent-doubling branch entirely
    sd.cfg_type = "none"

    sd.prompt_embeds = torch.randn(1, prompt_tokens, 8)
    sd.kvo_cache: List[torch.Tensor] = []
    sd.fio_cache: List[torch.Tensor] = []
    sd._fi_strength_tensor: Optional[torch.Tensor] = None
    sd._fi_threshold_tensor: Optional[torch.Tensor] = None

    sd.use_denoising_batch = True
    # scheduler_step_batch is orthogonal to this bug (it runs strictly after the UNet
    # call); stub it so the test doesn't need a real scheduler.
    sd.scheduler_step_batch = lambda model_pred_batch, x_t_latent_batch, idx=None: x_t_latent_batch

    def _noop_update_kvo_cache(kvo_cache_out, fio_cache_out=None):
        return None

    sd.update_kvo_cache = _noop_update_kvo_cache

    return sd


def _call_unet_step(sd: StreamDiffusion):
    x_t_latent = torch.randn(1, 4, 8, 8)
    t_list = torch.tensor([10])
    return sd.unet_step(x_t_latent, t_list, idx=0)


class TestSd15Sd21UnetCallBackendGate:
    def test_non_tensorrt_unet_call_does_not_raise(self):
        """RED before the fix: raises TypeError('unexpected keyword argument
        fio_cache'). GREEN after the fix: returns normally."""
        recording_unet = _RecordingUnet(_make_tiny_unet())
        sd = _make_pipeline(recording_unet)

        denoised_batch, model_pred = _call_unet_step(sd)

        assert model_pred is not None
        assert denoised_batch is not None

    def test_non_tensorrt_unet_receives_no_feature_injection_kwargs(self):
        """The patched PyTorch UNet only understands kvo_cache; fio_cache/fi_strength/
        fi_threshold must never reach it, TRT-only or not."""
        recording_unet = _RecordingUnet(_make_tiny_unet())
        sd = _make_pipeline(recording_unet)

        _call_unet_step(sd)

        assert "kvo_cache" in recording_unet.last_kwargs
        for key in ("fio_cache", "fi_strength", "fi_threshold"):
            assert key not in recording_unet.last_kwargs, (
                f"non-TRT UNet call must not receive {key!r}; got kwargs="
                f"{sorted(recording_unet.last_kwargs)}"
            )

    def test_tensorrt_engine_still_receives_feature_injection_kwargs(self):
        """Regression guard: the fix must not remove FI support from the real TRT path."""
        fake_engine = _FakeTrtEngine()
        sd = _make_pipeline(fake_engine)
        sd._fi_strength_tensor = torch.tensor(0.5)
        sd._fi_threshold_tensor = torch.tensor(0.1)

        _call_unet_step(sd)

        for key in ("kvo_cache", "fio_cache", "fi_strength", "fi_threshold"):
            assert key in fake_engine.last_kwargs, (
                f"TRT engine call must still receive {key!r}; got kwargs="
                f"{sorted(fake_engine.last_kwargs)}"
            )
