"""Unit tests for src/streamdiffusion/model_detection.py's is_turbo discrimination.

Covers `_detect_turbo_from_scheduler` directly and `detect_model`'s end-to-end
`is_turbo` result for the three checkpoints named in
docs/plans/SDXL-Turbo_Research_Verification.md's defect 2: sd-turbo, sdxl-turbo,
and sdxl-base-1.0. UNet config values below are the real published
unet/config.json / scheduler_config.json fields for each checkpoint (sdxl-base-1.0
is stubbed from its published config rather than downloading the ~7 GB checkpoint;
sd-turbo/sdxl-turbo are also stubbed here for speed and to keep the test
network-free and deterministic).

The point of this file: sdxl-turbo and sdxl-base-1.0 have an *identical* UNet-config
shape (`time_cond_proj_dim: null`, `addition_embed_type: "text_time"`) — that is
exactly why the old `time_cond_proj_dim is None` heuristic misclassified SDXL-Base
as Turbo. Only the scheduler discriminates them.

CPU-only, no CUDA required, no network access, no model weights loaded.
"""

from unittest.mock import MagicMock

from diffusers.configuration_utils import FrozenDict
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from streamdiffusion.model_detection import _detect_turbo_from_scheduler, detect_model


def _fake_unet(config: dict) -> MagicMock:
    """A UNet2DConditionModel-shaped mock carrying only the config fields
    detect_model reads. Mock(spec=...) makes isinstance() checks pass; the
    FrozenDict config supports both dict-style (`config.get(...)`) and
    attribute-style (`config.cross_attention_dim`) access, matching real
    diffusers configs."""
    unet = MagicMock(spec=UNet2DConditionModel)
    unet.config = FrozenDict(config)
    return unet


def _fake_pipe(class_name: str, timestep_spacing) -> MagicMock:
    scheduler_config = FrozenDict({"_class_name": class_name, "timestep_spacing": timestep_spacing})
    scheduler = MagicMock()
    scheduler.config = scheduler_config
    pipe = MagicMock()
    pipe.scheduler = scheduler
    return pipe


# Published unet/config.json fields (the ones detect_model reads).
SD_TURBO_UNET_CONFIG = {
    "addition_embed_type": None,
    "cross_attention_dim": 1024,
    "time_cond_proj_dim": None,
}
SDXL_TURBO_UNET_CONFIG = {
    "addition_embed_type": "text_time",
    "cross_attention_dim": 2048,
    "time_cond_proj_dim": None,
}
# Same shape as sdxl-turbo's -- the UNet config alone cannot tell them apart.
SDXL_BASE_UNET_CONFIG = {
    "addition_embed_type": "text_time",
    "cross_attention_dim": 2048,
    "time_cond_proj_dim": None,
}

# Published scheduler_config.json fields.
TURBO_SCHEDULER = ("EulerAncestralDiscreteScheduler", "trailing")
SDXL_BASE_SCHEDULER = ("EulerDiscreteScheduler", "leading")


class TestDetectTurboFromScheduler:
    def test_no_pipe_is_undecided(self):
        assert _detect_turbo_from_scheduler(None) is None

    def test_pipe_without_scheduler_is_undecided(self):
        pipe = MagicMock()
        pipe.scheduler = None
        assert _detect_turbo_from_scheduler(pipe) is None

    def test_euler_ancestral_trailing_is_turbo(self):
        pipe = _fake_pipe(*TURBO_SCHEDULER)
        assert _detect_turbo_from_scheduler(pipe) is True

    def test_euler_discrete_leading_is_not_turbo(self):
        pipe = _fake_pipe(*SDXL_BASE_SCHEDULER)
        assert _detect_turbo_from_scheduler(pipe) is False

    def test_euler_ancestral_without_trailing_is_not_turbo(self):
        """Class name alone isn't enough -- spacing must also be trailing."""
        pipe = _fake_pipe("EulerAncestralDiscreteScheduler", "leading")
        assert _detect_turbo_from_scheduler(pipe) is False


class TestDetectModelIsTurbo:
    """End-to-end: detect_model(unet, pipe)['is_turbo'] for the three checkpoints
    named in the verification doc's defect 2."""

    def test_sd_turbo_is_turbo(self):
        unet = _fake_unet(SD_TURBO_UNET_CONFIG)
        pipe = _fake_pipe(*TURBO_SCHEDULER)
        result = detect_model(unet, pipe)
        assert result["model_type"] == "SD2.1"
        assert result["is_turbo"] is True

    def test_sdxl_turbo_is_turbo(self):
        unet = _fake_unet(SDXL_TURBO_UNET_CONFIG)
        pipe = _fake_pipe(*TURBO_SCHEDULER)
        result = detect_model(unet, pipe)
        assert result["model_type"] == "SDXL"
        assert result["is_turbo"] is True

    def test_sdxl_base_is_not_turbo(self):
        """The regression case: sdxl-base-1.0's UNet config is indistinguishable
        from sdxl-turbo's (both have time_cond_proj_dim=None), so only the
        scheduler-based check keeps this False."""
        unet = _fake_unet(SDXL_BASE_UNET_CONFIG)
        pipe = _fake_pipe(*SDXL_BASE_SCHEDULER)
        result = detect_model(unet, pipe)
        assert result["model_type"] == "SDXL"
        assert result["is_turbo"] is False

    def test_no_pipe_defaults_is_turbo_false(self):
        """No pipe -> _detect_turbo_from_scheduler returns None -> is_turbo keeps
        its initial False rather than being set. Matches the two non-pipe call
        sites (unet_sdxl_export.py, controlnet_models.py) whose is_turbo output
        is confirmed dead downstream."""
        unet = _fake_unet(SDXL_TURBO_UNET_CONFIG)
        result = detect_model(unet, pipe=None)
        assert result["is_turbo"] is False
