"""Regression tests for the TensorRT SDXL config-dict sites touched by the
Turbo/model-detection unification (docs/plans -- detect_model's ``is_turbo``
key was renamed to ``turbo_from_scheduler``, see model_detection.py).

Both ``ControlNetSDXLTRT.__init__`` (controlnet_models.py) and
``get_sdxl_tensorrt_config`` (unet_sdxl_export.py) build a config dict from
``detect_model(unet)``'s output *without* passing a pipe -- so
``turbo_from_scheduler`` is always ``None`` there, and neither site ever read
the old ``is_turbo`` key. Removing it was still a real behavioural change in
principle (a ``KeyError`` risk had either site read the collapsed bool), even
though production never exercises either path today:

- ``ControlNetSDXLTRT``'s ``unet is not None`` branch is dead in production --
  the sole caller passes ``unet=None`` (wrapper.py -> engine_manager.py ->
  controlnet_models.py, verified end-to-end).
- ``get_sdxl_tensorrt_config`` has zero callers repo-wide.

These tests drive both sites directly with a mocked SDXL UNet so the branches
run at all, pinning the contract: constructs cleanly, no ``KeyError``,
``embedding_dim`` resolves to SDXL's 2048.

Skipped where TensorRT/onnx/polygraphy aren't installed, matching
test_engine_path_controlnet_tokens.py's guard -- controlnet_models.py's
BaseModel parent (models.py) imports onnx_graphsurgeon/onnx/polygraphy at
module load time, before any test in this file can run.
"""

from unittest.mock import MagicMock

import pytest
from diffusers.configuration_utils import FrozenDict
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

try:
    from streamdiffusion.acceleration.tensorrt.export_wrappers.unet_sdxl_export import (
        SDXLConditioningHandler,
        get_sdxl_tensorrt_config,
    )
    from streamdiffusion.acceleration.tensorrt.models.controlnet_models import ControlNetSDXLTRT

    IMPORT_OK = True
except ImportError:
    IMPORT_OK = False

pytestmark = pytest.mark.skipif(
    not IMPORT_OK,
    reason="acceleration.tensorrt.models.controlnet_models not importable (onnx/polygraphy missing)",
)

# Same shape as test_model_detection.py's SDXL_TURBO_UNET_CONFIG: addition_embed_type
# set (SDXL signal), cross_attention_dim=2048, time_cond_proj_dim=None.
SDXL_UNET_CONFIG = {
    "addition_embed_type": "text_time",
    "cross_attention_dim": 2048,
    "time_cond_proj_dim": None,
}


def _fake_sdxl_unet() -> MagicMock:
    """Same pattern as test_model_detection.py's _fake_unet: Mock(spec=...) so
    isinstance() checks pass, FrozenDict config for dict- and attribute-style
    access."""
    unet = MagicMock(spec=UNet2DConditionModel)
    unet.config = FrozenDict(SDXL_UNET_CONFIG)
    return unet


class TestControlNetSDXLTRTConstruction:
    """Drives the `unet is not None` branch that production never reaches
    (the sole real caller passes unet=None) -- pins the contract in case
    anyone ever wires a real UNet in."""

    def test_constructs_without_keyerror(self):
        # Must not raise -- specifically must not KeyError on a stale "is_turbo"
        # key that detect_model() no longer returns.
        model = ControlNetSDXLTRT(unet=_fake_sdxl_unet(), device="cpu")
        assert model.name == "ControlNet"

    def test_embedding_dim_resolves_to_sdxl_2048(self):
        model = ControlNetSDXLTRT(unet=_fake_sdxl_unet(), device="cpu")
        assert model.embedding_dim == 2048

    def test_unet_dim_defaults_to_sdxl_latent_channels(self):
        model = ControlNetSDXLTRT(unet=_fake_sdxl_unet(), device="cpu")
        assert model.unet_dim == 4


class TestGetSdxlTensorrtConfigSmokeTest:
    """get_sdxl_tensorrt_config is dead code repo-wide (zero callers), but it
    must remain dead-but-correct rather than dead-and-broken -- this exercises
    it directly with a mocked SDXL UNet."""

    def test_config_carries_expected_keys(self):
        config = get_sdxl_tensorrt_config("D:/fake/path.safetensors", _fake_sdxl_unet())

        assert config["is_sdxl"] is True
        assert config["has_time_cond"] is False  # time_cond_proj_dim=None in the fixture
        assert config["has_addition_embed"] is True  # addition_embed_type="text_time"
        assert "is_turbo" not in config  # the removed key must not reappear

    def test_config_is_consumable_by_conditioning_handler(self):
        """The whole point of the dict: SDXLConditioningHandler must be able
        to construct from it and produce a conditioning spec."""
        config = get_sdxl_tensorrt_config("D:/fake/path.safetensors", _fake_sdxl_unet())

        handler = SDXLConditioningHandler(config)
        spec = handler.get_conditioning_spec()

        assert spec["dual_encoders"] is True
        assert spec["context_dim"] == 2048
