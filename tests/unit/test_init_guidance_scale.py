"""The configured guidance_scale reaches the wrapper's warm-up prepare().

TD builds the wrapper via create_wrapper_from_config and never calls wrapper.prepare(), so the
config value only flowed through _extract_prepare_params and the stream kept a hard-coded 1.1.
"""

import pytest

from streamdiffusion.config import _extract_wrapper_params
from streamdiffusion.wrapper import _resolve_init_guidance_scale


@pytest.mark.parametrize("cfg_type", ["full", "self", "initialize"], ids=str)
def test_unconfigured_keeps_rcfg_default(cfg_type):
    assert _resolve_init_guidance_scale(cfg_type, None) == 1.1


def test_unconfigured_none_cfg_is_one():
    assert _resolve_init_guidance_scale("none", None) == 1.0


def test_configured_one_with_self_is_honored():
    # 1.0 turns RCFG off (pipeline: rcfg_active = guidance_scale > 1.0 and cfg_type != "none").
    assert _resolve_init_guidance_scale("self", 1.0) == 1.0


def test_configured_value_above_one_is_honored():
    assert _resolve_init_guidance_scale("self", 1.5) == 1.5


@pytest.mark.parametrize("cfg_type", ["full", "initialize"], ids=str)
@pytest.mark.parametrize("configured", [0.5, 1.0], ids=["below_one", "exactly_one"])
def test_full_and_initialize_are_clamped_to_1_2(cfg_type, configured, caplog):
    # Their static TRT batch assumes CFG doubling, which needs guidance > 1.0.
    with caplog.at_level("WARNING", logger="streamdiffusion.wrapper"):
        assert _resolve_init_guidance_scale(cfg_type, configured) == 1.2
    assert f"guidance_scale={configured}" in caplog.text


@pytest.mark.parametrize("cfg_type", ["full", "initialize"], ids=str)
def test_full_and_initialize_above_one_are_not_clamped(cfg_type):
    assert _resolve_init_guidance_scale(cfg_type, 1.4) == 1.4


@pytest.mark.parametrize("configured", [None, 0.5, 1.5], ids=["unset", "below_one", "above_one"])
def test_none_cfg_always_returns_one(configured):
    # The pipeline forces 1.0 for cfg 'none', so the logged initial value must match.
    assert _resolve_init_guidance_scale("none", configured) == 1.0


def test_extract_wrapper_params_forwards_guidance_scale():
    assert _extract_wrapper_params({"guidance_scale": 1.0})["guidance_scale"] == 1.0


def test_extract_wrapper_params_omits_guidance_scale_when_absent():
    assert "guidance_scale" not in _extract_wrapper_params({})
