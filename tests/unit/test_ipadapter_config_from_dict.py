"""
Regression test for A4 — ``IPAdapterConfig.from_dict``
(``streamdiffusion.modules.ipadapter_module.IPAdapterConfig``).

Both ``wrapper.py`` install sites (pre-TRT and post-TRT) used to build this dataclass by hand
with duplicated inline field lists and no validation. ``from_dict`` centralizes that and adds
one fail-fast check: a ``type="faceid"`` config with no ``insightface_model_name`` used to
construct successfully and only fail later, mid-stream, inside ``_get_faceid_embeds`` on the
first style-image update — the same silent/late-failure class as B1/B2. This test locks in
both the happy paths and that fail-fast behaviour.

Run with: pytest tests/unit/test_ipadapter_config_from_dict.py -v
"""

import pytest

from streamdiffusion.modules.ipadapter_module import IPAdapterConfig, IPAdapterType

_BASE_CFG = {
    "ipadapter_model_path": "h94/IP-Adapter/models/ip-adapter_sd15.bin",
    "image_encoder_path": "h94/IP-Adapter/models/image_encoder",
}


class TestFromDictHappyPaths:
    def test_regular_config_defaults(self):
        cfg = IPAdapterConfig.from_dict(dict(_BASE_CFG))

        assert cfg.type == IPAdapterType.REGULAR
        assert cfg.ipadapter_model_path == _BASE_CFG["ipadapter_model_path"]
        assert cfg.image_encoder_path == _BASE_CFG["image_encoder_path"]
        assert cfg.style_image_key == "ipadapter_main"
        assert cfg.num_image_tokens == 4
        assert cfg.scale == 1.0
        assert cfg.style_image is None
        assert cfg.insightface_model_name is None

    def test_faceid_config_with_insightface_model_name_succeeds(self):
        cfg = IPAdapterConfig.from_dict(
            dict(
                _BASE_CFG,
                type="faceid",
                insightface_model_name="buffalo_l",
            )
        )

        assert cfg.type == IPAdapterType.FACEID
        assert cfg.insightface_model_name == "buffalo_l"

    def test_explicit_style_image_key_is_preserved(self):
        cfg = IPAdapterConfig.from_dict(dict(_BASE_CFG, style_image_key="secondary"))

        assert cfg.style_image_key == "secondary"

    def test_empty_string_style_image_key_falls_back_to_default(self):
        """S4 alignment: an empty string (falsy) must still resolve to the same default
        'ipadapter_main' that build_embedding_hook and install() both use — not '' itself,
        which would silently desync the hook's cache lookup from the installed style key.
        """
        cfg = IPAdapterConfig.from_dict(dict(_BASE_CFG, style_image_key=""))

        assert cfg.style_image_key == "ipadapter_main"

    def test_custom_num_image_tokens_and_scale_pass_through(self):
        cfg = IPAdapterConfig.from_dict(dict(_BASE_CFG, num_image_tokens=16, scale=0.7))

        assert cfg.num_image_tokens == 16
        assert cfg.scale == 0.7

    def test_style_image_passes_through(self):
        sentinel = object()
        cfg = IPAdapterConfig.from_dict(dict(_BASE_CFG, style_image=sentinel))

        assert cfg.style_image is sentinel


class TestFromDictValidation:
    def test_faceid_without_insightface_model_name_raises_value_error(self):
        with pytest.raises(ValueError, match="insightface_model_name"):
            IPAdapterConfig.from_dict(dict(_BASE_CFG, type="faceid"))

    def test_faceid_with_empty_insightface_model_name_raises_value_error(self):
        with pytest.raises(ValueError, match="insightface_model_name"):
            IPAdapterConfig.from_dict(dict(_BASE_CFG, type="faceid", insightface_model_name=""))

    def test_missing_ipadapter_model_path_raises_key_error(self):
        cfg = {"image_encoder_path": _BASE_CFG["image_encoder_path"]}
        with pytest.raises(KeyError):
            IPAdapterConfig.from_dict(cfg)

    def test_missing_image_encoder_path_raises_key_error(self):
        cfg = {"ipadapter_model_path": _BASE_CFG["ipadapter_model_path"]}
        with pytest.raises(KeyError):
            IPAdapterConfig.from_dict(cfg)

    def test_invalid_type_string_raises_value_error(self):
        with pytest.raises(ValueError):
            IPAdapterConfig.from_dict(dict(_BASE_CFG, type="not-a-real-type"))
