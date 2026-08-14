"""
Workstream A / A9: unit coverage for _collect_lora_layers' loud-failure guard.

The guard (unet_unified_export.py:79-105) exists to stop the runtime lora_scale
tensor from silently becoming a no-op slider -- the exact bug class this whole
workstream exists to end. It had no test. Uses real peft LoraLayer instances
(not mocks) via inject_adapter_in_model, so the test tracks peft's actual
merged/disable_adapters/lora_variant state machine rather than an assumption
about it.
"""

import pytest
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model

from streamdiffusion.acceleration.tensorrt.export_wrappers.unet_unified_export import (
    _collect_lora_layers,
)


class _TinyUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8, bias=False)


def _make_adapted(use_dora=False, adapter_name="default"):
    config = LoraConfig(r=4, lora_alpha=4, target_modules=["to_q"], use_dora=use_dora)
    return inject_adapter_in_model(config, _TinyUnet(), adapter_name=adapter_name)


def test_vanilla_lora_collects_successfully():
    model = _make_adapted()
    layers = _collect_lora_layers(model, [("dummy.safetensors", "default")])
    assert list(layers.keys()) == ["default"]
    module, base_scaling = layers["default"][0]
    assert module is model.to_q
    assert base_scaling == pytest.approx(4 / 4)  # lora_alpha / r


def test_dora_adapter_raises_loudly():
    model = _make_adapted(use_dora=True)
    with pytest.raises(RuntimeError, match="non-vanilla PEFT variant"):
        _collect_lora_layers(model, [("dummy.safetensors", "default")])


def test_merged_adapter_raises_loudly():
    model = _make_adapted()
    model.to_q.merge()  # type: ignore[attr-defined]  -- inject_adapter_in_model swaps to_q for a lora.Linear
    with pytest.raises(RuntimeError, match="is merged into module"):
        _collect_lora_layers(model, [("dummy.safetensors", "default")])


def test_disabled_adapter_raises_loudly():
    model = _make_adapted()
    model.to_q.enable_adapters(False)  # type: ignore[attr-defined]  -- see test_merged_adapter_raises_loudly
    with pytest.raises(RuntimeError, match="is disabled on module"):
        _collect_lora_layers(model, [("dummy.safetensors", "default")])


def test_unmatched_adapter_name_raises_nothing_to_scale():
    model = _make_adapted()
    with pytest.raises(RuntimeError, match="nothing to scale"):
        _collect_lora_layers(model, [("dummy.safetensors", "not_the_real_adapter")])
