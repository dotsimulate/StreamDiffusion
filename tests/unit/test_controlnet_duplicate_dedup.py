"""Regression tests for the ControlNet duplicate-model dedup asymmetry.

Two ``Cn`` blocks pointing at the same model were collapsed to the highest
``conditioning_scale`` on TouchDesigner's *live* path (``Cnblock()``) but not on the
*startup* path (YAML export -> ``_prepare_controlnet_configs`` ->
``ControlNetModule.add_controlnet``), and the live-update path
(``_update_controlnet_config``) could not repair what startup produced:
``current_models`` was built index-keyed (``{0: canny, 1: canny}``) while
``desired_models`` was built model-id-keyed (``{canny: cfg}``), so the removal
predicate ``model_id not in desired_models`` never fired for a duplicate, and
``existing_index = next(...)`` only ever touched the *first* occurrence — a live
weight tweak could push effective conditioning from 1.3 up to 1.8 instead of down.

A second, previously undocumented vector: ``_update_controlnet_config`` built
``current_models`` once and never refreshed it after an ``add_controlnet`` call
inside the same pass, so a ``desired_config`` containing two entries for a model
that was not yet loaded produced *two* ``add_controlnet`` calls in one update.

Both are fixed by ``streamdiffusion.config.dedupe_controlnet_configs`` — a single
shared helper (collapse by ``model_id``, keep the highest ``conditioning_scale``,
preserve first-occurrence order) applied at both the config-prepare seam and the
live-update seam.

CPU-only, no GPU, no model download. ``FakeControlNetModule`` mirrors
``ControlNetModule.add_controlnet`` / ``remove_controlnet`` /
``reorder_controlnets_by_model_ids`` (controlnet_module.py) verbatim minus the real
``ControlNetModel.from_pretrained`` load, so the real bug pattern -- duplicate
model_id across positionally-indexed parallel lists -- is exercised exactly as it
occurs in the pipeline.
"""

import types
from typing import Any, List, Optional

import torch

from streamdiffusion.config import _prepare_controlnet_configs, dedupe_controlnet_configs
from streamdiffusion.modules.controlnet_module import ControlNetConfig
from streamdiffusion.stream_parameter_updater import StreamParameterUpdater

# ---------------------------------------------------------------------------
# Fake ControlNet model + module.
# ---------------------------------------------------------------------------


class _FakeLoadedModel:
    """Stand-in for the diffusers ControlNetModel a real load returns.

    Real code sets ``.model_id`` on the loaded model at controlnet_module.py:879;
    everything downstream (``getattr(cn, "model_id", ...)``) relies on that.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id


class FakeControlNetModule:
    """Duck-types the subset of ControlNetModule the updater touches, with
    add_controlnet/remove_controlnet/reorder_controlnets_by_model_ids copied
    verbatim from controlnet_module.py minus the real model load."""

    def __init__(self):
        self.controlnets: List[Any] = []
        self.controlnet_images: List[Any] = []
        self.controlnet_scales: List[float] = []
        self.preprocessors: List[Any] = []
        self.enabled_list: List[bool] = []
        self.add_controlnet_calls = 0

    def add_controlnet(self, cfg: ControlNetConfig, control_image: Optional[Any] = None) -> None:
        self.add_controlnet_calls += 1
        self.controlnets.append(_FakeLoadedModel(cfg.model_id))
        self.controlnet_images.append(None)
        self.controlnet_scales.append(float(cfg.conditioning_scale))
        self.preprocessors.append(None)
        self.enabled_list.append(bool(cfg.enabled))

    def remove_controlnet(self, index: int) -> None:
        if 0 <= index < len(self.controlnets):
            del self.controlnets[index]
            del self.controlnet_images[index]
            del self.controlnet_scales[index]
            del self.preprocessors[index]
            del self.enabled_list[index]

    def reorder_controlnets_by_model_ids(self, desired_model_ids: List[str]) -> None:
        current_ids = [getattr(cn, "model_id", f"controlnet_{i}") for i, cn in enumerate(self.controlnets)]
        picked = set()
        new_order: List[int] = []
        for mid in desired_model_ids:
            if mid in current_ids:
                idx = current_ids.index(mid)
                new_order.append(idx)
                picked.add(idx)
        for i in range(len(self.controlnets)):
            if i not in picked:
                new_order.append(i)
        if new_order == list(range(len(self.controlnets))):
            return

        def reindex(lst):
            return [lst[i] for i in new_order]

        self.controlnets = reindex(self.controlnets)
        self.controlnet_images = reindex(self.controlnet_images)
        self.controlnet_scales = reindex(self.controlnet_scales)
        self.preprocessors = reindex(self.preprocessors)
        self.enabled_list = reindex(self.enabled_list)


# ---------------------------------------------------------------------------
# Updater fixture -- mirrors the fake-stream + no-op attach_orchestrator recipe
# from tests/unit/test_param_updater_binding.py.
# ---------------------------------------------------------------------------


def _make_updater_with_module(module: FakeControlNetModule) -> StreamParameterUpdater:
    stream = types.SimpleNamespace()
    stream.device = torch.device("cpu")
    stream.dtype = torch.float32
    stream._preprocessing_orchestrator = None
    stream._controlnet_module = module

    orig_attach = StreamParameterUpdater.attach_orchestrator

    def _noop_attach(self, s):
        self._preprocessing_orchestrator = None

    StreamParameterUpdater.attach_orchestrator = _noop_attach
    try:
        updater = StreamParameterUpdater(stream)
    finally:
        StreamParameterUpdater.attach_orchestrator = orig_attach

    updater._embedding_orchestrator = None
    return updater


def _cn_config(model_id: str, conditioning_scale: float = 1.0, **overrides) -> dict:
    """Minimal YAML-shaped ControlNet block, as _prepare_controlnet_configs consumes it."""
    cfg = {"model_id": model_id, "conditioning_scale": conditioning_scale}
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# 1-3: dedupe_controlnet_configs / _prepare_controlnet_configs
# ---------------------------------------------------------------------------


def test_prepare_controlnet_configs_collapses_duplicates_keeping_highest_scale():
    config = {
        "controlnets": [
            _cn_config("canny", conditioning_scale=0.4),
            _cn_config("depth", conditioning_scale=0.6),
            _cn_config("canny", conditioning_scale=0.9),
        ]
    }

    result = _prepare_controlnet_configs(config)

    assert [cfg["model_id"] for cfg in result] == ["canny", "depth"], (
        "duplicate must collapse to one entry, in first-occurrence order"
    )
    canny = next(cfg for cfg in result if cfg["model_id"] == "canny")
    assert canny["conditioning_scale"] == 0.9, "the higher-weighted duplicate must win"


def test_prepare_controlnet_configs_winner_carries_its_own_preprocessor():
    config = {
        "controlnets": [
            _cn_config("canny", conditioning_scale=0.4, preprocessor="canny_edge", preprocessor_params={"low": 1}),
            _cn_config("canny", conditioning_scale=0.9, preprocessor="hed", preprocessor_params={"low": 2}),
        ]
    }

    result = _prepare_controlnet_configs(config)

    assert len(result) == 1
    winner = result[0]
    assert winner["conditioning_scale"] == 0.9
    assert winner["preprocessor"] == "hed", "the winner's own preprocessor must survive, not the loser's"
    assert winner["preprocessor_params"]["low"] == 2, "the winner's own params must survive, not the loser's"


def test_prepare_controlnet_configs_no_duplicates_passes_through_unchanged():
    config = {
        "controlnets": [
            _cn_config("canny", conditioning_scale=0.4),
            _cn_config("depth", conditioning_scale=0.6),
            _cn_config("openpose", conditioning_scale=0.8),
        ]
    }

    result = _prepare_controlnet_configs(config)

    assert [cfg["model_id"] for cfg in result] == ["canny", "depth", "openpose"]
    assert [cfg["conditioning_scale"] for cfg in result] == [0.4, 0.6, 0.8]


def test_dedupe_controlnet_configs_tie_keeps_first_occurrence():
    configs = [
        _cn_config("canny", conditioning_scale=0.5, preprocessor="first"),
        _cn_config("canny", conditioning_scale=0.5, preprocessor="second"),
    ]

    result = dedupe_controlnet_configs(configs)

    assert len(result) == 1
    assert result[0]["preprocessor"] == "first", "equal scale must keep the first occurrence, not flip-flop"


# ---------------------------------------------------------------------------
# 4-5: _update_controlnet_config
# ---------------------------------------------------------------------------


def test_update_controlnet_config_removes_pre_existing_duplicate_indices():
    """A stream that started with a startup-produced duplicate (pre-fix YAML export,
    or any other caller that skipped dedup) must self-heal on the next live update."""
    module = FakeControlNetModule()
    module.controlnets = [_FakeLoadedModel("canny"), _FakeLoadedModel("canny")]
    module.controlnet_images = [None, None]
    module.controlnet_scales = [0.4, 0.9]
    module.preprocessors = [None, None]
    module.enabled_list = [True, True]

    updater = _make_updater_with_module(module)
    updater._update_controlnet_config([_cn_config("canny", conditioning_scale=0.9)])

    assert len(module.controlnets) == 1, "duplicate index must be removed, not just left stale"
    assert module.controlnet_scales == [0.9]


def test_update_controlnet_config_duplicate_desired_entries_add_only_once():
    """The stale-current_models creation vector: desired_config carrying two entries
    for a model that is NOT currently loaded must not produce two add_controlnet
    calls -- dedupe_controlnet_configs must run on desired_config before the
    add/update loop, not just on what's already loaded."""
    module = FakeControlNetModule()  # nothing loaded yet

    updater = _make_updater_with_module(module)
    updater._update_controlnet_config(
        [
            _cn_config("canny", conditioning_scale=0.4),
            _cn_config("canny", conditioning_scale=0.9),
        ]
    )

    assert module.add_controlnet_calls == 1, "must add exactly once, not once per duplicate desired entry"
    assert len(module.controlnets) == 1
    assert module.controlnet_scales == [0.9], "the higher-weighted duplicate must be the one added"


# ---------------------------------------------------------------------------
# 6: non-duplicated multi-CN update -- regression guard against over-eager dedup
# ---------------------------------------------------------------------------


def test_update_controlnet_config_non_duplicated_setup_left_intact():
    module = FakeControlNetModule()
    module.controlnets = [_FakeLoadedModel("canny"), _FakeLoadedModel("depth")]
    module.controlnet_images = [None, None]
    module.controlnet_scales = [0.5, 0.7]
    module.preprocessors = [None, None]
    module.enabled_list = [True, True]

    updater = _make_updater_with_module(module)
    updater._update_controlnet_config(
        [
            _cn_config("canny", conditioning_scale=0.55),
            _cn_config("depth", conditioning_scale=0.7),
        ]
    )

    assert [cn.model_id for cn in module.controlnets] == ["canny", "depth"]
    assert module.controlnet_scales == [0.55, 0.7]
    assert module.add_controlnet_calls == 0, "no new controlnet should be added"


# ---------------------------------------------------------------------------
# 7: end-to-end -- duplicated YAML -> prepare -> load -> live update
# ---------------------------------------------------------------------------


def test_end_to_end_duplicated_startup_then_live_update_collapses_to_one():
    yaml_config = {
        "controlnets": [
            _cn_config("canny", conditioning_scale=0.4),
            _cn_config("canny", conditioning_scale=0.9),
        ]
    }

    # Startup: YAML -> _prepare_controlnet_configs -> load into the module.
    prepared = _prepare_controlnet_configs(yaml_config)
    assert len(prepared) == 1, "startup dedup must already collapse the duplicate"

    module = FakeControlNetModule()
    for cfg in prepared:
        module.add_controlnet(
            ControlNetConfig(
                model_id=cfg["model_id"],
                preprocessor=cfg.get("preprocessor"),
                conditioning_scale=cfg["conditioning_scale"],
                enabled=cfg.get("enabled", True),
            )
        )
    assert module.controlnet_scales == [0.9]

    # Live update with the same (already-deduped) desired state -- must be a no-op,
    # not an escalation (the original 1.3 -> 1.8 bug). add_controlnet_calls already
    # counts the one direct startup call above, so snapshot it before the live update.
    add_calls_after_startup = module.add_controlnet_calls
    updater = _make_updater_with_module(module)
    updater._update_controlnet_config(prepared)

    assert len(module.controlnets) == 1
    assert module.controlnet_scales == [0.9]
    assert module.add_controlnet_calls == add_calls_after_startup, "already-correct state must not re-add"
