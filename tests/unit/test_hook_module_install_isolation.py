"""One FX processor that fails to load must not drop the other FX of the same stage.

User report (Discord, 0.4.0 and 0.4.2): the log shows
"Processor 'feedback_loop' (...) not found in any module" and no FX runs. One cause:
wrapper._load_model added all processors of a stage inside a single try, so a single
unknown or broken processor (for example an FX name whose custom_processors file is
missing) aborted the whole stage. Then the healthy feedback_loop was missing too.
"""

import logging
import types

import pytest
import torch

from streamdiffusion.modules.image_processing_module import ImagePreprocessingModule
from streamdiffusion.preprocessing.processors import (
    PipelineAwareProcessor,
    _preprocessor_registry,
    register_preprocessor,
)
from streamdiffusion.wrapper import _IMAGE_HOOKS, _install_hook_module, _install_hook_stages


class FeedbackLoopPreprocessor(PipelineAwareProcessor):
    """Stand-in with the class name the TD OSC handler looks for (feedback_loop)."""

    zoom = 1.0

    def _process_core(self, image):
        return image


@pytest.fixture
def stream():
    return types.SimpleNamespace(
        width=512,
        height=512,
        device=torch.device("cpu"),
        dtype=torch.float32,
        image_preprocessing_hooks=[],
        image_postprocessing_hooks=[],
    )


@pytest.fixture
def fake_feedback_loop():
    name = "test_feedback_loop_isolation"
    register_preprocessor(name, FeedbackLoopPreprocessor)
    yield name
    _preprocessor_registry.pop(name, None)


def _cfg(*types_):
    return {
        "enabled": True,
        "processors": [{"type": t, "order": i, "enabled": True, "params": {}} for i, t in enumerate(types_, 1)],
    }


def test_unknown_sibling_does_not_drop_working_processor(stream, fake_feedback_loop, caplog):
    with caplog.at_level(logging.ERROR, logger="streamdiffusion.wrapper"):
        module = _install_hook_module(
            stream,
            ImagePreprocessingModule,
            _cfg(fake_feedback_loop, "no_such_fx"),
            "_image_preprocessing_module",
        )

    assert stream._image_preprocessing_module is module
    assert [type(p).__name__ for p in module.processors] == ["FeedbackLoopPreprocessor"]
    assert "no_such_fx" in caplog.text  # the failing FX is named in the log


def test_unregistered_only_processor_logs_name_and_keeps_stage(stream, caplog):
    with caplog.at_level(logging.ERROR, logger="streamdiffusion.wrapper"):
        module = _install_hook_module(
            stream, ImagePreprocessingModule, _cfg("no_such_fx"), "_image_preprocessing_module"
        )

    assert module is not None and module.processors == []
    assert "'no_such_fx'" in caplog.text and "custom_processors" in caplog.text


@pytest.mark.parametrize("cfg", [None, {"enabled": False, "processors": []}], ids=["absent", "disabled"])
def test_absent_or_disabled_stage_installs_nothing(stream, cfg):
    assert _install_hook_module(stream, ImagePreprocessingModule, cfg, "_image_preprocessing_module") is None
    assert not hasattr(stream, "_image_preprocessing_module")
    assert stream.image_preprocessing_hooks == []


def test_failed_module_import_disables_only_its_own_stage(stream, fake_feedback_loop, caplog):
    with caplog.at_level(logging.ERROR, logger="streamdiffusion.wrapper"):
        _install_hook_stages(
            stream,
            (
                (_cfg(fake_feedback_loop), _IMAGE_HOOKS, "ImagePreprocessingModule", "_image_preprocessing_module"),
                (_cfg("x"), "streamdiffusion.no_such_hook_module", "LatentPreprocessingModule", "_latent_pre"),
            ),
        )

    assert [type(p).__name__ for p in stream._image_preprocessing_module.processors] == ["FeedbackLoopPreprocessor"]
    assert not hasattr(stream, "_latent_pre")
    assert "Failed to install LatentPreprocessingModule" in caplog.text
