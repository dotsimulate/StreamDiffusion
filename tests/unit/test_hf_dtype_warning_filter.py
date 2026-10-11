"""The diffusers-fork-induced `torch_dtype` deprecation warning is filtered, others are not."""

import logging

import pytest

from streamdiffusion._patches import hf_dtype_warning_filter as mod

MESSAGE = "`torch_dtype` is deprecated! Use `dtype` instead!"
LOGGER_NAME = "transformers.modeling_utils"


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def target():
    mod.apply()
    logger = logging.getLogger(LOGGER_NAME)
    handler = _Collect()
    old_level, old_propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False  # collect on our handler only
    logger.collected = handler.messages
    yield logger
    del logger.collected
    logger.propagate = old_propagate
    logger.setLevel(old_level)
    logger.removeHandler(handler)


def _in_fake_diffusers_load_sub_model(logger, message):
    src = "def load_sub_model(logger, message):\n    logger.warning(message)\n"
    ns = {"__name__": "diffusers.pipelines.fake"}
    exec(compile(src, "fake_diffusers.py", "exec"), ns)
    ns["load_sub_model"](logger, message)


def test_message_from_diffusers_load_sub_model_is_dropped(target):
    _in_fake_diffusers_load_sub_model(target, MESSAGE)
    assert target.collected == []


def test_message_from_other_caller_is_kept(target):
    target.warning(MESSAGE)
    assert target.collected == [MESSAGE]


def test_message_from_same_named_function_outside_diffusers_is_kept(target):
    src = "def load_sub_model(logger, message):\n    logger.warning(message)\n"
    ns = {"__name__": "some_app.loader"}
    exec(compile(src, "fake_app.py", "exec"), ns)
    ns["load_sub_model"](target, MESSAGE)
    assert target.collected == [MESSAGE]


def test_other_warning_inside_load_sub_model_is_kept(target):
    _in_fake_diffusers_load_sub_model(target, "something else went wrong")
    assert target.collected == ["something else went wrong"]


def test_apply_is_idempotent():
    mod.apply()
    mod.apply()
    filters = [f for f in logging.getLogger(LOGGER_NAME).filters if isinstance(f, mod._DiffusersTorchDtypeFilter)]
    assert len(filters) == 1
