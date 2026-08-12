"""Regression tests for two parameter-binding bugs fixed in Stage 1:

  1. ``StreamDiffusion.__init__`` constructed ``StreamParameterUpdater`` with three
     *positional* args (``pipeline.py:156``), so the normalize flags landed on the
     wrong fields (``wrapper`` <- prompt flag, ``normalize_prompt_weights`` <- seed
     flag, ``normalize_seed_weights`` stuck at its default ``True``).  The updater
     ``__init__`` is now keyword-only past ``stream_diffusion`` so this mis-binding
     is impossible to reintroduce silently.

  2. ``create_wrapper_from_config`` passed ``interpolation_method=`` to
     ``wrapper.update_stream_params`` on the seed-only-blending path
     (``config.py:99``); that kwarg does not exist (the real name is
     ``seed_interpolation_method``), so any config with ``seed_blending`` but no
     ``prompt_blending`` raised ``TypeError`` at construction time.

All tests run on CPU and construct no real pipeline.
"""

import types
from unittest.mock import patch

import pytest
import torch

from streamdiffusion.stream_parameter_updater import StreamParameterUpdater

# ---------------------------------------------------------------------------
# Fixtures — mirror the minimal-fake-stream pattern from
# tests/unit/test_prompt_interpolation.py so __init__ runs without a real pipeline.
# ---------------------------------------------------------------------------


def _fake_stream():
    """Return a minimal namespace that looks like a StreamDiffusion instance."""
    stream = types.SimpleNamespace()
    stream.device = torch.device("cpu")
    stream.dtype = torch.float32
    stream.batch_size = 1
    stream.cfg_type = "none"
    stream.guidance_scale = 1.0
    stream.prompt_embeds = None
    stream.negative_prompt_embeds = None
    stream._preprocessing_orchestrator = None
    stream.embedding_hooks = []
    return stream


def _make_updater(**kwargs) -> StreamParameterUpdater:
    """Construct an updater with a fake stream, no-op'ing attach_orchestrator."""
    stream = _fake_stream()
    orig_attach = StreamParameterUpdater.attach_orchestrator

    def _noop_attach(self, s):
        self._preprocessing_orchestrator = None

    StreamParameterUpdater.attach_orchestrator = _noop_attach
    try:
        updater = StreamParameterUpdater(stream, **kwargs)
    finally:
        StreamParameterUpdater.attach_orchestrator = orig_attach

    updater._embedding_orchestrator = None
    return updater


# ---------------------------------------------------------------------------
# Bug 1 — normalize flags must bind to the correct fields.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt_flag, seed_flag",
    [(True, False), (False, True), (False, False), (True, True)],
)
def test_normalize_flags_bind_to_correct_fields(prompt_flag, seed_flag):
    updater = _make_updater(
        normalize_prompt_weights=prompt_flag,
        normalize_seed_weights=seed_flag,
    )
    assert updater.normalize_prompt_weights is prompt_flag
    assert updater.normalize_seed_weights is seed_flag
    # wrapper stays None at construction — the real reference is attached later
    # by StreamDiffusionWrapper (wrapper.py:454), never by a mis-bound bool.
    assert updater.wrapper is None


def test_getters_echo_constructed_flags():
    updater = _make_updater(normalize_prompt_weights=False, normalize_seed_weights=True)
    assert updater.get_normalize_prompt_weights() is False
    assert updater.get_normalize_seed_weights() is True


def test_positional_flag_binding_is_rejected():
    """The keyword-only barrier makes the original bug a hard TypeError."""
    stream = _fake_stream()
    with pytest.raises(TypeError):
        StreamParameterUpdater(stream, False, False)


# ---------------------------------------------------------------------------
# Bug 2 — seed-only blending config must reach update_stream_params with the
# correct kwarg name (seed_interpolation_method), not interpolation_method.
# ---------------------------------------------------------------------------


class _FakeWrapper:
    """Stub with the *real* update_stream_params kwarg name so the wrong name
    (interpolation_method) would raise TypeError, faithfully reproducing the bug."""

    def __init__(self, **kwargs):
        self.update_calls = []

    def prepare(self, **kwargs):  # not exercised on the seed-only path, present for safety
        pass

    def update_stream_params(self, *, seed_list=None, seed_interpolation_method="average"):
        self.update_calls.append({"seed_list": seed_list, "seed_interpolation_method": seed_interpolation_method})


def test_seed_only_blending_uses_seed_interpolation_method():
    from streamdiffusion import config as config_mod

    seed_list = [(1, 0.5), (2, 0.5)]
    cfg = {"seed_blending": {"seed_list": seed_list, "interpolation_method": "average"}}

    with patch("streamdiffusion.StreamDiffusionWrapper", _FakeWrapper):
        wrapper = config_mod.create_wrapper_from_config(cfg)

    assert isinstance(wrapper, _FakeWrapper)
    assert len(wrapper.update_calls) == 1
    call = wrapper.update_calls[0]
    assert call["seed_list"] == seed_list
    assert call["seed_interpolation_method"] == "average"
