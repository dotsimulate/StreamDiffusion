"""update_stream_params logs weight-normalization flags only when they change.

TD resends the whole prompt block every frame during startup; the unconditional INFO line
fired ~30 times per launch.
"""

import logging
import threading
from types import SimpleNamespace

import pytest

from streamdiffusion.stream_parameter_updater import StreamParameterUpdater


@pytest.fixture
def updater():
    upd = StreamParameterUpdater.__new__(StreamParameterUpdater)
    upd._update_lock = threading.RLock()
    upd.normalize_prompt_weights = True
    upd.normalize_seed_weights = True
    upd.stream = SimpleNamespace(kvo_cache=None, use_feature_injection=False)
    upd._last_prompt_interpolation_method = "average"
    upd._last_seed_interpolation_method = "average"
    # The re-blend itself is covered elsewhere; only the logging is under test here.
    upd.prompt_blends = []
    upd.seed_blends = []
    upd._apply_prompt_blending = upd.prompt_blends.append
    upd._apply_seed_blending = upd.seed_blends.append
    upd._embedding_orchestrator = None
    upd._preprocessing_orchestrator = None
    return upd


def _messages(caplog, needle):
    return [r.getMessage() for r in caplog.records if needle in r.getMessage()]


def test_unchanged_prompt_normalization_is_not_logged(updater, caplog):
    with caplog.at_level(logging.INFO, logger="streamdiffusion.stream_parameter_updater"):
        updater.update_stream_params(normalize_prompt_weights=True)
        updater.update_stream_params(normalize_prompt_weights=True)
    assert _messages(caplog, "Prompt weight normalization") == []


def test_changed_prompt_normalization_is_logged_once(updater, caplog):
    with caplog.at_level(logging.INFO, logger="streamdiffusion.stream_parameter_updater"):
        updater.update_stream_params(normalize_prompt_weights=False)
        updater.update_stream_params(normalize_prompt_weights=False)
    assert len(_messages(caplog, "Prompt weight normalization set to False")) == 1
    assert updater.normalize_prompt_weights is False


def test_changed_seed_normalization_is_logged_once(updater, caplog):
    with caplog.at_level(logging.INFO, logger="streamdiffusion.stream_parameter_updater"):
        updater.update_stream_params(normalize_seed_weights=False)
        updater.update_stream_params(normalize_seed_weights=False)
    assert len(_messages(caplog, "Seed weight normalization set to False")) == 1


def test_unchanged_flags_do_not_reblend(updater):
    updater.update_stream_params(normalize_prompt_weights=True, normalize_seed_weights=True)
    updater.update_stream_params(normalize_prompt_weights=True, normalize_seed_weights=True)
    assert updater.prompt_blends == []
    assert updater.seed_blends == []


def test_changed_prompt_flag_reblends_once(updater):
    updater.update_stream_params(normalize_prompt_weights=False)
    updater.update_stream_params(normalize_prompt_weights=False)
    assert updater.prompt_blends == ["average"]
    assert updater.seed_blends == []


def test_changed_seed_flag_reblends_once(updater):
    updater.update_stream_params(normalize_seed_weights=False)
    updater.update_stream_params(normalize_seed_weights=False)
    assert updater.seed_blends == ["average"]
    assert updater.prompt_blends == []


def test_interpolation_method_still_reblends_even_with_unchanged_flag(updater):
    updater.update_stream_params(prompt_interpolation_method="slerp", normalize_prompt_weights=True)
    assert updater.prompt_blends == ["slerp"]
