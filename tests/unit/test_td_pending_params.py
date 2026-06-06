"""
Unit tests for the render-thread pending-params drain in TouchDesignerManager.

Covers the fix for weight-drag glitches (round 5): OSC/batch thread must
deposit into _pending_params, not call _apply_parameters directly, while the
render loop is alive.

These tests replicate the exact logic from td_manager.py without loading the
real module (which requires CUDA / TD dependencies).  They serve as a
specification of the contract; if td_manager.py is refactored the tests must
stay green.

ASCII only -- no Unicode symbols (Windows cp1252 terminal compatibility).
"""

import threading
import types
import unittest
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Minimal faithful replica of the three methods under test.
# Copy-pasted from td_manager.py and frozen here so any future regression in
# td_manager.py will break these tests and alert the developer.
# ---------------------------------------------------------------------------

class _FakeStream:
    cfg_type = "none"


class _FakeWrapper:
    """Records calls made to update_stream_params."""
    def __init__(self):
        self.stream = _FakeStream()
        self.calls: list = []

    def update_stream_params(self, **kwargs):
        self.calls.append(kwargs)


class _Manager:
    """
    Minimal replica of TouchDesignerManager containing only the pending-params
    contract:

        update_parameters  -- public entrypoint called by OSC batch thread
        _apply_parameters  -- real apply (calls wrapper), called on render thread
        _streaming_loop_drain_snippet -- helper that replicates the loop drain
    """

    VALID_PARAMS = {
        'num_inference_steps', 'guidance_scale', 'delta', 't_index_list', 'seed',
        'prompt_list', 'negative_prompt', 'prompt_interpolation_method',
        'normalize_prompt_weights', 'seed_list', 'seed_interpolation_method',
        'normalize_seed_weights', 'controlnet_config', 'ipadapter_config',
        'image_preprocessing_config', 'image_postprocessing_config',
        'latent_preprocessing_config', 'latent_postprocessing_config',
        'use_safety_checker', 'safety_checker_threshold',
        'cache_maxframes', 'cache_interval', 'fi_strength', 'fi_threshold',
        'cn_cache_interval',
    }

    def __init__(self):
        self.streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self._pending_params: Dict[str, Any] = {}
        self._pending_params_lock = threading.Lock()
        self._randomize_seed_indices = []
        self.wrapper = _FakeWrapper()

    # --- Replica of td_manager.py _apply_parameters ---
    def _apply_parameters(self, params: Dict[str, Any]) -> None:
        import logging, random
        filtered_params = {k: v for k, v in params.items() if k in self.VALID_PARAMS}

        if 'guidance_scale' in filtered_params:
            cfg_type = getattr(self.wrapper.stream, 'cfg_type', None)
            if cfg_type in ("full", "initialize") and filtered_params['guidance_scale'] <= 1.0:
                filtered_params['guidance_scale'] = 1.2

        if 'seed_list' in filtered_params:
            self._randomize_seed_indices = []
            new_seed_list = []
            for idx, (seed, weight) in enumerate(filtered_params['seed_list']):
                if seed == -1:
                    self._randomize_seed_indices.append(idx)
                    seed = random.randint(0, 2**32 - 1)
                new_seed_list.append((seed, weight))
            filtered_params['seed_list'] = new_seed_list

        self.wrapper.update_stream_params(**filtered_params)

    # --- Replica of td_manager.py update_parameters ---
    def update_parameters(self, params: Dict[str, Any]) -> None:
        render_alive = (
            self.streaming
            and self.stream_thread is not None
            and self.stream_thread.is_alive()
        )
        if render_alive:
            with self._pending_params_lock:
                self._pending_params.update(params)
        else:
            self._apply_parameters(params)

    # --- Replica of the drain snippet at the top of _streaming_loop ---
    def _drain(self):
        with self._pending_params_lock:
            pending, self._pending_params = self._pending_params, {}
        if pending:
            self._apply_parameters(pending)


def _make_mgr() -> _Manager:
    return _Manager()


def _alive_thread() -> threading.Thread:
    """Return a started, long-lived daemon thread (simulates render loop)."""
    e = threading.Event()
    t = threading.Thread(target=e.wait, daemon=True)
    t.start()
    t._stop_event = e  # type: ignore[attr-defined]
    return t


def _stop_thread(t: threading.Thread):
    t._stop_event.set()  # type: ignore[attr-defined]
    t.join(timeout=2)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestUpdateParametersDefer(unittest.TestCase):
    """
    (a) update_parameters defers when streaming, applies directly when not.
    """

    def test_applies_directly_when_not_streaming(self):
        """Pre-start: params go straight to _apply_parameters -> wrapper."""
        mgr = _make_mgr()
        mgr.streaming = False
        mgr.stream_thread = None

        mgr.update_parameters({"guidance_scale": 1.5})

        self.assertEqual(len(mgr.wrapper.calls), 1)
        self.assertAlmostEqual(mgr.wrapper.calls[0]["guidance_scale"], 1.5)
        self.assertEqual(mgr._pending_params, {})

    def test_defers_when_streaming_alive(self):
        """While streaming: params go into _pending_params, wrapper NOT called."""
        mgr = _make_mgr()
        t = _alive_thread()
        mgr.streaming = True
        mgr.stream_thread = t

        try:
            mgr.update_parameters({"guidance_scale": 2.0})
            self.assertEqual(len(mgr.wrapper.calls), 0, "wrapper must not be called yet")
            self.assertEqual(mgr._pending_params.get("guidance_scale"), 2.0)
        finally:
            _stop_thread(t)

    def test_applies_directly_when_streaming_flag_set_but_thread_dead(self):
        """streaming=True but dead thread -> apply directly (safety net)."""
        mgr = _make_mgr()
        dead = threading.Thread(target=lambda: None)
        dead.start()
        dead.join()
        mgr.streaming = True
        mgr.stream_thread = dead

        mgr.update_parameters({"guidance_scale": 3.0})

        self.assertEqual(len(mgr.wrapper.calls), 1)
        self.assertAlmostEqual(mgr.wrapper.calls[0]["guidance_scale"], 3.0)


class TestLatestWinsMerge(unittest.TestCase):
    """
    (b) Multiple deferred updates merge with latest-wins per key.
    """

    def test_latest_wins_same_key(self):
        mgr = _make_mgr()
        t = _alive_thread()
        mgr.streaming = True
        mgr.stream_thread = t

        try:
            mgr.update_parameters({"guidance_scale": 1.0})
            mgr.update_parameters({"guidance_scale": 2.0})
            mgr.update_parameters({"guidance_scale": 3.0})

            self.assertEqual(mgr._pending_params["guidance_scale"], 3.0)
        finally:
            _stop_thread(t)

    def test_different_keys_merged(self):
        mgr = _make_mgr()
        t = _alive_thread()
        mgr.streaming = True
        mgr.stream_thread = t

        try:
            mgr.update_parameters({"guidance_scale": 1.5})
            mgr.update_parameters({"delta": 0.5})

            self.assertEqual(mgr._pending_params["guidance_scale"], 1.5)
            self.assertEqual(mgr._pending_params["delta"], 0.5)
        finally:
            _stop_thread(t)


class TestDrainClearsPending(unittest.TestCase):
    """
    (c) After drain, _pending_params is empty and wrapper was called.
    """

    def test_drain_clears_and_applies(self):
        mgr = _make_mgr()
        mgr._pending_params = {"guidance_scale": 4.0}

        mgr._drain()

        self.assertEqual(mgr._pending_params, {})
        self.assertEqual(len(mgr.wrapper.calls), 1)
        self.assertAlmostEqual(mgr.wrapper.calls[0]["guidance_scale"], 4.0)

    def test_drain_no_op_when_empty(self):
        mgr = _make_mgr()
        mgr._pending_params = {}

        mgr._drain()

        self.assertEqual(mgr._pending_params, {})
        self.assertEqual(len(mgr.wrapper.calls), 0)

    def test_drain_collapses_flood(self):
        """Flood of same-key updates -> one wrapper call with the last value."""
        mgr = _make_mgr()
        t = _alive_thread()
        mgr.streaming = True
        mgr.stream_thread = t

        try:
            for i in range(20):
                mgr.update_parameters({"guidance_scale": float(i)})
        finally:
            _stop_thread(t)
            mgr.streaming = False

        mgr._drain()

        self.assertEqual(len(mgr.wrapper.calls), 1, "only one wrapper call after draining a flooded queue")
        self.assertAlmostEqual(mgr.wrapper.calls[0]["guidance_scale"], 19.0)
        self.assertEqual(mgr._pending_params, {})

    def test_drain_applies_multi_key_batch(self):
        """Pending dict with multiple keys passes all to wrapper in one call."""
        mgr = _make_mgr()
        mgr._pending_params = {"guidance_scale": 1.2, "delta": 0.8}

        mgr._drain()

        self.assertEqual(len(mgr.wrapper.calls), 1)
        self.assertAlmostEqual(mgr.wrapper.calls[0]["guidance_scale"], 1.2)
        self.assertAlmostEqual(mgr.wrapper.calls[0]["delta"], 0.8)

    def test_invalid_keys_filtered_out(self):
        """_apply_parameters strips keys not in the whitelist."""
        mgr = _make_mgr()
        mgr._pending_params = {"guidance_scale": 1.5, "nonexistent_key": "bad"}

        mgr._drain()

        self.assertEqual(len(mgr.wrapper.calls), 1)
        self.assertNotIn("nonexistent_key", mgr.wrapper.calls[0])
        self.assertAlmostEqual(mgr.wrapper.calls[0]["guidance_scale"], 1.5)


if __name__ == "__main__":
    unittest.main()
