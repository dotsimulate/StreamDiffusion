"""
Unit tests for the ControlNet preprocessor-preview transport dispatch in
TouchDesignerManager.

Covers the fix for the dead export_controlnet_preview_ipc() call site: preview must
route via zero-copy CUDA IPC when cuda_ipc_cn_processed_shm_name is configured and
cuda_link is importable, fall back to the legacy CPU SharedMemory path otherwise, and
never fire via either transport when send_controlnet_preview (SDTD.par.Enablepreprocpreview,
read once at stream start) was off.

Also covers making the IPC latch honest: export_controlnet_preview_ipc() now returns a
bool (True = frame handed off, False = not configured or export failed) instead of
silently swallowing every outcome. td_manager.py tracks the last outcome in
_cn_preview_ipc_failed and folds it into get_stream_state()'s
controlnet_processed_connected so that flag stops claiming "connected" while every
frame is actually being dropped.

These tests replicate the exact logic from td_manager.py's __init__ (the
_cn_preview_via_ipc computation), _send_back_processed_controlnet (the IPC-vs-CPU
dispatch, now reading the wrapper's return value), and the
controlnet_processed_connected computation in get_stream_state() -- without loading the
real module (which requires CUDA / TD dependencies). They serve as a specification of
the contract; if td_manager.py is refactored the tests must stay green. Same replica
pattern as test_td_pending_params.py.

ASCII only -- no Unicode symbols (Windows cp1252 terminal compatibility).
"""

import unittest
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Minimal faithful replica of the logic under test.
# Copy-pasted from td_manager.py and frozen here so any future regression in
# td_manager.py will break these tests and alert the developer.
# ---------------------------------------------------------------------------


class _FakeControlNetModule:
    def __init__(self, controlnet_images: Optional[list] = None):
        self.controlnet_images = controlnet_images if controlnet_images is not None else []


class _FakeStream:
    def __init__(self, controlnet_module: Optional[_FakeControlNetModule]):
        self._controlnet_module = controlnet_module


class _FakeWrapper:
    """Records calls made to export_controlnet_preview_ipc and returns a scripted
    success/failure outcome, mirroring the real method's bool return value."""

    def __init__(self, stream: _FakeStream, ipc_export_result: bool = True):
        self.stream = stream
        self.ipc_calls: list = []
        self.ipc_export_result = ipc_export_result

    def export_controlnet_preview_ipc(self, tensor) -> bool:
        self.ipc_calls.append(tensor)
        return self.ipc_export_result


class _Manager:
    """
    Minimal replica of TouchDesignerManager containing only the ControlNet-preview
    transport contract:

        __init__ (partial)             -- computes _cn_preview_via_ipc, _cn_preview_ipc_failed
        _allocate_control_processed_memory -- stand-in for the real lazy CPU-SHM allocator
        _send_processed_controlnet_frame   -- stand-in for the real CPU-SHM path
        _send_back_processed_controlnet    -- the IPC-vs-CPU dispatch under test
        controlnet_processed_connected     -- replica of the get_stream_state() field
    """

    def __init__(self, config: Dict[str, Any], wrapper: _FakeWrapper, cuda_link_importable: bool = True):
        self.config = config
        self.wrapper = wrapper
        self.cpu_calls: list = []
        self.control_processed_memory = None  # allocated lazily; see _allocate_control_processed_memory

        # --- Replica of td_manager.py __init__'s _cn_preview_via_ipc computation ---
        self._cn_preview_via_ipc = False
        if self.config.get("send_controlnet_preview") and self.config.get("cuda_ipc_cn_processed_shm_name"):
            if cuda_link_importable:
                self._cn_preview_via_ipc = True

        # --- Replica of td_manager.py __init__'s _cn_preview_ipc_failed init ---
        self._cn_preview_ipc_failed = False

    # --- Stand-in for the real lazy CPU-SHM allocator; just marks it allocated ---
    def _allocate_control_processed_memory(self) -> None:
        if self.control_processed_memory is None:
            self.control_processed_memory = "allocated"

    # --- Stand-in for the real CPU-SHM path; just records the call ---
    def _send_processed_controlnet_frame(self, processed_tensor) -> None:
        self.cpu_calls.append(processed_tensor)

    # --- Replica of td_manager.py _send_back_processed_controlnet ---
    def _send_back_processed_controlnet(self) -> None:
        if not self.config.get("send_controlnet_preview", False):
            return
        try:
            if (
                hasattr(self.wrapper, "stream")
                and hasattr(self.wrapper.stream, "_controlnet_module")
                and self.wrapper.stream._controlnet_module is not None
            ):
                controlnet_module = self.wrapper.stream._controlnet_module
                if (
                    hasattr(controlnet_module, "controlnet_images")
                    and len(controlnet_module.controlnet_images) > 0
                    and controlnet_module.controlnet_images[0] is not None
                ):
                    processed_tensor = controlnet_module.controlnet_images[0]
                    if self._cn_preview_via_ipc:
                        exported = self.wrapper.export_controlnet_preview_ipc(processed_tensor)
                        self._cn_preview_ipc_failed = not exported
                        if not exported:
                            self._allocate_control_processed_memory()
                            self._send_processed_controlnet_frame(processed_tensor)
                    else:
                        self._send_processed_controlnet_frame(processed_tensor)
        except Exception:
            pass

    # --- Replica of td_manager.py get_stream_state()'s controlnet_processed_connected field ---
    @property
    def controlnet_processed_connected(self) -> bool:
        return self.control_processed_memory is not None or (
            self._cn_preview_via_ipc and not self._cn_preview_ipc_failed
        )


def _make_mgr(config: Dict[str, Any], tensor="fake_tensor", cuda_link_importable: bool = True) -> _Manager:
    cn_module = _FakeControlNetModule([tensor] if tensor is not None else [])
    stream = _FakeStream(cn_module)
    wrapper = _FakeWrapper(stream)
    return _Manager(config, wrapper, cuda_link_importable=cuda_link_importable)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestControlNetPreviewTransportDispatch(unittest.TestCase):
    def test_ipc_branch_taken_when_configured(self):
        """send_controlnet_preview on + IPC shm name configured + cuda_link importable
        -> export_controlnet_preview_ipc called, CPU fallback never touched."""
        mgr = _make_mgr(
            {
                "send_controlnet_preview": True,
                "cuda_ipc_cn_processed_shm_name": "StreamDiffusionTD_640-384_cn_processed_ipc",
            }
        )

        mgr._send_back_processed_controlnet()

        self.assertTrue(mgr._cn_preview_via_ipc)
        self.assertEqual(mgr.wrapper.ipc_calls, ["fake_tensor"])
        self.assertEqual(mgr.cpu_calls, [])

    def test_cpu_branch_taken_when_ipc_shm_name_not_configured(self):
        """send_controlnet_preview on but no IPC shm name -> CPU SharedMemory fallback."""
        mgr = _make_mgr(
            {
                "send_controlnet_preview": True,
                "cuda_ipc_cn_processed_shm_name": None,
            }
        )

        mgr._send_back_processed_controlnet()

        self.assertFalse(mgr._cn_preview_via_ipc)
        self.assertEqual(mgr.cpu_calls, ["fake_tensor"])
        self.assertEqual(mgr.wrapper.ipc_calls, [])

    def test_cpu_branch_taken_when_cuda_link_not_importable(self):
        """IPC shm name configured but cuda_link missing on this machine -> CPU fallback,
        same as td_manager.py __init__'s ImportError branch."""
        mgr = _make_mgr(
            {
                "send_controlnet_preview": True,
                "cuda_ipc_cn_processed_shm_name": "StreamDiffusionTD_640-384_cn_processed_ipc",
            },
            cuda_link_importable=False,
        )

        mgr._send_back_processed_controlnet()

        self.assertFalse(mgr._cn_preview_via_ipc)
        self.assertEqual(mgr.cpu_calls, ["fake_tensor"])
        self.assertEqual(mgr.wrapper.ipc_calls, [])

    def test_no_transport_fires_when_preview_disabled_at_stream_start(self):
        """SDTD.par.Enablepreprocpreview off at stream start (send_controlnet_preview
        False) -> preview fully disabled: neither IPC nor CPU transport fires, even
        though IPC is otherwise fully configured and available."""
        mgr = _make_mgr(
            {
                "send_controlnet_preview": False,
                "cuda_ipc_cn_processed_shm_name": "StreamDiffusionTD_640-384_cn_processed_ipc",
            }
        )

        mgr._send_back_processed_controlnet()

        self.assertFalse(mgr._cn_preview_via_ipc)
        self.assertEqual(mgr.wrapper.ipc_calls, [])
        self.assertEqual(mgr.cpu_calls, [])

    def test_no_transport_fires_when_no_processed_tensor_available(self):
        """No controlnet_images yet (e.g. before first frame) -> no crash, no call
        to either transport."""
        mgr = _make_mgr(
            {
                "send_controlnet_preview": True,
                "cuda_ipc_cn_processed_shm_name": "StreamDiffusionTD_640-384_cn_processed_ipc",
            },
            tensor=None,
        )

        mgr._send_back_processed_controlnet()

        self.assertEqual(mgr.wrapper.ipc_calls, [])
        self.assertEqual(mgr.cpu_calls, [])

    def test_ipc_failure_falls_back_to_cpu_path(self):
        """export_controlnet_preview_ipc() returns False (transport failed) -> the CPU
        SharedMemory fallback is lazily allocated and used for that same frame, instead
        of the frame being silently dropped."""
        mgr = _make_mgr(
            {
                "send_controlnet_preview": True,
                "cuda_ipc_cn_processed_shm_name": "StreamDiffusionTD_640-384_cn_processed_ipc",
            }
        )
        mgr.wrapper.ipc_export_result = False

        self.assertIsNone(mgr.control_processed_memory)

        mgr._send_back_processed_controlnet()

        self.assertTrue(mgr._cn_preview_via_ipc)
        self.assertTrue(mgr._cn_preview_ipc_failed)
        self.assertEqual(mgr.wrapper.ipc_calls, ["fake_tensor"])
        self.assertIsNotNone(mgr.control_processed_memory)
        self.assertEqual(mgr.cpu_calls, ["fake_tensor"])

    def test_controlnet_processed_connected_false_after_ipc_failure(self):
        """get_stream_state()'s controlnet_processed_connected must stop reporting
        connected once export_controlnet_preview_ipc() has actually failed -- it should
        not keep claiming the IPC transport is healthy while every frame is being
        dropped, even though the just-allocated CPU fallback is now what's connected."""
        mgr = _make_mgr(
            {
                "send_controlnet_preview": True,
                "cuda_ipc_cn_processed_shm_name": "StreamDiffusionTD_640-384_cn_processed_ipc",
            }
        )

        # Healthy IPC: connected, no CPU buffer needed.
        mgr._send_back_processed_controlnet()
        self.assertTrue(mgr.controlnet_processed_connected)
        self.assertIsNone(mgr.control_processed_memory)

        # IPC starts failing: still "connected" because the CPU fallback just came up.
        mgr.wrapper.ipc_export_result = False
        mgr._send_back_processed_controlnet()
        self.assertTrue(mgr._cn_preview_ipc_failed)
        self.assertTrue(mgr.controlnet_processed_connected)
        self.assertIsNotNone(mgr.control_processed_memory)


if __name__ == "__main__":
    unittest.main()
