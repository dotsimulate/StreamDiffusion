"""
Regression tests for Sub-phase 5.2 (output/PIL/IPC sync-free path,
docs/perf_bestpractices_audit_2026-07-10.md follow-ups #2/#... — see the
Phase 5 continuation plan).

Covers three independent output-side fixes, each converted from a
per-frame-allocation formula to a persistent-buffer / pinned-readback
formula. The reference (`_reference_*`) helpers below are frozen copies of
the pre-5.2 code, used as the numeric-parity oracle: every touched method
must produce byte-identical output to its old formula on the same input.

  - 5f: `_ipc_pack_rgba` / `_ipc_pack_unit_rgba` (wrapper.py) — persistent
    HWC x4 GPU buffer instead of a fresh `torch.full_like` alpha + `torch.cat`
    every frame.
  - 5d: `_tensor_to_pil_optimized` (wrapper.py) — routes through the shared
    `_output_pin_buf` / `_d2h_event` pinned-buffer + Event machinery instead
    of a blocking, unpinned `.cpu()` into pageable memory.
  - 5e: `_tensor_to_pil_safe` (preprocessing_orchestrator.py) — CPU-first
    reorder: the D2H transfer now happens before the tensor.min()/.max()
    range-check syncs, not after (kills 2 of 3 host syncs; same template as
    preprocessing/processors/base.py:tensor_to_pil).

5d and 5f exercise `torch.cuda.Event()` / `pin_memory=True`, which require a
real CUDA device — the whole module is skipped when unavailable.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from streamdiffusion._ipc_export import PackRing
from streamdiffusion.preprocessing.preprocessing_orchestrator import PreprocessingOrchestrator
from streamdiffusion.wrapper import StreamDiffusionWrapper

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Sub-phase 5.2 output paths require CUDA (pin_memory / cuda.Event)",
)


# ---------------------------------------------------------------------------
# helpers: minimal shells + frozen pre-5.2 reference formulas
# ---------------------------------------------------------------------------


def _make_wrapper_shell():
    """Minimal StreamDiffusionWrapper shell wired only with the attrs the
    touched methods read (object.__new__ pattern, see test_wrapper_exception_hygiene.py)."""
    w = object.__new__(StreamDiffusionWrapper)
    w._output_pin_buf = None
    w._d2h_event = None
    w._ipc_pack_ring = PackRing()
    w._ipc_pack_unit_ring = PackRing()
    w._cuda_ipc_exporter = None
    w._cuda_ipc_cn_exporter = None
    w._cuda_ipc_num_slots = 2
    w._cuda_ipc_async = False
    return w


def _reference_ipc_pack_rgba(wrapper, image_tensor):
    """Pre-5f formula: fresh alpha (full_like) + cat every call."""
    denorm = wrapper._denormalize_on_gpu(image_tensor)
    if denorm.dim() == 4:
        denorm = denorm[0]
    rgb_u8 = (denorm * 255).clamp(0, 255).to(torch.uint8)
    rgb_hwc = rgb_u8.permute(1, 2, 0).contiguous()
    alpha = torch.full_like(rgb_hwc[..., :1], 255)
    return torch.cat([rgb_hwc[..., 2:3], rgb_hwc[..., 1:2], rgb_hwc[..., 0:1], alpha], dim=-1).contiguous()


def _reference_ipc_pack_unit_rgba(image_tensor):
    """Pre-5f formula for the CN-preview twin (skips denormalize)."""
    t = image_tensor
    if t.dim() == 4:
        t = t[0]
    rgb_u8 = (t * 255).clamp(0, 255).to(torch.uint8)
    rgb_hwc = rgb_u8.permute(1, 2, 0).contiguous()
    alpha = torch.full_like(rgb_hwc[..., :1], 255)
    return torch.cat([rgb_hwc[..., 2:3], rgb_hwc[..., 1:2], rgb_hwc[..., 0:1], alpha], dim=-1).contiguous()


def _reference_tensor_to_pil_optimized(wrapper, image_tensor):
    """Pre-5d formula: blocking, unpinned .cpu()."""
    denormalized = wrapper._denormalize_on_gpu(image_tensor)
    uint8_tensor = (denormalized * 255).clamp(0, 255).to(torch.uint8)
    cpu_tensor = uint8_tensor.cpu()
    cpu_tensor = cpu_tensor.permute(0, 2, 3, 1)
    pil_images = []
    for i in range(cpu_tensor.shape[0]):
        img_array = cpu_tensor[i].numpy()
        if img_array.shape[-1] == 1:
            pil_images.append(Image.fromarray(img_array.squeeze(-1), mode="L"))
        else:
            pil_images.append(Image.fromarray(img_array))
    return pil_images


def _reference_tensor_to_pil_safe(tensor):
    """Pre-5e formula: range-check syncs (tensor.min()/.max()) before the D2H transfer."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3 and tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    if tensor.min() < 0:
        tensor = (tensor / 2.0 + 0.5).clamp(0, 1)
    if tensor.max() <= 1.0:
        tensor = tensor * 255.0
    numpy_image = tensor.detach().cpu().numpy().astype(np.uint8)
    return Image.fromarray(numpy_image)


# ---------------------------------------------------------------------------
# 5f: _ipc_pack_rgba / _ipc_pack_unit_rgba
# ---------------------------------------------------------------------------


class TestIpcPackPersistentBuffer:
    def test_ipc_pack_rgba_byte_identical_to_reference(self):
        wrapper = _make_wrapper_shell()
        torch.manual_seed(0)
        image_tensor = torch.rand(1, 3, 64, 96, device="cuda") * 2 - 1  # diffusion range [-1, 1]

        expected = _reference_ipc_pack_rgba(wrapper, image_tensor)
        actual = wrapper._ipc_pack_rgba(image_tensor)

        assert torch.equal(actual.cpu(), expected.cpu())

    def test_ipc_pack_rgba_reuses_same_buffer_across_calls(self):
        """5f's entire point: same shape -> same underlying allocation, no per-frame alloc."""
        wrapper = _make_wrapper_shell()
        t1 = torch.rand(1, 3, 64, 96, device="cuda") * 2 - 1
        t2 = torch.rand(1, 3, 64, 96, device="cuda") * 2 - 1

        buf1 = wrapper._ipc_pack_rgba(t1)
        ptr1 = buf1.data_ptr()
        buf2 = wrapper._ipc_pack_rgba(t2)

        assert buf2.data_ptr() == ptr1, "expected the persistent buffer to be reused, not reallocated"

    def test_ipc_pack_rgba_reallocs_on_shape_change(self):
        wrapper = _make_wrapper_shell()
        t1 = torch.rand(1, 3, 64, 96, device="cuda") * 2 - 1
        t2 = torch.rand(1, 3, 32, 48, device="cuda") * 2 - 1

        buf1 = wrapper._ipc_pack_rgba(t1)
        assert buf1.shape[:2] == (64, 96)
        buf2 = wrapper._ipc_pack_rgba(t2)
        assert buf2.shape[:2] == (32, 48)

    def test_ipc_pack_unit_rgba_byte_identical_to_reference(self):
        wrapper = _make_wrapper_shell()
        torch.manual_seed(1)
        image_tensor = torch.rand(1, 3, 48, 64, device="cuda")  # already [0, 1], no denorm

        expected = _reference_ipc_pack_unit_rgba(image_tensor)
        actual = wrapper._ipc_pack_unit_rgba(image_tensor)

        assert torch.equal(actual.cpu(), expected.cpu())

    def test_ipc_pack_rgba_and_unit_rgba_use_independent_buffers(self):
        """The two packers must not alias the same persistent buffer (separate exporters)."""
        wrapper = _make_wrapper_shell()
        t = torch.rand(1, 3, 40, 56, device="cuda")

        buf_main = wrapper._ipc_pack_rgba(t * 2 - 1)
        buf_unit = wrapper._ipc_pack_unit_rgba(t)

        assert buf_main.data_ptr() != buf_unit.data_ptr()


# ---------------------------------------------------------------------------
# 5d: _tensor_to_pil_optimized
# ---------------------------------------------------------------------------


class TestTensorToPilOptimizedPinnedReadback:
    def test_byte_identical_to_reference(self):
        wrapper = _make_wrapper_shell()
        torch.manual_seed(2)
        image_tensor = torch.rand(2, 3, 40, 56, device="cuda") * 2 - 1

        expected = _reference_tensor_to_pil_optimized(wrapper, image_tensor)
        actual = wrapper._tensor_to_pil_optimized(image_tensor)

        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert np.array_equal(np.array(a), np.array(e))

    def test_reuses_output_pin_buf_across_calls(self):
        """5d shares _output_pin_buf/_d2h_event with the 'np' output path (same shape/dtype guard)."""
        wrapper = _make_wrapper_shell()
        t1 = torch.rand(1, 3, 32, 32, device="cuda") * 2 - 1
        t2 = torch.rand(1, 3, 32, 32, device="cuda") * 2 - 1

        wrapper._tensor_to_pil_optimized(t1)
        buf_ptr_1 = wrapper._output_pin_buf.data_ptr()
        wrapper._tensor_to_pil_optimized(t2)
        buf_ptr_2 = wrapper._output_pin_buf.data_ptr()

        assert buf_ptr_1 == buf_ptr_2


# ---------------------------------------------------------------------------
# 5e: _tensor_to_pil_safe (CPU-first reorder)
# ---------------------------------------------------------------------------


class TestTensorToPilSafeCpuFirst:
    def test_byte_identical_vae_range(self):
        """VAE-range [-1, 1] input takes both the min()<0 and max()<=1.0 branches."""
        torch.manual_seed(3)
        tensor = torch.rand(3, 40, 56, device="cuda") * 2 - 1

        orchestrator = object.__new__(PreprocessingOrchestrator)
        expected = _reference_tensor_to_pil_safe(tensor.clone())
        actual = orchestrator._tensor_to_pil_safe(tensor.clone())

        assert np.array_equal(np.array(actual), np.array(expected))

    def test_byte_identical_unit_range(self):
        """[0, 1] input only takes the max()<=1.0 branch."""
        torch.manual_seed(4)
        tensor = torch.rand(3, 40, 56, device="cuda")

        orchestrator = object.__new__(PreprocessingOrchestrator)
        expected = _reference_tensor_to_pil_safe(tensor.clone())
        actual = orchestrator._tensor_to_pil_safe(tensor.clone())

        assert np.array_equal(np.array(actual), np.array(expected))

    def test_byte_identical_already_0_255_range(self):
        """Values already > 1.0 (pre-scaled to [0, 255]) skip both conditional branches."""
        torch.manual_seed(5)
        tensor = torch.rand(3, 40, 56, device="cuda") * 255.0

        orchestrator = object.__new__(PreprocessingOrchestrator)
        expected = _reference_tensor_to_pil_safe(tensor.clone())
        actual = orchestrator._tensor_to_pil_safe(tensor.clone())

        assert np.array_equal(np.array(actual), np.array(expected))
