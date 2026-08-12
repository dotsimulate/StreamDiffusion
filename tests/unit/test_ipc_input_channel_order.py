"""Tests for the R<->B channel-swap fix on the TD -> StreamDiffusion CUDA-IPC
input path (`TouchDesignerManager._unpack_hwc_to_rgb` in td_manager.py).

Root cause: TD's 8-bit 4-channel CUDA memory is BGRA-ordered (TD's own
documented preference), and the output side (`StreamDiffusionWrapper._pack_bgra`,
wrapper.py) already accounts for that with a `flip(-1)`. The input side did not
-- it only stripped alpha (`gpu_frame[:, :, :3]`) with no channel-order
correction -- so an 8-bit round trip came back with red and blue swapped.
`_unpack_hwc_to_rgb` is the fix: the mirror of `_pack_bgra` on the consumer
side, applying the same swap decision cuda-link's wire metadata supports
(`FLAGS_BGRA`) with a dtype fallback for senders that don't set it.

td_manager.py cannot be imported directly in a fast unit test -- it pulls in
timm/controlnet_aux/torch-dynamo at import time (~8s) -- so this file follows
the precedent in test_td_pending_params.py and freezes an exact replica of
`_unpack_hwc_to_rgb` below. Keep it in sync with td_manager.py's copy (and its
byte-identical Scripts/ mirror, checked at the bottom of this file); if the
swap logic changes there, update it here too or these tests go stale silently.

Round-trip tests pair this replica against the REAL
`StreamDiffusionWrapper._pack_bgra` (imported directly -- wrapper.py is light
enough to import, see test_ipc_pack_bgra_correctness.py) so the output-side and
input-side halves of the wire contract are exercised together, not just each
one's own self-consistency in isolation. Uses saturated primaries (R/G/B/W
quadrants), not random data: test_ipc_pack_bgra_correctness.py's `randint`
approach locks the pack's self-consistency but is blind to whether a matching
unpack exists -- that blindness is how the underlying bug shipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from streamdiffusion.wrapper import StreamDiffusionWrapper

# Mirrors cuda_link.shm_protocol.FLAGS_BGRA (bit2: 4-channel uint8 source is
# BGRA-ordered, not RGBA). Frozen here rather than imported so these tests
# don't require cuda_link to be installed.
FLAGS_BGRA = 0x0004

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
DEVICES = ["cpu", pytest.param("cuda", marks=requires_cuda)]


def _unpack_hwc_to_rgb(gpu_frame: torch.Tensor, flags: int) -> torch.Tensor:
    """Frozen replica of TouchDesignerManager._unpack_hwc_to_rgb (td_manager.py).

    Swap decision, in priority order:
      1. flags & FLAGS_BGRA set          -> swap
      2. flags == 0 and frame is uint8x4 -> swap (dtype fallback)
      3. otherwise                        -> no swap
    """
    is_4ch = gpu_frame.ndim == 3 and gpu_frame.shape[2] == 4
    should_swap = is_4ch and (bool(flags & FLAGS_BGRA) or (flags == 0 and gpu_frame.dtype == torch.uint8))
    if is_4ch:
        gpu_frame = gpu_frame[:, :, :3]
    if should_swap:
        gpu_frame = gpu_frame.flip(-1)
    return gpu_frame


def _rgbw_quadrants(size: int = 4, device: str = "cpu") -> torch.Tensor:
    """(2*size, 2*size, 3) uint8 HWC tensor: R top-left, G top-right, B
    bottom-left, W bottom-right -- the same saturated-primary pattern used to
    observe the bug (green/white are invariant under R<->B, which is what rules
    out a vertical flip as the explanation).
    """
    h = w = size
    img = torch.zeros((2 * h, 2 * w, 3), dtype=torch.uint8, device=device)
    img[:h, :w] = torch.tensor([255, 0, 0], dtype=torch.uint8, device=device)  # R top-left
    img[:h, w:] = torch.tensor([0, 255, 0], dtype=torch.uint8, device=device)  # G top-right
    img[h:, :w] = torch.tensor([0, 0, 255], dtype=torch.uint8, device=device)  # B bottom-left
    img[h:, w:] = torch.tensor([255, 255, 255], dtype=torch.uint8, device=device)  # W bottom-right
    return img


# ---------------------------------------------------------------------------
# Round-trip identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_round_trip_identity_saturated_primaries(device):
    """pack_bgra (output side) -> _unpack_hwc_to_rgb (input side) must be the
    identity on the RGBW pattern, both via the dtype fallback (today's Python
    td_exporter, which sets no flag) and via the explicit flag (e.g. cuda-link's
    C++ TOP)."""
    rgb = _rgbw_quadrants(device=device)
    packed = torch.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=torch.uint8, device=device)
    packed[..., 3] = 255
    StreamDiffusionWrapper._pack_bgra(rgb, packed)

    recovered_fallback = _unpack_hwc_to_rgb(packed, flags=0)
    assert torch.equal(recovered_fallback, rgb)

    recovered_flagged = _unpack_hwc_to_rgb(packed, flags=FLAGS_BGRA)
    assert torch.equal(recovered_flagged, rgb)


def test_round_trip_fails_without_fix_documents_the_bug():
    """Sanity check that this is a real bug, not a test artifact: stripping
    alpha with no channel-order correction (the pre-fix behaviour) does not
    recover the original RGB from a BGRA-ordered 4-channel uint8 frame, and
    reproduces exactly the reported symptom -- red and blue swapped, green and
    white unmoved."""
    rgb = _rgbw_quadrants()
    packed = torch.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=torch.uint8)
    packed[..., 3] = 255
    StreamDiffusionWrapper._pack_bgra(rgb, packed)

    pre_fix = packed[:, :, :3]  # strip alpha only -- the old td_manager.py behaviour
    assert not torch.equal(pre_fix, rgb)
    assert pre_fix[0, 0].tolist() == [0, 0, 255]  # red source read back as blue
    assert pre_fix[0, -1].tolist() == rgb[0, -1].tolist()  # green: unmoved
    assert pre_fix[-1, -1].tolist() == rgb[-1, -1].tolist()  # white: unmoved


# ---------------------------------------------------------------------------
# Helper truth table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags,dtype,should_swap",
    [
        pytest.param(FLAGS_BGRA, torch.uint8, True, id="flag_set-uint8"),
        pytest.param(FLAGS_BGRA, torch.float32, True, id="flag_set-float32"),
        pytest.param(0, torch.uint8, True, id="flag_clear-uint8_fallback"),
        pytest.param(0, torch.float32, False, id="flag_clear-float32_no_swap"),
    ],
)
def test_unpack_truth_table_4channel(flags, dtype, should_swap):
    if dtype == torch.uint8:
        frame = torch.tensor([[[10, 20, 30, 255]]], dtype=torch.uint8)
        expected_swapped = torch.tensor([[[30, 20, 10]]], dtype=torch.uint8)
    else:
        frame = torch.tensor([[[0.1, 0.2, 0.3, 1.0]]], dtype=torch.float32)
        expected_swapped = torch.tensor([[[0.3, 0.2, 0.1]]], dtype=torch.float32)
    expected_unswapped = frame[:, :, :3]

    result = _unpack_hwc_to_rgb(frame, flags)
    expected = expected_swapped if should_swap else expected_unswapped
    assert torch.equal(result, expected)


def test_unpack_3channel_never_swaps_regardless_of_flags():
    """FLAGS_BGRA is defined only for 4-channel uint8 sources (SHMProtocol.py);
    a 3-channel (no-alpha) frame must pass through untouched either way."""
    frame = torch.tensor([[[10, 20, 30]]], dtype=torch.uint8)
    for flags in (0, FLAGS_BGRA):
        assert torch.equal(_unpack_hwc_to_rgb(frame, flags), frame)


# ---------------------------------------------------------------------------
# Mirror parity -- the baked DAT mirror must carry the identical fix, or a
# backend-only change is silently overridden at runtime. td_manager.py is not
# yet registered in scripts/sync_td_mirror.py's PAIRS (only td_main.py and
# syphon_utils.py are), so this checks it directly rather than via that tool's
# test (test_td_mirror_sync.py).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL = _REPO_ROOT / "StreamDiffusionTD" / "td_manager.py"
_MIRROR = _REPO_ROOT.parent / "Scripts" / "streamdiffusionTD__Text__td_manager__td.py"

_skip_no_mirror = pytest.mark.skipif(
    not _MIRROR.is_file(),
    reason=f"{_MIRROR} not present (bare clone without the surrounding dev tree)",
)


@_skip_no_mirror
def test_td_manager_mirror_matches_canonical():
    assert _CANONICAL.is_file(), f"Canonical source not found: {_CANONICAL}"
    canonical_text = _CANONICAL.read_text(encoding="utf-8")
    mirror_text = _MIRROR.read_text(encoding="utf-8")
    assert canonical_text == mirror_text, (
        f"{_MIRROR.name} is out of sync with {_CANONICAL.name}. "
        "Hand-apply the same edit to both files (not wired into sync_td_mirror.py)."
    )
