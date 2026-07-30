"""Correctness test for the B1 `_pack_bgra` refactor (wrapper.py).

The three per-channel writes in `_ipc_pack_rgba` / `_ipc_pack_unit_rgba`
(`dst[..., 0] = src[..., 2]` etc.) were replaced by a single fused
`dst[..., :3] = src.flip(-1)`. Both are pure channel permutations, so the
result must be byte-identical — the frozen pre-B1 formula below is the
oracle and `torch.equal` compares all 4 channels (alpha must be untouched).

CPU runs always (flip is device-independent); CUDA runs behind the repo's
skip idiom (see test_sync_free_output_5_2.py).
"""

import pytest
import torch

from streamdiffusion.wrapper import StreamDiffusionWrapper

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")

# Both call sites pack an (H, W, 3) uint8 RGB into an (H, W, 4) BGRA buffer:
# the main IPC path at the pipeline output resolution, the CN-preview twin at
# the preprocessor output resolution. 512x512 is the production shape; the
# small non-square shape guards against stride/transpose mistakes.
SHAPES = [(512, 512), (33, 17)]
DEVICES = ["cpu", pytest.param("cuda", marks=requires_cuda)]


def _reference_three_write_pack(rgb_hwc: torch.Tensor, dst: torch.Tensor) -> None:
    """Frozen pre-B1 formula: three per-channel copies (B, G, R)."""
    dst[..., 0] = rgb_hwc[..., 2]  # B
    dst[..., 1] = rgb_hwc[..., 1]  # G
    dst[..., 2] = rgb_hwc[..., 0]  # R


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_pack_bgra_matches_three_write_reference(shape, device):
    h, w = shape
    gen = torch.Generator().manual_seed(2416333)
    rgb_hwc = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, generator=gen).to(device)

    # Alpha uses a sentinel (not the production 255) to prove the helper never
    # writes channel 3; RGB channels start as differing garbage so a no-op
    # would fail the comparison.
    dst_ref = torch.full((h, w, 4), 7, dtype=torch.uint8, device=device)
    dst_new = torch.full((h, w, 4), 11, dtype=torch.uint8, device=device)
    dst_ref[..., 3] = 123
    dst_new[..., 3] = 123

    _reference_three_write_pack(rgb_hwc, dst_ref)
    StreamDiffusionWrapper._pack_bgra(rgb_hwc, dst_new)

    assert torch.equal(dst_new, dst_ref)
    assert torch.equal(dst_new[..., 3], torch.full((h, w), 123, dtype=torch.uint8, device=device))
