"""
Regression coverage for the cuDNN-benchmark guard inside apply_edge_smoothness.

With the pipeline's global torch.backends.cudnn.benchmark=True, every new smoothness
strength is a new conv shape and would trigger a ~750 ms cuDNN autotune pass (a visible
freeze while dragging the Smoothing knob). apply_edge_smoothness works around this by
running its two separable conv2d passes under torch.backends.cudnn.flags(benchmark=False),
restoring the global flag afterward. This file asserts both halves of that contract.
"""

import torch

from streamdiffusion.preprocessing.processors.category_params import apply_edge_smoothness


def test_edge_smoothness_leaves_global_cudnn_benchmark_untouched():
    prev = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = True
        x = torch.rand(32, 48)
        y = apply_edge_smoothness(x, 0.7)
        assert y.shape == x.shape and y.dtype == x.dtype
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.benchmark = prev


def test_edge_smoothness_disables_benchmark_inside_conv(monkeypatch):
    import torch.nn.functional as F

    real_conv2d = F.conv2d
    seen = []

    def recording_conv2d(*args, **kwargs):
        seen.append(torch.backends.cudnn.benchmark)
        return real_conv2d(*args, **kwargs)

    monkeypatch.setattr(F, "conv2d", recording_conv2d)
    prev = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = True
        apply_edge_smoothness(torch.rand(1, 16, 16), 0.4)
    finally:
        torch.backends.cudnn.benchmark = prev
    assert len(seen) == 2  # separable: one horizontal + one vertical pass
    assert seen == [False, False]
