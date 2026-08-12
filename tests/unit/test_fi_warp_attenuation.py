"""
Regression tests for ``StreamDiffusion._fi_warp_attenuation`` (pipeline.py).

Feature Injection is a hard per-token gate that fights feedback_loop's outward warp
(see pipeline.py's fx_frame_transform docs) — a token that moved this frame still
looks "similar enough" to its cached neighbor and gets frozen back to the old value,
which is what turns endless magnification into freeze-then-lurch. This attenuates
_fi_strength_tensor by a scalar in [0, 1] derived from how far fx_frame_transform's
affine deviates from identity, so FI backs off in proportion to the warp while EA
(soft, attention-weighted) is left alone.

Pinned properties:
  - identity transform (fx_frame_transform is None) -> attenuation == 1.0 exactly,
    so a stationary feedback_loop (zoom=1, no pan/rotation) never touches FI strength;
  - attenuation strictly decreases as warp magnitude grows (zoom further from 1.0,
    larger pan, larger rotation);
  - attenuation is always clamped to [0, 1] even for large warps;
  - zoom=1.02 attenuates to ~0.66x (the empirical anchor point cited in pipeline.py's
    comment above _FI_WARP_ATTENUATION_K) — pins the K constant against silent drift.

CPU-only, model-free: StreamDiffusion built via ``__new__`` with only the attributes
_fi_warp_attenuation reads (same convention as test_kvo_cache_slot_latch.py).

Run with: pytest tests/unit/test_fi_warp_attenuation.py -v
"""

import itertools

import torch

from streamdiffusion.pipeline import StreamDiffusion


def _make_theta(zoom: float = 1.0, pan_x: float = 0.0, pan_y: float = 0.0, rotation: float = 0.0) -> torch.Tensor:
    """Same construction as feedback_loop.py's _make_theta — kept independent here
    so this test doesn't import the processor, only pipeline.py's consumer side."""
    angle = torch.tensor(rotation * 3.14159265 / 180.0)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    scale = 1.0 / zoom
    theta = torch.zeros(1, 2, 3)
    theta[:, 0, 0] = scale * cos_a
    theta[:, 0, 1] = scale * -sin_a
    theta[:, 0, 2] = -pan_x * 2.0
    theta[:, 1, 0] = scale * sin_a
    theta[:, 1, 1] = scale * cos_a
    theta[:, 1, 2] = -pan_y * 2.0
    return theta


def _make_stream() -> StreamDiffusion:
    stream = StreamDiffusion.__new__(StreamDiffusion)
    stream._FI_WARP_ATTENUATION_K = 15.0
    stream.fx_frame_transform = None
    return stream


class TestIdentityIsUnattenuated:
    def test_none_transform_gives_full_strength(self):
        stream = _make_stream()
        stream.fx_frame_transform = None
        assert stream._fi_warp_attenuation() == 1.0

    def test_explicit_identity_theta_gives_full_strength(self):
        stream = _make_stream()
        stream.fx_frame_transform = _make_theta(zoom=1.0)
        assert abs(stream._fi_warp_attenuation() - 1.0) < 1e-6


class TestAttenuationDecreasesWithWarp:
    def test_monotonic_decrease_with_zoom(self):
        stream = _make_stream()
        zooms = [1.0, 1.005, 1.02, 1.05, 1.1, 1.2]
        values = []
        for z in zooms:
            stream.fx_frame_transform = None if z == 1.0 else _make_theta(zoom=z)
            values.append(stream._fi_warp_attenuation())
        for a, b in itertools.pairwise(values):
            assert b < a, f"attenuation did not strictly decrease across zooms {zooms}: {values}"

    def test_monotonic_decrease_with_pan(self):
        stream = _make_stream()
        pans = [0.0, 0.02, 0.05, 0.1, 0.2]
        values = []
        for p in pans:
            stream.fx_frame_transform = None if p == 0.0 else _make_theta(pan_x=p)
            values.append(stream._fi_warp_attenuation())
        for a, b in itertools.pairwise(values):
            assert b < a, f"attenuation did not strictly decrease across pans {pans}: {values}"

    def test_monotonic_decrease_with_rotation(self):
        stream = _make_stream()
        rotations = [0.0, 1.0, 3.0, 10.0, 30.0]
        values = []
        for r in rotations:
            stream.fx_frame_transform = None if r == 0.0 else _make_theta(rotation=r)
            values.append(stream._fi_warp_attenuation())
        for a, b in itertools.pairwise(values):
            assert b < a, f"attenuation did not strictly decrease across rotations {rotations}: {values}"


class TestAttenuationIsClamped:
    def test_large_warp_clamps_to_zero_not_negative(self):
        stream = _make_stream()
        stream.fx_frame_transform = _make_theta(zoom=3.0, pan_x=0.9, rotation=90.0)
        value = stream._fi_warp_attenuation()
        assert 0.0 <= value <= 1.0

    def test_zero_zoom_edge_case_stays_in_range(self):
        # zoom -> scale = 1/zoom blows up as zoom shrinks toward 0; attenuation
        # must still clamp into [0, 1] rather than exp() overflow going negative/inf.
        stream = _make_stream()
        stream.fx_frame_transform = _make_theta(zoom=0.01)
        value = stream._fi_warp_attenuation()
        assert 0.0 <= value <= 1.0


class TestEmpiricalAnchor:
    def test_zoom_1_02_attenuates_to_roughly_two_thirds(self):
        """Pins _FI_WARP_ATTENUATION_K = 15.0 against silent drift: the comment
        above it in pipeline.py cites zoom=1.02 -> ~0.66x as the calibration point."""
        stream = _make_stream()
        stream.fx_frame_transform = _make_theta(zoom=1.02)
        value = stream._fi_warp_attenuation()
        assert abs(value - 0.66) < 0.03, f"zoom=1.02 attenuation {value:.4f}, expected ~0.66"
