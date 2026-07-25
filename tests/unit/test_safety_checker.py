"""
Regression tests for StreamDiffusionWrapper._apply_safety_checker.

These tests are deliberately CPU-only and model-free.  They construct a
minimal StreamDiffusionWrapper shell via object.__new__ and wire in only the
attributes the helper reads, so the full GPU/TRT stack is not required.

Root cause being guarded: when use_cuda_ipc_output=True and output_type='pt',
postprocess_image() exports the frame and returns None.  The old code fed that
None to self.safety_checker, which called torchvision T.Resize on None and
raised:
    TypeError: Unexpected type <class 'NoneType'>
This crashed the streaming loop every frame.

Fix: _apply_safety_checker() runs *before* postprocess_image, operating on the
raw diffusion-range [-1, 1] pipeline tensor so it is always a real tensor
regardless of output path.

Sub-phase 5.3 (delayed-emission model): the checker's verdict is read
1-frame-delayed from a pinned async buffer (mirrors the async-launch idea in
SimilarImageFilter), but unlike a plain delayed *readback*, the raw frame
itself is buffered in `_pending_frame` so each frame is gated on its OWN
verdict rather than a neighbor's. `safety_checker` is `(tensor, prob_pin) ->
None`: it writes into `prob_pin` rather than returning a bool. Each call to
`_apply_safety_checker`:
  1. reads the verdict for the frame buffered on the PREVIOUS call (now
     landed in the pin),
  2. launches classification for the CURRENT frame,
  3. emits the PENDING (previously buffered) frame, gated on its own verdict,
  4. buffers the CURRENT frame for the next call.
This delays output by exactly one frame. The very first call has no buffered
frame yet, so it emits a black startup frame and primes the pipeline instead
of passing raw pixels through unscreened. Tests below seed `w._nsfw_prob_pin`
and `w._pending_frame` directly where they need to bypass the first-call
priming and exercise a specific verdict against a specific frame.
"""

import torch

from streamdiffusion.wrapper import StreamDiffusionWrapper

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_wrapper(
    *,
    use_safety_checker: bool = True,
    fallback_type: str = "blank",
    threshold: float = 0.5,
):
    """Construct a minimal StreamDiffusionWrapper without model loading."""
    w = object.__new__(StreamDiffusionWrapper)
    w.use_safety_checker = use_safety_checker
    w.safety_checker_threshold = threshold
    w.safety_checker_fallback_type = fallback_type
    w._prev_clean_tensor = None
    w._nsfw_prob_pin = None
    w._pending_frame = None
    # safety_checker is set per-test
    w.safety_checker = None
    return w


def _black_denorm(t: torch.Tensor) -> bool:
    """True when t is all-zeros after _denormalize_on_gpu (i.e. all -1.0 raw)."""
    return torch.allclose(t, torch.zeros_like(t))


# ---------------------------------------------------------------------------
# test cases
# ---------------------------------------------------------------------------


class TestApplySafetyChecker:
    # ── Case 1: No-None contract (directly reproduces the old crash) ────────
    def test_checker_never_receives_none(self):
        """
        The safety checker must never be called with a None tensor.
        This is the exact condition that produced:
            TypeError: Unexpected type <class 'NoneType'>
        in the old post-hoc code path when output_type='pt' and IPC was active.
        """
        received: list = []

        def capturing_checker(tensor, prob_pin):
            received.append(tensor)
            prob_pin.fill_(1.0)

        w = _make_wrapper()
        w.safety_checker = capturing_checker

        dummy = torch.randn(1, 3, 64, 64)
        w._apply_safety_checker(dummy)

        assert len(received) == 1
        arg = received[0]
        assert arg is not None, "safety checker received None — regression"
        assert isinstance(arg, torch.Tensor), f"expected Tensor, got {type(arg)}"

    # ── Case 2: NSFW → black frame (blank fallback) ──────────────────────────
    def test_nsfw_blank_fallback_is_black(self):
        w = _make_wrapper(fallback_type="blank")
        w.safety_checker = lambda t, pin: None  # verdict is pre-seeded below

        pending = torch.randn(1, 3, 64, 64)
        w._pending_frame = pending
        w._nsfw_prob_pin = torch.tensor([1.0])  # pending frame's own verdict: flagged

        current = torch.randn(1, 3, 64, 64)
        result = w._apply_safety_checker(current)

        assert result is not pending, "flagged pending frame should be replaced"
        # Denormalize: (x/2+0.5).clamp(0,1); -1.0 → 0.0 = black
        denorm = (result / 2 + 0.5).clamp(0, 1)
        assert _black_denorm(denorm), "NSFW blank fallback should produce a black frame"

    # ── Case 3: NSFW → previous frame ────────────────────────────────────────
    def test_nsfw_previous_fallback_returns_cached_clean_frame(self):
        w = _make_wrapper(fallback_type="previous")
        w.safety_checker = lambda t, pin: None  # verdict is pre-seeded below

        cached_clean = torch.randn(1, 3, 64, 64)
        w._prev_clean_tensor = cached_clean

        pending = torch.randn(1, 3, 64, 64)
        w._pending_frame = pending
        w._nsfw_prob_pin = torch.tensor([1.0])  # pending frame's own verdict: flagged

        current = torch.randn(1, 3, 64, 64)
        result = w._apply_safety_checker(current)

        assert result is cached_clean, "previous-fallback should return the cached clean tensor"

    # ── Case 4: Clean frame passthrough ──────────────────────────────────────
    def test_clean_frame_returned_unchanged(self):
        w = _make_wrapper()
        w.safety_checker = lambda t, pin: None  # verdict is pre-seeded below

        pending = torch.randn(1, 3, 64, 64)
        w._pending_frame = pending
        w._nsfw_prob_pin = torch.tensor([0.0])  # pending frame's own verdict: clean

        current = torch.randn(1, 3, 64, 64)
        result = w._apply_safety_checker(current)

        assert torch.equal(result, pending), "clean pending frame should be emitted unchanged"

    # ── Case 5: use_safety_checker=False bypasses entirely ───────────────────
    def test_disabled_bypasses_checker(self):
        called = [False]

        def should_not_be_called(t, pin):
            called[0] = True

        w = _make_wrapper(use_safety_checker=False)
        w.safety_checker = should_not_be_called

        dummy = torch.randn(1, 3, 64, 64)
        result = w._apply_safety_checker(dummy)

        assert not called[0], "safety checker must not be called when disabled"
        assert torch.equal(result, dummy), "disabled checker should return input unchanged"

    # ── Case 6: flagged verdict with no cached frame → black ─────────────────
    def test_nsfw_previous_fallback_no_cache_falls_back_to_black(self):
        """
        If the pending frame is flagged (per its own landed verdict) and
        _prev_clean_tensor is None, the fallback should still produce a black
        frame rather than raise.
        """
        w = _make_wrapper(fallback_type="previous")
        w.safety_checker = lambda t, pin: pin.fill_(0.0)
        w._pending_frame = torch.randn(1, 3, 64, 64)
        w._nsfw_prob_pin = torch.tensor([1.0])  # pending frame's own verdict: flagged
        assert w._prev_clean_tensor is None  # no cache yet

        dummy = torch.randn(1, 3, 64, 64)
        result = w._apply_safety_checker(dummy)

        denorm = (result / 2 + 0.5).clamp(0, 1)
        assert _black_denorm(denorm), "when prev cache is empty and content is flagged, fallback must be black"

    # ── Case 7: clean frame caches for previous strategy ─────────────────────
    def test_clean_frame_cached_for_previous_strategy(self):
        w = _make_wrapper(fallback_type="previous")
        w.safety_checker = lambda t, pin: pin.fill_(0.0)
        pending = torch.randn(1, 3, 64, 64)
        w._pending_frame = pending
        w._nsfw_prob_pin = torch.tensor([0.0])  # pending frame's own verdict: clean

        assert w._prev_clean_tensor is None
        dummy = torch.randn(1, 3, 64, 64)
        w._apply_safety_checker(dummy)

        assert w._prev_clean_tensor is not None, (
            "clean pending frame with previous strategy should populate _prev_clean_tensor"
        )
        assert torch.equal(w._prev_clean_tensor, pending)

    # ── Case 8: clean frame NOT cached for blank strategy ────────────────────
    def test_clean_frame_not_cached_for_blank_strategy(self):
        w = _make_wrapper(fallback_type="blank")
        w.safety_checker = lambda t, pin: pin.fill_(0.0)
        w._pending_frame = torch.randn(1, 3, 64, 64)
        w._nsfw_prob_pin = torch.tensor([0.0])  # pending frame's own verdict: clean

        dummy = torch.randn(1, 3, 64, 64)
        w._apply_safety_checker(dummy)

        assert w._prev_clean_tensor is None, "_prev_clean_tensor should stay None when fallback_type='blank'"

    # ── Case 9: first call emits black and buffers ────────────────────────────
    def test_first_frame_emits_black_and_buffers(self):
        """
        The very first call has no buffered frame yet, so it cannot gate
        anything on its own verdict. It must emit a black startup frame
        (never ungated pixels) and buffer the raw frame for the next call.
        """
        w = _make_wrapper(fallback_type="blank")
        w.safety_checker = lambda t, pin: pin.fill_(1.0)  # would flag if read this call

        dummy = torch.randn(1, 3, 64, 64)
        result = w._apply_safety_checker(dummy)

        denorm = (result / 2 + 0.5).clamp(0, 1)
        assert _black_denorm(denorm), "first call must emit a black startup frame"
        assert w._pending_frame is not None and torch.equal(w._pending_frame, dummy), (
            "first call must buffer the raw frame for next call's emission"
        )

    # ── Case 10: _process_skip_diffusion wiring — checker is called, result is black ──
    def test_skip_diffusion_routes_through_safety_checker(self):
        """
        Verify that _process_skip_diffusion actually feeds its tensor through
        _apply_safety_checker before postprocess_image, and that a flagged
        pending frame produces the black substitution rather than passing
        through unscreened.

        This is a wiring test — the behaviour of _apply_safety_checker itself is
        covered by the earlier cases.  We stub all model-dependent calls with
        identity/no-op lambdas so the test is CPU-only and model-free.
        """
        received: list = []

        def capturing_checker(tensor, pin):
            received.append(tensor)
            pin.fill_(0.0)  # this frame's own async result — irrelevant to this call's read

        w = _make_wrapper(fallback_type="blank")
        w.safety_checker = capturing_checker
        w._pending_frame = torch.randn(1, 3, 64, 64)  # a prior frame awaiting emission
        w._nsfw_prob_pin = torch.tensor([1.0])  # that pending frame's own verdict: flagged
        w.mode = "img2img"
        w.device = torch.device("cpu")
        w.dtype = torch.float32

        # Stub the stream's pre/post hooks to identity
        class _FakeStream:
            def _apply_image_preprocessing_hooks(self, t):
                return t

            def _apply_image_postprocessing_hooks(self, t):
                return t

        w.stream = _FakeStream()

        # _normalize_on_gpu / _denormalize_on_gpu are identity for this test
        w._normalize_on_gpu = lambda t: t
        w._denormalize_on_gpu = lambda t: t

        # postprocess_image returns its input so we can inspect the tensor
        w.postprocess_image = lambda t, output_type=None: t
        w.output_type = "pt"

        dummy_image = torch.randn(1, 3, 64, 64)
        result = w._process_skip_diffusion(dummy_image)

        # Checker must have been called exactly once with a real tensor
        assert len(received) == 1, "safety checker should be called exactly once"
        assert isinstance(received[0], torch.Tensor), "checker arg must be a Tensor, not None"

        # Flagged pending frame (per the pre-seeded pinned verdict) must produce the black substitution
        denorm = (result / 2 + 0.5).clamp(0, 1)
        assert torch.allclose(denorm, torch.zeros_like(denorm)), "flagged pending frame should produce a black output"

    # ── Case 11: pin_memory is guarded on CPU-only builds ────────────────────
    def test_pin_memory_guarded_on_cpu(self):
        """
        Regression test: the first call used to unconditionally call
        .pin_memory(), which raises on CPU-only / no-CUDA-driver PyTorch
        builds. It must be guarded and still produce a usable CPU tensor.
        """
        w = _make_wrapper(fallback_type="blank")
        w.safety_checker = lambda t, pin: pin.fill_(0.0)

        dummy = torch.randn(1, 3, 64, 64)
        w._apply_safety_checker(dummy)  # must not raise

        assert isinstance(w._nsfw_prob_pin, torch.Tensor)
        assert w._nsfw_prob_pin.device.type == "cpu"

    # ── Case 12: isolated NSFW frame is blanked, not leaked ──────────────────
    def test_isolated_nsfw_frame_is_blanked_not_leaked(self):
        """
        Drives [clean, NSFW, clean] through the checker and asserts the NSFW
        frame is the one substituted (never emitted raw) and the surrounding
        clean frames pass through unscreened — proving the old bug (verdict
        misattributed to the wrong frame, causing an isolated NSFW frame to
        leak and the following clean frame to be wrongly blanked) is fixed.
        """
        verdicts = {"clean1": 0.0, "nsfw": 1.0, "clean2": 0.0}
        order = ["clean1", "nsfw", "clean2"]
        call_index = [0]

        def checker(t, pin):
            key = order[call_index[0]]
            call_index[0] += 1
            pin.fill_(verdicts[key])

        w = _make_wrapper(fallback_type="blank", threshold=0.5)
        w.safety_checker = checker

        clean1 = torch.randn(1, 3, 64, 64)
        nsfw = torch.randn(1, 3, 64, 64)
        clean2 = torch.randn(1, 3, 64, 64)

        r0 = w._apply_safety_checker(clean1)  # priming call: black startup frame
        r1 = w._apply_safety_checker(nsfw)  # emits clean1 (clean)
        r2 = w._apply_safety_checker(clean2)  # emits nsfw (flagged -> black)
        # A 4th call would be needed to emit clean2; verify what we can without it.

        denorm0 = (r0 / 2 + 0.5).clamp(0, 1)
        assert _black_denorm(denorm0), "priming call must emit black, not raw pixels"

        assert torch.equal(r1, clean1), "clean1 must be emitted unchanged, gated on its own verdict"

        denorm2 = (r2 / 2 + 0.5).clamp(0, 1)
        assert _black_denorm(denorm2), "the NSFW frame itself must be the one blanked, not a neighbor"
        assert not torch.equal(r2, clean2), "clean2 must never be substituted for the NSFW frame's verdict"
