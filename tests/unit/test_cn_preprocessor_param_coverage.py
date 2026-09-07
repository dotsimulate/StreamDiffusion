"""
Category-param GPU-coverage guard for ControlNet-coupled preprocessors.

Verification test for plan i-ve-noticed-that-in-graceful-kettle.md, Change 2.

Part A (below) is a static guard: for every gpu_native=True CN-coupled preprocessor, every
parameter key declared in get_preprocessor_metadata()["parameters"] must be textually
referenced somewhere in the class's GPU-reachable source -- i.e. any method the class (or a
base class between it and BasePreprocessor) defines, EXCLUDING _process_core (the CPU/PIL-only
path) and EXCLUDING get_preprocessor_metadata itself (whose own source trivially contains every
parameter name it declares, which would make the guard vacuous).

Excluding _process_core is the load-bearing part -- it is exactly what makes this guard fail on
an unfixed CannyPreprocessor: 'smoothness' was only read in its CPU path (canny.py
_process_core), never on the GPU tensor path that TouchDesigner's live pipeline actually uses
(BasePreprocessor.process_tensor always prefers _process_tensor_core when the class overrides
it, and every CN-coupled class here does).

Text-substring matching (not a `self.params.get("<name>", ...)` regex) is deliberate: some
processors read the category param via `self.params.get(...)` (canny -> smoothness, scribble ->
smoothness / scribble_threshold), others cache it once in __init__ as an instance attribute and
reference `self.<name>` in the hot path (feedback -> feedback_strength, via
`self.feedback_strength`). Both forms contain the parameter name as a literal substring.

Reference case for the "helper method, not _process_tensor_core itself" pattern:
ScribbleTensorrtPreprocessor does not define its own _process_tensor_core -- it inherits
SelfBuildingTRTPreprocessor._process_tensor_core (trt_base.py), which calls the abstract
self._postprocess(). The actual override -- where 'smoothness' and 'scribble_threshold' are
read -- lives only in ScribbleTensorrtPreprocessor._postprocess. A scan of only
cls._process_tensor_core's own source would miss both keys and produce a false positive here;
walking the MRO and scanning every method (except the two exclusions above) catches it
correctly. StandardLineartPreprocessor._compute_lineart_hwc (called by both _process_core and
_process_tensor_core) is a second instance of the same "separate helper" pattern.

Run with: pytest tests/unit/test_cn_preprocessor_param_coverage.py -v
"""

import inspect

import pytest
import torch

from streamdiffusion.preprocessing.processors import get_preprocessor_class, list_preprocessors
from streamdiffusion.preprocessing.processors.base import BasePreprocessor

# Mirrors CN_COUPLED_PREPROCESSORS in test_cn_preprocessor_residency.py (preprocessor names only;
# gpu_native is re-checked here rather than assumed, since that flag is already covered by the
# sibling test).
CN_COUPLED_PREPROCESSORS = [
    "canny",
    "soft_edge",
    "standard_lineart",
    "feedback",
    "passthrough",
    "depth_tensorrt",
    "pose_tensorrt",
    "hed_tensorrt",
    "scribble_tensorrt",
    "normal_bae_tensorrt",
]

# Methods whose source must never count as "GPU-reachable" evidence for a parameter:
#   - _process_core: the CPU/PIL-only path. Live TD frames never take this path once a class is
#     gpu_native=True, so a param read only here is exactly the Canny bug class.
#   - get_preprocessor_metadata: its own return-dict source literally contains every parameter
#     name it declares, which would make the guard trivially pass for anything.
_EXCLUDED_METHOD_NAMES = frozenset({"_process_core", "get_preprocessor_metadata"})

# Construction-time-only params that legitimately never appear in the per-frame GPU-reachable
# path (e.g. consumed once in __init__ to configure an engine, or read only via a base-class
# property this scan doesn't need to reach). Keep this short -- a growing allowlist is the
# signal that the guard below is being worked around rather than honoured.
#
# Empty today: every CN-coupled preprocessor's declared metadata parameters are all read
# somewhere in the GPU-reachable path once Change 1 (canny.py smoothness) lands.
CONSTRUCTION_TIME_ONLY_PARAMS: dict = {
    # "some_preprocessor_name": {"engine_path"},  # example shape
}


def _member_source(member) -> str | None:
    """Return source text for a class-`vars()` member the guard cares about, else None."""
    if isinstance(member, (staticmethod, classmethod)):
        member = member.__func__
    elif isinstance(member, property):
        if member.fget is None:
            return None
        member = member.fget

    if not inspect.isfunction(member):
        return None

    try:
        return inspect.getsource(member)
    except (OSError, TypeError):
        return None


def _gpu_reachable_source(cls: type) -> str:
    """
    Concatenate the source of every method `cls` defines itself, or inherits from a base class
    between it and (not including) BasePreprocessor, excluding _EXCLUDED_METHOD_NAMES.

    Walking the MRO (rather than only `cls`'s own dict) is what makes ScribbleTensorrtPreprocessor
    resolve correctly: it inherits _process_tensor_core from SelfBuildingTRTPreprocessor, several
    MRO steps up, while overriding _postprocess itself.
    """
    chunks = []
    for klass in cls.__mro__:
        if klass is BasePreprocessor:
            break  # BasePreprocessor and everything above it (ABC, object) is shared, not GPU-kernel code
        for name, member in vars(klass).items():
            if name in _EXCLUDED_METHOD_NAMES:
                continue
            source = _member_source(member)
            if source:
                chunks.append(source)
    return "\n".join(chunks)


def _cn_coupled_gpu_native_cases():
    """(preprocessor_name, cls) for every registered, gpu_native=True CN-coupled preprocessor."""
    registered = list_preprocessors()
    cases = []
    for name in CN_COUPLED_PREPROCESSORS:
        if name not in registered:
            continue  # unregistered (e.g. optional TensorRT dep missing) — covered by the residency test
        cls = get_preprocessor_class(name)
        if getattr(cls, "gpu_native", False):
            cases.append((name, cls))
    return cases


_CASES = _cn_coupled_gpu_native_cases()


@pytest.mark.parametrize("preprocessor_name,cls", _CASES, ids=[name for name, _ in _CASES])
def test_metadata_params_are_gpu_reachable(preprocessor_name, cls):
    """
    Every parameter declared in get_preprocessor_metadata()['parameters'] for a gpu_native
    preprocessor must be referenced somewhere outside _process_core -- otherwise a TD user can
    drag the Dyn* slider generated from that metadata and see zero effect on the live (GPU)
    pipeline, exactly as happened with Canny's 'smoothness' before this fix.
    """
    metadata = cls.get_preprocessor_metadata()
    declared_params = set(metadata.get("parameters", {}).keys())
    exempt = CONSTRUCTION_TIME_ONLY_PARAMS.get(preprocessor_name, set())
    source = _gpu_reachable_source(cls)

    missing = sorted(p for p in declared_params - exempt if p not in source)

    assert not missing, (
        f"{cls.__name__} ('{preprocessor_name}') declares parameter(s) {missing} in its "
        "metadata, but they are not referenced anywhere in its GPU-reachable source (everything "
        "the class defines or inherits down to BasePreprocessor, except _process_core). A "
        "TD-side Dyn* slider for this parameter would silently do nothing on the live tensor "
        "pipeline. Either wire the parameter into _process_tensor_core (or a helper it calls), "
        "or add it to CONSTRUCTION_TIME_ONLY_PARAMS with a comment explaining why it's exempt."
    )


# ---------------------------------------------------------------------------
# Part B — focused behavioural regression test for Canny
# ---------------------------------------------------------------------------


def _make_textured_test_image(H: int = 512, W: int = 512, block: int = 64, noise_amp: float = 0.5):
    """
    Fixed-seed synthetic image with macro structure (a large-block checkerboard, i.e. genuine
    'major contours') plus superimposed per-pixel noise (fine high-frequency 'texture'/'grain').
    This mirrors the plan's own description of what smoothness is for: as smoothness rises, the
    grain edges should progressively drop out while the checker-boundary edges persist longer.

    Pure uncorrelated `torch.rand` noise (no macro structure) was tried first and rejected: after
    the *fixed* 5x5 Gaussian that always runs before Sobel, its gradient magnitude never reaches
    even the low threshold at any smoothness level, so the sweep is a degenerate all-zero
    sequence that proves nothing. Macro structure is required for a meaningful monotonic-decrease
    signal.
    """
    torch.manual_seed(42)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    checker = (((yy // block) + (xx // block)) % 2).float()
    img = checker.unsqueeze(0).repeat(3, 1, 1)
    noise = (torch.rand(3, H, W) - 0.5) * noise_amp
    return (img + noise).clamp(0.0, 1.0)


def test_canny_smoothness_reduces_edges_on_gpu_path():
    """
    Canny's GPU path (_process_tensor_core) must honour 'smoothness' the same way its CPU path
    (_process_core) always has: higher smoothness -> more pre-blur -> fewer/sparser edges.

    Before Change 1, _process_tensor_core never read 'smoothness' at all, so all four sums below
    were byte-identical -- that identity is the exact regression signature this guards against.
    Runs entirely on CPU (device="cpu"), no TensorRT/GPU dependency.

    Thresholds (60/150) are deliberately lower than the plan sketch's illustrative 100/200: at
    100/200 the checkerboard's own boundary gradient falls below the *high* threshold once
    smoothness=1.0 stacks on top of the fixed 5x5 Gaussian, so Canny's hysteresis logic (a "weak"
    pixel only survives adjacent to a "strong" one) zeroes out the whole frame at the top of the
    sweep -- still a technically-monotonic [strictly decreasing, ending at 0] result, but it
    obscures the point of this test (texture drops out, *structure survives*). 60/150 keeps the
    checker-boundary edges above threshold at every sweep point, verified stable across 5
    independent noise seeds during test design.
    """
    from streamdiffusion.preprocessing.processors.canny import CannyPreprocessor

    textured_img = _make_textured_test_image()

    p = CannyPreprocessor(low_threshold=60, high_threshold=150, device="cpu", dtype=torch.float32)

    sums = []
    for s in (0.0, 0.25, 0.5, 1.0):
        p.params["smoothness"] = s
        result = p.process_tensor(textured_img)
        sums.append(result[0].sum().item())

    assert sums == sorted(sums, reverse=True), (
        f"Expected edge-pixel sum to be non-increasing as smoothness rises 0.0 -> 1.0, got "
        f"{sums}. Equal/identical values across the sweep is the pre-fix regression signature "
        "(the GPU path was ignoring 'smoothness' entirely)."
    )
    # The pre-fix bug produced four *identical* values (smoothness was read nowhere on this
    # path); assert the fix actually changes something, not just "doesn't increase".
    assert len(set(sums)) > 1, (
        f"All sums were identical ({sums[0]}) across the smoothness sweep -- this is the exact "
        "pre-fix symptom (GPU path ignoring 'smoothness' entirely), not a passing case."
    )
    # And confirm structure survives at max smoothness — the whole point of a *pre-blur* design
    # (vs. a naive post-blur) is that texture disappears while major contours remain detectable.
    assert sums[-1] > 0, (
        f"Edge sum at smoothness=1.0 was {sums[-1]} (no edges survived at all). Expected the "
        "checkerboard's macro boundaries to remain detectable even at maximum smoothness."
    )
