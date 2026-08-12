"""Single source of truth for the runtime-tunable StreamDiffusion parameter set.

This module owns parameter *identity* — the 26 names accepted by
:meth:`StreamDiffusionWrapper.update_stream_params`, which of those 24 are
forwarded to :meth:`StreamParameterUpdater.update_stream_params`, and each
param's *construction-time* default (the value a fresh wrapper gets when the
key is absent from config — see ``config._extract_wrapper_params`` and
``config._extract_prepare_params``).

It deliberately does **not** own order-dependent *apply* logic — the real
``update_stream_params`` implementations stay hand-written in
``wrapper.py`` / ``stream_parameter_updater.py``. A signature-parity test
(``tests/unit/test_param_schema.py``) asserts those signatures stay in sync
with ``PARAM_NAMES`` / ``UPDATER_PARAM_NAMES``, so drift is *caught*, not
*prevented*.

Note on ``default`` vs. the runtime signatures: every parameter in
``update_stream_params`` defaults to ``None`` at the call-site (meaning
"leave the current value unchanged") — including the two interpolation-method
Literals, which are additionally *sticky*: the last explicitly-supplied value
is persisted on the updater (``_last_prompt_interpolation_method`` /
``_last_seed_interpolation_method``) and re-applied whenever a caller omits
it, so a method-only update takes effect immediately and a later list-only
update doesn't silently revert it. ``ParamSpec.default`` here is a *different*
concept — the concrete construction-time value — and intentionally does not
mirror the ``None`` sentinels.

This module has no torch import and no dependency on the rest of the
``streamdiffusion`` package, so it stays cheap to import in isolation
(e.g. from a lightweight test or tool). In practice ``streamdiffusion``'s
own ``__init__.py`` eagerly imports ``.pipeline``/``.wrapper`` (torch-heavy)
before this module would ever be reached via ``from streamdiffusion...``,
so the "cheap import" property is good hygiene rather than a load-time win
for current consumers.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple

# Interpolation-method aliases shared by the wrapper and updater signatures.
PromptInterpolationMethod = Literal["average", "slerp", "cosine_weighted"]
SeedInterpolationMethod = Literal["average", "slerp", "cosine_weighted"]

# The four cfg_type values StreamDiffusion.__init__ accepts (pipeline.py's single
# assignment choke point, :85). cfg_type is construction-time only — it is not in
# PARAM_NAMES/UPDATER_PARAM_NAMES, so there is no live-update validation path to
# also guard; this constant is consumed once, at __init__.
VALID_CFG_TYPES: Tuple[str, ...] = ("none", "full", "self", "initialize")


@dataclass(frozen=True)
class ParamSpec:
    """Identity of one runtime-tunable parameter.

    Attributes
    ----------
    name:
        Keyword name, shared verbatim by the wrapper and (when ``updater``
        is True) the updater signatures.
    default:
        Construction-time default (see module docstring) — NOT the
        ``update_stream_params`` runtime default, which is ``None`` for
        every param.
    updater:
        Whether this param is forwarded to
        ``StreamParameterUpdater.update_stream_params``. False only for
        ``use_safety_checker`` / ``safety_checker_threshold``, which the
        wrapper handles itself (wrapper.py update_stream_params tail).
    """

    name: str
    default: Any
    updater: bool = True


# Order matches StreamDiffusionWrapper.update_stream_params exactly
# (wrapper.py:681-712).
PARAMS: Tuple[ParamSpec, ...] = (
    ParamSpec("num_inference_steps", 50),
    ParamSpec("guidance_scale", 1.2),
    ParamSpec("delta", 1.0),
    # Stored as a tuple so DEFAULTS never hands out a mutable list that a
    # caller could alias-mutate; consumers that need a list should do
    # list(DEFAULTS["t_index_list"]).
    ParamSpec("t_index_list", (0, 16, 32, 45)),
    ParamSpec("seed", 2),
    ParamSpec("prompt_list", None),
    ParamSpec("negative_prompt", ""),
    ParamSpec("prompt_interpolation_method", "slerp"),
    ParamSpec("normalize_prompt_weights", True),
    ParamSpec("seed_list", None),
    ParamSpec("seed_interpolation_method", "average"),
    ParamSpec("normalize_seed_weights", True),
    ParamSpec("controlnet_config", None),
    ParamSpec("ipadapter_config", None),
    ParamSpec("image_preprocessing_config", None),
    ParamSpec("image_postprocessing_config", None),
    ParamSpec("latent_preprocessing_config", None),
    ParamSpec("latent_postprocessing_config", None),
    ParamSpec("use_safety_checker", False, updater=False),
    ParamSpec("safety_checker_threshold", 0.5, updater=False),
    ParamSpec("cache_maxframes", 1),
    ParamSpec("cache_interval", 1),
    ParamSpec("cn_cache_interval", 1),
    ParamSpec("cn_cache_decay", 0.0),
    ParamSpec("fi_strength", 0.75),
    ParamSpec("fi_threshold", 0.98),
)

# All 26 params the wrapper accepts, in wrapper signature order.
PARAM_NAMES: Tuple[str, ...] = tuple(p.name for p in PARAMS)

# The 24 params forwarded to the updater, in updater signature order
# (a contiguous subsequence of PARAM_NAMES once use_safety_checker /
# safety_checker_threshold are removed).
UPDATER_PARAM_NAMES: Tuple[str, ...] = tuple(p.name for p in PARAMS if p.updater)

# name -> construction-time default.
DEFAULTS: Dict[str, Any] = {p.name: p.default for p in PARAMS}


def floor_num_inference_steps(num_inference_steps: int, max_t_index: int) -> int:
    """Raise ``num_inference_steps`` to ``max_t_index + 1`` if it's too small to
    hold the largest t_index value. Never lowers it.

    Extracted 1:1 from stream_parameter_updater.py's two
    ``if num_inference_steps <= max_t_index`` branches (~:328-348); callers
    keep their own branch-specific warning text — this helper only owns the
    arithmetic.
    """
    return max(num_inference_steps, max_t_index + 1)


# R-CFG delta bounds. The paper (2312.12491 Eq. 7) states no range for delta —
# it is only "a magnitude moderation coefficient for the virtual residual
# noise". The bounds below come from the combine's residual-noise-removal
# coefficient c = gamma - (gamma-1)*delta (pipeline.unet_step):
#   delta = 1 -> c = 1, exact cancellation (analytic optimum at
#     denoising_steps_num == 1, where stock_noise == the injected init_noise);
#   delta < 1 -> c > 1, over-subtraction -> additive Gaussian grain
#     (empirically confirmed in TD — hence the hard floor);
#   delta > 1 -> softening, useful up to gamma/(gamma-1) (see
#     delta_noise_cancellation_ceiling); 5.0 leaves headroom for low-guidance
#     use (gamma=1.2 -> ceiling 6.0) without letting fp16 magnitudes run away.
DELTA_MIN: float = 1.0
DELTA_MAX: float = 5.0


def clamp_delta(delta: float) -> Tuple[float, bool]:
    """Clamp ``delta`` to [DELTA_MIN, DELTA_MAX] = [1.0, 5.0]; returns
    ``(clamped, was_clamped)``. See the DELTA_MIN/DELTA_MAX comment for the
    derivation. Callers own their warning text.
    """
    clamped = min(max(delta, DELTA_MIN), DELTA_MAX)
    return clamped, clamped != delta


def delta_noise_cancellation_ceiling(guidance_scale: float) -> float:
    """The delta at which the R-CFG combine's residual-noise removal
    coefficient c = gamma - (gamma-1)*delta reaches zero: gamma/(gamma-1).
    Above it, the combine re-injects inverted noise (degraded output, not a
    crash). Unbounded (inf) at guidance_scale <= 1, where the uncond term
    never enters the combine.
    """
    if guidance_scale <= 1.0:
        return float("inf")
    return guidance_scale / (guidance_scale - 1.0)


def rescale_t_index_list(old_t_list: List[int], old_num_steps: int, new_num_steps: int) -> List[int]:
    """Proportionally rescale t_index values from an old step-count space to a
    new one, clamped to the new space's valid range.

    Extracted 1:1 from stream_parameter_updater.py:358-359.

    Example: rescale_t_index_list([0, 16, 32, 45], 50, 9) == [0, 3, 5, 7].
    (Not [0, 3, 6, 8] — that is the result for new_num_steps=10. The source
    comment this was extracted from had the same off-by-one; see
    tests/unit/test_param_schema.py for the corrected golden.)
    """
    scale_factor = (new_num_steps - 1) / (old_num_steps - 1) if old_num_steps > 1 else 1.0
    return [min(round(t * scale_factor), new_num_steps - 1) for t in old_t_list]


def compute_sub_timesteps(timesteps: Any, t_index_list: List[int]) -> List[Any]:
    """Select the UNet-facing timestep value for each configured t_index.

    Extracted 1:1 from pipeline.py's prepare() (``for t in self.t_list:
    self.sub_timesteps.append(self.timesteps[t])``) and
    stream_parameter_updater.py's ``_update_timestep_calculations``, which
    duplicate the exact same indexing.

    ``timesteps`` must already be the scheduler's *materialised* grid
    (``scheduler.timesteps`` after ``set_timesteps`` — and, in prepare(),
    after any sampler-specific spacing override has been applied). This
    helper deliberately does not call ``set_timesteps`` itself: doing so
    would silently discard prepare()'s spacing override for the
    ``_SPACING_SAMPLERS`` samplers (see pipeline.py's ``prepare()``).
    """
    return [timesteps[t] for t in t_index_list]


# LCM/TCD ignore timestep_spacing natively; only these samplers get the
# spacing override in materialise_timestep_grid. Mirrors pipeline.py
# prepare()'s local _SPACING_SAMPLERS (:615) — duplicated here too rather
# than imported, so this module stays import-free of the rest of the
# package (see module docstring).
_SPACING_SAMPLERS: Set[str] = frozenset({"simple", "sgm_uniform", "ddim"})


def materialise_timestep_grid(
    scheduler: Any,
    num_inference_steps: int,
    sampler_type: str,
    device: Any,
    get_spaced_timesteps: Callable[[str, int], Any],
) -> Any:
    """Build the scheduler's timestep grid, applying the sampler-specific
    spacing override StreamDiffusion's own schedules use.

    Extracted 1:1 from pipeline.py's prepare() (:612-620), which is also
    duplicated verbatim in wrapper.py's fp8-calibration path (~:2919-2926,
    self-documented there as a duplicate of this exact block). Both call
    sites — and any standalone probe — should call this instead of
    reimplementing the ``_SPACING_SAMPLERS`` override, so they can never
    silently drift apart on what "spacing override" means. This is the
    other half of the contract ``compute_sub_timesteps`` names in its own
    docstring: that helper deliberately won't call ``set_timesteps`` itself
    because doing so would discard this override.

    LCM/TCD ignore ``timestep_spacing`` natively — only samplers in
    ``_SPACING_SAMPLERS`` ("simple", "sgm_uniform", "ddim") get it; "normal"
    (LCM's native grid) passes through untouched.

    ``get_spaced_timesteps`` takes ``(spacing, num_inference_steps)`` and
    returns a tensor already placed on a device — see
    ``StreamDiffusion._get_spaced_timesteps``, which callers with a live
    pipeline should pass bound (``pipeline._get_spaced_timesteps``); a probe
    with no pipeline instance can pass any equivalent free function. Mutates
    ``scheduler.timesteps`` in place (matching prepare()'s own behaviour) and
    also returns the resulting grid, moved to ``device``, for callers that
    only want the return value (e.g. a standalone probe).
    """
    scheduler.set_timesteps(num_inference_steps, device)
    if sampler_type in _SPACING_SAMPLERS:
        spacing = getattr(scheduler.config, "timestep_spacing", "leading")
        if spacing in ("trailing", "linspace", "leading"):
            scheduler.timesteps = get_spaced_timesteps(spacing, num_inference_steps).to(device)
    return scheduler.timesteps.to(device)


# Ghost-bleed inter-step beta_sqrt threshold. See bleed_risk_message.
GHOST_BLEED_THRESHOLD: float = 0.75


def bleed_risk_message(
    inter_step_betas: Sequence[float],
    t_list: Sequence[int],
    use_denoising_batch: bool,
    do_add_noise: bool,
    threshold: float = GHOST_BLEED_THRESHOLD,
) -> Optional[str]:
    """Warning text if the current schedule risks ghost bleed from the
    previous frame's content, or None if it doesn't apply.

    Extracted 1:1 from stream_parameter_updater.py's
    ``_update_timestep_calculations`` (~:1216-1236). Only meaningful on the
    batched multi-step path: with ``do_add_noise=False`` the denoising-batch
    pipelining trick hands each inter-step slot the *previous frame's*
    partially-denoised latent with no noise re-injected, so a large
    inter-step beta_sqrt (marginal noise std at that step) means the model
    is being asked to resolve a residual it was never given the noise budget
    to explain — audible as bleed from the prior frame.

    ``inter_step_betas`` must be the per-step ``beta_prod_t_sqrt`` values for
    the sub_timesteps *after* the first (index 0 has no "previous step" to
    bleed from), pre-``repeat_interleave`` — i.e. ``beta_prod_t_sqrt[1:, 0,
    0, 0]`` in pipeline.py's tensor layout.

    Callers own their own logging call (this returns text, not a log side
    effect), so the same message can be shared between
    ``_update_timestep_calculations`` and a schedule-diagnostics dump without
    either one depending on the other's logger.
    """
    if not (use_denoising_batch and not do_add_noise and len(t_list) > 1):
        return None
    if not inter_step_betas:
        return None
    max_beta = max(inter_step_betas)
    if max_beta <= threshold:
        return None
    return (
        f"do_add_noise=False + use_denoising_batch: inter-step beta_sqrt={max_beta:.3f} "
        f"(t_index={list(t_list[1:])}) exceeds {threshold:.2f}. Previous-frame ghost bleed likely -- "
        "consider enabling do_add_noise (reference fork default)."
    )


def _band_interior_ints(span_len: int, count: int) -> List[int]:
    """Pick up to ``count`` offsets into ``range(span_len)``, at sub-interval
    *centers* rather than endpoints: ``floor((i + 0.5) * span_len / count)``
    for ``i in range(count)``.

    Unlike an endpoint-inclusive linspace, these offsets never land on 0 or
    ``span_len - 1`` unless the band is so narrow relative to ``count`` that
    centers collapse onto the edges. This matters because adjacent bands are
    half-open and touch at their boundary, and a caller's configured values
    often sit exactly on a decade boundary — an endpoint sampler wastes
    budget re-picking those already-``picked`` boundary values (see
    ``build_calibration_t_indices``). May return fewer than ``count``
    distinct values if the span is narrower than the request — callers dedup
    via a set.
    """
    if count <= 0 or span_len <= 0:
        return []
    return [int(min((i + 0.5) * span_len // count, span_len - 1)) for i in range(count)]


def _widest_gap_index(lo: int, hi: int, picked: Set[int]) -> Optional[int]:
    """The unpicked index in ``[lo, hi]`` sitting in the band's widest
    remaining gap — furthest from any already-picked value. Ties resolve
    toward the band midpoint (then the lowest index), keeping the pick
    interior for the same reason ``_band_interior_ints`` avoids endpoints.
    None when the band holds no unpicked index.
    """
    candidates = [i for i in range(lo, hi + 1) if i not in picked]
    if not candidates:
        return None
    mid = (lo + hi) / 2

    def _distance(i: int) -> int:
        return min(abs(i - p) for p in picked) if picked else 0

    return min(candidates, key=lambda i: (-_distance(i), abs(i - mid), i))


def build_calibration_t_indices(
    t_index_list: List[int],
    num_inference_steps: int,
    budget: int,
) -> List[int]:
    """Derive a calibration t_index schedule covering the timestep region
    deployment actually visits, per a neighbour-midpoint band rule: each
    distinct, clamped value in ``t_index_list`` owns the index range up to
    the midpoint with its neighbours, so bands are derived from the values
    themselves rather than from a fixed decade grid. The first band floors
    at 1 and the last extends to ``num_inference_steps - 1``, so the bands
    partition ``[1, num_inference_steps - 1]`` with no gaps and no overlaps,
    and every configured value is provably inside its own band — including
    the single-entry case, where one band spans the full range. Band 0's
    floor is 1, not 0 — index 0 is the highest-noise raw timestep (~999) and
    must never be calibrated implicitly; it only enters the result if
    ``t_index_list`` configures it explicitly. Bands scale automatically with
    the number of distinct values in ``t_index_list``, so this stays correct
    for whatever step count a static TRT engine has locked in for its
    lifetime.

    Every value in ``t_index_list`` is always included in the result — the
    exact deployment points must never be missed, even though only the
    *union* of bands matters for FP8 scale correctness (a per-tensor amax at
    a given raw timestep doesn't care which step index produced it). The
    remaining budget (``budget - len(set(t_index_list))``) is spread as
    evenly as possible across the bands, sampling each band's *interior*
    (see ``_band_interior_ints``) so spare slots don't collide with the
    band-boundary values ``t_index_list`` typically already occupies. Any
    per-band budget a narrow band can't absorb spills forward into the
    remaining bands; a later band whose own interior picks collide with
    already-picked values can still fall short even after that spill, so a
    final round-robin top-up (widest remaining gap per band) mops up
    whatever is left, guaranteeing the full budget is consumed whenever the
    bands have room for it. A live ``/t_list`` change that stays within its
    calibrated band doesn't need an engine rebuild.

    Returns a sorted-ascending, deduplicated list of t_index values. Normally
    ``len(result) <= budget``; if ``budget < len(set(t_index_list))`` the
    configured values still all win and the result exceeds budget — never
    drop a deployment point to fit a smaller budget.
    """
    max_idx = max(num_inference_steps - 1, 0)

    picked = {min(max(t, 0), max_idx) for t in t_index_list}

    # Bands derive from the configured values themselves (sorted ascending),
    # not a fixed decade grid -- band k owns the range up to the midpoint
    # with its neighbours, so every value is guaranteed to land inside the
    # band derived for it, regardless of len(t_index_list) or spacing.
    values = sorted(picked)
    n_bands = max(len(values), 1)
    bands = []
    if not values:
        bands.append((min(1, max_idx), max_idx))
    else:
        for k, t in enumerate(values):
            lo = 1 if k == 0 else (values[k - 1] + t) // 2 + 1
            hi = max_idx if k == n_bands - 1 else (t + values[k + 1]) // 2
            bands.append((min(max(lo, 0), max_idx), min(hi, max_idx)))

    remaining = max(budget - len(picked), 0)
    base, extra = divmod(remaining, n_bands)
    band_budgets = [base + (1 if i < extra else 0) for i in range(n_bands)]

    # Two passes so unused budget (band narrower than its share) spills
    # forward into later bands instead of being silently dropped.
    spill = 0
    for i, (lo, hi) in enumerate(bands):
        span_len = hi - lo + 1
        want = band_budgets[i] + spill
        span = list(range(lo, hi + 1))
        before = len(picked)
        for j in _band_interior_ints(span_len, want):
            picked.add(span[j])
        used = len(picked) - before
        spill = max(want - used, 0) if used < want else 0

    # Forward-only spill can still leave budget on the table -- the last band
    # has nowhere further to spill into if its own interior picks collide
    # with already-picked values (see module-level defect notes). Top up by
    # sweeping every band's widest remaining gap, round-robin, until the
    # budget is met or every band is saturated.
    while len(picked) < budget:
        progress = False
        for lo, hi in bands:
            if len(picked) >= budget:
                break
            idx = _widest_gap_index(lo, hi, picked)
            if idx is not None:
                picked.add(idx)
                progress = True
        if not progress:
            break

    return sorted(picked)
