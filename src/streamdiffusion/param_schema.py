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
"leave the current value unchanged"), except the two interpolation-method
Literals. ``ParamSpec.default`` here is a *different* concept — the
concrete construction-time value — and intentionally does not mirror the
``None`` sentinels.

This module has no torch import and no dependency on the rest of the
``streamdiffusion`` package, so it stays cheap to import in isolation
(e.g. from a lightweight test or tool). In practice ``streamdiffusion``'s
own ``__init__.py`` eagerly imports ``.pipeline``/``.wrapper`` (torch-heavy)
before this module would ever be reached via ``from streamdiffusion...``,
so the "cheap import" property is good hygiene rather than a load-time win
for current consumers.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

# Interpolation-method aliases shared by the wrapper and updater signatures.
PromptInterpolationMethod = Literal["linear", "slerp", "cosine_weighted"]
SeedInterpolationMethod = Literal["linear", "slerp"]


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
        all but the two interpolation-method params.
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
    ParamSpec("seed_interpolation_method", "linear"),
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
