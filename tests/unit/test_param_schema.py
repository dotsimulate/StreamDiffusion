"""Unit tests for src/streamdiffusion/param_schema.py.

Covers:
  - PARAM_NAMES / UPDATER_PARAM_NAMES counts and ordering
  - DEFAULTS golden values (construction-time defaults, not the
    update_stream_params None-sentinel defaults)
  - floor_num_inference_steps / rescale_t_index_list vs hand-computed values
  - signature parity: PARAM_NAMES / UPDATER_PARAM_NAMES must match the real
    StreamDiffusionWrapper.update_stream_params / StreamParameterUpdater.
    update_stream_params signatures — this is the regression lock that
    catches drift the moment either signature changes.

CPU-only, no CUDA required (importing streamdiffusion pulls in torch, but
no tensor is ever created).
"""

import inspect

import pytest

from streamdiffusion.param_schema import (
    DEFAULTS,
    PARAM_NAMES,
    UPDATER_PARAM_NAMES,
    clamp_delta,
    delta_noise_cancellation_ceiling,
    floor_num_inference_steps,
    rescale_t_index_list,
)
from streamdiffusion.stream_parameter_updater import StreamParameterUpdater
from streamdiffusion.wrapper import StreamDiffusionWrapper

WRAPPER_ONLY_PARAMS = {"use_safety_checker", "safety_checker_threshold"}


class TestParamNames:
    def test_param_names_count(self):
        assert len(PARAM_NAMES) == 26

    def test_updater_param_names_count(self):
        assert len(UPDATER_PARAM_NAMES) == 24

    def test_updater_param_names_is_ordered_subsequence_of_param_names(self):
        """Dropping the two wrapper-only names from PARAM_NAMES, in place,
        must yield exactly UPDATER_PARAM_NAMES (order preserved)."""
        filtered = tuple(n for n in PARAM_NAMES if n not in WRAPPER_ONLY_PARAMS)
        assert filtered == UPDATER_PARAM_NAMES

    def test_wrapper_only_params_excluded_from_updater(self):
        assert not (WRAPPER_ONLY_PARAMS & set(UPDATER_PARAM_NAMES))
        assert set(PARAM_NAMES) >= WRAPPER_ONLY_PARAMS

    def test_no_duplicate_names(self):
        assert len(PARAM_NAMES) == len(set(PARAM_NAMES))


class TestSignatureParity:
    """Regression lock: PARAM_NAMES / UPDATER_PARAM_NAMES must track the real
    signatures. If either signature changes without updating param_schema.py,
    these fail immediately."""

    def test_wrapper_signature_matches_param_names(self):
        sig = inspect.signature(StreamDiffusionWrapper.update_stream_params)
        params = [name for name in sig.parameters if name != "self"]
        assert tuple(params) == PARAM_NAMES

    def test_updater_signature_matches_updater_param_names(self):
        sig = inspect.signature(StreamParameterUpdater.update_stream_params)
        params = [name for name in sig.parameters if name != "self"]
        assert tuple(params) == UPDATER_PARAM_NAMES


class TestDefaultsGolden:
    """Spot-check construction-time defaults against the literals confirmed
    identical in both config.py (_extract_wrapper_params /
    _extract_prepare_params) and StreamDiffusionWrapper.__init__."""

    def test_prepare_time_defaults(self):
        assert DEFAULTS["num_inference_steps"] == 50
        assert DEFAULTS["guidance_scale"] == 1.2
        assert DEFAULTS["delta"] == 1.0

    def test_t_index_list_default_and_immutability(self):
        assert list(DEFAULTS["t_index_list"]) == [0, 16, 32, 45]
        # Must not be a mutable list a caller could alias-mutate.
        assert not isinstance(DEFAULTS["t_index_list"], list)

    def test_scalar_defaults(self):
        assert DEFAULTS["seed"] == 2
        assert DEFAULTS["negative_prompt"] == ""
        assert DEFAULTS["use_safety_checker"] is False
        assert DEFAULTS["safety_checker_threshold"] == 0.5
        assert DEFAULTS["normalize_prompt_weights"] is True
        assert DEFAULTS["normalize_seed_weights"] is True
        assert DEFAULTS["cache_maxframes"] == 1
        assert DEFAULTS["cache_interval"] == 1
        assert DEFAULTS["cn_cache_interval"] == 1
        assert DEFAULTS["cn_cache_decay"] == 0.0
        assert DEFAULTS["fi_strength"] == 0.75
        assert DEFAULTS["fi_threshold"] == 0.98

    def test_interpolation_method_defaults(self):
        assert DEFAULTS["prompt_interpolation_method"] == "slerp"
        assert DEFAULTS["seed_interpolation_method"] == "linear"

    def test_config_only_params_default_none(self):
        for name in (
            "prompt_list",
            "seed_list",
            "controlnet_config",
            "ipadapter_config",
            "image_preprocessing_config",
            "image_postprocessing_config",
            "latent_preprocessing_config",
            "latent_postprocessing_config",
        ):
            assert DEFAULTS[name] is None


class TestFloorNumInferenceSteps:
    def test_no_change_when_already_large_enough(self):
        assert floor_num_inference_steps(50, 45) == 50

    def test_raises_when_too_small(self):
        assert floor_num_inference_steps(9, 45) == 46

    def test_boundary_equal_to_max_t_index_is_too_small(self):
        # Original code: `if num_inference_steps <= max_t_index: ... = max_t_index + 1`
        assert floor_num_inference_steps(45, 45) == 46

    def test_boundary_one_above_max_t_index_is_fine(self):
        assert floor_num_inference_steps(46, 45) == 46


class TestClampDelta:
    def test_in_range_unchanged(self):
        assert clamp_delta(1.5) == (1.5, False)

    def test_below_min_clamped_to_one(self):
        # delta < 1 over-subtracts the residual (c = gamma - (gamma-1)*delta > 1)
        # and adds Gaussian grain — empirically confirmed in TD.
        assert clamp_delta(0.5) == (1.0, True)

    def test_negative_clamped_to_one(self):
        assert clamp_delta(-0.1) == (1.0, True)

    def test_above_max_clamped_to_max(self):
        assert clamp_delta(6.0) == (5.0, True)

    def test_boundary_min_not_clamped(self):
        assert clamp_delta(1.0) == (1.0, False)

    def test_boundary_max_not_clamped(self):
        assert clamp_delta(5.0) == (5.0, False)


class TestDeltaCeiling:
    """delta_noise_cancellation_ceiling: gamma/(gamma-1) — the delta at which
    the CFG combine's residual-noise removal coefficient c = gamma-(gamma-1)*delta
    reaches zero; above it, inverted noise is re-injected."""

    def test_gamma_1_4(self):
        assert delta_noise_cancellation_ceiling(1.4) == pytest.approx(3.5)

    def test_gamma_2_0(self):
        assert delta_noise_cancellation_ceiling(2.0) == pytest.approx(2.0)

    def test_gamma_at_or_below_one_is_unbounded(self):
        # At gamma <= 1 the uncond term never enters the combine — no ceiling.
        assert delta_noise_cancellation_ceiling(1.0) == float("inf")
        assert delta_noise_cancellation_ceiling(0.5) == float("inf")


class TestRescaleTIndexList:
    def test_golden_50_to_9(self):
        """Corrected golden — the code's scale_factor=(new-1)/(old-1) gives
        [0,3,5,7] for 50->9, NOT [0,3,6,8] (that off-by-one was in the
        original source comment at stream_parameter_updater.py:357 and was
        copied into an earlier draft of this extraction)."""
        assert rescale_t_index_list([0, 16, 32, 45], 50, 9) == [0, 3, 5, 7]

    def test_golden_50_to_10(self):
        """[0,3,6,8] is the correct result for new_num_steps=10, not 9."""
        assert rescale_t_index_list([0, 16, 32, 45], 50, 10) == [0, 3, 6, 8]

    def test_single_old_step_no_division_by_zero(self):
        assert rescale_t_index_list([0], 1, 9) == [0]

    def test_same_step_count_is_identity(self):
        assert rescale_t_index_list([0, 16, 32, 45], 50, 50) == [0, 16, 32, 45]

    def test_result_clamped_to_new_range(self):
        result = rescale_t_index_list([0, 49], 50, 5)
        assert max(result) <= 4
