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
    build_calibration_t_indices,
    clamp_delta,
    compute_sub_timesteps,
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
        assert DEFAULTS["seed_interpolation_method"] == "average"

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


class TestComputeSubTimesteps:
    def test_indexes_by_position(self):
        assert compute_sub_timesteps([10, 20, 30, 40], [0, 2, 3]) == [10, 30, 40]

    def test_reproduces_known_lcm_grid(self):
        """timesteps[j] = 999 - 20j for num_inference_steps=50,
        original_inference_steps=100 (confirmed by direct computation against
        diffusers' LCMScheduler.set_timesteps -- see the fp8-round-5-handoff
        plan). Deployment t_index_list [15, 21, 27] must map to raw UNet
        timesteps 699 / 579 / 459, not the turbo-schedule 999/499/249 that
        4-step calibration used to sample."""
        timesteps = [999 - 20 * j for j in range(50)]
        assert compute_sub_timesteps(timesteps, [15, 21, 27]) == [699, 579, 459]


class TestBuildCalibrationTIndices:
    """Band spec (user's decade rule): the k-th configured t_index_list entry
    anchors band [k*10, (k+1)*10). Verified against the production config
    (t_index_list=[15,21,27], num_inference_steps=50) throughout."""

    def test_always_includes_configured_t_index_list(self):
        result = build_calibration_t_indices([15, 21, 27], num_inference_steps=50, budget=8)
        assert {15, 21, 27} <= set(result)

    def test_result_is_sorted_and_deduplicated(self):
        result = build_calibration_t_indices([15, 21, 27], num_inference_steps=50, budget=8)
        assert result == sorted(set(result))

    def test_covers_every_band(self):
        """[15,21,27] sits in bands 1,2,2 (not 0,1,2) -- band 0 ([0,10]) has
        no configured point in it at all, so band coverage only holds if the
        remaining budget is genuinely spread across every band, not just
        parked next to the configured values."""
        result = build_calibration_t_indices([15, 21, 27], num_inference_steps=50, budget=8)
        for lo, hi in [(0, 10), (10, 20), (20, 30)]:
            assert any(lo <= t <= hi for t in result), f"band [{lo},{hi}] uncovered: {result}"

    def test_generalises_to_five_step_schedule(self):
        t_index_list = [5, 14, 23, 32, 41]
        result = build_calibration_t_indices(t_index_list, num_inference_steps=50, budget=12)
        assert set(t_index_list) <= set(result)
        assert result == sorted(set(result))
        assert len(result) <= 12

    def test_generalises_to_three_step_schedule(self):
        t_index_list = [15, 21, 27]
        result = build_calibration_t_indices(t_index_list, num_inference_steps=50, budget=8)
        assert set(t_index_list) <= set(result)
        assert len(result) <= 8

    def test_bands_clamp_at_max_index_no_out_of_range(self):
        # num_inference_steps=25 -> max valid t_index is 24. Configured
        # values sit near the top of the grid; bands must not run past it.
        result = build_calibration_t_indices([20, 22, 24], num_inference_steps=25, budget=8)
        assert all(0 <= t <= 24 for t in result)

    def test_never_negative(self):
        result = build_calibration_t_indices([0, 1, 2], num_inference_steps=10, budget=8)
        assert all(t >= 0 for t in result)

    def test_configured_values_win_when_budget_too_small(self):
        """budget smaller than len(t_index_list): every deployment point must
        still be present even though that means the result exceeds budget."""
        t_index_list = [15, 21, 27, 33, 39]
        result = build_calibration_t_indices(t_index_list, num_inference_steps=50, budget=3)
        assert set(t_index_list) <= set(result)

    def test_index_zero_absent_unless_configured(self):
        """fp8-round-5-handoff round 6: band 0's floor was k*band_width == 0,
        so t_index 0 (raw timestep ~999, pure noise) was silently included
        even when nothing configured it. Band 0 must now floor at 1."""
        result = build_calibration_t_indices([10, 20, 26], num_inference_steps=50, budget=8)
        assert 0 not in result

    def test_production_config_uses_full_budget(self):
        """The exact config from the round-5 log (t_index_list=[10,20,26],
        num_inference_steps=50, budget=8) must now consume all 8 slots as
        distinct in-band indices, matching the fp8-round-7 hand trace under
        neighbour-midpoint bands: [(1,15), (16,23), (24,49)]."""
        result = build_calibration_t_indices([10, 20, 26], num_inference_steps=50, budget=8)
        assert result == [4, 10, 12, 18, 20, 22, 26, 37]

    def test_no_value_repeated_across_adjacent_bands(self):
        """Regression for the old bug where _evenly_spaced_ints returned band
        endpoints and adjacent bands shared a boundary, so most of the spare
        budget re-picked an already-picked value instead of a new one."""
        result = build_calibration_t_indices([10, 20, 26], num_inference_steps=50, budget=8)
        assert len(result) == len(set(result)) == 8

    def test_terminal_band_spill_is_not_dropped(self):
        """fp8-round-6.1: the live t_index_list=[7,16,25] config left the last
        band's own interior pick colliding with an already-picked value, and
        forward-only spill has nowhere further to go from the last band --
        budget silently dropped to 7/8. A round-robin top-up must recover it."""
        result = build_calibration_t_indices([7, 16, 25], num_inference_steps=50, budget=8)
        assert len(result) == 8

    def test_full_budget_consumed_across_configs(self):
        """Every config traced during the fp8-round-6.1 investigation must
        consume its full budget now that the top-up sweep backstops spill."""
        configs = [
            [7, 16, 25],
            [10, 20, 26],
            [0, 16, 32, 45],
            [5, 15],
            [3, 13, 23, 33, 43],
        ]
        for t_index_list in configs:
            result = build_calibration_t_indices(t_index_list, num_inference_steps=50, budget=8)
            assert len(result) == 8, f"{t_index_list} -> {result}"

    def test_single_t_index_spreads_across_full_range(self):
        """fp8-round-7: under the old fixed-decade grid, a single configured
        value only anchored band [1,9], collapsing 7 of 8 rows into the
        noisiest indices (1..9) and leaving the deployment point isolated.
        Neighbour-midpoint bands must give a lone value the *entire* range,
        so the budget spreads across it instead of piling up near index 1."""
        result = build_calibration_t_indices([25], num_inference_steps=50, budget=8)
        assert len(result) == 8
        assert max(result) > 9, f"budget collapsed near index 1: {result}"

    def test_every_configured_value_lands_in_its_own_band(self):
        """fp8-round-7: the old fixed-decade grid assigned band k to
        t_index_list[k] positionally, with no guarantee the value actually
        fell inside it -- e.g. [0,16,32,45] put 32 in band 3 (decade
        [30,39], "belonging" to 45) and left 45 outside every band. Under
        neighbour-midpoint bands each value must provably sit inside the
        band derived for it -- except an explicit 0, whose own band 0 still
        floors at 1 by design (see test_index_zero_absent_unless_configured);
        0 stays in the result via direct picked-membership, just outside its
        derived band's interior-sampling range."""
        for t_index_list in ([7, 16, 25], [0, 16, 32, 45], [25], [5, 15]):
            values = sorted(set(t_index_list))
            n = len(values)
            for k, t in enumerate(values):
                if k == 0 and t == 0:
                    continue
                lo = 1 if k == 0 else (values[k - 1] + t) // 2 + 1
                hi = 49 if k == n - 1 else (t + values[k + 1]) // 2
                assert lo <= t <= hi, f"{t} not in its own band [{lo},{hi}] for {t_index_list}"


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
