"""Unit tests for src/streamdiffusion/model_detection.py's is_turbo discrimination.

Covers `_detect_turbo_from_scheduler` directly and `detect_model`'s end-to-end
`is_turbo` result for the three checkpoints named in
docs/plans/SDXL-Turbo_Research_Verification.md's defect 2: sd-turbo, sdxl-turbo,
and sdxl-base-1.0. UNet config values below are the real published
unet/config.json / scheduler_config.json fields for each checkpoint (sdxl-base-1.0
is stubbed from its published config rather than downloading the ~7 GB checkpoint;
sd-turbo/sdxl-turbo are also stubbed here for speed and to keep the test
network-free and deterministic).

The point of this file: sdxl-turbo and sdxl-base-1.0 have an *identical* UNet-config
shape (`time_cond_proj_dim: null`, `addition_embed_type: "text_time"`) — that is
exactly why the old `time_cond_proj_dim is None` heuristic misclassified SDXL-Base
as Turbo. Only the scheduler discriminates them.

CPU-only, no CUDA required, no network access, no model weights loaded.
"""

from unittest.mock import MagicMock

import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from safetensors.torch import save_file

from streamdiffusion.model_detection import (
    _detect_turbo_from_scheduler,
    detect_model,
    read_safetensors_metadata,
    resolve_is_turbo,
    turbo_from_checkpoint_metadata,
    turbo_hint_from_model_id,
)


def _fake_unet(config: dict) -> MagicMock:
    """A UNet2DConditionModel-shaped mock carrying only the config fields
    detect_model reads. Mock(spec=...) makes isinstance() checks pass; the
    FrozenDict config supports both dict-style (`config.get(...)`) and
    attribute-style (`config.cross_attention_dim`) access, matching real
    diffusers configs."""
    unet = MagicMock(spec=UNet2DConditionModel)
    unet.config = FrozenDict(config)
    return unet


def _fake_pipe(class_name: str, timestep_spacing) -> MagicMock:
    scheduler_config = FrozenDict({"_class_name": class_name, "timestep_spacing": timestep_spacing})
    scheduler = MagicMock()
    scheduler.config = scheduler_config
    pipe = MagicMock()
    pipe.scheduler = scheduler
    return pipe


# Published unet/config.json fields (the ones detect_model reads).
SD_TURBO_UNET_CONFIG = {
    "addition_embed_type": None,
    "cross_attention_dim": 1024,
    "time_cond_proj_dim": None,
}
SDXL_TURBO_UNET_CONFIG = {
    "addition_embed_type": "text_time",
    "cross_attention_dim": 2048,
    "time_cond_proj_dim": None,
}
# Same shape as sdxl-turbo's -- the UNet config alone cannot tell them apart.
SDXL_BASE_UNET_CONFIG = {
    "addition_embed_type": "text_time",
    "cross_attention_dim": 2048,
    "time_cond_proj_dim": None,
}

# Published scheduler_config.json fields.
TURBO_SCHEDULER = ("EulerAncestralDiscreteScheduler", "trailing")
SDXL_BASE_SCHEDULER = ("EulerDiscreteScheduler", "leading")


class TestDetectTurboFromScheduler:
    def test_no_pipe_is_undecided(self):
        assert _detect_turbo_from_scheduler(None) is None

    def test_pipe_without_scheduler_is_undecided(self):
        pipe = MagicMock()
        pipe.scheduler = None
        assert _detect_turbo_from_scheduler(pipe) is None

    def test_euler_ancestral_trailing_is_turbo(self):
        pipe = _fake_pipe(*TURBO_SCHEDULER)
        assert _detect_turbo_from_scheduler(pipe) is True

    def test_euler_discrete_leading_is_not_turbo(self):
        pipe = _fake_pipe(*SDXL_BASE_SCHEDULER)
        assert _detect_turbo_from_scheduler(pipe) is False

    def test_euler_ancestral_without_trailing_is_not_turbo(self):
        """Class name alone isn't enough -- spacing must also be trailing."""
        pipe = _fake_pipe("EulerAncestralDiscreteScheduler", "leading")
        assert _detect_turbo_from_scheduler(pipe) is False


class TestDetectModelIsTurbo:
    """End-to-end: detect_model(unet, pipe)['turbo_from_scheduler'] for the three
    checkpoints named in the verification doc's defect 2.

    detect_model() no longer collapses this into a bool of its own -- it reports
    the Optional[bool] scheduler signal verbatim; resolve_is_turbo() is the sole
    place that resolves it into an authoritative Turbo/non-Turbo decision."""

    def test_sd_turbo_is_turbo(self):
        unet = _fake_unet(SD_TURBO_UNET_CONFIG)
        pipe = _fake_pipe(*TURBO_SCHEDULER)
        result = detect_model(unet, pipe)
        assert result["model_type"] == "SD2.1"
        assert result["turbo_from_scheduler"] is True

    def test_sdxl_turbo_is_turbo(self):
        unet = _fake_unet(SDXL_TURBO_UNET_CONFIG)
        pipe = _fake_pipe(*TURBO_SCHEDULER)
        result = detect_model(unet, pipe)
        assert result["model_type"] == "SDXL"
        assert result["turbo_from_scheduler"] is True

    def test_sdxl_base_is_not_turbo(self):
        """The regression case: sdxl-base-1.0's UNet config is indistinguishable
        from sdxl-turbo's (both have time_cond_proj_dim=None), so only the
        scheduler-based check keeps this False."""
        unet = _fake_unet(SDXL_BASE_UNET_CONFIG)
        pipe = _fake_pipe(*SDXL_BASE_SCHEDULER)
        result = detect_model(unet, pipe)
        assert result["model_type"] == "SDXL"
        assert result["turbo_from_scheduler"] is False

    def test_no_pipe_is_undecided(self):
        """No pipe -> _detect_turbo_from_scheduler returns None -> turbo_from_scheduler
        stays None (undecided), not False. Matches the two non-pipe call sites
        (unet_sdxl_export.py, controlnet_models.py), which no longer even read this
        key -- Turbo status there is resolved authoritatively on the wrapper via
        resolve_is_turbo(), never re-derived from a pipe-less detect_model() call."""
        unet = _fake_unet(SDXL_TURBO_UNET_CONFIG)
        result = detect_model(unet, pipe=None)
        assert result["turbo_from_scheduler"] is None

    def test_is_turbo_key_not_reintroduced(self):
        """Tripwire: detect_model()'s output must never carry a collapsed
        "is_turbo" bool again. Both unet_sdxl_export.py's get_sdxl_tensorrt_config
        and controlnet_models.py's ControlNetSDXLTRT build a config dict straight
        from this return value without an intermediate allowlist -- if "is_turbo"
        reappeared here, either site would silently start propagating a stale
        pipe-less signal instead of the wrapper's authoritative resolve_is_turbo()
        verdict."""
        unet = _fake_unet(SDXL_TURBO_UNET_CONFIG)
        result = detect_model(unet, pipe=_fake_pipe(*TURBO_SCHEDULER))
        assert "is_turbo" not in result
        assert "turbo_from_scheduler" in result


class TestDetectTurboFromSchedulerClassNameFallback:
    """dotsimulate PR #58 gap, part 1: `_class_name` is only present on configs
    loaded from a JSON scheduler_config.json (from_pretrained). A scheduler built
    via `.from_config(...)` -- diffusers' `_legacy_load_scheduler` synthesized-
    defaults path for single-file loads -- has no `_class_name` key at all, so the
    old `getattr(scheduler_config, "_class_name", "")` silently degraded to "" and
    every such scheduler read as non-Turbo regardless of its real class."""

    def test_class_name_absent_falls_back_to_runtime_class(self):
        # Real diffusers object, not a FrozenDict stand-in: from_config never
        # stamps `_class_name` onto the resulting config (verified empirically).
        scheduler = EulerAncestralDiscreteScheduler.from_config(
            {"num_train_timesteps": 1000, "timestep_spacing": "trailing"}
        )
        assert "_class_name" not in scheduler.config

        pipe = MagicMock()
        pipe.scheduler = scheduler
        assert _detect_turbo_from_scheduler(pipe) is True

    def test_class_name_absent_non_trailing_is_not_turbo(self):
        scheduler = EulerAncestralDiscreteScheduler.from_config(
            {"num_train_timesteps": 1000, "timestep_spacing": "leading"}
        )
        pipe = MagicMock()
        pipe.scheduler = scheduler
        assert _detect_turbo_from_scheduler(pipe) is False


class TestSafetensorsMetadata:
    """dotsimulate PR #58 gap, part 2: for single-file loads, corroborate (or veto)
    the unreliable scheduler signal with the checkpoint's own embedded SAI Model
    Spec `modelspec.architecture` tag, read header-only (no weights loaded)."""

    def _write_checkpoint(self, tmp_path, metadata=None):
        path = str(tmp_path / "model.safetensors")
        save_file({"weight": torch.zeros(2, 2)}, path, metadata=metadata)
        return path

    def test_read_safetensors_metadata_returns_embedded_dict(self, tmp_path):
        path = self._write_checkpoint(tmp_path, {"modelspec.architecture": "stable-diffusion-xl-turbo-v1"})
        assert read_safetensors_metadata(path) == {"modelspec.architecture": "stable-diffusion-xl-turbo-v1"}

    def test_read_safetensors_metadata_missing_file_returns_empty_dict(self):
        assert read_safetensors_metadata("D:/does/not/exist.safetensors") == {}

    def test_read_safetensors_metadata_none_path_returns_empty_dict(self):
        assert read_safetensors_metadata(None) == {}

    def test_turbo_from_checkpoint_metadata_turbo_architecture(self, tmp_path):
        path = self._write_checkpoint(tmp_path, {"modelspec.architecture": "stable-diffusion-xl-turbo-v1"})
        assert turbo_from_checkpoint_metadata(path) is True

    def test_turbo_from_checkpoint_metadata_base_architecture_is_veto(self, tmp_path):
        """A confirmed non-Turbo architecture tag is a genuine negative -- this is
        the case that must be able to override a misleading "turbo" filename."""
        path = self._write_checkpoint(tmp_path, {"modelspec.architecture": "stable-diffusion-xl-v1-base"})
        assert turbo_from_checkpoint_metadata(path) is False

    def test_turbo_from_checkpoint_metadata_absent_is_undecided(self, tmp_path):
        """The common case for community merges: no modelspec metadata at all."""
        path = self._write_checkpoint(tmp_path, metadata=None)
        assert turbo_from_checkpoint_metadata(path) is None


class TestTurboHintFromModelId:
    def test_turbo_in_basename_is_true(self):
        assert turbo_hint_from_model_id("D:/models/SDXL-Turbo-merge.safetensors") is True

    def test_turbo_case_insensitive(self):
        assert turbo_hint_from_model_id("D:/models/sdxl_TURBO_merge.safetensors") is True

    def test_turbo_in_parent_dir_only_is_false(self):
        """Guard case: a base checkpoint staged under a `turbo_tests/` directory
        must not be misread as a Turbo checkpoint from its path alone."""
        assert turbo_hint_from_model_id("D:/turbo_tests/sdxl_base.safetensors") is False

    def test_no_hint_is_false(self):
        assert turbo_hint_from_model_id("D:/models/sdxl_base.safetensors") is False

    def test_none_is_false(self):
        assert turbo_hint_from_model_id(None) is False


class TestResolveIsTurbo:
    """Precedence-matrix tests covering the 7 validated cases from the plan plus
    both explicit-override directions."""

    def _checkpoint(self, tmp_path, metadata=None):
        path = str(tmp_path / "model.safetensors")
        save_file({"weight": torch.zeros(2, 2)}, path, metadata=metadata)
        return path

    def test_explicit_true_wins_outright(self, tmp_path):
        path = self._checkpoint(tmp_path, {"modelspec.architecture": "stable-diffusion-xl-v1-base"})
        is_turbo, source = resolve_is_turbo(explicit=True, model_id_or_path=path, loaded_via_single_file=True)
        assert (is_turbo, source) == (True, "explicit")

    def test_explicit_false_wins_outright(self):
        pipe = _fake_pipe(*TURBO_SCHEDULER)
        is_turbo, source = resolve_is_turbo(
            pipe=pipe, explicit=False, model_id_or_path="turbo.safetensors", loaded_via_single_file=False
        )
        assert (is_turbo, source) == (False, "explicit")

    def test_repo_id_load_trusts_scheduler(self):
        """stabilityai/sdxl-turbo / stabilityai/sd-turbo (repo id) case: unchanged,
        deployed behaviour."""
        pipe = _fake_pipe(*TURBO_SCHEDULER)
        is_turbo, source = resolve_is_turbo(
            pipe=pipe, model_id_or_path="stabilityai/sdxl-turbo", loaded_via_single_file=False
        )
        assert (is_turbo, source) == (True, "scheduler")

    def test_repo_id_base_load_trusts_scheduler(self):
        """sdxl-base-1.0 (repo id) case."""
        pipe = _fake_pipe(*SDXL_BASE_SCHEDULER)
        is_turbo, source = resolve_is_turbo(
            pipe=pipe, model_id_or_path="stabilityai/stable-diffusion-xl-base-1.0", loaded_via_single_file=False
        )
        assert (is_turbo, source) == (False, "scheduler")

    def test_single_file_ignores_scheduler_even_if_present(self, tmp_path):
        """The core gap: a from_single_file load's pipe carries a scheduler (the
        *reference repo's*, not the checkpoint's own), but it must never be
        trusted -- only metadata/filename corroborate single-file loads."""
        path = self._checkpoint(tmp_path, {"modelspec.architecture": "stable-diffusion-xl-turbo-v1"})
        pipe = _fake_pipe(*SDXL_BASE_SCHEDULER)  # would say False if trusted
        is_turbo, source = resolve_is_turbo(pipe=pipe, model_id_or_path=path, loaded_via_single_file=True)
        assert (is_turbo, source) == (True, "modelspec")

    def test_single_file_community_sdxl_turbo_merge(self, tmp_path):
        """community SDXL-Turbo merge `.safetensors`: no modelspec metadata, name
        says turbo."""
        path = str(tmp_path / "SDXL-Turbo-merge.safetensors")
        save_file({"weight": torch.zeros(2, 2)}, path)
        is_turbo, source = resolve_is_turbo(model_id_or_path=path, loaded_via_single_file=True)
        assert (is_turbo, source) == (True, "filename")

    def test_single_file_community_sd_turbo_merge(self, tmp_path):
        """community sd-turbo merge `.safetensors`: same shape, non-SDXL name."""
        path = str(tmp_path / "sd-turbo-merge.safetensors")
        save_file({"weight": torch.zeros(2, 2)}, path)
        is_turbo, source = resolve_is_turbo(model_id_or_path=path, loaded_via_single_file=True)
        assert (is_turbo, source) == (True, "filename")

    def test_single_file_metadata_vetoes_misleading_filename(self, tmp_path):
        """modelspec.architecture outranks the filename hint: a confirmed-base
        checkpoint stays False even if someone names the file "turbo"."""
        path = str(tmp_path / "totally_turbo.safetensors")
        save_file(
            {"weight": torch.zeros(2, 2)}, path, metadata={"modelspec.architecture": "stable-diffusion-xl-v1-base"}
        )
        is_turbo, source = resolve_is_turbo(model_id_or_path=path, loaded_via_single_file=True)
        assert (is_turbo, source) == (False, "modelspec")

    def test_guard_base_dir_under_turbo_tests(self, tmp_path):
        """guard: base checkpoint's directory contains "turbo" but its own
        basename doesn't -- must not be misclassified."""
        turbo_dir = tmp_path / "turbo_tests"
        turbo_dir.mkdir()
        path = str(turbo_dir / "sdxl_base.safetensors")
        save_file({"weight": torch.zeros(2, 2)}, path)
        is_turbo, source = resolve_is_turbo(model_id_or_path=path, loaded_via_single_file=True)
        assert (is_turbo, source) == (False, "undecided")

    def test_guard_base_safetensors_in_turbo_dir_with_metadata_veto(self, tmp_path):
        """guard: base checkpoint under a turbo-named directory, this time with an
        explicit base modelspec tag -- still False."""
        turbo_dir = tmp_path / "turbo_tests"
        turbo_dir.mkdir()
        path = str(turbo_dir / "sdxl_base.safetensors")
        save_file(
            {"weight": torch.zeros(2, 2)}, path, metadata={"modelspec.architecture": "stable-diffusion-xl-v1-base"}
        )
        is_turbo, source = resolve_is_turbo(model_id_or_path=path, loaded_via_single_file=True)
        assert (is_turbo, source) == (False, "modelspec")

    def test_single_file_nothing_decisive_is_undecided(self, tmp_path):
        """sdxl_merged.safetensors case: no metadata, no filename hint -- False,
        tagged "undecided" so the fp8 build path can warn."""
        path = str(tmp_path / "sdxl_merged.safetensors")
        save_file({"weight": torch.zeros(2, 2)}, path)
        is_turbo, source = resolve_is_turbo(model_id_or_path=path, loaded_via_single_file=True)
        assert (is_turbo, source) == (False, "undecided")

    def test_no_pipe_no_path_repo_id_load_is_undecided(self):
        """Defensive: a from_pretrained-flagged load with no pipe at all (pipe=None)
        has no scheduler to consult and falls straight to undecided."""
        is_turbo, source = resolve_is_turbo(pipe=None, model_id_or_path=None, loaded_via_single_file=False)
        assert (is_turbo, source) == (False, "undecided")
