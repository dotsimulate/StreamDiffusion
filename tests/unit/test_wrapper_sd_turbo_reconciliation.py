"""
Regression tests for the missing `sd_turbo` reconciliation in
`StreamDiffusionWrapper._load_model`.

Three independent Turbo detectors existed in this repo: `wrapper.sd_turbo` (a
crude heuristic, the only one actually read by `txt2img`/the LCM-LoRA gate),
`wrapper._is_turbo` (the authoritative verdict from `resolve_is_turbo`'s 5-level
precedence ladder, write-only before this fix), and `detect_model`'s own
scheduler-only signal (defaulted to a near-constant `False` for the four call
sites that pass no pipe, and unread by its only consumers). `_load_model`
documented a contract -- "resolves the authoritative verdict ... and reconciles
self.sd_turbo below" -- that didn't exist in code: `self._is_turbo` was assigned
exactly once and never read again.

Every existing `resolve_is_turbo` test (test_model_detection.py) calls it
directly, bypassing the wrapper entirely -- a green suite there coexisted with a
wrong runtime value here. These tests exercise the reconciliation through the
actual wrapper call path instead.

Follows the object.__new__ shell + patch.object(wrapper_module.*, ...) pattern
established in test_load_model_network_error_reporting.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from safetensors.torch import save_file

from streamdiffusion import wrapper as wrapper_module
from streamdiffusion.wrapper import StreamDiffusionWrapper


def _make_wrapper_shell(model_id_or_path: str) -> StreamDiffusionWrapper:
    """Construct a minimal StreamDiffusionWrapper without model loading, with
    __init__'s Step-2 pre-load state (no explicit override, basename-only
    turbo_hint_from_model_id guess) and just enough attributes to satisfy the
    StreamDiffusion(...) kwargs evaluated at the end of _load_model (that call
    itself is patched to raise -- see below -- but Python still evaluates its
    kwargs first).

    use_lcm_lora mirrors __init__'s `self.use_lcm_lora = use_lcm_lora` (default
    param value None -- see wrapper.py:355) rather than _load_model's separate,
    unused-in-the-gate `use_lcm_lora: bool = True` parameter (wrapper.py:1834);
    the gate at wrapper.py:2174 reads only the former."""
    w = object.__new__(StreamDiffusionWrapper)
    w.device = torch.device("cpu")
    w.dtype = torch.float32
    w.cleanup_gpu_memory = lambda: None
    w.width = 512
    w.height = 512
    w.frame_buffer_size = 1
    w.use_denoising_batch = True
    w._is_turbo_override = None
    w.sd_turbo = wrapper_module.turbo_hint_from_model_id(model_id_or_path)
    w.use_lcm_lora = None
    return w


def _fake_unet() -> MagicMock:
    """An SDXL-shaped UNet2DConditionModel mock -- the config fields
    detect_model reads. Mock(spec=...) makes isinstance() checks pass; FrozenDict
    supports both dict-style and attribute-style config access, matching real
    diffusers configs."""
    unet = MagicMock(spec=UNet2DConditionModel)
    unet.config = FrozenDict(
        {
            "addition_embed_type": "text_time",
            "cross_attention_dim": 2048,
            "time_cond_proj_dim": None,
        }
    )
    return unet


def _fake_sdxl_pipe(unet: MagicMock) -> MagicMock:
    """A minimal pipe stand-in: enough for detect_model's isinstance/config
    checks and for _load_model's post-load device-placement hasattr checks (all
    no-ops here since acceleration defaults to "tensorrt")."""
    pipe = MagicMock()
    pipe.to.return_value = pipe
    pipe.unet = unet
    # loaded_via_single_file=True means resolve_is_turbo never trusts this
    # scheduler anyway (see its docstring); explicitly None so
    # _detect_turbo_from_scheduler -- still computed for detect_model's now-
    # decoupled turbo_from_scheduler field -- short-circuits cleanly instead of
    # chasing an unconfigured MagicMock.
    pipe.scheduler = None
    return pipe


class TestSdTurboReconciliation:
    """Step 1 regression seam: wrapper.sd_turbo must adopt resolve_is_turbo's
    post-load verdict, not keep __init__'s pre-load basename guess.

    Fixture: a checkpoint whose filename has no "turbo" substring (pre-load
    guess -> False, via turbo_hint_from_model_id) but whose embedded
    modelspec.architecture metadata declares it Turbo (post-load verdict, via
    resolve_is_turbo -> True, source="modelspec"). Pre- and post-load values
    genuinely disagree, so this is the input shape that actually exercises
    Step 1's `if self.sd_turbo != is_turbo: ... self.sd_turbo = is_turbo`
    adoption -- unlike the turbo_tests/ parent-directory fixture (see
    TestSdTurboReconciliationGuardCase below), where Step 2's basename fix alone
    already makes pre-load and post-load values agree and the adoption branch
    never fires.

    Aborts right after adoption (patching wrapper_module.StreamDiffusion to
    raise) so the test stays cheap -- no real weights, no engine build.
    """

    def test_sd_turbo_flips_false_to_true_on_metadata_veto(self, tmp_path):
        path = str(tmp_path / "model.safetensors")
        save_file(
            {"weight": torch.zeros(2, 2)},
            path,
            metadata={"modelspec.architecture": "stable-diffusion-xl-turbo-v1"},
        )

        w = _make_wrapper_shell(path)
        assert w.sd_turbo is False, "sanity: pre-load basename guess must be False for this fixture"

        pipe = _fake_sdxl_pipe(_fake_unet())
        captured_kwargs = {}

        def fake_auto_from_pretrained(*args, **kwargs):
            raise RuntimeError("not a from_pretrained-loadable path")

        def fake_sd_from_single_file(*args, **kwargs):
            return pipe

        def fake_stream_diffusion(*args, **kwargs):
            captured_kwargs.update(kwargs)
            raise RuntimeError("aborted after reconciliation -- test only needs wrapper state")

        with (
            patch.object(
                wrapper_module.AutoPipelineForText2Image,
                "from_pretrained",
                staticmethod(fake_auto_from_pretrained),
            ),
            patch.object(
                wrapper_module.StableDiffusionPipeline,
                "from_single_file",
                staticmethod(fake_sd_from_single_file),
            ),
            patch.object(wrapper_module, "StreamDiffusion", fake_stream_diffusion),
        ):
            with pytest.raises(RuntimeError, match="aborted after reconciliation"):
                w._load_model(path, t_index_list=[0])

        assert w.sd_turbo is True
        assert w._is_turbo is True
        assert w._is_turbo_source == "modelspec"
        assert w.sd_turbo == w._is_turbo
        # Step 4: the resolved verdict must also be threaded into StreamDiffusion(...),
        # not just adopted onto self.sd_turbo.
        assert captured_kwargs["is_turbo"] is True


class TestSdTurboReconciliationGuardCase:
    """The plan's originally-cited fixture shape (base checkpoint staged under a
    `turbo_tests/` parent directory, basename itself has no "turbo") -- the
    guard case Step 2's basename-only fix protects. Doesn't exercise the Step 1
    adoption branch (pre-load and post-load values already agree once Step 2 is
    in place -- see TestSdTurboReconciliation above for that), but this is the
    first end-to-end check of that guard through the actual wrapper call path
    rather than a direct resolve_is_turbo() call.
    """

    def test_parent_dir_named_turbo_does_not_misclassify_wrapper_state(self, tmp_path):
        turbo_dir = tmp_path / "turbo_tests"
        turbo_dir.mkdir()
        path = str(turbo_dir / "base_checkpoint.safetensors")
        save_file({"weight": torch.zeros(2, 2)}, path)

        w = _make_wrapper_shell(path)
        assert w.sd_turbo is False, "sanity: pre-load basename guess must ignore the parent dir"

        pipe = _fake_sdxl_pipe(_fake_unet())

        def fake_auto_from_pretrained(*args, **kwargs):
            raise RuntimeError("not a from_pretrained-loadable path")

        def fake_sd_from_single_file(*args, **kwargs):
            return pipe

        def fake_stream_diffusion(*args, **kwargs):
            raise RuntimeError("aborted after reconciliation -- test only needs wrapper state")

        with (
            patch.object(
                wrapper_module.AutoPipelineForText2Image,
                "from_pretrained",
                staticmethod(fake_auto_from_pretrained),
            ),
            patch.object(
                wrapper_module.StableDiffusionPipeline,
                "from_single_file",
                staticmethod(fake_sd_from_single_file),
            ),
            patch.object(wrapper_module, "StreamDiffusion", fake_stream_diffusion),
        ):
            with pytest.raises(RuntimeError, match="aborted after reconciliation"):
                w._load_model(path, t_index_list=[0])

        assert w.sd_turbo is False
        assert w._is_turbo is False
        assert w._is_turbo_source == "undecided"
        assert w.sd_turbo == w._is_turbo


class TestLcmLoraGateFollowsResolvedVerdict:
    """`lora_dict` is the only route by which the Turbo verdict could ever
    alter a TensorRT engine's cache key (EngineManager._lora_signature,
    engine_manager.py:83-107) -- get_engine_path itself carries no turbo
    parameter of any kind. This pins that the deprecated LCM-LoRA gate at
    wrapper.py:2174-2195 keys off self.sd_turbo *after* Step 1's adoption
    (wrapper.py:2156-2161), not the pre-load basename guess -- i.e. that the
    unification actually reaches this consumer, not just the log line.

    Reuses the two fixtures above: the modelspec-veto checkpoint (Turbo=True)
    and the turbo_tests/ guard-case checkpoint (Turbo=False), both loaded
    through an SDXL-shaped pipe so `is_sdxl=True` selects lcm-lora-sdxl.
    """

    def test_turbo_verdict_skips_lcm_lora(self, tmp_path):
        path = str(tmp_path / "model.safetensors")
        save_file(
            {"weight": torch.zeros(2, 2)},
            path,
            metadata={"modelspec.architecture": "stable-diffusion-xl-turbo-v1"},
        )

        w = _make_wrapper_shell(path)
        w.use_lcm_lora = True

        pipe = _fake_sdxl_pipe(_fake_unet())
        captured_kwargs = {}

        def fake_auto_from_pretrained(*args, **kwargs):
            raise RuntimeError("not a from_pretrained-loadable path")

        def fake_sd_from_single_file(*args, **kwargs):
            return pipe

        def fake_stream_diffusion(*args, **kwargs):
            captured_kwargs.update(kwargs)
            raise RuntimeError("aborted after reconciliation -- test only needs wrapper state")

        with (
            patch.object(
                wrapper_module.AutoPipelineForText2Image,
                "from_pretrained",
                staticmethod(fake_auto_from_pretrained),
            ),
            patch.object(
                wrapper_module.StableDiffusionPipeline,
                "from_single_file",
                staticmethod(fake_sd_from_single_file),
            ),
            patch.object(wrapper_module, "StreamDiffusion", fake_stream_diffusion),
        ):
            with pytest.raises(RuntimeError, match="aborted after reconciliation"):
                w._load_model(path, t_index_list=[0])

        assert w.sd_turbo is True
        assert captured_kwargs["lora_dict"] is None
        # Turbo branch takes the `else:` arm (wrapper.py:2188-2195), which
        # clears the deprecated flag -- unlike the non-Turbo case below.
        assert w.use_lcm_lora is None

    def test_non_turbo_verdict_adds_lcm_lora_at_scale_one(self, tmp_path):
        turbo_dir = tmp_path / "turbo_tests"
        turbo_dir.mkdir()
        path = str(turbo_dir / "base_checkpoint.safetensors")
        save_file({"weight": torch.zeros(2, 2)}, path)

        w = _make_wrapper_shell(path)
        w.use_lcm_lora = True

        pipe = _fake_sdxl_pipe(_fake_unet())
        captured_kwargs = {}

        def fake_auto_from_pretrained(*args, **kwargs):
            raise RuntimeError("not a from_pretrained-loadable path")

        def fake_sd_from_single_file(*args, **kwargs):
            return pipe

        def fake_stream_diffusion(*args, **kwargs):
            captured_kwargs.update(kwargs)
            raise RuntimeError("aborted after reconciliation -- test only needs wrapper state")

        with (
            patch.object(
                wrapper_module.AutoPipelineForText2Image,
                "from_pretrained",
                staticmethod(fake_auto_from_pretrained),
            ),
            patch.object(
                wrapper_module.StableDiffusionPipeline,
                "from_single_file",
                staticmethod(fake_sd_from_single_file),
            ),
            patch.object(wrapper_module, "StreamDiffusion", fake_stream_diffusion),
        ):
            with pytest.raises(RuntimeError, match="aborted after reconciliation"):
                w._load_model(path, t_index_list=[0])

        assert w.sd_turbo is False
        assert captured_kwargs["lora_dict"] == {"latent-consistency/lcm-lora-sdxl": 1.0}
        # Non-Turbo branch takes the `if:` arm (wrapper.py:2176-2187), which
        # never touches self.use_lcm_lora.
        assert w.use_lcm_lora is True


@pytest.mark.skip(
    reason=(
        "fp8_guidance_scale (wrapper.py:3016) is the resolved verdict's second "
        "behavioural consumer, found during MCP-graph re-verification of the "
        "detection-unification plan -- see the plan's test 5. Reaching that line "
        "needs _load_model driven past real StreamDiffusion construction, LoRA "
        "adapter activation, and EngineManager.get_engine_path resolution (fp8=True, "
        "acceleration='tensorrt') -- far past the abort-after-reconciliation seam "
        "every other test in this file uses (patch.object(wrapper_module, "
        "'StreamDiffusion', <raiser>) short-circuits before that code ever runs). "
        "Driving it for real would mean either mocking ~750 lines of intervening "
        "engine-manager/pipeline plumbing (its own source of false confidence) or "
        "adding a GPU-dependent integration test, which is out of scope for the "
        "unit-tests-only, no-GPU constraint this plan was scoped to. Per the plan's "
        "documented fallback: skip with a pointer at the coupling (added at "
        "wrapper.py:3016) rather than a shallow/misleading unit test."
    )
)
def test_fp8_guidance_scale_follows_resolved_verdict():
    """Placeholder documenting the coupling and why it isn't unit-tested here:
    `_unet_build_opts["fp8_guidance_scale"] = 0.0 if _is_turbo else 7.5`
    (wrapper.py:3016) should be 0.0 for a Turbo verdict and 7.5 otherwise --
    i.e. FP8 calibration keys off resolve_is_turbo()'s answer, not a stale
    heuristic. Not reachable from the wrapper's live config anyway
    (StreamDiffusionTD/td_config.yaml has fp8: false), so this branch is dead
    for the current deployment regardless.
    """
