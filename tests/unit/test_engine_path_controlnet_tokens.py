"""
Regression tests for fp8-round-12's ControlNet cache-key fix.

``EngineManager.get_engine_path``'s ControlNet branch (acceleration/tensorrt/
engine_manager.py, ``EngineType.CONTROLNET``) builds
``controlnet_{model_id}--min_batch-N--max_batch-N--res-HxW`` and returns immediately --
unlike the UNet branch there is no hash compaction, so this literal prefix IS the cache
key. Before this fix it omitted two tokens the UNet branch already has:

- ``--batch-{static_batch_size}``: ControlNet is always built with
  ``build_static_batch=True`` (``_get_default_controlnet_build_options``) at
  ``opt_batch_size`` baked to the caller's ``stream.trt_unet_batch_size`` (=
  ``len(t_index_list) * frame_buffer_size`` for the default cfg_type). That value never
  reached ``get_engine_path`` -- only the *capacity* range (min/max_batch_size, fixed
  wrapper constructor defaults, not config-driven) did. Two different t_index_list
  lengths therefore baked two different static batches into the SAME ``cnet.engine``
  file, and the second config to load failed ``set_input_shape`` on the first frame.
- ``--trt{version}--cc{sm}``: TRT engines are not portable across TRT versions or GPU
  compute capabilities. Every other engine type auto-invalidates on a TRT upgrade or GPU
  change; ControlNet silently did not.

This test constructs ``EngineManager`` via ``__new__`` (bypassing ``__init__``'s
compile-fn imports, which need TensorRT/onnx/polygraphy installed) so it only exercises
the pure-path logic in ``get_engine_path`` -- same pattern as test_engine_path_length.py
and test_engine_path_ipadapter_suffixes.py.

Run with: pytest tests/unit/test_engine_path_controlnet_tokens.py -v
"""

from pathlib import Path

import pytest

try:
    from streamdiffusion.acceleration.tensorrt.engine_manager import EngineManager, EngineType

    IMPORT_OK = True
except ImportError:
    IMPORT_OK = False

pytestmark = pytest.mark.skipif(
    not IMPORT_OK,
    reason="acceleration.tensorrt.engine_manager not importable",
)

_ENGINE_DIR = r"C:\Users\deswh\Documents\sdtd040\StreamDiffusion\engines\td"

_FILENAMES = {
    EngineType.UNET: "unet.engine",
    EngineType.CONTROLNET: "cnet.engine",
}

_BASE_KWARGS = {
    "engine_type": EngineType.CONTROLNET,
    "model_id_or_path": "",  # not used for ControlNet
    "max_batch_size": 4,
    "min_batch_size": 1,
    "mode": "",  # not used for ControlNet
    "use_tiny_vae": False,  # not used for ControlNet
    "controlnet_model_id": "lllyasviel/sd-controlnet-canny",
    "resolution": (512, 512),
    "build_static_batch": True,
    "static_batch_size": 2,
    "fp8": False,
}

# Minimal viable UNET-branch call, reused only by TestTrtCcTagSharedAcrossBranches to
# prove both branches are driven by the same EngineManager._trt_cc_tag() method.
_UNET_BASE_KWARGS = {
    "engine_type": EngineType.UNET,
    "model_id_or_path": "stabilityai/sdxl-turbo",
    "max_batch_size": 4,
    "min_batch_size": 1,
    "mode": "img2img",
    "use_tiny_vae": True,
}


def _make_engine_manager(engine_dir: str = _ENGINE_DIR) -> EngineManager:
    """Build an EngineManager without running __init__'s heavy compile-fn imports."""
    em = EngineManager.__new__(EngineManager)
    em.engine_dir = Path(engine_dir)
    em._configs = {etype: {"filename": fname} for etype, fname in _FILENAMES.items()}
    return em


class TestControlNetBatchSizeCacheKey:
    """The core fp8-round-12 fix. Direct analog of
    test_engine_path_length.py::TestUnetEnginePathLength.test_distinct_configs_do_not_collide
    (same static_batch_size variable, same assertion shape) applied to the ControlNet
    branch. RED before fp8-round-12 (both paths collided into the same directory),
    GREEN after."""

    def test_distinct_static_batch_sizes_do_not_collide(self):
        em = _make_engine_manager()
        path_a = em.get_engine_path(**dict(_BASE_KWARGS, static_batch_size=1))
        path_b = em.get_engine_path(**dict(_BASE_KWARGS, static_batch_size=2))

        assert path_a != path_b
        assert path_a.parent != path_b.parent

    def test_sbatch_flag_present_when_static(self):
        em = _make_engine_manager()
        path = em.get_engine_path(**_BASE_KWARGS)
        assert "--sbatch1" in path.parent.name
        assert "--batch-2" in path.parent.name

    def test_batch_size_omitted_when_not_static_batch(self):
        """build_static_batch=False (a hypothetical future dynamic-batch ControlNet)
        must not bake a batch size into the path -- mirrors the UNet branch's guard at
        engine_manager.py:407-408 (``if build_static_batch and static_batch_size is
        not None``)."""
        em = _make_engine_manager()
        path = em.get_engine_path(**dict(_BASE_KWARGS, build_static_batch=False, static_batch_size=2))
        assert "--batch-2" not in path.parent.name
        assert "--sbatch0" in path.parent.name

    def test_engine_path_is_deterministic(self):
        """Same config -> same path, so a previously-built engine is still found on
        rebuild (ControlNet has no hash compaction, so this also guards against any
        accidental non-determinism creeping into the new token ordering)."""
        em = _make_engine_manager()
        path_a = em.get_engine_path(**_BASE_KWARGS)
        path_b = em.get_engine_path(**_BASE_KWARGS)
        assert path_a == path_b


class TestTrtCcTagSharedAcrossBranches:
    """Both the ControlNet and standard branches call the single
    EngineManager._trt_cc_tag() helper (extracted this round for the same anti-drift
    reason as the existing _fp8_recipe_tag extraction -- two hand-written copies of a
    cache-key token is how the fp8-round-8 prefix-vs-hash drift bug happened).
    Monkeypatched so this test needs no CUDA device."""

    def test_trt_cc_tag_present_in_controlnet_path(self, monkeypatch):
        em = _make_engine_manager()
        monkeypatch.setattr(EngineManager, "_trt_cc_tag", lambda self: "--trtTEST--cc89")
        path = em.get_engine_path(**_BASE_KWARGS)
        assert "--trtTEST--cc89" in path.parent.name

    def test_trt_cc_tag_forks_both_branches(self, monkeypatch):
        """Changing the tag must fork BOTH branches' output: literally, for
        ControlNet (unhashed prefix); via the hash, for UNet (whose directory name
        compacts the full prefix to a sha1 short hash, so the raw tag text is not
        expected to appear verbatim there -- only the fact that the path forks is)."""
        em = _make_engine_manager()

        monkeypatch.setattr(EngineManager, "_trt_cc_tag", lambda self: "--trtAAA--cc86")
        cn_a = em.get_engine_path(**_BASE_KWARGS)
        unet_a = em.get_engine_path(**_UNET_BASE_KWARGS)

        monkeypatch.setattr(EngineManager, "_trt_cc_tag", lambda self: "--trtBBB--cc90")
        cn_b = em.get_engine_path(**_BASE_KWARGS)
        unet_b = em.get_engine_path(**_UNET_BASE_KWARGS)

        assert cn_a != cn_b, "ControlNet path must fork when the TRT/CC tag changes"
        assert unet_a != unet_b, "UNet path must fork when the TRT/CC tag changes (via the hash)"
        assert "--trtAAA--cc86" in cn_a.parent.name
        assert "--trtBBB--cc90" in cn_b.parent.name

    def test_trt_cc_tag_absent_without_cuda_does_not_raise(self):
        """The real _trt_cc_tag() silently fails to "" if tensorrt/torch aren't
        importable or no CUDA device is present -- must not raise, matching the
        inline block it replaced."""
        em = _make_engine_manager()
        path = em.get_engine_path(**_BASE_KWARGS)
        assert path.parent.name  # constructed without raising


class TestControlNetPathExistingDiscrimination:
    """Coverage that did not exist before fp8-round-12 -- pins the behavior the fix
    must not regress while touching this branch."""

    def test_path_changes_with_model_id(self):
        em = _make_engine_manager()
        path_a = em.get_engine_path(**_BASE_KWARGS)
        path_b = em.get_engine_path(**dict(_BASE_KWARGS, controlnet_model_id="lllyasviel/sd-controlnet-openpose"))
        assert path_a != path_b

    def test_path_changes_with_resolution(self):
        em = _make_engine_manager()
        path_a = em.get_engine_path(**_BASE_KWARGS)
        path_b = em.get_engine_path(**dict(_BASE_KWARGS, resolution=(1024, 1024)))
        assert path_a != path_b

    def test_path_changes_with_fp8(self):
        em = _make_engine_manager()
        path_a = em.get_engine_path(**_BASE_KWARGS)
        path_b = em.get_engine_path(**dict(_BASE_KWARGS, fp8=True))
        assert path_a != path_b

    def test_model_id_slash_is_sanitized_to_underscore(self):
        em = _make_engine_manager()
        path = em.get_engine_path(**_BASE_KWARGS)
        assert "/" not in path.parent.name
