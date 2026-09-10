"""
Regression test for the UNet TensorRT engine path exceeding Windows MAX_PATH (260).

``EngineManager.get_engine_path`` (acceleration/tensorrt/engine_manager.py) encodes every
UNet build flag into the on-disk directory name. With a realistic ``engine_dir`` and a
config using fp8 + static batch + pin_cache_frames + optlvl + resolution, the generated
directory was 246 chars; the derived ``unet.engine.onnx`` path was 263 and
``unet.engine.opt.onnx`` was 267 — both over Windows' 260-char MAX_PATH. Because the
*directory* fit but the *file* didn't, ``mkdir`` silently succeeded while
``torch.onnx.export`` -> ``open(onnx_path, "wb")`` raised ``FileNotFoundError``, which
wrapper.py's OOM-only fallback does not catch — see
"Acceleration has failed: [Errno 2] No such file or directory: ...unet.engine.onnx".

This test constructs ``EngineManager`` via ``__new__`` (bypassing ``__init__``'s compile-fn
imports, which need TensorRT/onnx/polygraphy installed) so it only exercises the pure-path
logic in ``get_engine_path``.

Run with: pytest tests/unit/test_engine_path_length.py -v
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

# Mirrors the crash report: a nested Documents path, matching the real-world depth
# that pushed the UNet engine path over MAX_PATH.
_CRASH_ENGINE_DIR = r"C:\Users\deswh\Documents\sdtd040\StreamDiffusion\engines\td"

# The exact flag combination from the crash log's UNet directory name.
_CRASH_UNET_KWARGS = {
    "engine_type": EngineType.UNET,
    "model_id_or_path": "stabilityai/sd-turbo",
    "max_batch_size": 4,
    "min_batch_size": 1,
    "mode": "img2img",
    "use_tiny_vae": True,
    "fp8": True,
    "use_cached_attn": False,
    "use_feature_injection": False,
    "build_static_batch": True,
    "static_batch_size": 2,
    "pin_cache_frames": True,
    "cache_maxframes": 4,
    "builder_optimization_level": 4,
    "resolution": (512, 512),
}


_CRASH_VAE_KWARGS = {
    "engine_type": EngineType.VAE_ENCODER,
    "model_id_or_path": "stabilityai/sd-turbo",
    "max_batch_size": 4,
    "min_batch_size": 1,
    "mode": "img2img",
    "use_tiny_vae": True,
    "builder_optimization_level": 4,
    "resolution": (512, 512),
}


def _make_engine_manager(engine_dir: str) -> EngineManager:
    """Build an EngineManager without running __init__'s heavy compile-fn imports."""
    em = EngineManager.__new__(EngineManager)
    em.engine_dir = Path(engine_dir)
    em._configs = {
        EngineType.UNET: {"filename": "unet.engine"},
        EngineType.VAE_ENCODER: {"filename": "vae_encoder.engine"},
        EngineType.VAE_DECODER: {"filename": "vae_decoder.engine"},
    }
    return em


class TestUnetEnginePathLength:
    def test_onnx_export_paths_stay_under_max_path(self):
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        engine_path = em.get_engine_path(**_CRASH_UNET_KWARGS)

        onnx_path = str(engine_path) + ".onnx"
        opt_onnx_path = str(engine_path) + ".opt.onnx"

        assert len(onnx_path) < 260, f"onnx path is {len(onnx_path)} chars (MAX_PATH=260): {onnx_path}"
        assert len(opt_onnx_path) < 260, f"opt.onnx path is {len(opt_onnx_path)} chars (MAX_PATH=260): {opt_onnx_path}"

    def test_engine_path_is_deterministic(self):
        """Same config -> same path, so a previously-built engine is still found on rebuild."""
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        path_a = em.get_engine_path(**_CRASH_UNET_KWARGS)
        path_b = em.get_engine_path(**_CRASH_UNET_KWARGS)

        assert path_a == path_b

    def test_distinct_configs_do_not_collide(self):
        """A differing flag (static_batch_size) must still produce a distinct directory."""
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        kwargs_b = dict(_CRASH_UNET_KWARGS, static_batch_size=4)

        path_a = em.get_engine_path(**_CRASH_UNET_KWARGS)
        path_b = em.get_engine_path(**kwargs_b)

        assert path_a != path_b
        assert path_a.parent != path_b.parent


class TestFp8RecipeTagV4:
    """fp8-round-8: the fp8 tag bumped --fp8v3 -> --fp8v4 (the scale-conversion math
    changed — see fp8_quantize.py::_rescale_fp8_qdq_scales — so every existing v3
    engine is stale even with every recipe flag at its default) and gained new
    components for fp8_scale_headroom / fp8_exclude_attention. This tag is built by
    EngineManager._fp8_recipe_tag and used verbatim (not hashed) in the final UNet
    directory name at both engine_manager.py:~197 (feeds the canonical/hash string)
    and :~287 (the literal on-disk suffix) — the two call sites must stay in sync.
    """

    def test_default_recipe_uses_v4_base_tag(self):
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        path = em.get_engine_path(**dict(_CRASH_UNET_KWARGS, fp8_mha_qdq=False))
        assert "fp8v3" not in path.parent.name
        assert "fp8v4" in path.parent.name

    def test_all_new_recipe_flags_stay_under_max_path(self):
        """Worst case for the length regression this file guards: every fp8-round-8
        recipe flag active at once, on top of the already-tight crash config."""
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        kwargs = dict(
            _CRASH_UNET_KWARGS,
            fp8_mha_qdq=True,
            fp8_exclude_attention=True,
            fp8_scale_headroom=2.0,
        )
        engine_path = em.get_engine_path(**kwargs)
        onnx_path = str(engine_path) + ".onnx"
        opt_onnx_path = str(engine_path) + ".opt.onnx"

        assert len(onnx_path) < 260, f"onnx path is {len(onnx_path)} chars (MAX_PATH=260): {onnx_path}"
        assert len(opt_onnx_path) < 260, f"opt.onnx path is {len(opt_onnx_path)} chars (MAX_PATH=260): {opt_onnx_path}"

    def test_recipe_flags_fork_the_path_independently(self):
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        base = em.get_engine_path(**_CRASH_UNET_KWARGS)
        mhaq = em.get_engine_path(**dict(_CRASH_UNET_KWARGS, fp8_mha_qdq=True))
        noattn = em.get_engine_path(**dict(_CRASH_UNET_KWARGS, fp8_exclude_attention=True))
        hr2 = em.get_engine_path(**dict(_CRASH_UNET_KWARGS, fp8_scale_headroom=2.0))

        assert len({base, mhaq, noattn, hr2}) == 4, "each recipe flag must fork a distinct engine directory"

    def test_recipe_flags_compose(self):
        """All three flags together must differ from any single flag alone — guards
        against one flag's suffix silently overwriting another's in string concat."""
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        mhaq_only = em.get_engine_path(**dict(_CRASH_UNET_KWARGS, fp8_mha_qdq=True))
        all_three = em.get_engine_path(
            **dict(_CRASH_UNET_KWARGS, fp8_mha_qdq=True, fp8_exclude_attention=True, fp8_scale_headroom=2.0)
        )
        assert mhaq_only != all_three

    def test_recipe_tag_is_deterministic(self):
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        kwargs = dict(_CRASH_UNET_KWARGS, fp8_mha_qdq=True, fp8_exclude_attention=True, fp8_scale_headroom=2.0)

        path_a = em.get_engine_path(**kwargs)
        path_b = em.get_engine_path(**kwargs)

        assert path_a == path_b


class TestVaeEnginePathIdentity:
    """Custom-VAE fix, Step 5: two different `vae_id`s must never collide on one
    cached VAE_ENCODER/VAE_DECODER directory, existing users must see no rebuild,
    and — the one that would be invisible until users reported hour-long rebuilds
    after upgrading — the UNet directory must be byte-identical whether or not a
    `vae_id` is passed at all, since the token is scoped to VAE engine types only.
    """

    def test_distinct_vae_ids_produce_distinct_paths(self):
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        path_a = em.get_engine_path(**_CRASH_VAE_KWARGS, vae_id="stabilityai/sd-vae-ft-mse")
        path_b = em.get_engine_path(**_CRASH_VAE_KWARGS, vae_id="madebyollin/sdxl-vae-fp16-fix")

        assert path_a != path_b
        assert path_a.parent != path_b.parent

    def test_no_vae_id_reproduces_the_path_from_before_this_parameter_existed(self):
        """vae_id defaults to None, so an explicit vae_id=None call and a call that
        omits the kwarg entirely (how every pre-existing caller invokes this) must
        land on the identical path - no engine reuse breaks for existing users."""
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        path_omitted = em.get_engine_path(**_CRASH_VAE_KWARGS)
        path_explicit_none = em.get_engine_path(**_CRASH_VAE_KWARGS, vae_id=None)

        assert path_omitted == path_explicit_none
        assert "--vae-" not in path_omitted.parent.name

    def test_vae_id_token_present_only_when_given(self):
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        path = em.get_engine_path(**_CRASH_VAE_KWARGS, vae_id="stabilityai/sd-vae-ft-mse")

        assert "--vae-" in path.parent.name

    def test_unet_path_is_identical_with_and_without_vae_id(self):
        """The no-mass-rebuild guard: vae_id must not perturb the UNet cache key at
        all, even though get_engine_path accepts it for every engine type."""
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        path_without = em.get_engine_path(**_CRASH_UNET_KWARGS)
        path_with = em.get_engine_path(**_CRASH_UNET_KWARGS, vae_id="stabilityai/sd-vae-ft-mse")

        assert path_without == path_with

    def test_vae_engine_paths_with_vae_id_stay_under_max_path(self):
        """Mirrors test_all_new_recipe_flags_stay_under_max_path, but for the VAE
        branch, which has no hash compaction to fall back on (Step 5's constraint 2:
        only EngineType.UNET gets the short-hash treatment) - the fixed-width hashed
        --vae- token is exactly what keeps this bounded regardless of vae_id length."""
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        long_vae_id = "some-org/a-fairly-long-custom-vae-repository-name-for-margin-testing"

        for engine_type in (EngineType.VAE_ENCODER, EngineType.VAE_DECODER):
            kwargs = dict(_CRASH_VAE_KWARGS, engine_type=engine_type)
            engine_path = em.get_engine_path(**kwargs, vae_id=long_vae_id)
            onnx_path = str(engine_path) + ".onnx"
            opt_onnx_path = str(engine_path) + ".opt.onnx"

            assert len(onnx_path) < 260, f"onnx path is {len(onnx_path)} chars (MAX_PATH=260): {onnx_path}"
            assert len(opt_onnx_path) < 260, (
                f"opt.onnx path is {len(opt_onnx_path)} chars (MAX_PATH=260): {opt_onnx_path}"
            )

    def test_vae_id_path_is_deterministic(self):
        em = _make_engine_manager(_CRASH_ENGINE_DIR)
        path_a = em.get_engine_path(**_CRASH_VAE_KWARGS, vae_id="stabilityai/sd-vae-ft-mse")
        path_b = em.get_engine_path(**_CRASH_VAE_KWARGS, vae_id="stabilityai/sd-vae-ft-mse")

        assert path_a == path_b
