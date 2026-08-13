"""
Regression test for IP-Adapter/FaceID scoping of the TensorRT engine cache key.

``EngineManager.get_engine_path`` (acceleration/tensorrt/engine_manager.py) only folds
``is_faceid`` / ``ipadapter_tokens`` into the cache key inside the
``if engine_type == EngineType.UNET:`` block (:160-178). VAE-encoder and VAE-decoder engines
are IP-Adapter-agnostic (they never see cross-attention weights), so those two params must be
accepted but ignored for ``EngineType.VAE_ENCODER`` / ``EngineType.VAE_DECODER`` — otherwise
toggling FaceID or changing the token count would force a redundant VAE rebuild.

This is also the B3 regression test: bumping the marker from ``--fid`` to ``--fid2`` (a plain
generation bump, done because B2's LoRA fusion changes the exported UNet graph and would
otherwise let a stale ``--fid`` engine be silently reused) must still fork the UNet cache key
whenever ``is_faceid`` flips, independent of the literal marker string. The UNet prefix is
hashed (``engine_manager.py:240-247``) before becoming a directory name, so this test compares
*paths*, not literal substrings — pinning ``--fid2`` itself would break the moment the marker
is bumped again.

Reuses the ``_make_engine_manager`` / GPU-less-import pattern from test_engine_path_length.py.

Run with: pytest tests/unit/test_engine_path_ipadapter_suffixes.py -v
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

_BASE_KWARGS = {
    "model_id_or_path": "stabilityai/sdxl-turbo",
    "max_batch_size": 4,
    "min_batch_size": 1,
    "mode": "img2img",
    "use_tiny_vae": True,
}

# Extra filenames beyond UNET (test_engine_path_length.py's _make_engine_manager only wires
# UNET), so VAE_ENCODER/VAE_DECODER paths can be built too.
_FILENAMES = {
    EngineType.UNET: "unet.engine",
    EngineType.VAE_ENCODER: "vae_encoder.engine",
    EngineType.VAE_DECODER: "vae_decoder.engine",
}


def _make_engine_manager(engine_dir: str = _ENGINE_DIR) -> EngineManager:
    """Build an EngineManager without running __init__'s heavy compile-fn imports."""
    em = EngineManager.__new__(EngineManager)
    em.engine_dir = Path(engine_dir)
    em._configs = {etype: {"filename": fname} for etype, fname in _FILENAMES.items()}
    return em


class TestUnetPathForksOnFaceIdAndTokens:
    def test_unet_path_changes_when_faceid_toggled(self):
        em = _make_engine_manager()
        path_regular = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=False, **_BASE_KWARGS)
        path_faceid = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=True, **_BASE_KWARGS)

        assert path_regular != path_faceid
        assert path_regular.parent != path_faceid.parent

    def test_unet_path_changes_with_ipadapter_tokens(self):
        em = _make_engine_manager()
        path_4 = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=True, ipadapter_tokens=4, **_BASE_KWARGS)
        path_77 = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=True, ipadapter_tokens=77, **_BASE_KWARGS)

        assert path_4 != path_77

    def test_unet_path_is_deterministic(self):
        """Same FaceID/token config -> same path, so a previously-built engine is reused."""
        em = _make_engine_manager()
        kwargs = dict(_BASE_KWARGS, engine_type=EngineType.UNET, is_faceid=True, ipadapter_tokens=4)

        path_a = em.get_engine_path(**kwargs)
        path_b = em.get_engine_path(**kwargs)

        assert path_a == path_b


class TestVaePathsIgnoreIpAdapterFlags:
    def test_vae_encoder_path_unaffected_by_faceid(self):
        em = _make_engine_manager()
        path_regular = em.get_engine_path(engine_type=EngineType.VAE_ENCODER, is_faceid=False, **_BASE_KWARGS)
        path_faceid = em.get_engine_path(engine_type=EngineType.VAE_ENCODER, is_faceid=True, **_BASE_KWARGS)

        assert path_regular == path_faceid

    def test_vae_decoder_path_unaffected_by_faceid(self):
        em = _make_engine_manager()
        path_regular = em.get_engine_path(engine_type=EngineType.VAE_DECODER, is_faceid=False, **_BASE_KWARGS)
        path_faceid = em.get_engine_path(engine_type=EngineType.VAE_DECODER, is_faceid=True, **_BASE_KWARGS)

        assert path_regular == path_faceid

    def test_vae_paths_unaffected_by_ipadapter_tokens(self):
        em = _make_engine_manager()
        path_4 = em.get_engine_path(
            engine_type=EngineType.VAE_DECODER, is_faceid=True, ipadapter_tokens=4, **_BASE_KWARGS
        )
        path_77 = em.get_engine_path(
            engine_type=EngineType.VAE_DECODER, is_faceid=True, ipadapter_tokens=77, **_BASE_KWARGS
        )

        assert path_4 == path_77

    def test_vae_encoder_and_decoder_paths_differ_from_each_other(self):
        """Sanity check that the two VAE engine types don't collide via a shared filename bug."""
        em = _make_engine_manager()
        path_enc = em.get_engine_path(engine_type=EngineType.VAE_ENCODER, is_faceid=True, **_BASE_KWARGS)
        path_dec = em.get_engine_path(engine_type=EngineType.VAE_DECODER, is_faceid=True, **_BASE_KWARGS)

        assert path_enc != path_dec


class TestFp8RecipeTagOrthogonalToIpAdapterSuffixes:
    """fp8-round-8: the fp8 recipe tag (fp8_mha_qdq / fp8_scale_headroom /
    fp8_exclude_attention) is appended to the same UNet ``prefix`` string as the
    FaceID/token suffixes covered above, inside the same
    ``if engine_type == EngineType.UNET:`` block in engine_manager.py. Guard that
    toggling one axis doesn't silently cancel or alias the other's fork.
    """

    def test_fp8_recipe_forks_independently_of_faceid(self):
        em = _make_engine_manager()
        base = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=True, fp8=True, **_BASE_KWARGS)
        mhaq = em.get_engine_path(
            engine_type=EngineType.UNET, is_faceid=True, fp8=True, fp8_mha_qdq=True, **_BASE_KWARGS
        )
        no_faceid = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=False, fp8=True, **_BASE_KWARGS)
        no_faceid_mhaq = em.get_engine_path(
            engine_type=EngineType.UNET, is_faceid=False, fp8=True, fp8_mha_qdq=True, **_BASE_KWARGS
        )

        assert base != mhaq
        assert no_faceid != no_faceid_mhaq
        assert base != no_faceid
        assert mhaq != no_faceid_mhaq

    def test_fp8_exclude_ipadapter_forks_independently_of_faceid(self):
        """fp8-round-9: the new -noip tag (fp8_exclude_ipadapter) must fork the UNet
        cache key the same way -noattn does, independent of whether FaceID/IP-Adapter
        is even enabled for this build."""
        em = _make_engine_manager()
        base = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=True, fp8=True, **_BASE_KWARGS)
        noip = em.get_engine_path(
            engine_type=EngineType.UNET, is_faceid=True, fp8=True, fp8_exclude_ipadapter=True, **_BASE_KWARGS
        )
        no_faceid = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=False, fp8=True, **_BASE_KWARGS)
        no_faceid_noip = em.get_engine_path(
            engine_type=EngineType.UNET, is_faceid=False, fp8=True, fp8_exclude_ipadapter=True, **_BASE_KWARGS
        )

        assert base != noip
        assert no_faceid != no_faceid_noip
        assert base != no_faceid
        assert noip != no_faceid_noip

    def test_fp8_exclude_ipadapter_forks_independently_of_exclude_attention(self):
        """-noip and -noattn are independent levers (Fix 3's plan explicitly notes they
        cover non-overlapping node sets) -- toggling one must not alias the other, and
        both together must differ from either alone."""
        em = _make_engine_manager()
        neither = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=True, fp8=True, **_BASE_KWARGS)
        noattn = em.get_engine_path(
            engine_type=EngineType.UNET, is_faceid=True, fp8=True, fp8_exclude_attention=True, **_BASE_KWARGS
        )
        noip = em.get_engine_path(
            engine_type=EngineType.UNET, is_faceid=True, fp8=True, fp8_exclude_ipadapter=True, **_BASE_KWARGS
        )
        both = em.get_engine_path(
            engine_type=EngineType.UNET,
            is_faceid=True,
            fp8=True,
            fp8_exclude_attention=True,
            fp8_exclude_ipadapter=True,
            **_BASE_KWARGS,
        )

        assert len({neither, noattn, noip, both}) == 4


class TestUnetPathLoraSuffix:
    """Workstream A / A5: get_engine_path's LoRA cache-key suffix (_lora_signature)
    deliberately EXCLUDES weight -- dragging the runtime `Weight` slider must reuse the
    cached engine, not rebuild it, which is the entire point of making LoRA weight a
    live tensor instead of something fused into the exported UNet graph. It must still
    fork on the file's actual identity though: several coexisting LoRA files can share a
    basename (e.g. multiple training-run checkpoints all saved as
    sepiagraph_woolsey.safetensors in different folders), so basename alone would
    silently load the wrong weights into a matching cache dir. size+mtime is the guard
    against that collision.
    """

    def test_unet_path_same_when_lora_weight_changes(self, tmp_path):
        lora_file = tmp_path / "style_a.safetensors"
        lora_file.write_bytes(b"fake-lora-weights")
        em = _make_engine_manager()
        path_low = em.get_engine_path(
            engine_type=EngineType.UNET, is_faceid=False, lora_dict={str(lora_file): 0.3}, **_BASE_KWARGS
        )
        path_high = em.get_engine_path(
            engine_type=EngineType.UNET, is_faceid=False, lora_dict={str(lora_file): 0.9}, **_BASE_KWARGS
        )

        assert path_low == path_high

    def test_unet_path_differs_when_lora_file_changes(self, tmp_path):
        file_a = tmp_path / "style_a.safetensors"
        file_b = tmp_path / "style_b.safetensors"
        file_a.write_bytes(b"fake-lora-weights-a")
        file_b.write_bytes(b"fake-lora-weights-b")
        em = _make_engine_manager()
        path_a = em.get_engine_path(
            engine_type=EngineType.UNET, is_faceid=False, lora_dict={str(file_a): 0.5}, **_BASE_KWARGS
        )
        path_b = em.get_engine_path(
            engine_type=EngineType.UNET, is_faceid=False, lora_dict={str(file_b): 0.5}, **_BASE_KWARGS
        )

        assert path_a != path_b

    def test_unet_path_differs_when_basename_collides_but_content_differs(self, tmp_path):
        """Same basename, different folders/content (the sepiagraph_* scenario) must
        still fork -- basename-only signing would silently alias them."""
        dir_a = tmp_path / "run_a"
        dir_b = tmp_path / "run_b"
        dir_a.mkdir()
        dir_b.mkdir()
        file_a = dir_a / "style.safetensors"
        file_b = dir_b / "style.safetensors"
        file_a.write_bytes(b"a" * 100)
        file_b.write_bytes(b"b" * 250)
        em = _make_engine_manager()
        path_a = em.get_engine_path(
            engine_type=EngineType.UNET, is_faceid=False, lora_dict={str(file_a): 0.5}, **_BASE_KWARGS
        )
        path_b = em.get_engine_path(
            engine_type=EngineType.UNET, is_faceid=False, lora_dict={str(file_b): 0.5}, **_BASE_KWARGS
        )

        assert path_a != path_b

    def test_unet_path_no_lora_and_empty_lora_dict_are_equivalent(self):
        em = _make_engine_manager()
        path_none = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=False, lora_dict=None, **_BASE_KWARGS)
        path_empty = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=False, lora_dict={}, **_BASE_KWARGS)

        assert path_none == path_empty

    def test_unet_path_forks_on_lora_presence(self, tmp_path):
        lora_file = tmp_path / "style_a.safetensors"
        lora_file.write_bytes(b"fake-lora-weights")
        em = _make_engine_manager()
        path_none = em.get_engine_path(engine_type=EngineType.UNET, is_faceid=False, lora_dict=None, **_BASE_KWARGS)
        path_lora = em.get_engine_path(
            engine_type=EngineType.UNET, is_faceid=False, lora_dict={str(lora_file): 1.0}, **_BASE_KWARGS
        )

        assert path_none != path_lora

    def test_unet_path_is_deterministic_with_lora(self, tmp_path):
        """Same LoRA config -> same path, so a previously-built engine is reused."""
        lora_file = tmp_path / "style_a.safetensors"
        lora_file.write_bytes(b"fake-lora-weights")
        em = _make_engine_manager()
        kwargs = dict(_BASE_KWARGS, engine_type=EngineType.UNET, is_faceid=False, lora_dict={str(lora_file): 0.5})

        path_a = em.get_engine_path(**kwargs)
        path_b = em.get_engine_path(**kwargs)

        assert path_a == path_b

    def test_vae_paths_unaffected_by_lora_dict(self, tmp_path):
        lora_file = tmp_path / "style_a.safetensors"
        lora_file.write_bytes(b"fake-lora-weights")
        em = _make_engine_manager()
        path_none = em.get_engine_path(
            engine_type=EngineType.VAE_DECODER, is_faceid=False, lora_dict=None, **_BASE_KWARGS
        )
        path_lora = em.get_engine_path(
            engine_type=EngineType.VAE_DECODER, is_faceid=False, lora_dict={str(lora_file): 0.7}, **_BASE_KWARGS
        )

        assert path_none == path_lora
