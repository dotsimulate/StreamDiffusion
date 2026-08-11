"""
Regression tests for adapter-mode-aware FP8 calibration image resolution
(fp8-round-9.1 §10, wrapper.py / fp8_quantize.py / engine_manager.py).

fp8-round-9.1 wired real calibration images into FP8 IP-Adapter token
resolution, but a single flat folder is subject-constrained the moment the
config switches to `type: faceid`: InsightFace/ArcFace raises on any image
with no detectable face (diffusers_ipadapter/ip_adapter/face_utils.py's
extract_face_embeddings), and get_image_embeds is all-or-nothing, so one
non-face image in a shared folder would delete every other image's
contribution too and silently degrade the whole calibration set to zero-pad.

§10's fix is two folders keyed on encoder *modality* -- `general/` for
CLIP-based adapters (regular/plus), `faces/` for FaceID -- resolved once by
`_resolve_fp8_calibration_dir` and threaded into both the `--ci<hash>`
cache-key call and the actual image loader, so the two can never disagree
about which folder is in play. This file pins that resolver, plus the
`_list_calibration_images` helper now shared by the loader, the resolver,
and `EngineManager._calibration_image_signature`.
"""

import hashlib
import logging
from pathlib import Path

from PIL import Image

from streamdiffusion.acceleration.tensorrt.engine_manager import EngineManager
from streamdiffusion.acceleration.tensorrt.fp8_quantize import _list_calibration_images
from streamdiffusion.wrapper import _load_fp8_calibration_style_images, _resolve_fp8_calibration_dir


def _write_tiny_image(path: Path, fill=(10, 20, 30)) -> None:
    Image.new("RGB", (2, 2), color=fill).save(path)


def _make_engine_manager(engine_dir: str) -> EngineManager:
    """Build an EngineManager without running __init__'s heavy compile-fn
    imports -- same pattern as test_engine_path_ipadapter_suffixes.py.
    `_calibration_image_signature` only touches Path/hashlib, so `_configs`
    is never needed here."""
    em = EngineManager.__new__(EngineManager)
    em.engine_dir = Path(engine_dir)
    return em


class TestResolveFp8CalibrationDir:
    def test_regular_and_plus_resolve_to_general_faceid_resolves_to_faces(self, tmp_path):
        root = tmp_path / "calibration"
        general = root / "general"
        faces = root / "faces"
        general.mkdir(parents=True)
        faces.mkdir(parents=True)
        _write_tiny_image(general / "a.png")
        _write_tiny_image(faces / "b.png")

        assert _resolve_fp8_calibration_dir(str(root), "regular") == str(general)
        assert _resolve_fp8_calibration_dir(str(root), "plus") == str(general)
        assert _resolve_fp8_calibration_dir(str(root), "faceid") == str(faces)

    def test_unrecognized_type_defaults_to_general(self, tmp_path):
        """A type string this table doesn't know about must not crash --
        default to the CLIP-based folder, same as None/unset."""
        root = tmp_path / "calibration"
        general = root / "general"
        general.mkdir(parents=True)
        _write_tiny_image(general / "a.png")

        assert _resolve_fp8_calibration_dir(str(root), "some_future_type") == str(general)
        assert _resolve_fp8_calibration_dir(str(root), None) == str(general)

    def test_missing_subfolder_returns_original_path_unchanged_with_warning(self, tmp_path, caplog):
        root = tmp_path / "calibration"
        root.mkdir()
        _write_tiny_image(root / "flat.png")  # no faces/ subfolder under root

        with caplog.at_level(logging.WARNING, logger="streamdiffusion.wrapper"):
            result = _resolve_fp8_calibration_dir(str(root), "faceid")

        assert result == str(root)
        assert any("faces" in r.message and "faceid" in r.message for r in caplog.records)

    def test_empty_subfolder_treated_as_missing(self, tmp_path):
        """The subfolder exists but has no recognized-extension file in it --
        must fall back exactly like a missing subfolder, not return an
        empty directory the loader would then find nothing in."""
        root = tmp_path / "calibration"
        faces = root / "faces"
        faces.mkdir(parents=True)
        _write_tiny_image(root / "flat.png")

        result = _resolve_fp8_calibration_dir(str(root), "faceid")

        assert result == str(root)

    def test_single_file_path_returned_unchanged_no_mode_logic(self, tmp_path):
        f = tmp_path / "style.png"
        _write_tiny_image(f)

        assert _resolve_fp8_calibration_dir(str(f), "faceid") == str(f)
        assert _resolve_fp8_calibration_dir(str(f), "regular") == str(f)

    def test_none_path_returns_none(self):
        assert _resolve_fp8_calibration_dir(None, "faceid") is None

    def test_nonexistent_path_returns_unchanged(self, tmp_path):
        missing = tmp_path / "does_not_exist"

        assert _resolve_fp8_calibration_dir(str(missing), "faceid") == str(missing)


class TestListLoadHashAgreement:
    """The three call sites (`_list_calibration_images`,
    `_load_fp8_calibration_style_images`, `EngineManager.
    _calibration_image_signature`) must always agree on which files count --
    a disagreement would mean the `--ci<hash>` cache key could name a
    different image set than the one actually loaded into calibration."""

    def test_loader_lister_and_hasher_agree_on_which_files_count(self, tmp_path):
        root = tmp_path / "calibration" / "general"
        root.mkdir(parents=True)
        _write_tiny_image(root / "a.png")
        _write_tiny_image(root / "b.png")
        em = _make_engine_manager(str(tmp_path / "engines"))

        listed_before = _list_calibration_images(str(root))
        loaded_before = _load_fp8_calibration_style_images(str(root))
        sig_before = em._calibration_image_signature(str(root))

        assert len(listed_before) == 2
        assert loaded_before is not None and len(loaded_before) == 2
        assert sig_before is not None

        # An unrecognized extension must be invisible to all three -- proves
        # they're all working from the same file set, not just the same count.
        (root / "notes.txt").write_text("not an image")

        listed_after = _list_calibration_images(str(root))
        loaded_after = _load_fp8_calibration_style_images(str(root))
        sig_after = em._calibration_image_signature(str(root))

        assert len(listed_after) == len(listed_before)
        assert loaded_after is not None and len(loaded_after) == len(loaded_before)
        assert sig_after == sig_before

    def test_nested_subdirectory_image_is_invisible_to_all_three(self, tmp_path):
        """Deliberately non-recursive: a mode-folder split one level below
        `path` (§10.1's `general/`/`faces/`) must not itself be picked up by
        a listing rooted at `path`'s parent, and any other accidental
        nesting must stay invisible too."""
        root = tmp_path / "calibration" / "general"
        nested = root / "nested"
        nested.mkdir(parents=True)
        _write_tiny_image(root / "a.png")
        _write_tiny_image(nested / "hidden.png")
        em = _make_engine_manager(str(tmp_path / "engines"))

        listed = _list_calibration_images(str(root))
        loaded = _load_fp8_calibration_style_images(str(root))
        sig = em._calibration_image_signature(str(root))

        assert len(listed) == 1
        assert loaded is not None and len(loaded) == 1
        expected_sig = hashlib.sha1((root / "a.png").read_bytes()).hexdigest()[:6]
        assert sig == expected_sig
