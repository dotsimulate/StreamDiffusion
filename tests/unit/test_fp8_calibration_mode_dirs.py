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
about which folder is in play. This file pins that resolver.

The companion `_list_calibration_images` helper shared by the loader, the
resolver, and `EngineManager._calibration_image_signature` is pinned in
pr/trt-engine-cache-tags instead of here: this PR's cherry-pick does not
touch engine_manager.py, so a three-way agreement test against it would
fail against an engine_manager.py that predates that PR's changes.
"""

import logging
from pathlib import Path

from PIL import Image

from streamdiffusion.wrapper import _resolve_fp8_calibration_dir


def _write_tiny_image(path: Path, fill=(10, 20, 30)) -> None:
    Image.new("RGB", (2, 2), color=fill).save(path)


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
