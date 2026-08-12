"""
Regression test for B1 — feeding InsightFace BGR instead of RGB
(``streamdiffusion.modules.faceid_compat.apply_faceid_patches``), plus S1 — collapsing the
two-detection-pass ``prepare_face_conditioning`` into one.

Every InsightFace ONNX model (arcface_onnx.py, retinaface.py, scrfd.py) preprocesses with
``swapRB=True`` — i.e. it contractually expects BGR. The vendored ``face_utils.py`` hands it a
plain RGB numpy array (``np.array(PIL_image)``), so ArcFace computes the 512-d identity
embedding from a channel-swapped face — no error, just a corrupted vector. B1's patch wraps
``detect_faces_multires`` to reverse the channel axis before the array reaches the real
detector.

This test fakes out ``diffusers_ipadapter`` entirely (it is a pip-installed, pinned-SHA vendor
package — not something to import for real in a unit test) with a minimal stand-in module tree
that has the same shape as the two functions ``apply_faceid_patches`` touches, so the patch
logic itself is exercised without any InsightFace/torch/PIL dependency beyond what the patch
code already imports at module scope.

Run with: pytest tests/unit/test_faceid_bgr_patch.py -v
"""

import sys
import types

import numpy as np
import pytest


def _install_fake_diffusers_ipadapter():
    """Build a minimal fake ``diffusers_ipadapter.ip_adapter.{face_utils,ip_adapter}`` module
    pair and register it in sys.modules, so ``faceid_compat`` can ``import`` it normally.

    Mirrors the real shape just enough for the patch under test:
    - ``face_utils.detect_faces_multires(insightface_model, image)`` — records the array it
      was called with and returns a canned "faces" list.
    - ``face_utils.prepare_face_conditioning`` — a two-pass stub matching the vendored
      original's signature (so the S1 replacement's call signature is exercised too).
    - ``ip_adapter.py``'s ``from .face_utils import ... prepare_face_conditioning`` is modeled
      by copying the *same function object* into the fake ``ip_adapter`` module's namespace —
      this is the exact aliasing behaviour the S1 patch has to defeat.
    """
    calls = {"detect": [], "prepare_two_pass": 0}

    pkg = types.ModuleType("diffusers_ipadapter")
    subpkg = types.ModuleType("diffusers_ipadapter.ip_adapter")
    face_utils = types.ModuleType("diffusers_ipadapter.ip_adapter.face_utils")
    ip_adapter_mod = types.ModuleType("diffusers_ipadapter.ip_adapter.ip_adapter")

    class _FakeFace:
        def __init__(self):
            self.normed_embedding = np.zeros(512, dtype=np.float32)
            self.embedding = np.zeros(512, dtype=np.float32)
            self.kps = np.zeros((5, 2), dtype=np.float32)

    def detect_faces_multires(insightface_model, image, *args, **kwargs):
        calls["detect"].append(np.array(image, copy=True))
        return [_FakeFace()]

    def get_face_crop_size(is_sdxl=False, is_kolors=False):
        if is_kolors:
            return 336
        if is_sdxl:
            return 256
        return 224

    def prepare_face_conditioning(
        insightface_model, images, is_sdxl=False, is_kolors=False, normalize_embeddings=True
    ):
        """Stand-in for the vendored *original* two-pass implementation: detects once inside
        an 'extract_face_embeddings'-style helper, then a second time to re-crop at the real
        model size. Only used to prove the S1 replacement stops calling this shape twice —
        the replacement doesn't call this function at all, it calls detect_faces_multires
        directly, so this counter would stay at 0 once patched.
        """
        calls["prepare_two_pass"] += 1
        detect_faces_multires(insightface_model, images)
        detect_faces_multires(insightface_model, images)
        return None, None

    face_utils.detect_faces_multires = detect_faces_multires
    face_utils.get_face_crop_size = get_face_crop_size
    face_utils.prepare_face_conditioning = prepare_face_conditioning

    # Mirrors `from .face_utils import get_insightface_model, prepare_face_conditioning` in
    # the real ip_adapter.py: copies the function *object* into this module's own namespace.
    ip_adapter_mod.prepare_face_conditioning = face_utils.prepare_face_conditioning

    sys.modules["diffusers_ipadapter"] = pkg
    sys.modules["diffusers_ipadapter.ip_adapter"] = subpkg
    sys.modules["diffusers_ipadapter.ip_adapter.face_utils"] = face_utils
    sys.modules["diffusers_ipadapter.ip_adapter.ip_adapter"] = ip_adapter_mod

    return face_utils, ip_adapter_mod, calls


@pytest.fixture
def fake_vendor(monkeypatch):
    """Install the fake vendor modules for the duration of one test and always clean up
    sys.modules afterward, plus reload faceid_compat so each test gets fresh patch state
    (apply_faceid_patches is idempotent-guarded on the *patched function*, which is a fresh
    object every time the fake module is rebuilt).
    """
    for name in (
        "diffusers_ipadapter",
        "diffusers_ipadapter.ip_adapter",
        "diffusers_ipadapter.ip_adapter.face_utils",
        "diffusers_ipadapter.ip_adapter.ip_adapter",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    face_utils, ip_adapter_mod, calls = _install_fake_diffusers_ipadapter()
    yield face_utils, ip_adapter_mod, calls


class TestBgrPatch:
    def test_reverses_channel_axis_before_detection(self, fake_vendor):
        face_utils, _ip_adapter_mod, calls = fake_vendor
        from streamdiffusion.modules import faceid_compat

        faceid_compat.apply_faceid_patches()

        rgb_image = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb_image[..., 0] = 10  # R
        rgb_image[..., 1] = 20  # G
        rgb_image[..., 2] = 30  # B

        face_utils.detect_faces_multires(None, rgb_image)

        assert len(calls["detect"]) == 1
        received = calls["detect"][0]
        assert received[0, 0, 0] == 30  # was B, now first channel
        assert received[0, 0, 1] == 20  # G unchanged (middle channel)
        assert received[0, 0, 2] == 10  # was R, now last channel

    def test_leaves_non_hwc3_arrays_untouched(self, fake_vendor):
        """Only ndim==3 and shape[-1]==3 arrays are channel-swapped — anything else (e.g. a
        grayscale or already-preprocessed array) must pass through unmodified rather than
        raising or silently mis-slicing.
        """
        face_utils, _ip_adapter_mod, calls = fake_vendor
        from streamdiffusion.modules import faceid_compat

        faceid_compat.apply_faceid_patches()

        grayscale = np.full((8, 8), 42, dtype=np.uint8)
        face_utils.detect_faces_multires(None, grayscale)

        assert np.array_equal(calls["detect"][0], grayscale)

    def test_patch_is_idempotent(self, fake_vendor):
        face_utils, _ip_adapter_mod, calls = fake_vendor
        from streamdiffusion.modules import faceid_compat

        faceid_compat.apply_faceid_patches()
        once_patched = face_utils.detect_faces_multires
        faceid_compat.apply_faceid_patches()
        twice_patched = face_utils.detect_faces_multires

        assert once_patched is twice_patched

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        face_utils.detect_faces_multires(None, image)
        assert len(calls["detect"]) == 1  # not wrapped twice -> not swapped twice either

    def test_single_pass_patch_detects_once_per_image_on_both_bindings(self, fake_vendor):
        """S1: after apply_faceid_patches, both face_utils.prepare_face_conditioning and
        ip_adapter.prepare_face_conditioning (the separately-imported binding real
        ip_adapter.py's _get_faceid_embeds actually calls) must resolve to the same
        single-detection-pass replacement.
        """
        face_utils, ip_adapter_mod, calls = fake_vendor
        from PIL import Image

        from streamdiffusion.modules import faceid_compat

        faceid_compat.apply_faceid_patches()

        assert face_utils.prepare_face_conditioning is ip_adapter_mod.prepare_face_conditioning

        image = Image.new("RGB", (8, 8))
        face_utils.prepare_face_conditioning(None, image, is_sdxl=True)

        assert calls["prepare_two_pass"] == 0  # replacement never calls the two-pass original
        assert len(calls["detect"]) == 1  # exactly one detection for one image

    def test_single_pass_patch_uses_correct_crop_size(self, fake_vendor):
        face_utils, ip_adapter_mod, _calls = fake_vendor
        from PIL import Image

        from streamdiffusion.modules import faceid_compat

        faceid_compat.apply_faceid_patches()

        image = Image.new("RGB", (16, 16))
        _embeds, cropped = ip_adapter_mod.prepare_face_conditioning(None, image, is_sdxl=True)

        assert cropped[0].size == (256, 256)
