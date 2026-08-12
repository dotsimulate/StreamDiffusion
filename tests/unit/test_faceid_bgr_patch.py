"""
Regression test for B1 — feeding InsightFace BGR instead of RGB
(``streamdiffusion.modules.faceid_compat.apply_faceid_patches``), S1 — collapsing the
two-detection-pass ``prepare_face_conditioning`` into one, and the cost-reduction patches
(``_apply_insightface_loader_patch`` / ``_apply_detect_faces_multires_patch``) that cut the
InsightFace work done per "update image" press.

Every InsightFace ONNX model (arcface_onnx.py, retinaface.py, scrfd.py) preprocesses with
``swapRB=True`` — i.e. it contractually expects BGR. The vendored ``face_utils.py`` hands it a
plain RGB numpy array (``np.array(PIL_image)``), so ArcFace computes the 512-d identity
embedding from a channel-swapped face — no error, just a corrupted vector.

``detect_faces_multires`` is *replaced*, not wrapped: the patch reimplements the
resolution-sweep loop itself (BGR swap, ``max_num=1``, a sweep sized to the image and bounded
to 3 attempts) rather than delegating to the vendored implementation, because the vendored
loop offers no way to inject ``max_num`` or change its fixed 640-start / 7-attempt sweep from
outside. That means the fake harness below models ``insightface_model`` as an object with
``det_model.input_size`` and ``get(img, max_num)`` — the surface the replacement actually
calls — rather than modeling ``detect_faces_multires`` itself as the thing under patch.

This test fakes out ``diffusers_ipadapter`` entirely (it is a pip-installed, pinned-SHA vendor
package — not something to import for real in a unit test) with a minimal stand-in module tree
that has the same shape as the functions ``apply_faceid_patches`` touches, so the patch logic
itself is exercised without any InsightFace/torch/PIL dependency beyond what the patch code
already imports at module scope. The one exception is ``insightface.app.FaceAnalysis``, which
the loader patch monkeypatches for real (see ``TestInsightfaceLoaderPatch``) — that is the
actual real package, exercised the same way production code exercises it.

Run with: pytest tests/unit/test_faceid_bgr_patch.py -v
"""

import sys
import types

import numpy as np
import pytest


class _FakeFace:
    def __init__(self):
        self.normed_embedding = np.zeros(512, dtype=np.float32)
        self.embedding = np.zeros(512, dtype=np.float32)
        self.kps = np.zeros((5, 2), dtype=np.float32)


class _FakeDetModel:
    def __init__(self):
        self.input_size = None


class _FakeInsightFaceModel:
    """Stands in for a real InsightFace ``FaceAnalysis`` instance — only the surface the
    ``detect_faces_multires`` replacement actually touches: ``det_model.input_size``
    (settable, mirrors the sweep's resolution) and ``get(img, max_num)``.

    ``succeed_at_size`` lets a test control which sweep attempt first returns a face, to
    exercise the fallback-to-smaller-size path. ``always_faces=False`` simulates a frame with
    no detectable face at any size, to exercise the bounded-sweep-exhausted path.
    """

    def __init__(self, calls, always_faces=True, succeed_at_size=None):
        self.det_model = _FakeDetModel()
        self._calls = calls
        self._always_faces = always_faces
        self._succeed_at_size = succeed_at_size

    def get(self, img, max_num=0):
        size = self.det_model.input_size
        self._calls["get"].append({"image": np.array(img, copy=True), "max_num": max_num, "size": size})
        if self._succeed_at_size is not None:
            return [_FakeFace()] if size == self._succeed_at_size else []
        return [_FakeFace()] if self._always_faces else []


def _install_fake_diffusers_ipadapter():
    """Build a minimal fake ``diffusers_ipadapter.ip_adapter.{face_utils,ip_adapter}`` module
    pair and register it in sys.modules, so ``faceid_compat`` can ``import`` it normally.

    Mirrors the real shape just enough for the patches under test:
    - ``face_utils.detect_faces_multires(insightface_model, image)`` — a stand-in for the
      *un-patched vendored* implementation. Only ``prepare_face_conditioning``'s internal
      closure-captured call to it (below) still exercises this; the real patch under test
      never calls it, since it replaces the vendored sweep outright rather than wrapping it.
    - ``face_utils.prepare_face_conditioning`` — a two-pass stub matching the vendored
      original's signature (so the S1 replacement's call signature is exercised too).
    - ``face_utils.get_insightface_model`` — returns a fresh ``_FakeInsightFaceModel``.

    Both aliasing sites the real vendor package has for these names are modeled too, so a
    patch that only reaches ``face_utils`` fails a test here exactly as it fails in
    production:
    - ``ip_adapter.py``'s ``from .face_utils import get_insightface_model,
      prepare_face_conditioning`` — copies both function *objects* into the fake
      ``ip_adapter`` module's own namespace. (A prior version of this harness copied only
      ``prepare_face_conditioning``, which is why the loader patch's missing second
      binding shipped without a failing test — see ``TestInsightfaceLoaderPatch``.)
    - ``diffusers_ipadapter/ip_adapter/__init__.py``'s ``from .face_utils import *`` —
      copies every fake public name into the package module (``sys.modules
      ["diffusers_ipadapter.ip_adapter"]``, i.e. ``subpkg`` below) as well.
    """
    calls = {"get": [], "prepare_two_pass": 0, "get_insightface_model": []}

    pkg = types.ModuleType("diffusers_ipadapter")
    subpkg = types.ModuleType("diffusers_ipadapter.ip_adapter")
    face_utils = types.ModuleType("diffusers_ipadapter.ip_adapter.face_utils")
    ip_adapter_mod = types.ModuleType("diffusers_ipadapter.ip_adapter.ip_adapter")

    def detect_faces_multires(insightface_model, image, *args, **kwargs):
        return insightface_model.get(image)

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

    def get_insightface_model(model_name="buffalo_l", providers=None):
        calls["get_insightface_model"].append({"model_name": model_name, "providers": providers})
        return _FakeInsightFaceModel(calls)

    face_utils.detect_faces_multires = detect_faces_multires
    face_utils.get_face_crop_size = get_face_crop_size
    face_utils.prepare_face_conditioning = prepare_face_conditioning
    face_utils.get_insightface_model = get_insightface_model

    # Mirrors `from .face_utils import get_insightface_model, prepare_face_conditioning` in
    # the real ip_adapter.py: copies both function *objects* into this module's own namespace.
    ip_adapter_mod.get_insightface_model = face_utils.get_insightface_model
    ip_adapter_mod.prepare_face_conditioning = face_utils.prepare_face_conditioning

    # Mirrors the real package __init__'s `from .face_utils import *`: every fake public
    # face_utils name gets its own copy in the package module too.
    subpkg.detect_faces_multires = face_utils.detect_faces_multires
    subpkg.get_face_crop_size = face_utils.get_face_crop_size
    subpkg.prepare_face_conditioning = face_utils.prepare_face_conditioning
    subpkg.get_insightface_model = face_utils.get_insightface_model

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


def _make_model(calls, **kwargs):
    return _FakeInsightFaceModel(calls, **kwargs)


class TestBgrPatch:
    def test_reverses_channel_axis_before_detection(self, fake_vendor):
        face_utils, _ip_adapter_mod, calls = fake_vendor
        from streamdiffusion.modules import faceid_compat

        faceid_compat.apply_faceid_patches()

        rgb_image = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb_image[..., 0] = 10  # R
        rgb_image[..., 1] = 20  # G
        rgb_image[..., 2] = 30  # B

        model = _make_model(calls)
        face_utils.detect_faces_multires(model, rgb_image)

        assert len(calls["get"]) == 1
        received = calls["get"][0]["image"]
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
        model = _make_model(calls)
        face_utils.detect_faces_multires(model, grayscale)

        assert np.array_equal(calls["get"][0]["image"], grayscale)

    def test_patch_is_idempotent(self, fake_vendor):
        face_utils, _ip_adapter_mod, calls = fake_vendor
        from streamdiffusion.modules import faceid_compat

        faceid_compat.apply_faceid_patches()
        once_patched = face_utils.detect_faces_multires
        faceid_compat.apply_faceid_patches()
        twice_patched = face_utils.detect_faces_multires

        assert once_patched is twice_patched

        model = _make_model(calls)
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        face_utils.detect_faces_multires(model, image)
        assert len(calls["get"]) == 1  # not wrapped twice -> not swapped twice either

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

        model = _make_model(calls)
        image = Image.new("RGB", (8, 8))
        face_utils.prepare_face_conditioning(model, image, is_sdxl=True)

        assert calls["prepare_two_pass"] == 0  # replacement never calls the two-pass original
        assert len(calls["get"]) == 1  # exactly one detection for one image

    def test_single_pass_patch_uses_correct_crop_size(self, fake_vendor):
        face_utils, ip_adapter_mod, calls = fake_vendor
        from PIL import Image

        from streamdiffusion.modules import faceid_compat

        faceid_compat.apply_faceid_patches()

        model = _make_model(calls)
        image = Image.new("RGB", (16, 16))
        _embeds, cropped = ip_adapter_mod.prepare_face_conditioning(model, image, is_sdxl=True)

        assert cropped[0].size == (256, 256)


class TestDetectFacesCostReduction:
    """Coverage for the sweep-size / max_num cost cuts folded into the same replacement as B1."""

    def test_uses_max_num_one_and_starts_from_image_native_size(self, fake_vendor):
        face_utils, _ip_adapter_mod, calls = fake_vendor
        from streamdiffusion.modules import faceid_compat

        faceid_compat.apply_faceid_patches()

        model = _make_model(calls)  # always finds a face
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        faces = face_utils.detect_faces_multires(model, image)

        assert len(faces) == 1
        assert len(calls["get"]) == 1  # first attempt succeeds -> no wasted sweep
        assert calls["get"][0]["max_num"] == 1
        assert calls["get"][0]["size"] == (512, 512)  # native size, never upscaled to 640

    def test_sweep_falls_back_to_smaller_size(self, fake_vendor):
        face_utils, _ip_adapter_mod, calls = fake_vendor
        from streamdiffusion.modules import faceid_compat

        faceid_compat.apply_faceid_patches()

        model = _make_model(calls, succeed_at_size=(384, 384))
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        faces = face_utils.detect_faces_multires(model, image)

        assert len(faces) == 1
        sizes_tried = [c["size"] for c in calls["get"]]
        assert sizes_tried == [(512, 512), (448, 448), (384, 384)]

    def test_sweep_is_bounded_to_three_attempts(self, fake_vendor):
        """The vendored original sweeps up to 7 sizes (640 down to 256) before giving up.
        The replacement must not reproduce that worst case."""
        face_utils, _ip_adapter_mod, calls = fake_vendor
        from streamdiffusion.modules import faceid_compat

        faceid_compat.apply_faceid_patches()

        model = _make_model(calls, always_faces=False)  # no face at any size
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        faces = face_utils.detect_faces_multires(model, image)

        assert faces == []
        assert len(calls["get"]) == 3


class TestInsightfaceLoaderPatch:
    """Coverage for _apply_insightface_loader_patch: get_insightface_model must construct
    FaceAnalysis with allowed_modules=['detection', 'recognition'], and must restore the
    real insightface.app.FaceAnalysis binding afterward regardless of what called it.
    """

    def test_forwards_allowed_modules_to_face_analysis_and_restores_binding(self, fake_vendor, monkeypatch):
        face_utils, ip_adapter_mod, _calls = fake_vendor
        import insightface.app as insightface_app_module

        fa_calls = []

        class _RecordingFaceAnalysis:
            def __init__(self, *args, **kwargs):
                fa_calls.append(kwargs)

        monkeypatch.setattr(insightface_app_module, "FaceAnalysis", _RecordingFaceAnalysis)

        def fake_get_insightface_model(model_name="buffalo_l", providers=None):
            # Mirrors the real vendored function's actual construction call
            # (face_utils.py:49): FaceAnalysis(name=..., root=..., providers=...).
            return insightface_app_module.FaceAnalysis(name=model_name, providers=providers)

        # Install the SAME function object on both bindings, mirroring how the real
        # `from .face_utils import get_insightface_model` in ip_adapter.py copies one
        # function object into two module namespaces. Setting only `face_utils`'s binding
        # (as this test used to) leaves `ip_adapter_mod`'s pointing at the fixture's
        # original get_insightface_model, so apply_faceid_patches's identity-guarded
        # rebind — see _rebind_across_modules — would skip it, and the test would call
        # through a binding the patch never touches: exactly the wrong-seam gap that let
        # the shipped bug (loader patch never reaching ip_adapter.py:56) through review.
        face_utils.get_insightface_model = fake_get_insightface_model
        ip_adapter_mod.get_insightface_model = fake_get_insightface_model

        from streamdiffusion.modules import faceid_compat

        faceid_compat.apply_faceid_patches()

        assert face_utils.get_insightface_model is ip_adapter_mod.get_insightface_model

        # The real call site is ip_adapter.py:56, which reads *its own* copied-in name —
        # invoke through that binding, not face_utils's, so this test can only pass if the
        # patch actually reached the real caller.
        ip_adapter_mod.get_insightface_model("buffalo_l", None)

        assert len(fa_calls) == 1
        assert fa_calls[0]["allowed_modules"] == ["detection", "recognition"]

        # The patch must not leave insightface.app.FaceAnalysis pointing at its own
        # injector after the call returns — it should be restored to whatever was bound
        # (here, the test's own monkeypatched recorder) before the call started.
        assert insightface_app_module.FaceAnalysis is _RecordingFaceAnalysis

    def test_does_not_override_an_explicit_allowed_modules(self, fake_vendor, monkeypatch):
        """setdefault semantics: if a future vendored version starts passing its own
        allowed_modules, this patch must not silently clobber it."""
        face_utils, ip_adapter_mod, _calls = fake_vendor
        import insightface.app as insightface_app_module

        fa_calls = []

        class _RecordingFaceAnalysis:
            def __init__(self, *args, **kwargs):
                fa_calls.append(kwargs)

        monkeypatch.setattr(insightface_app_module, "FaceAnalysis", _RecordingFaceAnalysis)

        def fake_get_insightface_model(model_name="buffalo_l", providers=None):
            return insightface_app_module.FaceAnalysis(
                name=model_name, providers=providers, allowed_modules=["detection"]
            )

        # Same-object install on both bindings — see the comment in the sibling test above.
        face_utils.get_insightface_model = fake_get_insightface_model
        ip_adapter_mod.get_insightface_model = fake_get_insightface_model

        from streamdiffusion.modules import faceid_compat

        faceid_compat.apply_faceid_patches()
        ip_adapter_mod.get_insightface_model("buffalo_l", None)

        assert fa_calls[0]["allowed_modules"] == ["detection"]

    def test_patch_is_idempotent(self, fake_vendor):
        face_utils, _ip_adapter_mod, _calls = fake_vendor
        from streamdiffusion.modules import faceid_compat

        faceid_compat.apply_faceid_patches()
        once_patched = face_utils.get_insightface_model
        faceid_compat.apply_faceid_patches()
        twice_patched = face_utils.get_insightface_model

        assert once_patched is twice_patched


class _FakeRecognitionModel:
    """Stands in for the InsightFace 'recognition' entry of FaceAnalysis.models — only
    the surface warmup_faceid actually touches: get_feat(img)."""

    def __init__(self, calls):
        self._calls = calls

    def get_feat(self, img):
        self._calls.append(np.array(img, copy=True))
        return np.zeros((1, 512), dtype=np.float32)


class _FakeInsightFaceModelForWarmup:
    """Like _FakeInsightFaceModel, but also exposes a `models` dict (keyed by taskname,
    mirroring face_analysis.py:39) so warmup_faceid can reach 'recognition' directly."""

    def __init__(self, det_calls, recognition_calls):
        self.det_model = _FakeDetModel()
        self._det_calls = det_calls
        self.models = {"recognition": _FakeRecognitionModel(recognition_calls)}

    def get(self, img, max_num=0):
        self._det_calls.append(
            {"image": np.array(img, copy=True), "max_num": max_num, "size": self.det_model.input_size}
        )
        return []  # a blank warmup frame never contains a face


class _FakeIPAdapterForWarmup:
    def __init__(self, insightface_model, is_sdxl=True):
        self.insightface_model = insightface_model
        self.is_sdxl = is_sdxl


class TestWarmupFaceid:
    """Coverage for warmup_faceid: it must not be reachable from apply_faceid_patches
    (the fake harness above calls that on every test and must stay unaffected), must warm
    the detection session at 512x512/max_num=1, must warm the recognition session
    directly via models['recognition'].get_feat, and must never raise — a warmup failure
    is a perf regression (slow first press), not a correctness failure.
    """

    def test_warms_detection_and_recognition_sessions(self, fake_vendor):
        from streamdiffusion.modules import faceid_compat

        det_calls = []
        rec_calls = []
        model = _FakeInsightFaceModelForWarmup(det_calls, rec_calls)
        ipadapter = _FakeIPAdapterForWarmup(model, is_sdxl=True)

        faceid_compat.warmup_faceid(ipadapter)

        assert len(det_calls) == 1
        assert det_calls[0]["max_num"] == 1
        assert det_calls[0]["size"] == (512, 512)
        assert model.det_model.input_size == (512, 512)

        assert len(rec_calls) == 1
        assert rec_calls[0].shape == (112, 112, 3)

    def test_no_insightface_model_is_a_noop(self):
        from streamdiffusion.modules import faceid_compat

        class _NoInsightfaceModel:
            pass

        faceid_compat.warmup_faceid(_NoInsightfaceModel())  # must not raise

    def test_detection_failure_does_not_block_recognition_warmup(self, fake_vendor):
        from streamdiffusion.modules import faceid_compat

        rec_calls = []

        class _RaisingModel:
            def __init__(self):
                self.det_model = _FakeDetModel()
                self.models = {"recognition": _FakeRecognitionModel(rec_calls)}

            def get(self, img, max_num=0):
                raise RuntimeError("simulated detection session failure")

        ipadapter = _FakeIPAdapterForWarmup(_RaisingModel(), is_sdxl=False)

        faceid_compat.warmup_faceid(ipadapter)  # must not raise

        assert len(rec_calls) == 1  # later step still ran despite the earlier failure

    def test_missing_recognition_model_is_a_noop_for_that_step(self, fake_vendor):
        from streamdiffusion.modules import faceid_compat

        det_calls = []

        class _NoRecognitionModel:
            def __init__(self):
                self.det_model = _FakeDetModel()
                self.models = {}  # allowed_modules=['detection'] only, e.g.

            def get(self, img, max_num=0):
                det_calls.append(max_num)
                return []

        ipadapter = _FakeIPAdapterForWarmup(_NoRecognitionModel(), is_sdxl=True)

        faceid_compat.warmup_faceid(ipadapter)  # must not raise

        assert det_calls == [1]

    def test_apply_faceid_patches_does_not_invoke_warmup(self, fake_vendor, monkeypatch):
        """apply_faceid_patches must stay warmup-free: every other test in this file calls
        it against a fake vendor tree with no insightface_model at all, so if warmup ever
        got folded in there, those tests would start failing."""
        from streamdiffusion.modules import faceid_compat

        calls = []
        monkeypatch.setattr(faceid_compat, "warmup_faceid", lambda ipadapter: calls.append(ipadapter))

        faceid_compat.apply_faceid_patches()

        assert calls == []
