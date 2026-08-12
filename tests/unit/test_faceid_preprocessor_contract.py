"""
Regression test for A5 — ``FaceIDEmbeddingPreprocessor``
(``streamdiffusion.preprocessing.processors.faceid_embedding``).

Two things changed: the dead ``update_faceid_v2_weight`` method (zero callers anywhere in the
tree, confirmed by repo-wide search) was deleted, and the ``faceid_v2_weight`` metadata
description was corrected — it previously implied ``faceid_v2_weight`` affects all FaceID
models, but tracing its only read site (vendored ``ip_adapter.py``'s ``_get_faceid_embeds``,
inside ``if self.is_plus:`` branches) shows it only affects FaceID-Plus/v2, never plain FaceID.
This test locks in the deletion and the corrected metadata wording, plus the preprocessor's
existing construction/processing contract so a future edit can't silently regress either.

Run with: pytest tests/unit/test_faceid_preprocessor_contract.py -v
"""

import pytest
import torch
from PIL import Image

from streamdiffusion.preprocessing.processors.faceid_embedding import FaceIDEmbeddingPreprocessor


class _FakeIPAdapter:
    def __init__(self, insightface_model="buffalo_l", raise_on_embed=False):
        self.insightface_model = insightface_model
        self.raise_on_embed = raise_on_embed
        self.calls = []

    def get_image_embeds(self, images, faceid_v2_weight=None):
        self.calls.append({"images": images, "faceid_v2_weight": faceid_v2_weight})
        if self.raise_on_embed:
            raise RuntimeError("boom")
        return torch.zeros(1, 4, 8), torch.ones(1, 4, 8)


class TestMetadataCorrection:
    def test_faceid_v2_weight_description_scopes_to_plus_v2_only(self):
        meta = FaceIDEmbeddingPreprocessor.get_preprocessor_metadata()
        description = meta["parameters"]["faceid_v2_weight"]["description"]

        assert "Plus" in description or "v2" in description
        assert "no effect" in description.lower() or "only" in description.lower()

    def test_dead_update_method_was_removed(self):
        """A5: update_faceid_v2_weight had zero callers anywhere in the tree — deleted
        outright rather than left as unreachable code.
        """
        assert not hasattr(FaceIDEmbeddingPreprocessor, "update_faceid_v2_weight")


class TestConstructionContract:
    def test_requires_insightface_model_attribute(self):
        class _NoInsightface:
            def get_image_embeds(self, **kwargs):
                return None

        with pytest.raises(ValueError, match="InsightFace"):
            FaceIDEmbeddingPreprocessor(ipadapter=_NoInsightface())

    def test_requires_insightface_model_not_none(self):
        with pytest.raises(ValueError, match="InsightFace"):
            FaceIDEmbeddingPreprocessor(ipadapter=_FakeIPAdapter(insightface_model=None))

    def test_requires_get_image_embeds_method(self):
        class _NoEmbedMethod:
            insightface_model = "buffalo_l"

        with pytest.raises(ValueError, match="get_image_embeds"):
            FaceIDEmbeddingPreprocessor(ipadapter=_NoEmbedMethod())

    def test_default_faceid_v2_weight_is_one(self):
        pre = FaceIDEmbeddingPreprocessor(ipadapter=_FakeIPAdapter())
        assert pre.faceid_v2_weight == 1.0

    def test_custom_faceid_v2_weight_is_coerced_to_float(self):
        pre = FaceIDEmbeddingPreprocessor(ipadapter=_FakeIPAdapter(), faceid_v2_weight=2)
        assert pre.faceid_v2_weight == 2.0
        assert isinstance(pre.faceid_v2_weight, float)


class TestProcessCore:
    def test_passes_image_and_weight_through_to_get_image_embeds(self):
        fake = _FakeIPAdapter()
        pre = FaceIDEmbeddingPreprocessor(ipadapter=fake, faceid_v2_weight=0.5)
        image = Image.new("RGB", (8, 8))

        positive, negative = pre._process_core(image)

        assert len(fake.calls) == 1
        assert fake.calls[0]["images"] == [image]
        assert fake.calls[0]["faceid_v2_weight"] == 0.5
        assert positive.shape == (1, 4, 8)
        assert negative.shape == (1, 4, 8)

    def test_wraps_embed_failure_in_runtime_error(self):
        fake = _FakeIPAdapter(raise_on_embed=True)
        pre = FaceIDEmbeddingPreprocessor(ipadapter=fake)
        image = Image.new("RGB", (8, 8))

        with pytest.raises(RuntimeError, match="Failed to extract face embeddings"):
            pre._process_core(image)
