"""
Regression test for B2 — fusing the FaceID checkpoint's discarded rank-128 LoRA into the
UNet's attention linears (``streamdiffusion.modules.faceid_compat.fuse_faceid_lora``).

``ip_adapter.py``'s strict ``load_state_dict`` silently drops every ``to_{q,k,v,out}_lora``
key because no LoRA-aware attention-processor class exists in the vendored
``attention_processor.py``, and the TensorRT export path rebuilds every processor before ONNX
export regardless (``IPAdapterUNetExportWrapper.__init__``), so a processor-resident LoRA would
be lost there too. ``fuse_faceid_lora`` instead adds ``(up @ down) * lora_scale`` directly into
the base ``to_q``/``to_k``/``to_v``/``to_out[0]`` linears, which no processor rebuild can touch.

This test uses a tiny synthetic attention module (real ``torch.nn.Module`` instances, so
``unet.get_submodule("layers.<i>.to_out.0")`` resolves exactly like it does on the real UNet's
``attn1``/``attn2`` — a ``ModuleList`` index is just a string-keyed submodule) and a synthetic
LoRA state dict — no GPU, no checkpoint download, no diffusers UNet construction.

Run with: pytest tests/unit/test_faceid_lora_fusion.py -v
"""

import pytest
import torch
import torch.nn as nn

from streamdiffusion.modules.faceid_compat import _LORA_TARGET_ATTR, fuse_faceid_lora

_DIM = 4
_RANK = 2
_N_LAYERS = 3


class _FakeAttnModule(nn.Module):
    """Mirrors the shape of a real diffusers Attention module: to_q/to_k/to_v linears plus
    a to_out ModuleList (index 0 is the projection linear, matching ``to_out.0``), and a
    ``.processor`` submodule standing in for the real ``AttnProcessor``/``IPAttnProcessor``.
    """

    def __init__(self, dim: int = _DIM):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        self.processor = nn.Module()


class _FakeUNet(nn.Module):
    """Minimal stand-in for UNet2DConditionModel: only ``attn_processors`` and
    ``get_submodule`` (inherited from nn.Module, unmodified) are used by ``fuse_faceid_lora``.
    """

    def __init__(self, n_layers: int = _N_LAYERS, dim: int = _DIM):
        super().__init__()
        self.layers = nn.ModuleList([_FakeAttnModule(dim) for _ in range(n_layers)])

    @property
    def attn_processors(self):
        # Real diffusers_ipadapter iterates unet.attn_processors.keys() in this same
        # ".processor"-suffixed form to resolve module_path via proc_key[:-len(".processor")].
        return {f"layers.{i}.processor": self.layers[i].processor for i in range(len(self.layers))}


def _make_lora_checkpoint(tmp_path, n_layers=_N_LAYERS, dim=_DIM, rank=_RANK, only_indices=None):
    """Build a synthetic FaceID-shaped checkpoint and save it to a real file (fuse_faceid_lora
    loads via torch.load(ckpt_path, ...), not from an in-memory dict).
    """
    indices = range(n_layers) if only_indices is None else only_indices
    sd = {}
    for i in indices:
        for lora_name in _LORA_TARGET_ATTR:
            sd[f"{i}.{lora_name}.down.weight"] = torch.randn(rank, dim)
            sd[f"{i}.{lora_name}.up.weight"] = torch.randn(dim, rank)
    ckpt_path = tmp_path / "fake_faceid.bin"
    torch.save({"ip_adapter": sd}, ckpt_path)
    return ckpt_path, sd


class TestFuseFaceIdLora:
    def test_fuses_up_at_down_into_target_linears(self, tmp_path):
        unet = _FakeUNet()
        ckpt_path, sd = _make_lora_checkpoint(tmp_path)

        original_to_q = [layer.to_q.weight.detach().clone() for layer in unet.layers]
        original_to_out = [layer.to_out[0].weight.detach().clone() for layer in unet.layers]

        fused_count = fuse_faceid_lora(unet, str(ckpt_path))

        assert fused_count == _N_LAYERS
        for i, layer in enumerate(unet.layers):
            down = sd[f"{i}.to_q_lora.down.weight"]
            up = sd[f"{i}.to_q_lora.up.weight"]
            expected = original_to_q[i] + (up @ down)
            assert torch.allclose(layer.to_q.weight, expected, atol=1e-6)

            down_out = sd[f"{i}.to_out_lora.down.weight"]
            up_out = sd[f"{i}.to_out_lora.up.weight"]
            expected_out = original_to_out[i] + (up_out @ down_out)
            assert torch.allclose(layer.to_out[0].weight, expected_out, atol=1e-6)

    def test_lora_scale_scales_the_delta(self, tmp_path):
        unet = _FakeUNet()
        ckpt_path, sd = _make_lora_checkpoint(tmp_path)
        original_to_k = [layer.to_k.weight.detach().clone() for layer in unet.layers]

        fuse_faceid_lora(unet, str(ckpt_path), lora_scale=0.5)

        for i, layer in enumerate(unet.layers):
            down = sd[f"{i}.to_k_lora.down.weight"]
            up = sd[f"{i}.to_k_lora.up.weight"]
            expected = original_to_k[i] + (up @ down) * 0.5
            assert torch.allclose(layer.to_k.weight, expected, atol=1e-6)

    def test_second_fusion_is_a_noop(self, tmp_path):
        unet = _FakeUNet()
        ckpt_path, _ = _make_lora_checkpoint(tmp_path)

        first_count = fuse_faceid_lora(unet, str(ckpt_path))
        after_first = [layer.to_v.weight.detach().clone() for layer in unet.layers]

        second_count = fuse_faceid_lora(unet, str(ckpt_path))
        after_second = [layer.to_v.weight.detach().clone() for layer in unet.layers]

        assert first_count == _N_LAYERS
        assert second_count == 0
        for a, b in zip(after_first, after_second):
            assert torch.equal(a, b)

    def test_shape_mismatch_raises(self, tmp_path):
        unet = _FakeUNet(n_layers=1, dim=_DIM)
        # up/down shaped for a *different* dim than the target linear -> matmul shape
        # (wrong_dim, wrong_dim) can never equal the real to_q weight's (dim, dim).
        wrong_dim = _DIM + 1
        sd = {
            "0.to_q_lora.down.weight": torch.randn(_RANK, wrong_dim),
            "0.to_q_lora.up.weight": torch.randn(wrong_dim, _RANK),
        }
        ckpt_path = tmp_path / "bad_shape.bin"
        torch.save({"ip_adapter": sd}, ckpt_path)

        with pytest.raises(RuntimeError, match="shape mismatch"):
            fuse_faceid_lora(unet, str(ckpt_path))

    def test_no_lora_keys_is_a_noop(self, tmp_path):
        """A non-FaceID (or already to_k_ip/to_v_ip-only) checkpoint has no '_lora.' keys —
        fuse_faceid_lora must recognise this and skip cleanly rather than fusing nothing
        per-layer via 140 no-op inner loops.
        """
        unet = _FakeUNet()
        original_to_q = [layer.to_q.weight.detach().clone() for layer in unet.layers]

        sd = {"0.to_k_ip.weight": torch.randn(_DIM, _DIM)}
        ckpt_path = tmp_path / "no_lora.bin"
        torch.save({"ip_adapter": sd}, ckpt_path)

        fused_count = fuse_faceid_lora(unet, str(ckpt_path))

        assert fused_count == 0
        for i, layer in enumerate(unet.layers):
            assert torch.equal(layer.to_q.weight, original_to_q[i])

    def test_partial_lora_only_fuses_layers_present_in_checkpoint(self, tmp_path):
        """Real FaceID checkpoints only carry LoRA on layers that also have IP cross-attn
        (odd indices / attn2) — even-index attn1 processors are paramless. fuse_faceid_lora
        must fuse exactly the layers present and leave the rest untouched.
        """
        unet = _FakeUNet(n_layers=4)
        ckpt_path, sd = _make_lora_checkpoint(tmp_path, n_layers=4, only_indices=[1, 3])
        original = [layer.to_q.weight.detach().clone() for layer in unet.layers]

        fused_count = fuse_faceid_lora(unet, str(ckpt_path))

        assert fused_count == 2
        for i, layer in enumerate(unet.layers):
            if i in (1, 3):
                down = sd[f"{i}.to_q_lora.down.weight"]
                up = sd[f"{i}.to_q_lora.up.weight"]
                expected = original[i] + (up @ down)
                assert torch.allclose(layer.to_q.weight, expected, atol=1e-6)
            else:
                assert torch.equal(layer.to_q.weight, original[i])
