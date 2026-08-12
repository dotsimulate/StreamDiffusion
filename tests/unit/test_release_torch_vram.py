"""Regression test for `_release_torch_vram`'s `ipadapter_ref` holder (builder.py).

Before this fix, `_release_torch_vram` only ever walked `pipe_ref` (== `stream.pipe`)
for `unet`/`vae`/`text_encoder`/`text_encoder_2` plus `builder.network` -- it never
saw the IP-Adapter/FaceID CLIP image encoder at all, because that module hangs off
`stream._ipadapter_module.ipadapter.image_encoder`, a completely separate object
graph not reachable from `stream.pipe`. For FaceID-non-plus configs that encoder is
dead weight during FP8 calibration (never called -- see ip_adapter.py's non-plus
branch) but stayed resident on the GPU through the VRAM-hungry ONNX quantize stage
regardless, at ~3.4+ GiB for ViT-bigG/14. This test exercises the real function with
a real (tiny) CUDA-resident nn.Module standing in for the image encoder and asserts
it is actually moved and restored -- a wiring bug here would leave the module on GPU
and this test silently passing if it only checked "no exception raised".
"""

import types

import pytest
import torch

from streamdiffusion.acceleration.tensorrt.builder import _release_torch_vram

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


def _fake_builder():
    return types.SimpleNamespace()


def _fake_ipadapter_ref(image_encoder):
    """Mirrors stream._ipadapter_module: an object with a `.ipadapter` attribute
    that in turn has `.image_encoder` (see modules/ipadapter_module.py:381 and
    diffusers_ipadapter/ip_adapter/ip_adapter.py:62)."""
    inner = types.SimpleNamespace(image_encoder=image_encoder)
    return types.SimpleNamespace(ipadapter=inner)


@requires_cuda
def test_ipadapter_image_encoder_moved_and_restored():
    encoder = torch.nn.Linear(4, 4).to("cuda")
    ipadapter_ref = _fake_ipadapter_ref(encoder)

    restore = _release_torch_vram(_fake_builder(), pipe_ref=None, ipadapter_ref=ipadapter_ref)

    assert next(ipadapter_ref.ipadapter.image_encoder.parameters()).device.type == "cpu"

    restore()

    assert next(ipadapter_ref.ipadapter.image_encoder.parameters()).device.type == "cuda"


def test_ipadapter_ref_without_installed_adapter_is_a_noop():
    """IPAdapterModule exists (stream._ipadapter_module is set) but .install() hasn't
    run yet / failed, so .ipadapter is still None -- must not raise."""
    ipadapter_ref = types.SimpleNamespace(ipadapter=None)

    restore = _release_torch_vram(_fake_builder(), pipe_ref=None, ipadapter_ref=ipadapter_ref)
    restore()


def test_no_ipadapter_ref_is_a_noop():
    restore = _release_torch_vram(_fake_builder(), pipe_ref=None, ipadapter_ref=None)
    restore()
