"""EngineBuilder frees the PyTorch UNet weights after its ONNX export (builder.py).

The network passed to EngineBuilder is the pipeline's own UNet inside an export
wrapper, so the old `.to("cpu")` after the export kept a full host copy of the
weights (~5 GB fp16 SDXL) through optimize, the TRT build and the session. These
tests stop the build at the optimize stage and check what the export left behind.
"""

import types

import pytest
import torch

from streamdiffusion.acceleration.tensorrt import builder as builder_mod
from streamdiffusion.acceleration.tensorrt import compile_unet


class _StopAtOptimize(Exception):
    pass


class _Wrapper(torch.nn.Module):
    """Stands in for UnifiedExportWrapper: holds the pipeline's UNet as a submodule."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet


def _shared_unet():
    unet = torch.nn.Linear(8, 8)
    unet.register_buffer("stat", torch.ones(8))
    unet.config = types.SimpleNamespace(in_channels=4)
    pipe = types.SimpleNamespace(unet=unet)
    return pipe, _Wrapper(unet)


def _run_to_optimize(monkeypatch, tmp_path, network, cached_onnx=False, **build_kwargs):
    exported = []
    monkeypatch.setattr(builder_mod, "export_onnx", lambda model, **kw: exported.append(model))

    def _optimize(**kw):
        raise _StopAtOptimize

    monkeypatch.setattr(builder_mod, "optimize_onnx", _optimize)
    onnx_path = tmp_path / "unet.engine.onnx"
    if cached_onnx:
        onnx_path.write_bytes(b"x")
        monkeypatch.setattr(builder_mod, "_onnx_cache_valid", lambda path: True)

    b = builder_mod.EngineBuilder(model=types.SimpleNamespace(), network=network, device=torch.device("cpu"))
    with pytest.raises(_StopAtOptimize):
        b.build(
            str(onnx_path),
            str(tmp_path / "unet.engine.opt.onnx"),
            str(tmp_path / "unet.engine"),
            opt_batch_size=1,
            **build_kwargs,
        )
    return b, exported


def test_fresh_export_releases_shared_weights(monkeypatch, tmp_path):
    pipe, wrapper = _shared_unet()
    b, exported = _run_to_optimize(monkeypatch, tmp_path, wrapper, release_network_weights=True)

    assert exported == [wrapper]
    assert not hasattr(b, "network")
    # pipe.unet is the same module: its weights go too, its structure and config stay.
    assert all(p.is_meta for p in pipe.unet.parameters())
    assert all(t.is_meta for t in pipe.unet.buffers())
    assert pipe.unet.config.in_channels == 4
    assert pipe.unet.weight.shape == (8, 8)


def test_cached_onnx_also_releases_weights(monkeypatch, tmp_path):
    # Cached ONNX skips the export but compile_unet already moved the UNet to the GPU:
    # it would stay there through the TRT build and the session.
    pipe, wrapper = _shared_unet()
    b, exported = _run_to_optimize(monkeypatch, tmp_path, wrapper, cached_onnx=True, release_network_weights=True)

    assert exported == []
    assert not hasattr(b, "network")
    assert all(p.is_meta for p in pipe.unet.parameters())


def test_default_keeps_weights_on_cpu(monkeypatch, tmp_path):
    # VAE / ControlNet / safety checker / FP8 UNet builds keep the old behaviour.
    pipe, wrapper = _shared_unet()
    before = pipe.unet.weight.detach().clone()
    b, _ = _run_to_optimize(monkeypatch, tmp_path, wrapper)

    assert not hasattr(b, "network")
    assert pipe.unet.weight.device.type == "cpu"
    assert torch.equal(pipe.unet.weight.detach(), before)


def test_default_cached_onnx_leaves_network_alone(monkeypatch, tmp_path):
    pipe, wrapper = _shared_unet()
    b, _ = _run_to_optimize(monkeypatch, tmp_path, wrapper, cached_onnx=True)

    assert b.network is wrapper
    assert not any(p.is_meta for p in pipe.unet.parameters())


@pytest.mark.parametrize("fp8, expected", [(False, True), (True, False)])
def test_compile_unet_releases_only_without_fp8(monkeypatch, fp8, expected):
    seen = {}

    def _build(self, *args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(builder_mod.EngineBuilder, "build", _build)

    class _Unet:
        def to(self, *args, **kwargs):  # compile_unet moves it to CUDA first
            return self

    compile_unet(
        _Unet(),
        types.SimpleNamespace(),
        "u.onnx",
        "u.opt.onnx",
        "u.engine",
        opt_batch_size=1,
        engine_build_options={"fp8": fp8},
    )
    assert seen["release_network_weights"] is expected
