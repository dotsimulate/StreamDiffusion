"""
Regression tests: the TensorRT ONNX optimize path must never call ModelProto.ByteSize().

Field crash (v0.4.1, SDXL UNet build): `MemoryError` at `Optimizer.fold_constants` ->
`onnx_graph.ByteSize()`. protobuf's upb backend implements ByteSize() by serializing the
whole message, so a ~5 GB fp16 UNet ModelProto transiently needs ~4.5x its size (~23 GB) and
fails on 32-64 GB hosts. The same gate was fixed once (dfae7d7, os.path.getsize) and silently
reverted by a cherry-pick (ef50c0d), so an AST guard locks the whole package down.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from streamdiffusion.acceleration.tensorrt.models import models as models_module
from streamdiffusion.acceleration.tensorrt.models.models import Optimizer, _onnx_tensor_bytes

SRC_PKG = Path(models_module.__file__).resolve().parents[3]  # .../streamdiffusion


def _small_model(weight_mb: float = 1.0) -> onnx.ModelProto:
    n = int(weight_mb * 1024 * 1024 / 4)
    w = numpy_helper.from_array(np.ones((n,), dtype=np.float32), name="w")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [n])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [n])
    graph = helper.make_graph([helper.make_node("Add", ["x", "w"], ["y"])], "g", [x], [y], initializer=[w])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class _ByteSizeBomb:
    """Delegates to a real ModelProto but raises like the field crash if ByteSize() is called."""

    def __init__(self, model):
        self._m = model

    def __getattr__(self, name):
        return getattr(self._m, name)

    def ByteSize(self):
        raise MemoryError()


class TestHelperAccuracy:
    def test_matches_bytesize_within_one_percent(self):
        model = _small_model(4.0)
        exact = model.ByteSize()
        approx = _onnx_tensor_bytes(model)
        assert approx <= exact
        assert approx >= exact * 0.99

    def test_typed_field_storage_is_counted(self):
        t = helper.make_tensor("c", TensorProto.FLOAT, [1000], [0.0] * 1000)  # float_data, no raw_data
        graph = helper.make_graph([], "g", [], [], initializer=[t])
        assert _onnx_tensor_bytes(helper.make_model(graph)) == 4000

    def test_external_data_length_counted_without_loading(self):
        t = TensorProto()
        t.name = "big"
        t.data_type = TensorProto.FLOAT16
        t.dims.extend([5 * 1024**3 // 2])
        t.data_location = TensorProto.EXTERNAL
        for k, v in (("location", "weights.pb"), ("offset", "0"), ("length", str(5 * 1024**3))):
            e = t.external_data.add()
            e.key, e.value = k, v
        graph = helper.make_graph([], "g", [], [], initializer=[t])
        assert _onnx_tensor_bytes(helper.make_model(graph)) == 5 * 1024**3

    def test_constant_node_and_subgraph_tensors_counted(self):
        const = numpy_helper.from_array(np.zeros((256,), dtype=np.float32), name="k")
        node = helper.make_node("Constant", [], ["out"], value=const)
        graph = helper.make_graph([node], "g", [], [])
        assert _onnx_tensor_bytes(helper.make_model(graph)) == 1024


class TestOptimizerNeverCallsByteSize:
    """Red before the fix: both calls raised MemoryError from the injected ByteSize bomb."""

    def _optimizer(self):
        opt = Optimizer(_small_model(), verbose=False)
        real_export = models_module.gs.export_onnx
        return opt, lambda g: _ByteSizeBomb(real_export(g))

    def test_fold_constants_does_not_serialize_the_model(self):
        opt, bomb_export = self._optimizer()
        with (
            patch.object(models_module.gs, "export_onnx", bomb_export),
            patch.object(models_module, "fold_constants", lambda m, **kw: m._m),
        ):
            opt.fold_constants()
            assert opt.graph is not None

    def test_infer_shapes_does_not_serialize_the_model(self):
        opt, bomb_export = self._optimizer()
        with (
            patch.object(models_module.gs, "export_onnx", bomb_export),
            patch.object(models_module.shape_inference, "infer_shapes", lambda m: m._m),
        ):
            opt.infer_shapes()
            assert opt.graph is not None


def _is_bytesize_reference(node: ast.AST) -> bool:
    """Any `x.ByteSize` attribute access (call or bound reference), or a string-keyed lookup of it
    such as getattr(x, "ByteSize") / hasattr / operator.methodcaller("ByteSize")."""
    if isinstance(node, ast.Attribute):
        return node.attr == "ByteSize"
    if isinstance(node, ast.Call):
        return any(isinstance(a, ast.Constant) and a.value == "ByteSize" for a in node.args)
    return False


def _bytesize_calls(root: Path):
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if _is_bytesize_reference(node):
                yield f"{path.relative_to(root)}:{node.lineno}"


def test_no_bytesize_calls_anywhere_in_streamdiffusion_package():
    offenders = list(_bytesize_calls(SRC_PKG))
    assert not offenders, (
        "ModelProto.ByteSize() serializes the whole proto under protobuf-upb (~4.5x transient, MemoryError "
        f"on SDXL). Use models.models._onnx_tensor_bytes instead: {offenders}"
    )


@pytest.mark.parametrize(
    "source",
    [
        "def f(m):\n    return m.ByteSize()\n",
        "def f(m):\n    size = m.ByteSize\n    return size()\n",
        "def f(m):\n    return getattr(m, 'ByteSize')()\n",
    ],
    ids=["call", "bound-reference", "getattr"],
)
def test_ast_guard_detects_planted_variants(tmp_path, source):
    (tmp_path / "bad.py").write_text(source, encoding="utf-8")
    assert len(list(_bytesize_calls(tmp_path))) == 1


def test_ast_guard_ignores_docstring_mentions(tmp_path):
    (tmp_path / "ok.py").write_text('def f():\n    """ByteSize() is banned."""\n', encoding="utf-8")
    assert list(_bytesize_calls(tmp_path)) == []


@pytest.mark.parametrize("fn", ["_onnx_tensor_bytes"])
def test_utilities_uses_the_helper(fn):
    from streamdiffusion.acceleration.tensorrt import utilities

    assert getattr(utilities, fn) is getattr(models_module, fn)
