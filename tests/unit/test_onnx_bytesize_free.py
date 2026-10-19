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


def _foldable_model(weight_elems: int = 300_000, weight_math: bool = False) -> onnx.ModelProto:
    """y = x * w + n, where n comes from a shape computation on w that constant folding removes:
    Shape(w) -> Cast -> ReduceSum. Only w's shape is needed by that fold, never its values (the SDXL
    UNet case). weight_math=True uses x + w * c instead: folding w * c needs w's values."""
    w = numpy_helper.from_array(np.arange(weight_elems, dtype=np.float32), name="w")
    c = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name="c")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [weight_elems])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [weight_elems])
    head = (
        [helper.make_node("Mul", ["w", "c"], ["wc"]), helper.make_node("Add", ["x", "wc"], ["s"])]
        if weight_math
        else [helper.make_node("Mul", ["x", "w"], ["s"])]
    )
    nodes = head + [
        helper.make_node("Shape", ["w"], ["w_shape"]),
        helper.make_node("Cast", ["w_shape"], ["w_shape_f"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["w_shape_f"], ["n"], keepdims=0),
        helper.make_node("Add", ["s", "n"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "g", [x], [y], initializer=[w, c])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _graph_signature(model: onnx.ModelProto):
    g = model.graph
    nodes = [(n.op_type, tuple(n.input), tuple(n.output)) for n in g.node]
    inits = {t.name: (t.data_type, tuple(t.dims), numpy_helper.to_array(t).tobytes()) for t in g.initializer}
    return nodes, inits, [i.name for i in g.input], [o.name for o in g.output]


class TestLargeGraphFoldsWithoutWeights:
    """Room note 158: a >2GB graph is folded on a skeleton whose big constants are graph inputs, so
    Polygraphy's per-pass export / ONNX Runtime / shape inference never copy the weights."""

    def test_graph_constant_bytes_matches_the_export_based_helper(self):
        const = numpy_helper.from_array(np.zeros((256,), dtype=np.float32), name="k")
        model = _foldable_model(1000)
        model.graph.node.append(helper.make_node("Constant", [], ["unused_k"], value=const))
        opt = Optimizer(model)
        assert models_module._graph_constant_bytes(opt.graph) == _onnx_tensor_bytes(models_module.gs.export_onnx(opt.graph))

    def _fold(self, large: bool, weight_math: bool = False):
        opt = Optimizer(_foldable_model(weight_math=weight_math))
        opt.cleanup()
        seen = []
        real_fold = models_module.fold_constants

        def spy(model, **kw):
            seen.append(max((numpy_helper.to_array(t).nbytes for t in model.graph.initializer), default=0))
            return real_fold(model, **kw)

        with (
            patch.object(models_module, "_ONNX_2GB", 0 if large else models_module._ONNX_2GB),
            patch.object(models_module, "_SKELETON_CONSTANT_BYTES", 1024),
            patch.object(models_module, "fold_constants", spy),
        ):
            opt.fold_constants()
        opt.cleanup()
        return models_module.gs.export_onnx(opt.graph), seen

    def test_skeleton_fold_equals_the_normal_fold(self):
        normal, _ = self._fold(large=False)
        skeleton, _ = self._fold(large=True)
        assert _graph_signature(skeleton) == _graph_signature(normal)
        assert not any(n.op_type == "Shape" for n in skeleton.graph.node)  # the shape computation was folded

    def test_polygraphy_never_sees_the_big_weight(self):
        _, seen_normal = self._fold(large=False)
        _, seen_skeleton = self._fold(large=True)
        assert seen_normal[0] == 300_000 * 4
        assert seen_skeleton and max(seen_skeleton) <= 1024

    def test_weights_restored_as_constants_not_inputs(self):
        model, _ = self._fold(large=True)
        assert [i.name for i in model.graph.input] == ["x"]
        w = next(t for t in model.graph.initializer if t.name == "w")
        assert np.array_equal(numpy_helper.to_array(w), np.arange(300_000, dtype=np.float32))

    def test_infer_shapes_on_a_large_graph_does_not_export(self):
        opt = Optimizer(_foldable_model())

        def no_export(_g):
            raise AssertionError("infer_shapes exported a >2GB graph only to skip shape inference")

        with patch.object(models_module, "_ONNX_2GB", 0), patch.object(models_module.gs, "export_onnx", no_export):
            assert opt.infer_shapes() is None
        assert opt.graph is not None

    def test_a_fold_that_needs_weight_values_is_left_for_tensorrt(self):
        """Documented limit: w * c can't be folded without w's values, so the skeleton keeps the Mul
        (TensorRT folds constant layers at build time). The normal path folds it into a new constant."""
        normal, _ = self._fold(large=False, weight_math=True)
        skeleton, _ = self._fold(large=True, weight_math=True)
        assert not any(n.op_type == "Mul" for n in normal.graph.node)
        assert any(n.op_type == "Mul" and list(n.input) == ["w", "c"] for n in skeleton.graph.node)
        assert not any(n.op_type == "Shape" for n in skeleton.graph.node)
