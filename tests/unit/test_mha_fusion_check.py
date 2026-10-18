"""
Regression tests for builder.py's fused-MHA check.

The "No fused MHA layers detected" warning fired on TAESD's vae_encoder/vae_decoder
engines, which have no attention at all, so zero fused-MHA kernels there is correct.
The layer names below are real engine-inspector output (TRT 10.16, Ada, 512x512).

Run with: pytest tests/unit/test_mha_fusion_check.py -v
"""

import onnx
from onnx import TensorProto, helper

from streamdiffusion.acceleration.tensorrt.builder import _check_mha_fusion, _count_attention_ops

TAESD_DECODER_LAYERS = [
    "PWN(PWN(PWN(/decoder/Constant_output_0 + ONNXTRT_Broadcast, PWN(/decoder/Div)), PWN(/decoder/Tanh)), "
    "/decoder/Constant_1_output_0 + ONNXTRT_Broadcast_1 + /decoder/Mul)",
    "/decoder/layers/layers.0/Cast",
    "Reformatting CopyNode for Input Tensor 0 to /decoder/layers/layers.0/Conv + /decoder/layers/layers.1/Relu",
    "/decoder/layers/layers.0/Conv + /decoder/layers/layers.1/Relu",
    "/decoder/layers/layers.2/conv/conv.0/Conv + /decoder/layers/layers.2/conv/conv.1/Relu",
]
UNET_LAYERS = [
    "Reformatting CopyNode for Input Tensor 0 to /unet/conv_in/Conv",
    "/unet/conv_in/Conv",
    "_gemm_mha_v2_myl25_6",
    "_gemm_mha_v2_myl25_11",
]


class _Log:
    def __init__(self):
        self.infos, self.warnings = [], []

    def info(self, msg):
        self.infos.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)


def _save_graph(path, with_softmax):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    op = helper.make_node("Softmax", ["x"], ["y"], axis=-1) if with_softmax else helper.make_node("Relu", ["x"], ["y"])
    onnx.save(helper.make_model(helper.make_graph([op], "g", [x], [y])), str(path))
    return str(path)


class TestCheckMhaFusion:
    def test_no_warning_for_model_without_attention(self):
        log = _Log()
        names = _check_mha_fusion(TAESD_DECODER_LAYERS, "fp16", log, attention_ops=0)
        assert names == []
        assert log.warnings == []

    def test_fused_unet_reports_kernels_without_warning(self):
        log = _Log()
        names = _check_mha_fusion(UNET_LAYERS, "fp16", log, attention_ops=70)
        assert names == ["_gemm_mha_v2_myl25_6", "_gemm_mha_v2_myl25_11"]
        assert log.warnings == []

    def test_warns_when_attention_present_but_not_fused(self):
        log = _Log()
        _check_mha_fusion(TAESD_DECODER_LAYERS, "fp16", log, attention_ops=1)
        assert len(log.warnings) == 1 and "No fused MHA layers detected" in log.warnings[0]

    def test_warns_when_attention_unknown(self):
        log = _Log()
        _check_mha_fusion(TAESD_DECODER_LAYERS, "fp16", log, attention_ops=None)
        assert len(log.warnings) == 1


class TestCountAttentionOps:
    def test_counts_softmax_nodes(self, tmp_path):
        assert _count_attention_ops(_save_graph(tmp_path / "a.onnx", with_softmax=True)) == 1

    def test_zero_without_softmax(self, tmp_path):
        assert _count_attention_ops(_save_graph(tmp_path / "b.onnx", with_softmax=False)) == 0

    def test_none_for_missing_file(self, tmp_path):
        assert _count_attention_ops(str(tmp_path / "missing.onnx")) is None
