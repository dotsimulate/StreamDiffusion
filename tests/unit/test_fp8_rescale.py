"""
Regression tests for fp8-round-8's correction of modelopt's inverted INT8->FP8
scale-conversion factor.

modelopt/onnx/quantization/fp8.py's _int8_scale_to_fp8_scale computes
    np_fp8_scale = (np_scale * 448.0) / 127.0
where np_scale is the *INT8* scale (amax / 127) -- the correct FP8 scale is
amax / 448 = np_scale * 127 / 448. modelopt applies the reciprocal factor instead,
so every calibrated amax lands at 127**2/448 = 36.0 in E4M3 space instead of 448.0,
a uniform 12.4437x under-utilization that manifests as blur in high-dynamic-range
tensors (attention softmax outputs). See fp8_quantize.py::_rescale_fp8_qdq_scales
and the fp8-round-8 plan for the full derivation.

These fixtures build minimal QuantizeLinear/DequantizeLinear graphs by hand rather
than running real modelopt quantization (no GPU/modelopt/TensorRT dependency), with
scale values computed via the *same* (buggy) formula modelopt actually emits, so the
tests exercise the real correction math end to end.

Run with: pytest tests/unit/test_fp8_rescale.py -v
"""

import re

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


def _modelopt_scale(amax: float) -> float:
    """Reproduce modelopt's actual (inverted-factor) int8->fp8 conversion for a
    calibrated amax, so fixtures start from the same as-shipped numbers
    _rescale_fp8_qdq_scales is meant to correct (fp8.py:70-74)."""
    np_scale = amax / 127.0
    return float(np.float16((np_scale * 448.0) / 127.0))


def _build_qdq_model(weight_amax=10.0, scale_value=None, scale_name="s", weight_name="w"):
    """A weight initializer feeding a QuantizeLinear/DequantizeLinear pair that
    *share* one scale initializer -- mirrors modelopt's own processed_tensor guard
    (fp8.py:85-96) closely enough to exercise the dedupe requirement.

    scale_value defaults to modelopt's as-shipped conversion of weight_amax; pass an
    explicit value to plant other scale states (e.g. the uncalibrated default).
    """
    if scale_value is None:
        scale_value = _modelopt_scale(weight_amax)
    w = np.array([weight_amax, -weight_amax * 0.5, 0.1], dtype=np.float16)
    s = np.array([scale_value], dtype=np.float16)
    w_init = helper.make_tensor(weight_name, TensorProto.FLOAT16, [3], w.tobytes(), raw=True)
    s_init = helper.make_tensor(scale_name, TensorProto.FLOAT16, [], s.tobytes(), raw=True)
    q = helper.make_node("QuantizeLinear", inputs=[weight_name, scale_name], outputs=["q"], axis=0)
    dq = helper.make_node("DequantizeLinear", inputs=["q", scale_name], outputs=["out"], axis=0)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT16, [3])
    g = helper.make_graph([q, dq], "min", [], [out], initializer=[w_init, s_init])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])


def _make_qdq_onnx(path, **kwargs):
    onnx.save(_build_qdq_model(**kwargs), path)


def _realized_peak(onnx_path, weight_name="w", scale_name="s"):
    model = onnx.load(onnx_path)
    w = next(i for i in model.graph.initializer if i.name == weight_name)
    s = next(i for i in model.graph.initializer if i.name == scale_name)
    wa = onnx.numpy_helper.to_array(w).astype(np.float32)
    sa = onnx.numpy_helper.to_array(s).astype(np.float32).reshape(-1)
    return float(np.abs(wa).max() / sa[0])


def test_fp8_rescale_factor(tmp_path):
    """After correction, max|w|/scale must land at the full E4M3 range (448.0), not
    modelopt's as-shipped 36.0 -- and scale with headroom (224.0 at headroom=2.0)."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _rescale_fp8_qdq_scales

    p1 = str(tmp_path / "headroom1.fp8.onnx")
    _make_qdq_onnx(p1, weight_amax=10.0)
    peak_before = _realized_peak(p1)
    assert peak_before == pytest.approx(36.0, rel=0.02), (
        f"fixture sanity: as-shipped peak should be ~36.0, got {peak_before}"
    )

    report1 = _rescale_fp8_qdq_scales(p1, headroom=1.0)
    assert report1["scales_corrected"] == 1
    assert _realized_peak(p1) == pytest.approx(448.0, rel=0.02)

    p2 = str(tmp_path / "headroom2.fp8.onnx")
    _make_qdq_onnx(p2, weight_amax=10.0)
    _rescale_fp8_qdq_scales(p2, headroom=2.0)
    assert _realized_peak(p2) == pytest.approx(224.0, rel=0.02)


def test_fp8_rescale_applied_once_per_shared_scale(tmp_path):
    """A Q and its paired DQ reference the *same* scale initializer -- the rescale
    pass must dedupe by name and correct it once, not twice. Applying k twice would
    silently produce a ~12.4x error in the wrong direction (peak ~5575 instead of 448),
    which test_fp8_rescale_factor's tolerance would also catch, but this test pins the
    dedupe count directly so a regression here fails close to its actual cause."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _rescale_fp8_qdq_scales

    onnx_path = str(tmp_path / "shared.fp8.onnx")
    _make_qdq_onnx(onnx_path, weight_amax=10.0)

    report = _rescale_fp8_qdq_scales(onnx_path, headroom=1.0)

    assert report["scales_seen"] == 1
    assert report["scales_corrected"] == 1


def test_fp8_rescale_external_data_roundtrip(tmp_path):
    """A scale stored in external data must be patched in place at its existing
    offset/length (never re-serialized wholesale) -- reload after the pass must yield
    the corrected value, and the backing data file must not change size."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _rescale_fp8_qdq_scales

    onnx_path = tmp_path / "ext.fp8.onnx"
    model = _build_qdq_model(weight_amax=10.0)
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="ext.fp8.onnx_data",
        size_threshold=0,  # force even these tiny tensors external
    )
    data_path = tmp_path / "ext.fp8.onnx_data"
    assert data_path.exists(), "fixture sanity: tensors should have externalized"
    size_before = data_path.stat().st_size

    report = _rescale_fp8_qdq_scales(str(onnx_path), headroom=1.0)

    assert report["scales_corrected"] == 1
    assert data_path.stat().st_size == size_before, "in-place patch must not resize the external data file"
    assert _realized_peak(str(onnx_path)) == pytest.approx(448.0, rel=0.02)


def test_fp8_rescale_reports_uncalibrated(tmp_path):
    """A scale planted at exactly FP16(448/127) -- ORT's default when calibration
    recorded no range (INT8 amax == 127.0 exactly) -- must be counted and named,
    not silently corrected as if it were a real calibrated value."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _rescale_fp8_qdq_scales

    onnx_path = str(tmp_path / "uncalibrated.fp8.onnx")
    uncalibrated_scale = 448.0 / 127.0
    _make_qdq_onnx(
        onnx_path,
        weight_amax=1.0,
        scale_value=uncalibrated_scale,
        scale_name="attn2_never_calibrated_scale",
    )

    report = _rescale_fp8_qdq_scales(onnx_path, headroom=1.0)

    assert report["uncalibrated_count"] == 1
    assert "attn2_never_calibrated_scale" in report["uncalibrated_samples"]


def test_fp8_exclude_attention_patterns():
    """_ATTENTION_EXCLUDE_PATTERNS must catch the two attention BMMs (QK^T,
    softmax@V) -- attn{1,2}/MatMul and its _N siblings -- while sparing the
    projection MatMuls (to_q/to_k/to_v/to_out/to_k_ip/...), which are correctly
    calibrated and must stay FP8. modelopt matches these via re.match (anchored at
    start only, graph_utils.py:1018), so the pattern's trailing '$' is load-bearing:
    without it, '.*attn1.*' would also catch every projection MatMul."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _ATTENTION_EXCLUDE_PATTERNS

    assert len(_ATTENTION_EXCLUDE_PATTERNS) == 1
    pattern = _ATTENTION_EXCLUDE_PATTERNS[0]

    should_match = [
        "/unet/down_blocks.1/attentions.0/transformer_blocks.0/attn1/MatMul",
        "/unet/down_blocks.1/attentions.0/transformer_blocks.0/attn1/MatMul_1",
        "/unet/down_blocks.1/attentions.0/transformer_blocks.0/attn2/MatMul_2",
        "/unet/down_blocks.1/attentions.0/transformer_blocks.0/attn2/MatMul_3",
    ]
    should_not_match = [
        "/unet/down_blocks.1/attentions.0/transformer_blocks.0/attn1/to_q/MatMul",
        "/unet/down_blocks.1/attentions.0/transformer_blocks.0/attn2/to_out.0/MatMul",
        "/unet/down_blocks.1/attentions.0/transformer_blocks.0/attn2/to_k_ip/MatMul",
        "/unet/down_blocks.1/attentions.0/transformer_blocks.0/attn1/MatMul_1_extra",
    ]

    for name in should_match:
        assert re.match(pattern, name), f"expected match: {name}"
    for name in should_not_match:
        assert not re.match(pattern, name), f"expected no match: {name}"


def test_fp8_exclude_ipadapter_patterns():
    """fp8-round-9: _IPADAPTER_ACTIVATION_EXCLUDE_PATTERNS must catch only the three
    IP-Adapter activation tensors that fed the 420-uncalibrated-initializer defect
    (Mul_4 = k_ip, Transpose_4 = v_ip, Mul_5 = scale*ip_out) -- not the already-FP16
    to_k_ip/to_v_ip projections, not the shared-query Mul_3, and not attn1's
    unrelated Mul_4 (attn1 has no IP-Adapter branch, so a pattern that dropped the
    '/attn2/' anchor would over-match there). Anchored like
    _ATTENTION_EXCLUDE_PATTERNS: modelopt's expand_node_names_from_patterns uses
    re.match (start-anchored only), so the trailing '$' is load-bearing here too."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import (
        _IPADAPTER_ACTIVATION_EXCLUDE_PATTERNS,
    )

    assert len(_IPADAPTER_ACTIVATION_EXCLUDE_PATTERNS) == 1
    pattern = _IPADAPTER_ACTIVATION_EXCLUDE_PATTERNS[0]

    should_match = [
        "/unet/down_blocks.2/attentions.0/transformer_blocks.0/attn2/Mul_4",
        "/unet/down_blocks.2/attentions.0/transformer_blocks.0/attn2/Mul_5",
        "/unet/down_blocks.2/attentions.0/transformer_blocks.0/attn2/Transpose_4",
    ]
    should_not_match = [
        "/unet/down_blocks.2/attentions.0/transformer_blocks.0/attn2/to_k_ip/MatMul",
        "/unet/down_blocks.2/attentions.0/transformer_blocks.0/attn2/to_v_ip/MatMul",
        "/unet/down_blocks.2/attentions.0/transformer_blocks.0/attn2/Mul_3",
        "/unet/down_blocks.2/attentions.0/transformer_blocks.0/attn1/Mul_4",
        "/unet/down_blocks.2/attentions.0/transformer_blocks.0/attn2/Mul_4_output_0",
        "/unet/down_blocks.2/attentions.0/transformer_blocks.0/attn2/Mul_40",
    ]

    for name in should_match:
        assert re.match(pattern, name), f"expected match: {name}"
    for name in should_not_match:
        assert not re.match(pattern, name), f"expected no match: {name}"


def test_fp8_ipadapter_bmms_covered_by_attention_patterns_not_ipadapter_patterns():
    """Division of labour pinned by the plan: the two IP-Adapter BMMs
    (attn2/MatMul_2 = k_ip^T @ q, attn2/MatMul_3 = softmax @ v_ip) are already
    excluded by _ATTENTION_EXCLUDE_PATTERNS (same family as the text-branch BMMs)
    -- fp8_exclude_ipadapter must not duplicate them, or the two flags would
    silently overlap and double-count in nodes_to_exclude."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import (
        _ATTENTION_EXCLUDE_PATTERNS,
        _IPADAPTER_ACTIVATION_EXCLUDE_PATTERNS,
    )

    ipa_bmms = [
        "/unet/down_blocks.2/attentions.0/transformer_blocks.0/attn2/MatMul_2",
        "/unet/down_blocks.2/attentions.0/transformer_blocks.0/attn2/MatMul_3",
    ]
    for name in ipa_bmms:
        assert any(re.match(p, name) for p in _ATTENTION_EXCLUDE_PATTERNS), (
            f"expected _ATTENTION_EXCLUDE_PATTERNS to cover: {name}"
        )
        assert not any(re.match(p, name) for p in _IPADAPTER_ACTIVATION_EXCLUDE_PATTERNS), (
            f"expected _IPADAPTER_ACTIVATION_EXCLUDE_PATTERNS to NOT cover: {name}"
        )
