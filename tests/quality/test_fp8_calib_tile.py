"""Regression: per-input-aware calibration row alignment for FP8 quantize.

Reproduces the kvo_cache_in dim-0=2 vs synthesized dim-0=1 split mismatch
hit by SDXL-Turbo + use_cached_attn + cfg_type=self configs, and the later
ipadapter_scale dim-0 drift (capture writes n_itr×num_ip_layers rows for a
graph input whose ONNX dim 0 is dynamic; quantize used to resolve that same
dim 0 to 1, inflating n_itr 70× and tiling every kvo/fio tensor into an
~290 GiB attempted allocation).
"""

import math

import numpy as np
import onnx
from onnx import TensorProto, helper


def _make_min_onnx(path):
    """Minimal 3-input ONNX: symbolic-batch 'sample', static-dim0=2
    'kvo_cache_in_0', and dynamic-dim0 'ipadapter_scale' (mirrors the "L_ip"
    axis at models/models.py:716)."""
    sample = helper.make_tensor_value_info("sample", TensorProto.FLOAT, ["2B", 4, 64, 64])
    kvo = helper.make_tensor_value_info("kvo_cache_in_0", TensorProto.FLOAT, [2, 4, "2B", 64, 64])
    ipa_scale = helper.make_tensor_value_info("ipadapter_scale", TensorProto.FLOAT, ["L_ip"])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, ["2B", 4, 64, 64])
    ident = helper.make_node("Identity", inputs=["sample"], outputs=["out"])
    g = helper.make_graph([ident], "min", [sample, kvo, ipa_scale], [out])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), path)


def test_per_input_tile_preserves_static_dim0(tmp_path):
    """Row alignment: sample stays at n_itr rows, kvo stays at 2×n_itr rows."""
    onnx_path = str(tmp_path / "min.onnx")
    _make_min_onnx(onnx_path)

    calib = {
        "sample": np.zeros((5, 4, 64, 64), dtype=np.float32),
        "kvo_cache_in_0": np.zeros((10, 4, 5, 64, 64), dtype=np.float32),
    }

    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _align_calibration_rows, _read_onnx_input_specs

    specs = _read_onnx_input_specs(onnx_path)
    out = _align_calibration_rows(calib, specs, num_ip_layers=0)

    # n_itr=5, resolved_dim0(sample)=1 → 5 rows; resolved_dim0(kvo)=2 → 10 rows
    assert out["sample"].shape == (5, 4, 64, 64)
    assert out["kvo_cache_in_0"].shape == (10, 4, 5, 64, 64)

    # Verify modelopt split math: n_itr chunks each of shape (resolved_dim0, ...)
    n_itr = 5
    sample_chunks = np.array_split(out["sample"], n_itr, axis=0)
    kvo_chunks = np.array_split(out["kvo_cache_in_0"], n_itr, axis=0)
    assert sample_chunks[0].shape[0] == 1
    assert kvo_chunks[0].shape[0] == 2  # static dim 0 must be preserved


def test_ipadapter_scale_does_not_inflate_n_itr(tmp_path):
    """THE regression: ipadapter_scale's dynamic 'L_ip' dim must not drive n_itr
    up. capture_calibration_data writes it at n_itr×num_ip_layers rows because
    the traced graph hardcodes Gather(scale_vec, idx=0..num_ip_layers-1); the
    quantize side must resolve the same per-chunk length or it elects a wildly
    inflated n_itr and tiles every other input to match.
    """
    onnx_path = str(tmp_path / "min.onnx")
    _make_min_onnx(onnx_path)

    num_ip_layers = 70
    n_itr = 8
    calib = {
        "sample": np.zeros((n_itr, 4, 64, 64), dtype=np.float32),
        "kvo_cache_in_0": np.zeros((n_itr * 2, 4, 5, 64, 64), dtype=np.float32),
        "ipadapter_scale": np.ones((n_itr * num_ip_layers,), dtype=np.float32),
    }

    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _align_calibration_rows, _read_onnx_input_specs

    specs = _read_onnx_input_specs(onnx_path)
    out = _align_calibration_rows(calib, specs, num_ip_layers=num_ip_layers)

    # n_itr must stay 8 (not 560=8×70) — every input already matches its target,
    # so _align_calibration_rows must return each array untouched (by identity).
    assert out["sample"].shape[0] == 8
    assert out["kvo_cache_in_0"].shape[0] == 16
    assert out["ipadapter_scale"].shape[0] == 560
    for k in calib:
        assert out[k] is calib[k], f"'{k}' was needlessly tiled/copied"

    chunks = np.array_split(out["ipadapter_scale"], n_itr, axis=0)
    assert all(c.shape[0] == num_ip_layers for c in chunks)


## FP8 Round 11 -- the two tests below document a *different* face of the same
## ipadapter_scale divisibility confusion the test above pins for
## _align_calibration_rows: here it's _select_calibration_calls's own
## max_multiplier inference (fp8_quantize.py's own docstring/module comment
## flags this as "not firing today" only because capture_calibration_data
## synthesizes ipadapter_scale *after* calling _select_calibration_calls --
## verified by direct read, not testable end-to-end without a live pipeline.
## These document the hazard rather than lock down a "fix"; if either ever
## starts failing because the two statements' order changed, that is a real
## regression to investigate, not an assertion to update.


def test_select_calibration_calls_ipadapter_scale_present_collapses_to_one_call():
    """If ipadapter_scale (shape num_calls*70) were ever passed into
    _select_calibration_calls, its multiplier (70) would dominate
    max_multiplier and collapse calls_budget to max(1, 8 // 70) == 1 -- a
    single-call collapse that would starve calibration to one prompt entirely.
    See the module note above for why this does not fire in production."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _select_calibration_calls

    num_calls = 8
    distinct_timesteps = [699, 579, 459, 339, 219, 99, 979, 819]
    num_ip_layers = 70
    calib = {
        "timestep": np.array(distinct_timesteps, dtype=np.float16).reshape(num_calls, 1),
        "ipadapter_scale": np.ones((num_calls * num_ip_layers,), dtype=np.float32),
    }

    selected = _select_calibration_calls(calib, max_rows=8)

    assert len(selected) == 1, (
        "documents the ipadapter_scale multiplier trap collapsing calls_budget "
        "to 1 -- see this test's docstring for why it does not fire in production"
    )


def test_select_calibration_calls_without_ipadapter_scale_yields_full_budget():
    """Companion to the collapse test above: the real hook-captured key set
    (timestep + sample + encoder_hidden_states -- never ipadapter_scale, which
    is synthesized afterward) must yield the full 8-call budget, not the
    collapsed 1-call budget the trap above would produce if the statement
    ordering ever changed."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _select_calibration_calls

    num_calls = 8
    distinct_timesteps = [699, 579, 459, 339, 219, 99, 979, 819]
    calib = {
        "timestep": np.array(distinct_timesteps, dtype=np.float16).reshape(num_calls, 1),
        "sample": np.arange(num_calls, dtype=np.float32).reshape(num_calls, 1),
        "encoder_hidden_states": np.arange(num_calls, dtype=np.float32).reshape(num_calls, 1),
    }

    selected = _select_calibration_calls(calib, max_rows=8)
    assert len(selected) == 8


def _make_ehs_onnx(path, ehs_seq=81):
    """Minimal ONNX with an encoder_hidden_states input declared at a static
    seq_len (81 = 77 text tokens + 4 IP-Adapter image tokens, mirroring
    UnifiedExportWrapper's concatenation) plus an unrelated already-matching
    static-dim input, so _reconcile_calib_to_onnx_dims's "only touch what's
    declared" behavior is exercised alongside the IP-Adapter special case
    (fp8-round-9)."""
    ehs = helper.make_tensor_value_info("encoder_hidden_states", TensorProto.FLOAT, ["B", ehs_seq, 2048])
    fi = helper.make_tensor_value_info("fi_strength", TensorProto.FLOAT, ["B", 3])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, ["B", ehs_seq, 2048])
    ident = helper.make_node("Identity", inputs=["encoder_hidden_states"], outputs=["out"])
    g = helper.make_graph([ident], "min_ehs", [ehs, fi], [out])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), path)


def test_reconcile_fills_ipadapter_token_region_from_real_tokens(tmp_path):
    """fp8-round-9: undersized encoder_hidden_states (77 captured text tokens vs.
    81 ONNX-declared) concatenates real IP-Adapter projection tokens onto the
    missing region instead of zero-padding — to_k_ip/to_v_ip are bias-free
    nn.Linear layers, so a zero-padded region starves that branch's FP8
    calibration entirely (the defect this fix corrects)."""
    onnx_path = str(tmp_path / "ehs.onnx")
    _make_ehs_onnx(onnx_path, ehs_seq=81)

    from streamdiffusion.acceleration.tensorrt.fp8_quantize import (
        _read_onnx_input_specs,
        _reconcile_calib_to_onnx_dims,
    )

    specs = _read_onnx_input_specs(onnx_path)
    n_rows = 3
    calib = {
        "encoder_hidden_states": np.random.default_rng(0).normal(size=(n_rows, 77, 2048)).astype(np.float32),
    }
    tokens = np.full((1, 4, 2048), 5.0, dtype=np.float32)  # nonzero stub tokens

    out = _reconcile_calib_to_onnx_dims(calib, specs, ipadapter_tokens=tokens, num_ip_layers=70)

    assert out["encoder_hidden_states"].shape == (n_rows, 81, 2048)
    appended = out["encoder_hidden_states"][:, 77:, :]
    assert np.abs(appended).max() > 0
    np.testing.assert_allclose(appended, np.broadcast_to(tokens, (n_rows, 4, 2048)))
    # Original 77-token region is untouched, and the input dict itself is not
    # mutated in place (capture_calibration_data reassigns calib_data from the
    # return value; a stray in-place mutation would silently double-apply).
    np.testing.assert_array_equal(out["encoder_hidden_states"][:, :77, :], calib["encoder_hidden_states"])
    assert calib["encoder_hidden_states"].shape == (n_rows, 77, 2048)


def test_reconcile_tiles_multi_image_tokens_round_robin(tmp_path):
    """fp8-round-9.1: N>1 ipadapter_tokens (multiple `fp8_calibration_style_image`
    files) must tile round-robin across the calibration rows and trim to the
    row count, not go through the N==1 `np.broadcast_to` path -- which only
    accepts source dim 1 or equal to the target and raises ValueError on any
    other N (unreachable before fp8-round-9.1, since every prior token source
    was N==1: the zeros-surrogate or a single cached embedding). 2 images
    across 8 rows must land as [img0, img1, img0, img1, ...] -- an
    interleave, not img0 repeated 4x then img1 repeated 4x -- so both
    images' activation ranges land in the same amax statistic throughout the
    schedule rather than only the first half."""
    onnx_path = str(tmp_path / "ehs.onnx")
    _make_ehs_onnx(onnx_path, ehs_seq=81)

    from streamdiffusion.acceleration.tensorrt.fp8_quantize import (
        _read_onnx_input_specs,
        _reconcile_calib_to_onnx_dims,
    )

    specs = _read_onnx_input_specs(onnx_path)
    n_rows = 8
    calib = {
        "encoder_hidden_states": np.random.default_rng(0).normal(size=(n_rows, 77, 2048)).astype(np.float32),
    }
    # Two distinct, easily-distinguished per-image token blocks.
    img0 = np.full((1, 4, 2048), 2.0, dtype=np.float32)
    img1 = np.full((1, 4, 2048), 5.0, dtype=np.float32)
    tokens = np.concatenate([img0, img1], axis=0)  # shape (2, 4, 2048)

    out = _reconcile_calib_to_onnx_dims(calib, specs, ipadapter_tokens=tokens, num_ip_layers=70)

    assert out["encoder_hidden_states"].shape == (n_rows, 81, 2048)
    appended = out["encoder_hidden_states"][:, 77:, :]
    assert np.abs(appended).max() > 0

    for row in range(n_rows):
        expected = img0[0] if row % 2 == 0 else img1[0]
        np.testing.assert_array_equal(
            appended[row], expected, err_msg=f"row {row} did not get the expected round-robin image token"
        )

    # Original 77-token region untouched, no in-place mutation of the input.
    np.testing.assert_array_equal(out["encoder_hidden_states"][:, :77, :], calib["encoder_hidden_states"])


def test_reconcile_falls_back_to_zero_pad_and_warns_without_tokens(tmp_path, caplog):
    """No ipadapter_tokens given -> the prior zero-pad behavior still applies
    (kvo/fio contract unchanged), plus the new zero-region warning fires:
    num_ip_layers>0 with an all-zero appended region is exactly the
    precondition that produced the original 420-uncalibrated-initializer
    defect, and must never be silent again."""
    onnx_path = str(tmp_path / "ehs.onnx")
    _make_ehs_onnx(onnx_path, ehs_seq=81)

    from streamdiffusion.acceleration.tensorrt.fp8_quantize import (
        _read_onnx_input_specs,
        _reconcile_calib_to_onnx_dims,
    )

    specs = _read_onnx_input_specs(onnx_path)
    calib = {"encoder_hidden_states": np.ones((2, 77, 2048), dtype=np.float32)}

    with caplog.at_level("WARNING", logger="streamdiffusion.acceleration.tensorrt.fp8_quantize"):
        out = _reconcile_calib_to_onnx_dims(calib, specs, ipadapter_tokens=None, num_ip_layers=70)

    assert out["encoder_hidden_states"].shape == (2, 81, 2048)
    appended = out["encoder_hidden_states"][:, 77:, :]
    assert np.abs(appended).max() == 0.0
    assert any("identically zero" in r.message for r in caplog.records)


def test_reconcile_no_warning_when_no_ipadapter_layers(tmp_path, caplog):
    """num_ip_layers=0 (no IP-Adapter installed) must not warn about a zero
    token region — the defect this guards against is IPA-specific, and every
    other feature (kvo/fio/controlnet) legitimately zero-pads without it."""
    onnx_path = str(tmp_path / "ehs.onnx")
    _make_ehs_onnx(onnx_path, ehs_seq=81)

    from streamdiffusion.acceleration.tensorrt.fp8_quantize import (
        _read_onnx_input_specs,
        _reconcile_calib_to_onnx_dims,
    )

    specs = _read_onnx_input_specs(onnx_path)
    calib = {"encoder_hidden_states": np.ones((2, 77, 2048), dtype=np.float32)}

    with caplog.at_level("WARNING", logger="streamdiffusion.acceleration.tensorrt.fp8_quantize"):
        _reconcile_calib_to_onnx_dims(calib, specs, ipadapter_tokens=None, num_ip_layers=0)

    assert not any("identically zero" in r.message for r in caplog.records)


def test_reconcile_trims_oversized_and_leaves_matching_tensors_untouched(tmp_path):
    """Oversized (85 -> 81) still trims — unaffected by the IP-Adapter special
    case, which only applies to the undersized branch — and an unrelated
    input that already matches its declared dims is left alone (same object,
    not a needless copy)."""
    onnx_path = str(tmp_path / "ehs.onnx")
    _make_ehs_onnx(onnx_path, ehs_seq=81)

    from streamdiffusion.acceleration.tensorrt.fp8_quantize import (
        _read_onnx_input_specs,
        _reconcile_calib_to_onnx_dims,
    )

    specs = _read_onnx_input_specs(onnx_path)
    calib = {
        "encoder_hidden_states": np.arange(2 * 85 * 2048, dtype=np.float32).reshape(2, 85, 2048),
        "fi_strength": np.zeros((2, 3), dtype=np.float32),
    }
    tokens = np.full((1, 4, 2048), 9.0, dtype=np.float32)

    out = _reconcile_calib_to_onnx_dims(calib, specs, ipadapter_tokens=tokens, num_ip_layers=70)

    assert out["encoder_hidden_states"].shape == (2, 81, 2048)
    np.testing.assert_array_equal(out["encoder_hidden_states"], calib["encoder_hidden_states"][:, :81, :])
    assert out["fi_strength"] is calib["fi_strength"]


def test_naive_max_rows_tile_would_break(tmp_path):
    """Confirms the OLD naïve tile produces the 'Got 1 Expected 2' symptom."""
    onnx_path = str(tmp_path / "min.onnx")
    _make_min_onnx(onnx_path)

    calib = {
        "sample": np.zeros((5, 4, 64, 64), dtype=np.float32),
        "kvo_cache_in_0": np.zeros((10, 4, 5, 64, 64), dtype=np.float32),
    }

    # Reproduce the buggy logic
    _max_rows = max(a.shape[0] for a in calib.values())
    for k, a in list(calib.items()):
        if a.shape[0] < _max_rows:
            calib[k] = np.tile(a, (math.ceil(_max_rows / a.shape[0]),) + (1,) * (a.ndim - 1))[:_max_rows]

    # modelopt: n_itr = sample.shape[0] / symbolic_dim0(1) = 10
    # splits kvo into 10 chunks → each has shape[0]=1 → ORT rejects (expected 2)
    n_itr_bad = calib["sample"].shape[0]  # 10 (doubled by naïve tile)
    kvo_chunk = np.array_split(calib["kvo_cache_in_0"], n_itr_bad, axis=0)[0]
    assert kvo_chunk.shape[0] == 1  # this is the "Got 1 Expected 2" symptom


## NOTE: the four tests that used to live here (test_naive_linspace_aliases_timesteps,
## test_calibration_row_selection_covers_all_timesteps,
## test_calibration_row_selection_two_rows_per_step,
## test_calibration_row_selection_preserves_cross_key_alignment) were removed.
## They covered a 4-step-schedule-specific stride fix that Step 1 of the
## fp8-round-5-handoff plan reverted in favor of isolating the
## enable_gemv_detection_for_trt variable. Step 2e below replaces row
## selection wholesale with a band-based, call-aligned, stratified-over-
## timesteps scheme (see _select_calibration_rows).


def test_select_calibration_rows_covers_all_timesteps():
    """Row selection must represent every distinct calibration timestep before
    doubling any of them, unlike a naive row-index linspace stride, which can
    alias onto the same handful of timestep phases across many prompts (the
    original round-5 defect Step 1 of the plan isolated)."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _select_calibration_rows

    distinct_timesteps = [699, 579, 459, 339, 219, 99, 979, 819]  # 8 distinct t values
    num_prompts = 4
    timestep_seq = distinct_timesteps * num_prompts  # prompt-major call order
    num_calls = len(timestep_seq)

    calib = {
        "timestep": np.array(timestep_seq, dtype=np.float16).reshape(num_calls, 1),
        "sample": np.arange(num_calls, dtype=np.float32).reshape(num_calls, 1, 1, 1),
    }

    out = _select_calibration_rows(calib, max_rows=8)

    assert out["timestep"].shape[0] == 8
    selected_values = sorted(set(out["timestep"].reshape(-1).tolist()))
    assert selected_values == sorted(distinct_timesteps), (
        "every distinct calibrated timestep must be represented in an 8-row budget "
        "when exactly 8 distinct timesteps were captured"
    )


def test_select_calibration_rows_preserves_cross_key_alignment_under_cfg():
    """CFG-doubled keys (sample, encoder_hidden_states) must stay aligned with
    the 1-row-per-call 'timestep' key: a selected call's cond+uncond pair in a
    doubled key must sit next to that same call's timestep row."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _select_calibration_rows

    distinct_timesteps = [699, 579, 459, 339]
    num_prompts = 6
    timestep_seq = distinct_timesteps * num_prompts
    num_calls = len(timestep_seq)  # 24

    timestep_arr = np.array(timestep_seq, dtype=np.float16).reshape(num_calls, 1)
    # 2 rows/call (CFG cond+uncond), each tagged call_idx*10 + {0: uncond, 1: cond}
    # so the pairing can be recovered from data content, not just algorithm knowledge.
    sample_arr = np.zeros((num_calls * 2, 1), dtype=np.float32)
    for c in range(num_calls):
        sample_arr[2 * c, 0] = c * 10 + 0
        sample_arr[2 * c + 1, 0] = c * 10 + 1

    calib = {"timestep": timestep_arr, "sample": sample_arr}
    out = _select_calibration_rows(calib, max_rows=8)

    # max_multiplier=2 -> calls_budget = 8 // 2 = 4 calls selected.
    assert out["timestep"].shape[0] == 4
    assert out["sample"].shape[0] == 8
    assert sorted(out["timestep"].reshape(-1).tolist()) == sorted(distinct_timesteps)

    for i in range(4):
        uncond_tag, cond_tag = out["sample"][2 * i, 0], out["sample"][2 * i + 1, 0]
        call_idx = int(uncond_tag // 10)
        assert int(cond_tag // 10) == call_idx, "cond/uncond pair split apart across selection"
        assert out["timestep"][i, 0] == timestep_seq[call_idx], (
            "sample rows for a call must stay aligned with that call's timestep row"
        )


def test_select_calibration_rows_no_downsample_within_budget():
    """When captured calls already fit the budget, nothing is dropped."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _select_calibration_rows

    calib = {
        "timestep": np.array([699, 579, 459], dtype=np.float16).reshape(3, 1),
        "sample": np.arange(6, dtype=np.float32).reshape(6, 1),
    }
    out = _select_calibration_rows(calib, max_rows=8)
    assert out["timestep"].shape[0] == 3
    assert out["sample"].shape[0] == 6


## FP8 Round 11 -- calibration-diversity stagger fix. _select_calibration_calls
## used bucket[round_idx] for every distinct timestep in a round; whenever
## calls_budget matched the distinct-timestep count (the common case: 8
## timesteps, budget 8), only round 0 ever ran, and round 0 = occurrence 0 of
## every timestep = the first pipe() batch, for all 8 selected calls. The tests
## below pin the position-staggered replacement directly (not through
## _select_calibration_rows, which only exposes the row-expanded result).


def test_select_calibration_calls_stagger_spans_distinct_batches():
    """The red-capable pin for FP8 Round 11 Item 2: with calls_budget matching
    the distinct-timestep count, the pre-fix code selected occurrence 0 of
    every timestep -- all 8 calls from the same (first) prompt batch. The
    stagger fix must instead spread the 8 selected calls across as many
    distinct batches as the capture holds occurrences for."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _select_calibration_calls

    distinct_timesteps = [699, 579, 459, 339, 219, 99, 979, 819]  # 8 distinct
    num_batches = 8
    timestep_seq = distinct_timesteps * num_batches  # prompt-major call order,
    # matching real capture: one pipe() batch walks every distinct timestep
    # once, in the same order, before the next batch starts.
    num_calls = len(timestep_seq)

    calib = {"timestep": np.array(timestep_seq, dtype=np.float16).reshape(num_calls, 1)}
    selected = _select_calibration_calls(calib, max_rows=8)

    assert len(selected) == 8

    def _batch_of(call_idx):
        return call_idx // len(distinct_timesteps)

    batches = {_batch_of(int(c)) for c in selected}
    assert len(batches) == 8, (
        f"expected all 8 batches represented, got batches={sorted(batches)} "
        f"from selected calls={sorted(int(c) for c in selected)}"
    )

    # Every distinct timestep still represented exactly once before any repeat.
    selected_timesteps = sorted(float(timestep_seq[int(c)]) for c in selected)
    assert selected_timesteps == sorted(float(t) for t in distinct_timesteps)


def test_select_calibration_calls_short_bucket_falls_back_without_dropping_timestep():
    """A timestep with fewer occurrences than the round count needs (a "short
    bucket") must still contribute at least once, and the stagger must never
    select a duplicate call index or raise -- it falls back to the earliest
    unused occurrence in that bucket instead of re-selecting one already taken,
    and skips (not crashes) once a bucket is genuinely exhausted."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _select_calibration_calls

    # 500 and 300 occur 3x each; 400 occurs only once -- short relative to the
    # multiple rounds a 6-call budget over these 7 calls will need.
    timestep_seq = [500, 400, 300, 500, 300, 500, 300]
    calib = {"timestep": np.array(timestep_seq, dtype=np.float16).reshape(len(timestep_seq), 1)}

    selected = _select_calibration_calls(calib, max_rows=6)

    selected_list = [int(c) for c in selected]
    assert len(selected_list) == len(set(selected_list)), "duplicate call index selected"
    assert len(selected_list) <= 6
    selected_timesteps = {timestep_seq[c] for c in selected_list}
    assert selected_timesteps == {500, 400, 300}, (
        "every distinct timestep must be represented at least once, including the short (single-occurrence) bucket"
    )


def test_pool_for_layer_skips_out_of_window_call_without_crashing():
    """FP8 Round 11 Item 2/3: a selected call whose index isn't in `records`
    (fell outside the K/V/FI recorder's kept window) must be skipped, not
    raise, and reported into `missed_calls` -- not silently lost with no
    visibility, per the WARNING _pool_for_layer's caller now logs."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _pool_for_layer

    records = {
        0: np.zeros((1, 4, 8), dtype=np.float32),
        2: np.ones((1, 4, 8), dtype=np.float32),
        # call 1 is "out of window" -- deliberately no entry
    }
    missed: set = set()

    pool = _pool_for_layer(records, [0, 1, 2], missed)

    assert len(pool) == 2  # calls 0 and 2 each contributed one row; call 1 skipped
    assert missed == {1}


def test_pool_for_layer_missed_calls_none_is_a_no_op():
    """missed_calls is optional -- omitting it must not raise, matching a
    caller that doesn't care about out-of-window visibility."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _pool_for_layer

    records = {5: np.zeros((2, 4, 8), dtype=np.float32)}
    pool = _pool_for_layer(records, [5, 6])
    assert len(pool) == 2


def test_fill_kvo_from_pool_cyclic_shift_with_mixed_provenance_pool():
    """Verified-safe claim (FP8 Round 11): _fill_kvo_from_pool indexes its pool
    cyclically by position, provenance-agnostic. Under the stagger fix the pool
    now holds entries from scattered, non-adjacent calls (different prompt
    batches) instead of a contiguous early-prefix run, including different real
    captured seq lengths per entry. The cyclic src=(m+1)%pool_len tiling must
    still produce a well-formed, correctly-shaped, real (non-degenerate)
    array."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _fill_kvo_from_pool

    # Mixed provenance: three pool entries with different real seq lengths,
    # standing in for K/V rows recorded from three different prompt batches.
    k_pool = [
        np.full((3, 8), 1.0, dtype=np.float32),
        np.full((5, 8), 2.0, dtype=np.float32),
        np.full((2, 8), 3.0, dtype=np.float32),
    ]
    v_pool = [
        np.full((3, 8), 10.0, dtype=np.float32),
        np.full((5, 8), 20.0, dtype=np.float32),
        np.full((2, 8), 30.0, dtype=np.float32),
    ]

    arr_shape = [2 * 4, 8, 8]  # n_itr=4 -> dim0 = 2*n_itr (K/V pair); seq_t=hidden_t=8
    out = _fill_kvo_from_pool(k_pool, v_pool, arr_shape, np.float32)

    assert out.shape == tuple(arr_shape)
    assert not np.isnan(out).any()
    # Frame-shift-by-one, cyclic: m=0 draws pool[1], not pool[0] (see
    # _fill_kvo_from_pool's docstring on the deliberate neighbour-frame shift).
    assert out[0, 0, 0] == 2.0  # K, m=0 -> src=(0+1)%3=1 -> k_pool[1] filled with 2.0
    assert out[1, 0, 0] == 20.0  # V, m=0 -> same src -> v_pool[1] filled with 20.0


## FP8 Round 14 -- predictive call recording. _MAX_CALIB_RECORD_CALLS (64) was
## raised in Round 11 to hold a full 8-distinct-timestep x 8-batch stagger
## pass, but the companion _MAX_CALIB_RECORD_BYTES (4 GiB) trips first at
## ~33 of the 64 needed calls, silently dropping calls 36/45/54/63 and
## halving kvo/fio calibration diversity (measured on disk: 8 distinct K
## sources pre-Round-11, 4 post-Round-11, in both calv7 engines). The fix
## inverts the recorder: predict which calls _select_calibration_calls will
## keep *before* the capture loop runs, and gate the hooks on membership in
## that predicted set instead of a call-count prefix.


def test_predict_selected_calls_pins_production_case():
    """Pins _predict_selected_calls's output against the real 640x384
    verification build's calib_data.meta.json ("selected_calls":
    [0, 9, 18, 27, 36, 45, 54, 63], for 32 prompts x 8 steps). A future
    change to the stagger in _select_calibration_calls that silently changes
    what gets predicted must fail here first."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _predict_selected_calls

    predicted = _predict_selected_calls(schedule_len=8, num_batches=32, max_rows=8)

    assert [int(c) for c in predicted] == [0, 9, 18, 27, 36, 45, 54, 63]


def test_predict_selected_calls_is_superset_under_budget_shrink():
    """The real post-hoc call inside capture_calibration_data passes the full
    captured calib_data, whose CFG-doubled keys (e.g. `sample` at 2 rows/call
    under guidance_scale > 1) shrink _select_calibration_calls's effective
    calls_budget below _MAX_CALIB_ROWS -- see _predict_selected_calls's own
    superset-guarantee docstring. That shrink must only ever truncate the
    max-budget prediction to a prefix, never diverge from it."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import (
        _MAX_CALIB_ROWS,
        _predict_selected_calls,
        _select_calibration_calls,
    )

    schedule_len, num_batches = 8, 32
    predicted = frozenset(int(c) for c in _predict_selected_calls(schedule_len, num_batches, _MAX_CALIB_ROWS))

    timestep_seq = np.tile(np.arange(schedule_len, dtype=np.float32)[::-1], num_batches)
    num_calls = timestep_seq.shape[0]
    calib_data = {
        "timestep": timestep_seq.reshape(num_calls, 1),
        "sample": np.zeros((num_calls * 2, 1), dtype=np.float32),  # CFG-doubled key -> 2x multiplier
    }
    actual = _select_calibration_calls(calib_data, max_rows=_MAX_CALIB_ROWS)
    actual_set = {int(c) for c in actual}

    assert actual_set.issubset(predicted)
    assert len(actual_set) < len(predicted), (
        "the 2x-multiplier key must actually shrink the budget for this test to "
        "exercise the truncation path, not just the trivial equal-budget case"
    )


def test_fill_kvo_from_pool_eight_recorded_calls_yield_eight_distinct_sources():
    """The direct regression this whole round fixes: when all 8 predicted
    calls have a recording, _fill_kvo_from_pool's 8 n_itr chunks must draw
    from 8 distinct sources -- matching the pre-Round-11 (calv6) on-disk
    measurement. See the companion 4-record test below for the regressed
    (calv7) behaviour this replaces."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _fill_kvo_from_pool, _pool_for_layer

    selected_calls = [0, 9, 18, 27, 36, 45, 54, 63]
    # One distinguishable marker value per call, single sub-row each.
    records = {c: np.full((1, 2, 4), float(i), dtype=np.float32) for i, c in enumerate(selected_calls)}
    missed: set = set()

    k_pool = _pool_for_layer(records, selected_calls, missed)
    v_pool = _pool_for_layer(records, selected_calls, missed)

    assert missed == set(), "all 8 predicted calls have a recording -- nothing should be missed"
    assert len(k_pool) == 8

    n_itr = 8
    arr_shape = [2 * n_itr, 2, 4]
    out = _fill_kvo_from_pool(k_pool, v_pool, arr_shape, np.float32)

    distinct_k_sources = {float(out[m * 2, 0, 0]) for m in range(n_itr)}
    assert len(distinct_k_sources) == 8


def test_fill_kvo_from_pool_four_recorded_calls_yield_four_distinct_sources():
    """Companion regression-documentation case: only the first 4 of 8
    predicted calls survive recording (calls 36/45/54/63 fell outside the
    byte-capped recorder window, the exact calv7 defect) -- _pool_for_layer
    reports them missed, and _fill_kvo_from_pool's cyclic src=(m+1)%pool_len
    reuse collapses the 8 n_itr chunks onto only 4 distinct sources instead
    of 8. This is the shipped calv7 behaviour this plan's fix (predictive
    recording) corrects -- kept as documentation of what regressed."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _fill_kvo_from_pool, _pool_for_layer

    selected_calls = [0, 9, 18, 27, 36, 45, 54, 63]
    recorded_calls = selected_calls[:4]  # 36/45/54/63 never recorded
    records = {c: np.full((1, 2, 4), float(i), dtype=np.float32) for i, c in enumerate(recorded_calls)}
    missed: set = set()

    k_pool = _pool_for_layer(records, selected_calls, missed)
    v_pool = _pool_for_layer(records, selected_calls, missed)

    assert missed == {36, 45, 54, 63}
    assert len(k_pool) == 4

    n_itr = 8
    arr_shape = [2 * n_itr, 2, 4]
    out = _fill_kvo_from_pool(k_pool, v_pool, arr_shape, np.float32)

    distinct_k_sources = {float(out[m * 2, 0, 0]) for m in range(n_itr)}
    assert len(distinct_k_sources) == 4


def test_recorder_admits_call_falls_back_to_prefix_after_batch_failure():
    """FP8 Round 14 batch-failure fallback: a mid-capture pipe() failure
    shifts every subsequent call index and invalidates the predicted-calls
    set (capture_calibration_data flips _predict_valid["ok"] to False in
    that except-block). Once invalid, a call index outside the prediction
    must still be admitted -- degrading to the pre-Round-14 call-count-prefix
    rule -- rather than being silently dropped for the rest of the capture."""
    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _recorder_admits_call

    predicted_calls = frozenset([0, 9, 18, 27, 36, 45, 54, 63])
    max_calls = 64

    # While the prediction is trustworthy: only predicted calls are admitted.
    assert _recorder_admits_call(9, True, predicted_calls, max_calls) is True
    assert _recorder_admits_call(10, True, predicted_calls, max_calls) is False

    # After a batch failure invalidates the prediction: call 10 (outside the
    # predicted set) must now be admitted, bounded only by the call-count cap.
    assert _recorder_admits_call(10, False, predicted_calls, max_calls) is True
    assert _recorder_admits_call(64, False, predicted_calls, max_calls) is False


## FP8 Round 11 -- attn_bmm_dq_fed (builder.py's engine-inspector-adjacent
## evidence-record metric, not fp8_quantize.py, but exercised here alongside
## the other calib-tile ONNX-graph tests since it shares this file's
## _make_min_onnx-style synthetic-graph convention).


def _make_attn_bmm_onnx(path):
    """Synthetic graph covering all four cases _count_attn_bmm_dq_fed must
    distinguish:
    - '.../attn1/MatMul' fed by two DequantizeLinear outputs -> counted, DQ-fed
    - '.../attn1/MatMul_1' fed by one non-DQ input -> counted, NOT DQ-fed
    - '.../attn2/MatMul' with one initializer input -> a projection MatMul
      (one operand is a weight), excluded from the count entirely despite the
      name matching _ATTN_BMM_NAME_RE
    - '.../to_q/MatMul' -> name doesn't match _ATTN_BMM_NAME_RE at all, excluded
    """
    q_i8 = helper.make_tensor_value_info("q_i8", TensorProto.FLOAT, [2, 8, 64])
    q_scale = helper.make_tensor_value_info("q_scale", TensorProto.FLOAT, [])
    k_i8 = helper.make_tensor_value_info("k_i8", TensorProto.FLOAT, [2, 8, 64])
    k_scale = helper.make_tensor_value_info("k_scale", TensorProto.FLOAT, [])
    v_raw = helper.make_tensor_value_info("v_raw", TensorProto.FLOAT, [2, 8, 64])
    softmax_out = helper.make_tensor_value_info("softmax_out", TensorProto.FLOAT, [2, 8, 8])
    some_act = helper.make_tensor_value_info("some_act", TensorProto.FLOAT, [2, 8, 64])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 8, 64])

    w_proj = helper.make_tensor("w_proj", TensorProto.FLOAT, [64, 64], [0.0] * (64 * 64))
    w_proj2 = helper.make_tensor("w_proj2", TensorProto.FLOAT, [64, 64], [0.0] * (64 * 64))

    dq_q = helper.make_node("DequantizeLinear", ["q_i8", "q_scale"], ["q_dq"], name="dq_q")
    dq_k = helper.make_node("DequantizeLinear", ["k_i8", "k_scale"], ["k_dq"], name="dq_k")
    identity_v = helper.make_node("Identity", ["v_raw"], ["v_ident"], name="identity_v")

    dq_fed_bmm = helper.make_node("MatMul", ["q_dq", "k_dq"], ["attn1_scores"], name="/blocks.0/attn1/MatMul")
    non_dq_fed_bmm = helper.make_node(
        "MatMul", ["softmax_out", "v_ident"], ["attn1_out"], name="/blocks.0/attn1/MatMul_1"
    )
    projection_matmul_matching_name = helper.make_node(
        "MatMul", ["some_act", "w_proj"], ["proj_out"], name="/blocks.0/attn2/MatMul"
    )
    unrelated_matmul = helper.make_node("MatMul", ["x", "w_proj2"], ["to_q_out"], name="/blocks.0/to_q/MatMul")

    outs = [
        helper.make_tensor_value_info("attn1_scores", TensorProto.FLOAT, [2, 8, 8]),
        helper.make_tensor_value_info("attn1_out", TensorProto.FLOAT, [2, 8, 64]),
        helper.make_tensor_value_info("proj_out", TensorProto.FLOAT, [2, 8, 64]),
        helper.make_tensor_value_info("to_q_out", TensorProto.FLOAT, [2, 8, 64]),
    ]
    g = helper.make_graph(
        [dq_q, dq_k, identity_v, dq_fed_bmm, non_dq_fed_bmm, projection_matmul_matching_name, unrelated_matmul],
        "attn_bmm",
        [q_i8, q_scale, k_i8, k_scale, v_raw, softmax_out, some_act, x],
        outs,
        initializer=[w_proj, w_proj2],
    )
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), path)


def test_count_attn_bmm_dq_fed_distinguishes_dq_fed_from_non_dq_fed_and_projections(tmp_path):
    """Pins the attn_bmm_dq_fed evidence-record metric (builder.py, FP8 Round
    11): of the two real attention BMMs in the synthetic graph, only the one
    fed by two DequantizeLinear outputs counts as dq_fed; the projection
    MatMul (one operand a weight initializer) and the unrelated MatMul (name
    doesn't match /attn[12]/MatMul) are excluded from both counts entirely --
    466/466 on a real -mhaq build is this same (dq_fed, total) shape, just
    larger."""
    from streamdiffusion.acceleration.tensorrt.builder import _count_attn_bmm_dq_fed

    onnx_path = str(tmp_path / "attn_bmm.onnx")
    _make_attn_bmm_onnx(onnx_path)

    result = _count_attn_bmm_dq_fed(onnx_path)

    assert result == (1, 2)


def test_count_attn_bmm_dq_fed_missing_path_returns_none(tmp_path):
    """Best-effort posture matching the rest of the inspector block: a
    nonexistent onnx_path must return None, not raise."""
    from streamdiffusion.acceleration.tensorrt.builder import _count_attn_bmm_dq_fed

    result = _count_attn_bmm_dq_fed(str(tmp_path / "does_not_exist.onnx"))

    assert result is None
