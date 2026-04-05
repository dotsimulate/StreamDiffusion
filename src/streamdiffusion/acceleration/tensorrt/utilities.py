#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/utilities.py

#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gc

# Set up logger for this module
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch


# cuda-python 13.x renamed 'cudart' to 'cuda.bindings.runtime'
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart
from PIL import Image
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes

from .models.models import CLIP, VAE, BaseModel, UNet, VAEEncoder


logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

from ...model_detection import detect_model


# ---------------------------------------------------------------------------
# GPU Hardware Profile — hardware-aware TRT builder configuration
# ---------------------------------------------------------------------------

@dataclass
class GPUBuildProfile:
    """
    Hardware-aware TRT builder configuration derived from CUDA device properties.

    All parameters are auto-selected based on GPU architecture tier:
      - Ampere  (CC 8.0–8.8): Conservative — small L2, preserve VRAM
      - Ada     (CC 8.9):      Balanced   — large L2, benefit from deeper tiling/opt
      - Blackwell (CC 12.0+):  Aggressive — massive L2, max search depth
    """
    gpu_name: str
    compute_capability: tuple
    l2_cache_bytes: int
    vram_bytes: int
    sm_count: int
    tier: str  # "ampere", "ada", "blackwell", "unknown"

    # IBuilderConfig parameters
    builder_optimization_level: int     # 0–5; higher = better kernels, longer build
    tiling_optimization_level: str      # "NONE"/"FAST"/"MODERATE"/"FULL"
    l2_limit_for_tiling: int            # bytes; target L2 budget for tiling
    max_aux_streams: int                # reserved; NOT applied (TRT heuristic is better)
    sparse_weights: bool                # examine weights for 2:4 sparsity (Ampere+)
    enable_runtime_activation_resize: bool  # RUNTIME_ACTIVATION_RESIZE_10_10
    max_workspace_cap_bytes: int        # hard cap on workspace (before free-mem calc)


def detect_gpu_profile(device: int = 0) -> GPUBuildProfile:
    """
    Detect the current GPU and return hardware-optimal TRT builder parameters.

    Called once at the start of every engine build so that all IBuilderConfig
    settings are tuned to the exact GPU running the build.

    Tiers and rationale
    -------------------
    Ampere (CC 8.0–8.8, e.g. RTX 3090 — 6 MiB L2, 82 SMs):
      - Opt level 4: always compiles dynamic kernels (better than level-3 heuristics)
      - Tiling FAST (static shapes only): small L2 gains little from deep search
      - 8 GiB workspace cap: conserve VRAM on 24 GB cards

    Ada Lovelace (CC 8.9, e.g. RTX 4090 — 72 MiB L2, 128 SMs):
      - Opt level 4: dynamic kernels without level-5 profiling OOM risk
      - Tiling MODERATE (static shapes only): 12× more L2 makes tiling worthwhile
      - 12 GiB workspace cap

    Blackwell (CC 12.0+, e.g. RTX 5090 — 128 MiB L2, ~170 SMs):
      - Opt level 4: same rationale — level 5 causes OOM during tactic profiling
      - Tiling FULL (static shapes only): massive L2 warrants widest search
      - 16 GiB workspace cap

    Note: tiling_optimization_level and l2_limit_for_tiling are only effective for
    static-shape engines. TRT confirms: "Graph contains symbolic shape, l2tc doesn't
    take effect." For dynamic-shape builds (our default), these are skipped entirely
    to avoid warning spam and wasted build time.

    max_aux_streams is NOT set — TRT's own heuristic is better than a fixed value.
    Setting it explicitly causes "[MS] Multi stream is disabled" warnings on simple
    models (VAE) without proven benefit on complex ones (UNet).
    """
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception as e:
        logger.warning(f"[TRT Build] Could not query GPU properties: {e} — using fallback profile")
        return _fallback_profile()

    cc = (props.major, props.minor)
    l2 = props.L2_cache_size
    vram = props.total_memory
    sms = props.multi_processor_count

    # --- Tier selection ---
    # opt_level=4 for all tiers: always compiles dynamic kernels (better kernel
    # selection than level-3 heuristics, even for static shapes). Level 5 avoided —
    # causes OOM during tactic profiling (160 GiB requests observed).
    # NOTE: tactic 0x3e9 "Assertion g.nodes.size() == 0" errors in TRT 10.12 are
    # a known TRT bug — benign, the tactic is skipped and build succeeds.
    if cc >= (12, 0):
        tier = "blackwell"
        opt_level = 4
        tiling = "FULL"
        max_ws_cap = 16 * (2 ** 30)   # 16 GiB cap
    elif cc >= (8, 9):                 # Ada Lovelace (8.9 exactly)
        tier = "ada"
        opt_level = 4
        tiling = "MODERATE"
        max_ws_cap = 12 * (2 ** 30)   # 12 GiB cap
    elif cc >= (8, 0):                 # Ampere (8.0 – 8.8)
        tier = "ampere"
        opt_level = 4
        tiling = "FAST"
        max_ws_cap = 8 * (2 ** 30)    # 8 GiB cap
    else:
        # Pre-Ampere or unknown — use conservative defaults
        tier = "unknown"
        opt_level = 3
        tiling = "NONE"
        max_ws_cap = 8 * (2 ** 30)

    profile = GPUBuildProfile(
        gpu_name=props.name,
        compute_capability=cc,
        l2_cache_bytes=l2,
        vram_bytes=vram,
        sm_count=sms,
        tier=tier,
        builder_optimization_level=opt_level,
        tiling_optimization_level=tiling,
        l2_limit_for_tiling=l2,        # use full L2 as tiling budget (static builds only)
        max_aux_streams=0,              # 0 = let TRT decide (avoids "[MS] disabled" spam)
        sparse_weights=True,            # always examine; no downside if not sparse
        enable_runtime_activation_resize=True,
        max_workspace_cap_bytes=max_ws_cap,
    )

    logger.info(
        f"[TRT Build] GPU detected: {props.name} | "
        f"CC {cc[0]}.{cc[1]} | Tier: {tier} | "
        f"L2: {l2 // (1024 * 1024)} MiB | VRAM: {vram // (1024 ** 3)} GiB | "
        f"opt_level={opt_level}"
    )
    return profile


def _fallback_profile() -> GPUBuildProfile:
    """Conservative fallback when GPU detection fails."""
    return GPUBuildProfile(
        gpu_name="unknown",
        compute_capability=(8, 0),
        l2_cache_bytes=6 * 1024 * 1024,
        vram_bytes=24 * (2 ** 30),
        sm_count=82,
        tier="unknown",
        builder_optimization_level=3,
        tiling_optimization_level="NONE",
        l2_limit_for_tiling=6 * 1024 * 1024,
        max_aux_streams=0,              # reserved; NOT applied
        sparse_weights=False,
        enable_runtime_activation_resize=True,
        max_workspace_cap_bytes=8 * (2 ** 30),
    )


def _apply_gpu_profile_to_config(
    config: "trt.IBuilderConfig",
    gpu_profile: Optional[GPUBuildProfile],
    dynamic_shapes: bool = True,
) -> None:
    """
    Apply hardware-aware IBuilderConfig parameters that Polygraphy does not expose.

    Called for both FP16 and FP8 builds after the config object is created.
    All settings gracefully degrade if the TRT version doesn't support a feature.

    Args:
        config: TRT IBuilderConfig to modify.
        gpu_profile: Hardware-detected build parameters from detect_gpu_profile().
        dynamic_shapes: Whether this engine uses dynamic input shapes.
            - True  (default): tiling and l2_limit skipped — TRT confirms these have
              no effect on symbolic-shape graphs and only produce warning spam.
            - False (static): tiling and l2_limit applied for full L2 cache benefit.
    """
    if gpu_profile is None:
        return

    # builder_optimization_level (0–5):
    #   4 = always compiles dynamic kernels (better than level-3 heuristics)
    #   5 = additionally compares dynamic vs static kernels — causes OOM during
    #       tactic profiling on dynamic-shape engines (160 GiB requests observed).
    # We use level 4 for all tiers to get the dynamic-kernel benefit without the
    # level-5 exhaustive comparison that OOMs.
    try:
        config.builder_optimization_level = gpu_profile.builder_optimization_level
        logger.info(f"[TRT Config] builder_optimization_level={gpu_profile.builder_optimization_level}")
    except AttributeError:
        logger.debug("[TRT Config] builder_optimization_level not supported — skipping")

    # tiling_optimization_level + l2_limit_for_tiling:
    # TRT's L2 tiling cache optimization requires static/concrete shapes to work.
    # For dynamic-shape engines, TRT emits: "Graph contains symbolic shape, l2tc
    # doesn't take effect" for every applicable layer — pure warning spam with zero
    # benefit. Skipped when dynamic_shapes=True.
    if not dynamic_shapes and gpu_profile.tiling_optimization_level != "NONE":
        try:
            tiling_map = {
                "NONE": trt.TilingOptimizationLevel.NONE,
                "FAST": trt.TilingOptimizationLevel.FAST,
                "MODERATE": trt.TilingOptimizationLevel.MODERATE,
                "FULL": trt.TilingOptimizationLevel.FULL,
            }
            tiling_level = tiling_map.get(gpu_profile.tiling_optimization_level, trt.TilingOptimizationLevel.NONE)
            config.tiling_optimization_level = tiling_level
            logger.info(f"[TRT Config] tiling_optimization_level={gpu_profile.tiling_optimization_level}")
        except AttributeError:
            logger.debug("[TRT Config] tiling_optimization_level not supported — skipping")

        try:
            if gpu_profile.l2_limit_for_tiling > 0:
                config.l2_limit_for_tiling = gpu_profile.l2_limit_for_tiling
                logger.info(
                    f"[TRT Config] l2_limit_for_tiling={gpu_profile.l2_limit_for_tiling // (1024 * 1024)} MiB"
                )
        except AttributeError:
            logger.debug("[TRT Config] l2_limit_for_tiling not supported — skipping")
    elif dynamic_shapes:
        logger.debug(
            "[TRT Config] tiling_optimization_level/l2_limit skipped — dynamic shapes "
            "(would produce '[l2tc] VALIDATE FAIL' warnings with no effect)"
        )

    # max_aux_streams: NOT SET — let TRT use its own heuristic.
    # Setting an explicit value causes "[MS] Multi stream is disabled" warnings on
    # any model where TRT can't assign that many streams (e.g. VAE decoder which is
    # too sequential). TRT's heuristic silently chooses the right value per model.

    # SPARSE_WEIGHTS: let TRT examine weight tensors for structured 2:4 sparsity
    # and use Sparse Tensor Core kernels if suitable. Zero downside for dense weights.
    if gpu_profile.sparse_weights:
        try:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
            logger.info("[TRT Config] SPARSE_WEIGHTS enabled")
        except Exception:
            logger.debug("[TRT Config] SPARSE_WEIGHTS not supported — skipping")

    # RUNTIME_ACTIVATION_RESIZE_10_10: allows update_device_memory_size_for_shapes()
    # to shrink activation memory when actual input shapes are smaller than max profile
    # dims. Our engines use dynamic shapes (min 256 → max 1024), so running at 512x512
    # can save ~50–75% of peak activation VRAM compared to always allocating for 1024.
    if gpu_profile.enable_runtime_activation_resize:
        try:
            config.set_preview_feature(trt.PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10, True)
            logger.info("[TRT Config] RUNTIME_ACTIVATION_RESIZE_10_10 enabled")
        except Exception:
            logger.debug("[TRT Config] RUNTIME_ACTIVATION_RESIZE_10_10 not supported — skipping")

    # avg_timing_iterations: number of timing runs averaged per tactic candidate.
    # Default 1 produces noisy measurements — occasional slow GPU clocks or cache
    # miss can unfairly disqualify the best kernel. Value of 4 gives stable rankings
    # with minimal extra build time (4× timing overhead, which is tiny vs. compilation).
    # TRT 10.12 confirmed to support this property.
    try:
        config.avg_timing_iterations = 4
        logger.info("[TRT Config] avg_timing_iterations=4")
    except AttributeError:
        logger.debug("[TRT Config] avg_timing_iterations not supported — skipping")


# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


class TRTProfiler(trt.IProfiler):
    """
    Per-layer TRT timing profiler.

    Activated by setting the STREAMDIFFUSION_PROFILE_TRT environment variable.
    Attach to Engine.context after create_execution_context(); TRT will call
    report_layer_time() once per layer per inference pass.

    NOTE: Attaching a profiler disables CUDA graph replay for that engine
    (IProfiler cannot report per-layer times through a captured graph).
    Production inference always runs without profiler — zero overhead.

    Usage:
        set STREAMDIFFUSION_PROFILE_TRT=1
        python td_main.py
        # After N iterations, call engine.dump_profile()

    Nsight Systems workflow (standalone .engine files):
        # Build with profilingVerbosity=DETAILED (done automatically at build time)
        # Profile with trtexec:
        trtexec --loadEngine=unet.engine --noDataTransfers --useSpinWait \\
                --warmUp=0 --duration=0 --iterations=50 \\
                --profilingVerbosity=detailed --dumpProfile --separateProfileRun
        # For CUDA graph per-kernel view, add --useCudaGraph --cuda-graph-trace=node
        # and wrap with: nsys profile --capture-range cudaProfilerApi trtexec ...
    """

    def __init__(self, name: str = ""):
        super().__init__()
        self.name = name
        self._runs: list = []        # list of lists: [[( layer_name, ms ), ...], ...]
        self._current: list = []     # accumulator for the in-progress inference

    def report_layer_time(self, layer_name: str, ms: float) -> None:  # noqa: N802
        self._current.append((layer_name, ms))

    def start_run(self) -> None:
        self._current = []

    def end_run(self) -> None:
        if self._current:
            self._runs.append(self._current)
        self._current = []

    def get_summary(self, last_n: int = 10) -> str:
        if not self._runs:
            return f"[{self.name}] No profiling data collected yet."

        runs = self._runs[-last_n:]
        from collections import defaultdict
        totals: dict = defaultdict(list)
        for run in runs:
            for layer_name, ms in run:
                totals[layer_name].append(ms)

        # Sort by median descending
        def _median(v):
            s = sorted(v)
            return s[len(s) // 2]

        sorted_layers = sorted(totals.items(), key=lambda x: _median(x[1]), reverse=True)
        total_ms = sum(_median(v) for _, v in sorted_layers)

        lines = [
            f"[{self.name}] Layer Profile — {len(runs)} runs, "
            f"{total_ms:.2f} ms total (median per layer):"
        ]
        for layer_name, times in sorted_layers[:25]:
            med = _median(times)
            pct = (med / total_ms * 100) if total_ms > 0 else 0
            lines.append(f"  {med:8.3f} ms  {pct:5.1f}%  {layer_name}")
        remaining = len(sorted_layers) - 25
        if remaining > 0:
            rest_ms = sum(_median(v) for _, v in sorted_layers[25:])
            lines.append(f"  ... {remaining} more layers  ({rest_ms:.2f} ms)")
        return "\n".join(lines)


class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph

        # Buffer reuse optimization tracking
        self._last_shape_dict = None
        self._last_device = None
        # Cached set of input tensor names — immutable after engine build
        self._allowed_inputs = None

    def __del__(self):
        # Check if AttributeError: 'Engine' object has no attribute 'buffers'
        if not hasattr(self, "buffers"):
            return
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray)]

        if hasattr(self, "cuda_graph_instance") and self.cuda_graph_instance is not None:
            try:
                CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instance))
            except:
                pass
        if hasattr(self, "graph") and self.graph is not None:
            try:
                CUASSERT(cudart.cudaGraphDestroy(self.graph))
            except:
                pass

        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, onnx_path, onnx_refit_path):
        def convert_int64(arr):
            # TODO: smarter conversion
            if len(arr.shape) == 0:
                return np.int32(arr)
            return arr

        def add_to_map(refit_dict, name, values):
            if name in refit_dict:
                assert refit_dict[name] is None
                if values.dtype == np.int64:
                    values = convert_int64(values)
                refit_dict[name] = values

        logger.info(f"Refitting TensorRT engine with {onnx_refit_path} weights")
        refit_nodes = gs.import_onnx(onnx.load(onnx_refit_path)).toposort().nodes

        # Construct mapping from weight names in refit model -> original model
        name_map = {}
        for n, node in enumerate(gs.import_onnx(onnx.load(onnx_path)).toposort().nodes):
            refit_node = refit_nodes[n]
            assert node.op == refit_node.op
            # Constant nodes in ONNX do not have inputs but have a constant output
            if node.op == "Constant":
                name_map[refit_node.outputs[0].name] = node.outputs[0].name
            # Handle scale and bias weights
            elif node.op == "Conv":
                if node.inputs[1].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTKERNEL"] = node.name + "_TRTKERNEL"
                if node.inputs[2].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTBIAS"] = node.name + "_TRTBIAS"
            # For all other nodes: find node inputs that are initializers (gs.Constant)
            else:
                for i, inp in enumerate(node.inputs):
                    if inp.__class__ == gs.Constant:
                        name_map[refit_node.inputs[i].name] = inp.name

        def map_name(name):
            if name in name_map:
                return name_map[name]
            return name

        # Construct refit dictionary
        refit_dict = {}
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()
        for layer_name, role in zip(all_weights[0], all_weights[1]):
            # for speciailized roles, use a unique name in the map:
            if role == trt.WeightsRole.KERNEL:
                name = layer_name + "_TRTKERNEL"
            elif role == trt.WeightsRole.BIAS:
                name = layer_name + "_TRTBIAS"
            else:
                name = layer_name

            assert name not in refit_dict, "Found duplicate layer: " + name
            refit_dict[name] = None

        for n in refit_nodes:
            # Constant nodes in ONNX do not have inputs but have a constant output
            if n.op == "Constant":
                name = map_name(n.outputs[0].name)
                add_to_map(refit_dict, name, n.outputs[0].values)

            # Handle scale and bias weights
            elif n.op == "Conv":
                if n.inputs[1].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTKERNEL")
                    add_to_map(refit_dict, name, n.inputs[1].values)

                if n.inputs[2].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTBIAS")
                    add_to_map(refit_dict, name, n.inputs[2].values)

            # For all other nodes: find node inputs that are initializers (AKA gs.Constant)
            else:
                for inp in n.inputs:
                    name = map_name(inp.name)
                    if inp.__class__ == gs.Constant:
                        add_to_map(refit_dict, name, inp.values)

        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if weights_role == trt.WeightsRole.KERNEL:
                custom_name = layer_name + "_TRTKERNEL"
            elif weights_role == trt.WeightsRole.BIAS:
                custom_name = layer_name + "_TRTBIAS"
            else:
                custom_name = layer_name

            # Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
            if layer_name.startswith("onnx::Trilu"):
                continue

            if refit_dict[custom_name] is not None:
                refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
            else:
                logger.warning(f"No refit weights for layer: {layer_name}")

        if not refitter.refit_cuda_engine():
            logger.error("Failed to refit!")
            raise RuntimeError("TensorRT engine refit failed")

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_refit=False,
        timing_cache=None,
        workspace_size=0,
        fp8=False,
        gpu_profile: Optional["GPUBuildProfile"] = None,
        dynamic_shapes: bool = True,
    ):
        logger.info(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")

        if fp8:
            self._build_fp8(
                onnx_path, input_profile, workspace_size,
                timing_cache=timing_cache, gpu_profile=gpu_profile,
                dynamic_shapes=dynamic_shapes,
            )
            return

        # --- Build using raw TRT API for full IBuilderConfig access ---
        # Polygraphy's CreateConfig does not expose: tiling_optimization_level,
        # l2_limit_for_tiling, max_aux_streams, builder_optimization_level,
        # set_preview_feature, or SPARSE_WEIGHTS. We use the raw API (same as
        # the FP8 path) so all parameters are available for both precision paths.

        build_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(build_logger)

        network_flags = 0
        network = builder.create_network(network_flags)

        parser = trt.OnnxParser(network, build_logger)
        parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
        success = parser.parse_from_file(onnx_path)
        if not success:
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(
                f"TRT ONNX parser failed for FP16 engine: {onnx_path}\n"
                + "\n".join(str(e) for e in errors)
            )

        config = builder.create_builder_config()

        # Embed layer names + tactic IDs in the engine for runtime IProfiler support.
        # Zero runtime cost — only affects engine metadata size (a few KB).
        try:
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        except AttributeError:
            pass

        # Precision flags
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.TF32)

        if enable_refit:
            config.set_flag(trt.BuilderFlag.REFIT)

        # Workspace
        if workspace_size > 0:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        # Optimization profile
        if input_profile:
            profile = builder.create_optimization_profile()
            for name, dims in input_profile.items():
                assert len(dims) == 3, f"Expected (min, opt, max) for {name}"
                profile.set_shape(name, min=dims[0], opt=dims[1], max=dims[2])
            config.add_optimization_profile(profile)

        # Timing cache — load existing or create fresh
        cache_data = b""
        if timing_cache and os.path.exists(timing_cache):
            try:
                with open(timing_cache, "rb") as f:
                    cache_data = f.read()
                logger.info(f"[TRT Build] Loaded timing cache: {timing_cache} ({len(cache_data) // 1024} KB)")
            except Exception as e:
                logger.warning(f"[TRT Build] Could not load timing cache {timing_cache}: {e} — starting fresh")
                cache_data = b""
        trt_cache = config.create_timing_cache(cache_data)
        config.set_timing_cache(trt_cache, ignore_mismatch=False)

        # Apply hardware-aware profile parameters
        _apply_gpu_profile_to_config(config, gpu_profile, dynamic_shapes=dynamic_shapes)

        # Build and serialize
        logger.info(f"[TRT Build] Building FP16 engine (raw API): {self.engine_path}")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(
                f"TRT FP16 engine build failed for {onnx_path}. "
                "Check TRT logs above for details."
            )

        with open(self.engine_path, "wb") as f:
            f.write(serialized)

        # Save timing cache for next build
        if timing_cache:
            try:
                updated_cache = config.get_timing_cache()
                if updated_cache is not None:
                    os.makedirs(os.path.dirname(timing_cache), exist_ok=True)
                    with open(timing_cache, "wb") as f:
                        f.write(updated_cache.serialize())
                    logger.info(f"[TRT Build] Saved timing cache: {timing_cache}")
            except Exception as e:
                logger.warning(f"[TRT Build] Could not save timing cache: {e}")

        size_bytes = getattr(serialized, 'nbytes', None) or len(serialized)
        logger.info(f"[TRT Build] FP16 engine saved: {self.engine_path} ({size_bytes / 1024 / 1024:.0f} MB)")

    def _build_fp8(
        self,
        onnx_path,
        input_profile,
        workspace_size,
        timing_cache=None,
        gpu_profile: Optional["GPUBuildProfile"] = None,
        dynamic_shapes: bool = True,
    ):
        """
        Build a TRT engine from a Q/DQ-annotated FP8 ONNX using the raw TRT builder API.

        Polygraphy 0.49.26's CreateConfig does not support fp8=, so we use the raw
        TensorRT Python API directly. The STRONGLY_TYPED network flag is required to
        preserve the Q/DQ precision annotations inserted by nvidia-modelopt.

        Args:
            onnx_path: Path to *.fp8.onnx (Q/DQ-annotated by fp8_quantize.py).
            input_profile: Dict of {name: (min, opt, max)} shapes.
            workspace_size: TRT workspace limit in bytes.
            timing_cache: Path to timing cache file for load/save.
            gpu_profile: Hardware-aware build parameters from detect_gpu_profile().
            dynamic_shapes: Whether the engine uses dynamic input shapes.
        """
        build_logger = trt.Logger(trt.Logger.WARNING)

        builder = trt.Builder(build_logger)

        # STRONGLY_TYPED: required for FP8. Tells TRT to use the data-type annotations
        # from Q/DQ nodes rather than running its own precision heuristics.
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = builder.create_network(network_flags)

        parser = trt.OnnxParser(network, build_logger)
        # NATIVE_INSTANCENORM: use TRT's fused InstanceNorm/GroupNorm kernel instead
        # of decomposing into primitive ops. Diffusion UNets use GroupNorm heavily.
        parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
        success = parser.parse_from_file(onnx_path)
        if not success:
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(
                f"TRT ONNX parser failed for FP8 engine: {onnx_path}\n"
                + "\n".join(str(e) for e in errors)
            )

        config = builder.create_builder_config()

        # Embed layer names + tactic IDs in the engine for runtime IProfiler support.
        try:
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        except AttributeError:
            pass

        # BuilderFlag.STRONGLY_TYPED was removed in TRT 10.12; the network-level flag
        # (NetworkDefinitionCreationFlag.STRONGLY_TYPED, set on network creation above)
        # is now the only mechanism. On older TRT versions where BuilderFlag.STRONGLY_TYPED
        # still exists, we also set precision flags on the config.
        if hasattr(trt.BuilderFlag, 'STRONGLY_TYPED'):
            # TRT < 10.12: BuilderFlag.STRONGLY_TYPED exists — set precision flags and
            # the builder-level STRONGLY_TYPED flag alongside the network-level flag.
            config.set_flag(trt.BuilderFlag.FP8)
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.TF32)
            config.set_flag(trt.BuilderFlag.STRONGLY_TYPED)
        # else: TRT 10.12+ — NetworkDefinitionCreationFlag.STRONGLY_TYPED (set on network
        # creation above) is sufficient; Q/DQ node annotations dictate precision directly.

        if workspace_size > 0:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        if input_profile:
            profile = builder.create_optimization_profile()
            for name, dims in input_profile.items():
                assert len(dims) == 3, f"Expected (min, opt, max) for {name}"
                profile.set_shape(name, min=dims[0], opt=dims[1], max=dims[2])
            config.add_optimization_profile(profile)

        # Timing cache — load existing or create fresh
        cache_data = b""
        if timing_cache and os.path.exists(timing_cache):
            try:
                with open(timing_cache, "rb") as f:
                    cache_data = f.read()
                logger.info(f"[FP8] Loaded timing cache: {timing_cache} ({len(cache_data) // 1024} KB)")
            except Exception as e:
                logger.warning(f"[FP8] Could not load timing cache {timing_cache}: {e} — starting fresh")
                cache_data = b""
        trt_cache = config.create_timing_cache(cache_data)
        config.set_timing_cache(trt_cache, ignore_mismatch=False)

        # Apply hardware-aware profile parameters
        _apply_gpu_profile_to_config(config, gpu_profile, dynamic_shapes=dynamic_shapes)

        logger.info(f"[FP8] Building TRT FP8 engine (STRONGLY_TYPED): {self.engine_path}")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(
                f"TRT FP8 engine build failed for {onnx_path}. "
                "Check TRT logs above for details."
            )

        with open(self.engine_path, "wb") as f:
            f.write(serialized)

        # Save timing cache for next build
        if timing_cache:
            try:
                updated_cache = config.get_timing_cache()
                if updated_cache is not None:
                    os.makedirs(os.path.dirname(timing_cache), exist_ok=True)
                    with open(timing_cache, "wb") as f:
                        f.write(updated_cache.serialize())
                    logger.info(f"[FP8] Saved timing cache: {timing_cache}")
            except Exception as e:
                logger.warning(f"[FP8] Could not save timing cache: {e}")

        size_bytes = getattr(serialized, 'nbytes', None) or len(serialized)
        logger.info(f"[FP8] Engine saved: {self.engine_path} ({size_bytes / 1024 / 1024:.0f} MB)")

    def load(self):
        logger.info(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

        # Attach per-layer profiler when STREAMDIFFUSION_PROFILE_TRT is set.
        # Requires engines built with profiling_verbosity=DETAILED for meaningful names.
        # NOTE: profiler presence disables CUDA graph replay in infer() — IProfiler
        # cannot report per-layer times through a captured graph.
        self.profiler: Optional[TRTProfiler] = None
        if os.environ.get("STREAMDIFFUSION_PROFILE_TRT"):
            self.profiler = TRTProfiler(name=os.path.basename(self.engine_path))
            self.context.profiler = self.profiler
            logger.info(f"[TRTProfiler] Attached to {os.path.basename(self.engine_path)} (CUDA graphs disabled)")

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        # Check if we can reuse existing buffers (OPTIMIZATION)
        if self._can_reuse_buffers(shape_dict, device):
            return

        # Clear existing buffers before reallocating
        self.tensors.clear()

        # Reset CUDA graph when buffers are reallocated
        # The captured graph becomes invalid with new memory addresses
        if self.cuda_graph_instance is not None:
            CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instance))
            self.cuda_graph_instance = None
            if hasattr(self, "graph") and self.graph is not None:
                CUASSERT(cudart.cudaGraphDestroy(self.graph))
                self.graph = None

        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)

            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)

            trt_dtype = self.engine.get_tensor_dtype(name)
            try:
                dtype_np = trt.nptype(trt_dtype)
                torch_dtype = numpy_to_torch_dtype_dict[dtype_np]
            except TypeError:
                # FP8 (FLOAT8E4M3FN) has no numpy equivalent — map directly to torch
                if trt_dtype == trt.DataType.FP8:
                    torch_dtype = torch.float8_e4m3fn
                else:
                    raise
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                if not self.context.set_input_shape(name, shape):
                    raise RuntimeError(
                        f"TensorRT: set_input_shape failed for '{name}' with shape {shape}"
                    )

            tensor = torch.empty(tuple(shape), dtype=torch_dtype).to(device=device)
            self.tensors[name] = tensor

        # Cache allocation parameters for reuse check
        self._last_shape_dict = shape_dict.copy() if shape_dict else None
        self._last_device = device

    def _can_reuse_buffers(self, shape_dict=None, device="cuda"):
        """
        Check if existing buffers can be reused (avoiding expensive reallocation)

        Returns:
            bool: True if buffers can be reused, False if reallocation needed
        """
        # No existing tensors - need to allocate
        if not self.tensors:
            return False

        # Device changed - need to reallocate
        if not hasattr(self, "_last_device") or self._last_device != device:
            return False

        # No cached shape_dict - need to allocate
        if not hasattr(self, "_last_shape_dict"):
            return False

        # Compare current vs cached shape_dict
        if shape_dict is None and self._last_shape_dict is None:
            return True
        elif shape_dict is None or self._last_shape_dict is None:
            return False

        # Quick check: if tensor counts differ, can't reuse
        if len(shape_dict) != len(self._last_shape_dict):
            return False

        # Compare shapes for all tensors in the new shape_dict
        for name, new_shape in shape_dict.items():
            # Check if tensor exists in cached shapes
            cached_shape = self._last_shape_dict.get(name)
            if cached_shape is None:
                return False

            # Compare shapes (handle different types consistently)
            if tuple(cached_shape) != tuple(new_shape):
                return False

        return True

    def reset_cuda_graph(self):
        if self.cuda_graph_instance is not None:
            CUASSERT(cudart.cudaGraphExecDestroy(self.cuda_graph_instance))
            self.cuda_graph_instance = None
        if hasattr(self, "graph") and self.graph is not None:
            CUASSERT(cudart.cudaGraphDestroy(self.graph))
            self.graph = None

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        # IProfiler cannot report per-layer times through CUDA graph replay — disable graphs
        # when profiler is attached. This is automatically set when STREAMDIFFUSION_PROFILE_TRT
        # is set in activate(), so callers do not need to change anything.
        if self.profiler is not None:
            use_cuda_graph = False

        # Filter inputs to only those the engine actually exposes to avoid binding errors
        # _allowed_inputs is cached on first call — IO tensor names are immutable after engine build
        if self._allowed_inputs is None:
            try:
                self._allowed_inputs = set()
                for idx in range(self.engine.num_io_tensors):
                    name = self.engine.get_tensor_name(idx)
                    if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        self._allowed_inputs.add(name)
            except Exception:
                self._allowed_inputs = None  # Will retry next call

        if self._allowed_inputs:
            # Drop any extra keys (e.g., text_embeds/time_ids) that the engine was not built to accept
            filtered_feed_dict = {k: v for k, v in feed_dict.items() if k in self._allowed_inputs}
            if len(filtered_feed_dict) != len(feed_dict):
                missing = [k for k in feed_dict.keys() if k not in self._allowed_inputs]
                if missing:
                    logger.debug(
                        "TensorRT Engine: filtering unsupported inputs %s (allowed=%s)",
                        missing,
                        sorted(list(self._allowed_inputs)),
                    )
            feed_dict = filtered_feed_dict

        if self.profiler is not None:
            self.profiler.start_run()

        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(
                    f"TensorRT: set_tensor_address failed for '{name}'"
                )

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream.ptr))
                # No cudaStreamSynchronize — graph replay is async; stream ordering ensures
                # downstream GPU ops (copy_, attention) wait for graph completion.
                # CPU sync happens only via end.synchronize() in pipeline.__call__.
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream.ptr)
                if not noerror:
                    raise ValueError("ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(
                    cudart.cudaStreamBeginCapture(stream.ptr, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
                )
                self.context.execute_async_v3(stream.ptr)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream.ptr))
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            noerror = self.context.execute_async_v3(stream.ptr)
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        if self.profiler is not None:
            # Synchronize to ensure all IProfiler.report_layer_time() callbacks have fired
            # before end_run() stores the accumulated per-layer data.
            stream.synchronize()
            self.profiler.end_run()

        return self.tensors

    def dump_profile(self, last_n: int = 10) -> None:
        """Log a per-layer timing summary for the last N profiled inference runs.

        No-op when STREAMDIFFUSION_PROFILE_TRT is not set (profiler is None).
        """
        if self.profiler is not None:
            logger.info(self.profiler.get_summary(last_n))


def decode_images(images: torch.Tensor):
    images = (
        ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
    )
    return [Image.fromarray(x) for x in images]


def preprocess_image(image: Image.Image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h))
    init_image = np.array(image).astype(np.float32) / 255.0
    init_image = init_image[None].transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image).contiguous()
    return 2.0 * init_image - 1.0


def prepare_mask_and_masked_image(image: Image.Image, mask: Image.Image):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32).contiguous() / 127.5 - 1.0
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(dtype=torch.float32).contiguous()

    masked_image = image * (mask < 0.5)

    return mask, masked_image


def create_models(
    model_id: str,
    use_auth_token: Optional[str],
    device: Union[str, torch.device],
    max_batch_size: int,
    unet_in_channels: int = 4,
    embedding_dim: int = 768,
):
    models = {
        "clip": CLIP(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
        "unet": UNet(
            hf_token=use_auth_token,
            fp16=True,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            unet_dim=unet_in_channels,
        ),
        "vae": VAE(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
        "vae_encoder": VAEEncoder(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
    }
    return models


def build_engine(
    engine_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    build_static_batch: bool = False,
    build_dynamic_shape: bool = False,
    build_enable_refit: bool = False,
    fp8: bool = False,
):
    # --- Step 0: Detect GPU and select hardware-optimal build parameters ---
    gpu_profile = detect_gpu_profile(device=torch.cuda.current_device())

    # --- Workspace sizing: leave 2 GiB for activations, cap per GPU tier ---
    _, free_mem, _ = cudart.cudaMemGetInfo()
    GiB = 2 ** 30
    if free_mem > 6 * GiB:
        activation_carveout = 2 * GiB
        max_workspace_size = min(
            free_mem - activation_carveout,
            gpu_profile.max_workspace_cap_bytes,
        )
    else:
        max_workspace_size = 0
    logger.info(
        f"[TRT Build] Workspace: free={free_mem / GiB:.1f} GiB, "
        f"cap={gpu_profile.max_workspace_cap_bytes / GiB:.1f} GiB, "
        f"allocated={max_workspace_size / GiB:.1f} GiB"
    )

    # --- Timing cache: shared per engine directory ---
    # Cache is stored alongside the engine files so it persists across rebuilds.
    engine_dir = os.path.dirname(engine_path)
    timing_cache_path = os.path.join(engine_dir, "timing.cache")

    engine = Engine(engine_path)
    input_profile = model_data.get_input_profile(
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        static_batch=build_static_batch,
        static_shape=not build_dynamic_shape,
    )
    engine.build(
        onnx_opt_path,
        fp16=True,
        input_profile=input_profile,
        enable_refit=build_enable_refit,
        timing_cache=timing_cache_path,
        workspace_size=max_workspace_size,
        fp8=fp8,
        gpu_profile=gpu_profile,
        dynamic_shapes=build_dynamic_shape,
    )

    return engine


def export_onnx(
    model,
    onnx_path: str,
    model_data: BaseModel,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    onnx_opset: int,
):
    # TODO: Not 100% happy about this function - needs refactoring

    is_sdxl = False
    is_sdxl_controlnet = False

    # Detect if this is a ControlNet model (vs UNet model)
    is_controlnet = (hasattr(model, "__class__") and "ControlNet" in model.__class__.__name__) or (
        hasattr(model, "config") and hasattr(model.config, "_class_name") and "ControlNet" in model.config._class_name
    )

    # Detect if this is an SDXL model via detect_model
    if hasattr(model, "unet"):
        detection_result = detect_model(model.unet)
        if detection_result is not None:
            is_sdxl = detection_result.get("is_sdxl", False)
    elif hasattr(model, "config"):
        detection_result = detect_model(model)
        if detection_result is not None:
            is_sdxl = detection_result.get("is_sdxl", False)

    # Detect if this is an SDXL ControlNet
    is_sdxl_controlnet = is_controlnet and (
        is_sdxl or (hasattr(model, "config") and getattr(model.config, "addition_embed_type", None) == "text_time")
    )

    wrapped_model = model  # Default: use model as-is

    # Apply SDXL wrapper for SDXL models (in practice, always UnifiedExportWrapper)
    # Skip SDXLExportWrapper if model is already a UnifiedExportWrapper — it handles
    # SDXL conditioning internally and has strict positional arg requirements (e.g.
    # ipadapter_scale) that SDXLExportWrapper's forward-test probe would violate.
    from .export_wrappers.unet_unified_export import UnifiedExportWrapper

    if is_sdxl and not is_controlnet and not isinstance(model, UnifiedExportWrapper):
        embedding_dim = getattr(model_data, "embedding_dim", "unknown")
        logger.info(f"Detected SDXL model (embedding_dim={embedding_dim}), using wrapper for ONNX export...")
        from .export_wrappers.unet_sdxl_export import SDXLExportWrapper

        wrapped_model = SDXLExportWrapper(model)
    elif not is_controlnet:
        embedding_dim = getattr(model_data, "embedding_dim", "unknown")
        logger.info(f"Detected non-SDXL model (embedding_dim={embedding_dim}), using model as-is for ONNX export...")

    # SDXL ControlNet models need special wrapper for added_cond_kwargs
    elif is_sdxl_controlnet:
        logger.info("Detected SDXL ControlNet model, using specialized wrapper...")
        from .export_wrappers.controlnet_export import SDXLControlNetExportWrapper

        wrapped_model = SDXLControlNetExportWrapper(model)

    # Regular ControlNet models are exported directly
    elif is_controlnet:
        logger.info("Detected ControlNet model, exporting directly...")
        wrapped_model = model

    with torch.inference_mode(), torch.autocast("cuda"):
        inputs = model_data.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)

        # Determine if we need external data format for large models (like SDXL)
        is_large_model = is_sdxl or (hasattr(model, "config") and getattr(model.config, "sample_size", 32) >= 64)

        # Export ONNX normally first
        torch.onnx.export(
            wrapped_model,
            inputs,
            onnx_path,
            export_params=True,
            opset_version=onnx_opset,
            do_constant_folding=True,
            input_names=model_data.get_input_names(),
            output_names=model_data.get_output_names(),
            dynamic_axes=model_data.get_dynamic_axes(),
            dynamo=False,
        )

        # Convert to external data format for large models (SDXL)
        if is_large_model:
            import os

            # Load the exported model
            onnx_model = onnx.load(onnx_path)

            # Check if model is large enough to need external data
            if onnx_model.ByteSize() > 2147483648:  # 2GB
                # Create directory for external data
                onnx_dir = os.path.dirname(onnx_path)

                # Re-save with external data format
                onnx.save_model(
                    onnx_model,
                    onnx_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="weights.pb",
                    convert_attribute=False,
                )
                logger.info("Converted to external data format with weights in weights.pb")

                # Delete individual tensor files left by torch.onnx.export (~4 GB for SDXL)
                # They are now consolidated into weights.pb and no longer needed
                for f in os.listdir(onnx_dir):
                    if f.startswith("onnx__"):
                        try:
                            os.remove(os.path.join(onnx_dir, f))
                        except OSError:
                            pass  # Caught by builder.py final cleanup if still present

            del onnx_model
    del wrapped_model
    gc.collect()
    torch.cuda.empty_cache()


def optimize_onnx(
    onnx_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
):
    import os

    # Check if external data files exist (indicating external data format was used)
    onnx_dir = os.path.dirname(onnx_path)
    external_data_files = [f for f in os.listdir(onnx_dir) if f.endswith(".pb")]
    uses_external_data = len(external_data_files) > 0

    if uses_external_data:
        logger.info(f"Optimizing ONNX with external data (found: {external_data_files})")
        # Load model with external data
        onnx_model = onnx.load(onnx_path, load_external_data=True)
        onnx_opt_graph = model_data.optimize(onnx_model)

        # Create output directory
        opt_dir = os.path.dirname(onnx_opt_path)
        os.makedirs(opt_dir, exist_ok=True)

        # Clean up existing files in output directory
        if os.path.exists(opt_dir):
            for f in os.listdir(opt_dir):
                if f.endswith(".pb") or f.endswith(".onnx"):
                    os.remove(os.path.join(opt_dir, f))

        # Save optimized model with external data format
        onnx.save_model(
            onnx_opt_graph,
            onnx_opt_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )
        logger.info("ONNX optimization complete with external data")

    else:
        # Standard optimization for smaller models
        onnx_opt_graph = model_data.optimize(onnx.load(onnx_path))
        onnx.save(onnx_opt_graph, onnx_opt_path)

    del onnx_opt_graph
    gc.collect()
    torch.cuda.empty_cache()
