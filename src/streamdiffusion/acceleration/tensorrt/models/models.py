#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/models.py

#
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

import logging

import onnx_graphsurgeon as gs
import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from onnx import shape_inference
from polygraphy.backend.onnx.loader import fold_constants

logger = logging.getLogger(__name__)


class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        # ORT's symbolic_shape_infer is unreliable on large SDXL graphs: it can't handle >2GB
        # protobufs (raw fp16 UNet / ControlNet-wrapped export) and crashes on FP8 QDQ (upstream
        # ort bug on Expand nodes). In every case polygraphy logs a scary [W] "Falling back..."
        # block, then succeeds via onnx.shape_inference anyway (byte-identical output). The QDQ
        # check alone misses the warning because fold_constants runs during ONNX optimize --
        # BEFORE FP8 quantization inserts QDQ nodes (builder.py: export -> optimize/fold -> fp8
        # quantize), so is_fp8 is always False here for the UNet/ControlNet path. Gate on size
        # too (same >2GB threshold Optimizer.infer_shapes uses below) so the doomed ORT attempt
        # is skipped for any large graph. Small fp16/CLIP/VAE graphs keep the faster ORT path.
        onnx_graph = gs.export_onnx(self.graph)
        is_fp8 = any(n.op in ("QuantizeLinear", "DequantizeLinear") for n in self.graph.nodes)
        is_large = onnx_graph.ByteSize() > 2147483648
        onnx_graph = fold_constants(
            onnx_graph,
            allow_onnxruntime_shape_inference=not (is_fp8 or is_large),
        )
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            print(
                f"[WARN] Model size ({onnx_graph.ByteSize() / (1024**3):.2f} GB) exceeds 2GB - this is normal for SDXL models"
            )
            print("[INFO] ONNX shape inference will be skipped for large models to avoid memory issues")
            # For large models like SDXL, skip shape inference to avoid memory/size issues
            # The model will still work with TensorRT's own shape inference during engine building
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def fix_layernorm_dtypes(self, return_onnx=False):
        """
        Fix LN dtype mismatch in FP8-quantized UNet without breaking Q/DQ adjacency.

        nvidia-modelopt DequantizeLinear outputs FP32; LN scale/bias stay FP16
        from the original weights. STRONGLY_TYPED rejects the mismatch
        (TRT 10.x: "INormalizationLayer 'input' and 'scale' must have identical types").

        Fix: promote scale/bias FP16→FP32 to match the FP32 input, and promote
        the LN output dtype to FP32 so consumers see consistent types.

        Earlier versions also inserted a Cast(FP32→FP16) on each LN output.
        That Cast pollutes Q/DQ adjacency — TRT's quantization fusion expects
        the LN→Q edge to be direct. The Cast made the engine *build* (Q/DQ
        count looked healthy, ~3082) but the DQ scale was applied to the
        post-Cast tensor instead of the actually-quantized tensor → numerically
        broken → pure noise at inference. Per SDXL UNet structure every LN
        output feeds only QuantizeLinear (qkv/ff projections) which accepts
        FP32 directly, so the Cast was unnecessary.
        """
        import numpy as np

        promoted = 0
        out_promoted = 0
        non_q_consumers_seen = 0

        for node in self.graph.nodes:
            if node.op != "LayerNormalization":
                continue
            if not node.inputs:
                continue

            for param in node.inputs[1:]:  # scale, then optional bias
                if param is None or not hasattr(param, "values") or param.values is None:
                    continue
                if param.values.dtype == np.float16:
                    param.values = param.values.astype(np.float32)
                    promoted += 1

            out_var = node.outputs[0]
            if hasattr(out_var, "dtype") and out_var.dtype == np.float16:
                out_var.dtype = np.float32
                out_promoted += 1

            # Sanity: warn if any LN feeds something other than QuantizeLinear,
            # since FP32 promotion of the output edge could then introduce a
            # downstream type mismatch the original Cast was masking.
            for consumer in self.graph.nodes:
                if out_var in consumer.inputs and consumer.op != "QuantizeLinear":
                    non_q_consumers_seen += 1

        if promoted or out_promoted:
            logger.info(
                f"[Optimizer] fix_layernorm_dtypes: promoted {promoted} initializer(s) "
                f"and {out_promoted} LN output dtype(s) FP16→FP32 (no Cast insertion)"
            )
        if non_q_consumers_seen:
            logger.warning(
                f"[Optimizer] fix_layernorm_dtypes: {non_q_consumers_seen} non-QuantizeLinear "
                f"LN consumer(s) detected — FP32 LN output may need a downstream Cast for "
                f"STRONGLY_TYPED build to succeed. Standard SDXL UNet should report 0."
            )
        if return_onnx:
            return gs.export_onnx(self.graph)


class BaseModel:
    def __init__(
        self,
        fp16=False,
        device="cuda",
        verbose=True,
        max_batch_size=4,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
    ):
        self.name = "SD Model"
        self.fp16 = fp16
        self.device = device
        self.verbose = verbose

        self.min_batch = min_batch_size
        self.max_batch = max_batch_size
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_model(self):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ": original")
        if len(opt.graph.nodes) == 0:
            # Guards against a corrupt/truncated source ONNX (e.g. left behind by an
            # interrupted export) slipping into fold_constants, where it would otherwise
            # surface as an opaque polygraphy "'NoneType' object has no attribute 'graph'"
            # once ORT shape inference bails on the missing/invalid opset.
            raise RuntimeError(
                f"{self.name}: input ONNX graph has 0 nodes -- the source ONNX is empty or "
                f"corrupt. Delete the cached .onnx file for this engine and rebuild."
            )
        opt.cleanup()
        opt.info(self.name + ": cleanup")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        if any(n.op in ("QuantizeLinear", "DequantizeLinear") for n in opt.graph.nodes):
            opt.fix_layernorm_dtypes()
            opt.info(self.name + ": fp8 LN dtype fix")
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        # Make batch size check more flexible for ONNX export
        if hasattr(self, "_allow_export_batch_override") and self._allow_export_batch_override:
            # During ONNX export, allow different batch sizes
            effective_min_batch = min(self.min_batch, batch_size)
            effective_max_batch = max(self.max_batch, batch_size)
        else:
            effective_min_batch = self.min_batch
            effective_max_batch = self.max_batch

        assert batch_size >= effective_min_batch and batch_size <= effective_max_batch, (
            f"Batch size {batch_size} not in range [{effective_min_batch}, {effective_max_batch}]"
        )
        assert image_height % 8 == 0 and image_width % 8 == 0, (
            f"image_height ({image_height}) and image_width ({image_width}) must both be divisible by 8"
        )
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        if static_batch:
            # Fully static: min=opt=max so TRT sees no symbolic batch dim.
            # Required for l2tc (L2 tiling) which checks that ALL dims are concrete.
            min_batch = batch_size
            max_batch = batch_size
        else:
            min_batch = self.min_batch
            max_batch = self.max_batch

        latent_height = image_height // 8
        latent_width = image_width // 8

        if static_shape:
            # Static: min=opt=max — TRT selects geometry-specific kernels,
            # enables L2 tiling, and CUDA graphs avoid worst-case allocation.
            min_image_height = max_image_height = image_height
            min_image_width = max_image_width = image_width
            min_latent_height = max_latent_height = latent_height
            min_latent_width = max_latent_width = latent_width
        else:
            # Dynamic: full range for runtime resolution flexibility
            min_image_height = self.min_image_shape
            max_image_height = self.max_image_shape
            min_image_width = self.min_image_shape
            max_image_width = self.max_image_shape
            min_latent_height = self.min_latent_shape
            max_latent_height = self.max_latent_shape
            min_latent_width = self.min_latent_shape
            max_latent_width = self.max_latent_shape

        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )


class CLIP(BaseModel):
    def __init__(self, device, max_batch_size, embedding_dim, min_batch_size=1):
        super(CLIP, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
        )
        self.name = "CLIP"

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        return ["text_embeddings", "pooler_output"]

    def get_dynamic_axes(self):
        return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(
            batch_size, image_height, image_width, static_batch, static_shape
        )
        return {
            "input_ids": [
                (min_batch, self.text_maxlen),
                (batch_size, self.text_maxlen),
                (max_batch, self.text_maxlen),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph)
        opt.info(self.name + ": original")
        opt.select_outputs([0])
        opt.cleanup()
        opt.info(self.name + ": remove output[1]")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        opt.select_outputs([0], names=["text_embeddings"])
        opt.info(self.name + ": remove output[0]")
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ": finished")
        return opt_onnx_graph


class SafetyChecker(BaseModel):
    def __init__(self, device, max_batch_size=1, min_batch_size=1):
        super(SafetyChecker, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
        )
        self.name = "safety_checker"

    def get_input_names(self):
        return ["clip_input"]

    def get_output_names(self):
        return ["has_nsfw_concepts"]

    def get_dynamic_axes(self):
        return {"clip_input": {0: "B"}}

    def get_input_profile(self, batch_size, *args, **kwargs):
        return {
            "clip_input": [
                (self.min_batch, 3, 224, 224),
                (batch_size, 3, 224, 224),
                (self.max_batch, 3, 224, 224),
            ],
        }

    def get_shape_dict(self, batch_size, *args, **kwargs):
        return {
            "clip_input": (batch_size, 3, 224, 224),
            "has_nsfw_concepts": (batch_size,),
        }

    def get_sample_input(self, batch_size, *args, **kwargs):
        return (torch.randn(batch_size, 3, 224, 224, dtype=torch.float16, device=self.device),)


class NSFWDetector(BaseModel):
    def __init__(self, device, max_batch_size=1, min_batch_size=1):
        super(NSFWDetector, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
        )
        self.name = "nsfw_detector"

    def get_input_names(self):
        return ["pixel_values"]

    def get_output_names(self):
        return ["logits"]

    def get_dynamic_axes(self):
        return {"pixel_values": {0: "B"}}

    def get_input_profile(self, batch_size, *args, **kwargs):
        return {
            "pixel_values": [
                (self.min_batch, 3, 448, 448),
                (batch_size, 3, 448, 448),
                (self.max_batch, 3, 448, 448),
            ],
        }

    def get_shape_dict(self, batch_size, *args, **kwargs):
        return {
            "pixel_values": (batch_size, 3, 448, 448),
            "logits": (batch_size, 2),
        }

    def get_sample_input(self, batch_size, *args, **kwargs):
        return (torch.randn(batch_size, 3, 448, 448, dtype=torch.float16, device=self.device),)


class UNet(BaseModel):
    def __init__(
        self,
        unet: UNet2DConditionModel = None,
        fp16=False,
        device="cuda",
        max_batch_size=4,
        min_batch_size=1,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4,
        use_control=False,
        unet_arch=None,
        image_height=512,
        image_width=512,
        use_ipadapter=False,
        num_image_tokens=4,
        num_ip_layers: int = None,
        use_cached_attn: bool = False,
        cache_maxframes: int = 1,
        min_cache_maxframes: int = 1,
        max_cache_maxframes: int = 4,
        use_feature_injection: bool = False,
        max_fi_up_blocks: int = 2,
    ):
        super(UNet, self).__init__(
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet = unet
        self.unet_dim = unet_dim
        self.name = "UNet"
        self.image_height = image_height
        self.image_width = image_width

        self.use_control = use_control
        self.unet_arch = unet_arch or {}
        self.use_ipadapter = use_ipadapter
        self.num_image_tokens = num_image_tokens
        self.num_ip_layers = num_ip_layers

        # Baked-in IPAdapter configuration
        if self.use_ipadapter:
            # With baked-in processors, we extend text_maxlen to include image tokens
            # TODO: Consider making this dynamic instead of fixed per IPAdapter variant
            # Could use dynamic shapes: min=77 (text only), max=93 (text + 16 tokens)
            # This would allow a single engine to handle all IPAdapter types instead of separate engines
            self.text_maxlen = text_maxlen + self.num_image_tokens
            if self.num_ip_layers is None:
                raise ValueError("UNet model requires num_ip_layers when use_ipadapter=True")

        if self.use_control and self.unet_arch:
            self.control_inputs = self.get_control(image_height, image_width)
            self._add_control_inputs()
        else:
            self.control_inputs = {}

        self.use_cached_attn = use_cached_attn
        self.cache_maxframes = cache_maxframes
        self.min_cache_maxframes = min_cache_maxframes
        self.max_cache_maxframes = max_cache_maxframes
        if self.use_cached_attn and self.unet is not None:
            from .utils import get_kvo_cache_info

            self.kvo_cache_shapes, self.kvo_cache_structure, self.kvo_cache_count = get_kvo_cache_info(
                self.unet, image_height, image_width
            )

            self.min_kvo_cache_shapes, _, _ = get_kvo_cache_info(self.unet, image_height, image_width)
            self.max_kvo_cache_shapes, _, _ = get_kvo_cache_info(self.unet, image_height, image_width)

        # Feature Injection output-cache (requires use_cached_attn and a live unet)
        self.use_feature_injection = use_feature_injection and use_cached_attn
        self.max_fi_up_blocks = max_fi_up_blocks
        if self.use_feature_injection and self.unet is not None:
            from .utils import get_fi_eligible_mask

            self.fi_eligible_mask = get_fi_eligible_mask(self.unet, image_height, image_width, max_fi_up_blocks)
            # fi_layer_indices: global kvo-layer index for each FI-eligible layer.
            # Used by wrapper.py to allocate fi_cache tensors in walk order.
            # Engine binding names use fi-local sequential indices (fio_cache_in_0 …)
            # rather than global indices — the engine only needs the count.
            self.fi_layer_indices = [i for i, e in enumerate(self.fi_eligible_mask) if e]
            self.fi_cache_count = len(self.fi_layer_indices)
            # Shapes in fi-local order (same walk order as fi_layer_indices)
            self.fi_cache_shapes = [self.kvo_cache_shapes[i] for i in self.fi_layer_indices]

    @property
    def has_symbolic_cache_dims(self) -> bool:
        """Whether the KVO/FI cache-frames axis (kvo "C" / fio "FC" in get_dynamic_axes)
        is still symbolic. True unless pin_cache_frames has pinned
        min_cache_maxframes == max_cache_maxframes, in which case the axis is dropped
        from get_dynamic_axes and the profile collapses to a single concrete shape —
        required for TRT's l2tc (L2 tiling) pass to validate a fully-static graph."""
        if not self.use_cached_attn:
            return False
        return self.min_cache_maxframes != self.max_cache_maxframes

    def get_control(self, image_height: int = 512, image_width: int = 512) -> dict:
        """Generate ControlNet input configurations with dynamic spatial dimensions based on input resolution."""
        block_out_channels = self.unet_arch.get("block_out_channels", (320, 640, 1280, 1280))

        # Calculate latent space dimensions
        latent_height = image_height // 8
        latent_width = image_width // 8

        control_inputs = {}

        if len(block_out_channels) == 3:
            # SDXL architecture: Match UNet's exact down_block_res_samples structure
            # UNet down_block_res_samples = [initial_sample] + [block0_residuals] + [block1_residuals] + [block2_residuals]
            # Pattern: [88x88] + [88x88, 88x88, 44x44] + [44x44, 44x44, 22x22] + [22x22, 22x22]
            # Total: 9 control tensors needed
            control_tensors = [
                # Initial sample (after conv_in: 4->320 channels, no downsampling)
                (block_out_channels[0], 1),  # 320 channels, 88x88
                # Block 0 residuals (320 channels)
                (block_out_channels[0], 1),  # 320 channels, 88x88
                (block_out_channels[0], 1),  # 320 channels, 88x88
                (block_out_channels[0], 2),  # 320 channels, 44x44 (downsampled)
                # Block 1 residuals (640 channels)
                (block_out_channels[1], 2),  # 640 channels, 44x44
                (block_out_channels[1], 2),  # 640 channels, 44x44
                (block_out_channels[1], 4),  # 640 channels, 22x22 (downsampled)
                # Block 2 residuals (1280 channels)
                (block_out_channels[2], 4),  # 1280 channels, 22x22
                (block_out_channels[2], 4),  # 1280 channels, 22x22
            ]
        else:
            # SD1.5/SD2.1 architecture: 4 down blocks with 12 control tensors
            control_tensors = [
                # Block 0: No downsampling from latent space (factor = 1)
                (320, 1),
                (320, 1),
                (320, 1),
                # Block 1: 2x downsampling from latent space (factor = 2)
                (320, 2),
                (640, 2),
                (640, 2),
                # Block 2: 4x downsampling from latent space (factor = 4)
                (640, 4),
                (1280, 4),
                (1280, 4),
                # Block 3: 8x downsampling from latent space (factor = 8)
                (1280, 8),
                (1280, 8),
                (1280, 8),
            ]

        # Generate control inputs with proper spatial dimensions
        for i, (channels, downsample_factor) in enumerate(control_tensors):
            input_name = f"input_control_{i:02d}"

            # Calculate spatial dimensions for this level
            control_height = max(1, latent_height // downsample_factor)
            control_width = max(1, latent_width // downsample_factor)

            control_inputs[input_name] = {
                "batch": self.min_batch,
                "channels": channels,
                "height": control_height,
                "width": control_width,
                "downsampling_factor": downsample_factor,
            }

        # Middle block uses the most downsampled resolution based on architecture
        if len(block_out_channels) == 3:
            # SDXL: middle block at 4x downsampling (after 3 down blocks)
            middle_downsample_factor = 4
        else:
            # SD1.5: middle block at 8x downsampling (after 4 down blocks)
            middle_downsample_factor = 8

        control_inputs["input_control_middle"] = {
            "batch": self.min_batch,
            "channels": 1280,
            "height": max(1, latent_height // middle_downsample_factor),
            "width": max(1, latent_width // middle_downsample_factor),
            "downsampling_factor": middle_downsample_factor,
        }

        return control_inputs

    def get_kvo_cache_names(self, in_out: str):
        return [f"kvo_cache_{in_out}_{idx}" for idx in range(self.kvo_cache_count)]

    def get_fi_cache_names(self, in_out: str):
        """Return FI output-cache binding names using fi-local sequential indices.

        Sequential local indices (0, 1, … fi_cache_count-1) keep the engine runtime
        simple: like kvo, it only needs the count to reconstruct binding names and
        does not need to know global kvo-layer indices.
        """
        return [f"fio_cache_{in_out}_{i}" for i in range(self.fi_cache_count)]

    def get_fi_cache_input_profile(self, min_batch, batch_size, max_batch):
        """TRT input-profile triples for each fio_cache_in binding.

        Shape: (cache_maxframes, batch, seq, hidden) — no K/V pair dim.
        cache_maxframes is dynamic (min/opt/max follow kvo_cache convention).
        """
        profiles = []
        for global_idx in self.fi_layer_indices:
            shape = self.kvo_cache_shapes[global_idx]
            profile = [
                (self.min_cache_maxframes, min_batch, shape[0], shape[1]),
                (self.cache_maxframes, batch_size, shape[0], shape[1]),
                (self.max_cache_maxframes, max_batch, shape[0], shape[1]),
            ]
            profiles.append(profile)
        return profiles

    def _add_control_inputs(self):
        """Add ControlNet inputs to the model's input/output specifications"""
        if not self.control_inputs:
            return

        self._original_get_input_names = self.get_input_names
        self._original_get_dynamic_axes = self.get_dynamic_axes
        self._original_get_input_profile = self.get_input_profile
        self._original_get_shape_dict = self.get_shape_dict
        self._original_get_sample_input = self.get_sample_input

    def get_input_names(self):
        """Get input names including ControlNet inputs"""
        base_names = ["sample", "timestep", "encoder_hidden_states"]
        if self.use_ipadapter:
            base_names.append("ipadapter_scale")
            try:
                import logging

                logging.getLogger(__name__).debug(f"TRT Models: get_input_names with ipadapter -> {base_names}")
            except Exception:
                pass
        if self.use_control and self.control_inputs:
            control_names = sorted(self.control_inputs.keys())
            base_names = base_names + control_names
        if self.use_cached_attn:
            base_names = base_names + self.get_kvo_cache_names("in")
        if self.use_feature_injection:
            # FI output-cache inputs, then scalar tunables (fi_strength, fi_threshold)
            base_names = base_names + self.get_fi_cache_names("in")
            base_names = base_names + ["fi_strength", "fi_threshold"]
        return base_names

    def get_output_names(self):
        base_names = ["latent"]
        if self.use_cached_attn:
            base_names = base_names + self.get_kvo_cache_names("out")
        if self.use_feature_injection:
            base_names = base_names + self.get_fi_cache_names("out")
        return base_names

    def get_kvo_cache_input_profile(self, min_batch, batch_size, max_batch):
        profiles = []
        for min_shape, shape, max_shape in zip(
            self.min_kvo_cache_shapes, self.kvo_cache_shapes, self.max_kvo_cache_shapes
        ):
            profile = [
                (2, self.min_cache_maxframes, min_batch, min_shape[0], min_shape[1]),
                (2, self.cache_maxframes, batch_size, shape[0], shape[1]),
                (2, self.max_cache_maxframes, max_batch, max_shape[0], max_shape[1]),
            ]
            profiles.append(profile)
        return profiles

    def get_dynamic_axes(self):
        base_axes = {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "timestep": {0: "2B"},
            "encoder_hidden_states": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
        }
        if self.use_ipadapter:
            base_axes["ipadapter_scale"] = {0: "L_ip"}
            try:
                import logging

                logging.getLogger(__name__).debug(
                    f"TRT Models: dynamic axes include ipadapter_scale with L_ip={getattr(self, 'num_ip_layers', None)}"
                )
            except Exception:
                pass

        if self.use_control and self.control_inputs:
            for name, shape_spec in self.control_inputs.items():
                height = shape_spec["height"]
                width = shape_spec["width"]
                spatial_suffix = f"{height}x{width}"
                base_axes[name] = {0: "2B", 2: f"H_{spatial_suffix}", 3: f"W_{spatial_suffix}"}
        if self.use_cached_attn:
            # hardcoded resolution for now due to VRAM limitations
            # NOTE: dim[0]=2 (K/V pair) must stay static — attention Gather nodes
            # index into it at idx=0 and idx=1, so dim[0]<2 causes OOB errors.
            # The "C" (cache-frames) axis is itself dropped when pin_cache_frames has
            # pinned min_cache_maxframes == max_cache_maxframes (has_symbolic_cache_dims
            # False) so the exported graph has no symbolic dims left and TRT's l2tc
            # (L2 tiling) pass can validate.
            for i in range(self.kvo_cache_count):
                base_axes[f"kvo_cache_in_{i}"] = {1: "C", 2: "2B"} if self.has_symbolic_cache_dims else {2: "2B"}
                base_axes[f"kvo_cache_out_{i}"] = {2: "2B"}

        if self.use_feature_injection:
            # fio_cache shape: (maxframes, batch, S, H) — no K/V dim
            # dim 0 (maxframes) and dim 1 (batch) are dynamic; dims 2,3 (S,H) are static.
            # See kvo note above: "FC" is dropped when has_symbolic_cache_dims is False.
            for i in range(self.fi_cache_count):
                base_axes[f"fio_cache_in_{i}"] = {0: "FC", 1: "2B"} if self.has_symbolic_cache_dims else {1: "2B"}
                base_axes[f"fio_cache_out_{i}"] = {1: "2B"}
            # fi_strength, fi_threshold: static [1] — no dynamic axes

        return base_axes

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        # Following TensorRT documentation: ensure proper min ≤ opt ≤ max constraints for ALL dimensions
        # Calculate optimal latent dimensions that fall within min/max range
        opt_latent_height = min(max(latent_height, min_latent_height), max_latent_height)
        opt_latent_width = min(max(latent_width, min_latent_width), max_latent_width)

        # For dynamic shapes, ensure opt != min to satisfy TRT constraint (min < opt <= max).
        # For static shapes min == opt == max is correct and intentional — skip separation.
        if not static_shape:
            if opt_latent_height == min_latent_height and min_latent_height < max_latent_height:
                opt_latent_height = min(min_latent_height + 8, max_latent_height)
            if opt_latent_width == min_latent_width and min_latent_width < max_latent_width:
                opt_latent_width = min(min_latent_width + 8, max_latent_width)

        # Image dimensions for ControlNet inputs
        if static_shape:
            min_image_h = max_image_h = image_height
            min_image_w = max_image_w = image_width
            opt_image_height = image_height
            opt_image_width = image_width
        else:
            min_image_h, max_image_h = self.min_image_shape, self.max_image_shape
            min_image_w, max_image_w = self.min_image_shape, self.max_image_shape
            opt_image_height = min(max(image_height, min_image_h), max_image_h)
            opt_image_width = min(max(image_width, min_image_w), max_image_w)
            if opt_image_height == min_image_h and min_image_h < max_image_h:
                opt_image_height = min(min_image_h + 64, max_image_h)
            if opt_image_width == min_image_w and min_image_w < max_image_w:
                opt_image_width = min(min_image_w + 64, max_image_w)

        profile = {
            "sample": [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width),
                (batch_size, self.unet_dim, opt_latent_height, opt_latent_width),
                (max_batch, self.unet_dim, max_latent_height, max_latent_width),
            ],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "encoder_hidden_states": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim),
            ],
        }
        if self.use_ipadapter:
            # scalar per-layer vector, length fixed to num_ip_layers
            profile["ipadapter_scale"] = [
                (1,),
                (self.num_ip_layers,),
                (self.num_ip_layers,),
            ]
            try:
                import logging

                logging.getLogger(__name__).debug(
                    f"TRT Models: profile ipadapter_scale min/opt/max={(1,), (self.num_ip_layers,), (self.num_ip_layers,)}"
                )
            except Exception:
                pass

        if self.use_control and self.control_inputs:
            # Use the actual calculated spatial dimensions for each ControlNet input
            # Each control input has its own specific spatial resolution based on UNet architecture
            for name, shape_spec in self.control_inputs.items():
                channels = shape_spec["channels"]
                control_height = shape_spec["height"]
                control_width = shape_spec["width"]

                if static_shape:
                    # Static: all three identical — exact resolution, no padding
                    min_control_h = max_control_h = opt_control_h = control_height
                    min_control_w = max_control_w = opt_control_w = control_width
                else:
                    # Dynamic: scale proportionally with latent range
                    scale_h = opt_latent_height / latent_height if latent_height > 0 else 1.0
                    scale_w = opt_latent_width / latent_width if latent_width > 0 else 1.0

                    min_control_h = max(1, int(control_height * min_latent_height / latent_height))
                    max_control_h = max(min_control_h + 1, int(control_height * max_latent_height / latent_height))
                    opt_control_h = max(min_control_h, min(int(control_height * scale_h), max_control_h))

                    min_control_w = max(1, int(control_width * min_latent_width / latent_width))
                    max_control_w = max(min_control_w + 1, int(control_width * max_latent_width / latent_width))
                    opt_control_w = max(min_control_w, min(int(control_width * scale_w), max_control_w))

                profile[name] = [
                    (min_batch, channels, min_control_h, min_control_w),  # min
                    (batch_size, channels, opt_control_h, opt_control_w),  # opt
                    (max_batch, channels, max_control_h, max_control_w),  # max
                ]
        if self.use_cached_attn:
            profile.update(
                zip(
                    self.get_kvo_cache_names("in"),
                    self.get_kvo_cache_input_profile(min_batch, batch_size, max_batch),
                )
            )

        if self.use_feature_injection:
            profile.update(
                zip(
                    self.get_fi_cache_names("in"),
                    self.get_fi_cache_input_profile(min_batch, batch_size, max_batch),
                )
            )
            # fi_strength and fi_threshold are static [1] fp32 scalars
            profile["fi_strength"] = [(1,), (1,), (1,)]
            profile["fi_threshold"] = [(1,), (1,), (1,)]

        return profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        shape_dict = {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (2 * batch_size,),
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }
        if self.use_ipadapter:
            shape_dict["ipadapter_scale"] = (self.num_ip_layers,)
            try:
                import logging

                logging.getLogger(__name__).debug(f"TRT Models: shape_dict ipadapter_scale={(self.num_ip_layers,)}")
            except Exception:
                pass

        if self.use_control and self.control_inputs:
            # Use the actual calculated spatial dimensions for each ControlNet input
            for name, shape_spec in self.control_inputs.items():
                channels = shape_spec["channels"]
                control_height = shape_spec["height"]
                control_width = shape_spec["width"]
                shape_dict[name] = (2 * batch_size, channels, control_height, control_width)

        if self.use_cached_attn:
            for in_name, out_name, shape in zip(
                self.get_kvo_cache_names("in"), self.get_kvo_cache_names("out"), self.kvo_cache_shapes
            ):
                shape_dict[in_name] = (2, self.cache_maxframes, batch_size, shape[0], shape[1])
                shape_dict[out_name] = (2, 1, batch_size, shape[0], shape[1])

        if self.use_feature_injection:
            for in_name, out_name, shape in zip(
                self.get_fi_cache_names("in"), self.get_fi_cache_names("out"), self.fi_cache_shapes
            ):
                # fio_cache_in: all cached frames; fio_cache_out: current frame only
                shape_dict[in_name] = (self.cache_maxframes, batch_size, shape[0], shape[1])
                shape_dict[out_name] = (1, batch_size, shape[0], shape[1])
            shape_dict["fi_strength"] = (1,)
            shape_dict["fi_threshold"] = (1,)

        return shape_dict

    def get_sample_input(self, batch_size, image_height, image_width):
        # Enable flexible batch size checking for ONNX export
        self._allow_export_batch_override = True

        try:
            latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        finally:
            # Clean up the override flag
            if hasattr(self, "_allow_export_batch_override"):
                delattr(self, "_allow_export_batch_override")

        dtype = torch.float16 if self.fp16 else torch.float32

        # Use smaller batch size for memory efficiency during ONNX export
        export_batch_size = min(batch_size, 1)  # Use batch size 1 for ONNX export to save memory

        base_inputs = [
            # sample dtype matches self.fp16 so the ONNX `sample` input is FP16 when
            # the unet runs FP16 — eliminates an FP32→FP16 Cast at conv_in, and
            # avoids a dtype mismatch when modelopt's ORT inference probe (used in
            # FP8 calibration) feeds FP16 captures into the graph.
            torch.randn(
                2 * export_batch_size,
                self.unet_dim,
                latent_height,
                latent_width,
                dtype=dtype,
                device=self.device,
            ),
            # timestep stays FP32 — diffusers' sinusoidal time_proj needs FP32 for
            # numerical stability; this is also what the FP16 unet expects upstream.
            torch.ones((2 * export_batch_size,), dtype=torch.float32, device=self.device),
            torch.randn(2 * export_batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
        ]

        if self.use_ipadapter:
            base_inputs.append(torch.ones(self.num_ip_layers, dtype=torch.float32, device=self.device))

        if self.use_control and self.control_inputs:
            control_inputs = []

            # Use the ACTUAL calculated spatial dimensions for each control input
            # This ensures each control input matches its expected UNet feature map resolution

            for name in sorted(self.control_inputs.keys()):
                shape_spec = self.control_inputs[name]
                channels = shape_spec["channels"]

                # KEY FIX: Use the specific spatial dimensions calculated for this control input
                control_height = shape_spec["height"]
                control_width = shape_spec["width"]

                control_input = torch.randn(
                    2 * export_batch_size, channels, control_height, control_width, dtype=dtype, device=self.device
                )
                control_inputs.append(control_input)

                # Clear cache periodically to prevent memory buildup
                if len(control_inputs) % 4 == 0:
                    torch.cuda.empty_cache()

            base_inputs = base_inputs + control_inputs

        if self.use_cached_attn:
            base_inputs = base_inputs + [
                torch.randn(
                    2, self.cache_maxframes, 2 * export_batch_size, shape[0], shape[1], dtype=torch.float16
                ).to(self.device)
                for shape in self.kvo_cache_shapes
            ]

        if self.use_feature_injection:
            # FI output cache — zeros so the first frame sees no ghost features
            base_inputs = base_inputs + [
                torch.zeros(self.cache_maxframes, 2 * export_batch_size, shape[0], shape[1], dtype=torch.float16).to(
                    self.device
                )
                for shape in self.fi_cache_shapes
            ]
            # fi_strength default 0.75 (thesis α=0.75); fi_threshold 0.98
            base_inputs.append(torch.tensor([0.75], dtype=torch.float32, device=self.device))
            base_inputs.append(torch.tensor([0.98], dtype=torch.float32, device=self.device))

        return tuple(base_inputs)


class VAE(BaseModel):
    def __init__(self, device, max_batch_size, min_batch_size=1):
        super(VAE, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=None,
        )
        self.name = "VAE decoder"

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {
            "latent": {0: "B", 2: "H", 3: "W"},
            "images": {0: "B", 2: "8H", 3: "8W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            _,
            _,
            _,
            _,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "latent": [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(
            batch_size,
            4,
            latent_height,
            latent_width,
            dtype=torch.float32,
            device=self.device,
        )


class VAEEncoder(BaseModel):
    def __init__(self, device, max_batch_size, min_batch_size=1):
        super(VAEEncoder, self).__init__(
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            embedding_dim=None,
        )
        self.name = "VAE encoder"

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "images": {0: "B", 2: "8H", 3: "8W"},
            "latent": {0: "B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            _,
            _,
            _,
            _,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            "images": [
                (min_batch, 3, min_image_height, min_image_width),
                (batch_size, 3, image_height, image_width),
                (max_batch, 3, max_image_height, max_image_width),
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "images": (batch_size, 3, image_height, image_width),
            "latent": (batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(
            batch_size,
            3,
            image_height,
            image_width,
            dtype=torch.float32,
            device=self.device,
        )
