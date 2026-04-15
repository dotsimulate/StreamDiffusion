"""
Depth Anything TensorRT Engine Builder

Builds TensorRT engines for the Depth Anything model used by depth_tensorrt preprocessor.
Based on: https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt

Usage:
    python -m streamdiffusion.tools.compile_depth_anything_tensorrt --output_dir ./engines/preprocessors
    python -m streamdiffusion.tools.compile_depth_anything_tensorrt --model_size small --resolution 518
"""

import torch
import logging
import os
from pathlib import Path
from typing import Optional
import fire

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available. Please install it first.")

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Depth Anything model configs
DEPTH_ANYTHING_MODELS = {
    "small": {
        "repo": "LiheYoung/depth-anything-small-hf",
        "input_size": 518,
    },
    "base": {
        "repo": "LiheYoung/depth-anything-base-hf",
        "input_size": 518,
    },
    "large": {
        "repo": "LiheYoung/depth-anything-large-hf",
        "input_size": 518,
    },
}


def export_depth_anything_to_onnx(
    onnx_path: Path,
    model_size: str = "small",
    resolution: int = 518,
    device: str = "cuda"
) -> bool:
    """Export Depth Anything model to ONNX format"""
    try:
        from transformers import AutoModelForDepthEstimation
    except ImportError:
        logger.error("transformers library required. Install with: pip install transformers")
        return False

    if model_size not in DEPTH_ANYTHING_MODELS:
        logger.error(f"Unknown model size: {model_size}. Choose from: {list(DEPTH_ANYTHING_MODELS.keys())}")
        return False

    model_config = DEPTH_ANYTHING_MODELS[model_size]
    repo_id = model_config["repo"]

    logger.info(f"Loading Depth Anything {model_size} from {repo_id}...")

    try:
        model = AutoModelForDepthEstimation.from_pretrained(repo_id)
        model = model.to(device)
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, resolution, resolution, device=device)

        logger.info(f"Exporting to ONNX: {onnx_path}")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

        logger.info("ONNX export successful")
        return True

    except Exception as e:
        logger.error(f"Failed to export ONNX: {e}")
        import traceback
        traceback.print_exc()
        return False


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    resolution: int = 518,
    fp16: bool = True,
) -> bool:
    """Build TensorRT engine from ONNX model"""
    if not TENSORRT_AVAILABLE:
        logger.error("TensorRT not available")
        return False

    logger.info(f"Building TensorRT engine: {engine_path}")

    try:
        trt_logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt_logger)

        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(i)}")
                return False

        # Build config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Using FP16 precision")

        # Set optimization profile for fixed resolution
        profile = builder.create_optimization_profile()
        profile.set_shape("input",
                          (1, 3, resolution, resolution),  # min
                          (1, 3, resolution, resolution),  # opt
                          (1, 3, resolution, resolution))  # max
        config.add_optimization_profile(profile)

        # Build engine
        logger.info("Building engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            logger.error("Failed to build TensorRT engine")
            return False

        # Save engine
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        logger.info(f"TensorRT engine saved: {engine_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to build TensorRT engine: {e}")
        import traceback
        traceback.print_exc()
        return False


def compile_depth_anything(
    output_dir: str = "./engines/preprocessors",
    model_size: str = "small",
    resolution: int = 518,
    fp16: bool = True,
    keep_onnx: bool = False,
    device: str = "cuda",
):
    """
    Compile Depth Anything model to TensorRT engine

    Args:
        output_dir: Directory to save the engine
        model_size: Model size (small, base, large)
        resolution: Input resolution (default 518 for Depth Anything)
        fp16: Use FP16 precision
        keep_onnx: Keep intermediate ONNX file
        device: Device for export (cuda recommended)
    """
    if not TENSORRT_AVAILABLE:
        logger.error("TensorRT is required. Install with: pip install tensorrt")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    engine_name = f"depth_anything_{model_size}_{resolution}.engine"
    onnx_name = f"depth_anything_{model_size}_{resolution}.onnx"

    onnx_path = output_path / onnx_name
    engine_path = output_path / engine_name

    # Check if engine already exists
    if engine_path.exists():
        logger.info(f"Engine already exists: {engine_path}")
        overwrite = input("Overwrite? (y/N): ").lower().strip() == 'y'
        if not overwrite:
            return

    # Export to ONNX
    logger.info("Step 1/2: Exporting model to ONNX...")
    if not export_depth_anything_to_onnx(onnx_path, model_size, resolution, device):
        logger.error("ONNX export failed")
        return

    # Build TensorRT engine
    logger.info("Step 2/2: Building TensorRT engine...")
    if not build_tensorrt_engine(onnx_path, engine_path, resolution, fp16):
        logger.error("TensorRT build failed")
        return

    # Cleanup ONNX if not keeping
    if not keep_onnx and onnx_path.exists():
        onnx_path.unlink()
        logger.info("Removed intermediate ONNX file")

    logger.info(f"\nSuccess! Engine saved to: {engine_path}")
    logger.info(f"\nTo use in config:")
    logger.info(f'  preprocessor: "depth_tensorrt"')
    logger.info(f'  preprocessor_params:')
    logger.info(f'    engine_path: "{engine_path}"')


if __name__ == "__main__":
    fire.Fire(compile_depth_anything)
