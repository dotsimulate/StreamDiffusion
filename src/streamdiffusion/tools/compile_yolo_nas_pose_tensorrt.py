"""
YOLO-NAS Pose TensorRT Engine Builder

Builds the TensorRT engine consumed by the `pose_tensorrt` preprocessor
(streamdiffusion.preprocessing.processors.pose_tensorrt.YoloNasPoseTensorrtPreprocessor).
Based on: https://github.com/yuvraj108c/ComfyUI-YoloNasPose-Tensorrt

Unlike compile_depth_anything_tensorrt.py, this tool does not export a torch model to ONNX:
the repo has no YOLO-NAS torch checkpoint, and neither `super-gradients` nor `ultralytics`
(the libraries that could produce one) is installed. Instead it downloads the prebuilt,
publicly available ONNX export that the ported ComfyUI-YoloNasPose-Tensorrt implementation
was written against.

Usage:
    python -m streamdiffusion.tools.compile_yolo_nas_pose_tensorrt --output_dir ./engines/td/preprocessors
    python -m streamdiffusion.tools.compile_yolo_nas_pose_tensorrt --model_size l --resolution 640
"""

import importlib.util
import logging
import shutil
from pathlib import Path

import fire

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    import tensorrt as trt

    from streamdiffusion.acceleration.tensorrt.utilities import BUILD_TRT_LOGGER

    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    BUILD_TRT_LOGGER = None
    logger.warning("TensorRT not available. Please install it first.")

ONNX_AVAILABLE = importlib.util.find_spec("onnx") is not None

# Public, ungated HF Hub repo publishing prebuilt YOLO-NAS Pose ONNX exports at several
# score-threshold variants. Filenames encode {model_size}_{score_threshold}. Only the "l"
# (large) / 0.8 variant has been verified downloadable and matches this repo's own example
# engine basename (yolo_nas_pose_l_0.8-fp16.engine, see configs/*.yaml.example).
YOLO_NAS_POSE_HF_REPO = "yuvraj108c/yolo-nas-pose-onnx"
YOLO_NAS_POSE_VARIANTS = {
    "l": "yolo_nas_pose_l_0.8.onnx",
}


def export_yolo_nas_pose_to_onnx(
    onnx_path: Path, model_size: str = "l", resolution: int = 640, device: str = "cuda"
) -> bool:
    """
    Stage a prebuilt YOLO-NAS Pose ONNX at `onnx_path` by downloading it from the HF Hub.

    Named/shaped like the other tools' `export_*` functions (and called identically by
    td_manager.py's `external` build_registry dispatch: `export_fn(onnx_path, model_size,
    resolution, 'cuda')`) even though nothing is actually exported here — there is no local
    torch model to export from.

    `device` is accepted only for call-signature compatibility with that dispatch; a download
    has no device to run on.
    """
    del device  # unused; kept for signature parity with the `external` build_registry dispatch

    if model_size not in YOLO_NAS_POSE_VARIANTS:
        logger.error(f"Unknown model_size: {model_size}. Choose from: {list(YOLO_NAS_POSE_VARIANTS.keys())}")
        return False

    if resolution != 640:
        logger.warning(
            f"resolution={resolution} was requested, but the published ONNX is a fixed 640x640 "
            "export -- the downloaded model will still be 640x640 regardless of this argument."
        )

    filename = YOLO_NAS_POSE_VARIANTS[model_size]

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub is required to download the YOLO-NAS Pose ONNX. Install with: pip install huggingface_hub"
        )
        return False

    try:
        logger.info(
            f"Downloading {filename} from {YOLO_NAS_POSE_HF_REPO} (reuses the shared HF cache on repeat calls)..."
        )
        downloaded_path = hf_hub_download(repo_id=YOLO_NAS_POSE_HF_REPO, filename=filename)
    except Exception as e:
        logger.error(f"Failed to download YOLO-NAS Pose ONNX from {YOLO_NAS_POSE_HF_REPO}: {e}")
        return False

    try:
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded_path, onnx_path)
    except Exception as e:
        logger.error(f"Failed to stage downloaded ONNX at {onnx_path}: {e}")
        return False

    logger.info(f"YOLO-NAS Pose ONNX staged at: {onnx_path}")
    return True


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    resolution: int = 640,
    fp16: bool = True,
) -> bool:
    """
    Build a TensorRT engine from the YOLO-NAS Pose ONNX model.

    Deliberately deviates from compile_depth_anything_tensorrt.build_tensorrt_engine in two
    ways (see plan i-ve-noticed-that-in-graceful-kettle.md, Change 1):

    1. The optimization profile is added only if the parsed network's input is actually
       dynamic (a dim == -1). The published ONNX is expected to be static (1,3,640,640), and
       unconditionally calling profile.set_shape() for a fully-static input errors in TRT 10.x
       the way compile_depth_anything_tensorrt.py does it (that tool's ONNX has dynamic axes,
       so it never hits this).
    2. The engine's I/O contract is asserted right after parsing, not left to fail on frame 1.
       pose_tensorrt.py's _process_core/_process_tensor_core both call
       `self.engine.infer({"input": image_resized_uint8}, cuda_stream)` with a hardcoded input
       name "input" and a uint8 tensor -- if the downloaded ONNX doesn't match that contract,
       failing here (at build time, with an explicit message) is the entire point of this
       change: it converts "silent skip -> 17-minute-delayed per-frame crash" into "loud
       failure at build time", which is the same fix this tool exists to make for pose_tensorrt
       in the first place.
    """
    if not TENSORRT_AVAILABLE:
        logger.error("TensorRT not available")
        return False

    logger.info(f"Building TensorRT engine: {engine_path}")

    try:
        builder = trt.Builder(BUILD_TRT_LOGGER)
        network = builder.create_network()  # EXPLICIT_BATCH deprecated/ignored in TRT 10.x
        parser = trt.OnnxParser(network, BUILD_TRT_LOGGER)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(i)}")
                return False

        if network.num_inputs == 0:
            logger.error("Parsed network has no inputs -- cannot validate the I/O contract")
            return False

        input_tensor = network.get_input(0)
        input_shape = tuple(input_tensor.shape)
        expected_shape = (1, 3, resolution, resolution)

        logger.info(f"Parsed input: name={input_tensor.name!r} dtype={input_tensor.dtype} shape={input_shape}")

        if input_tensor.name != "input":
            logger.error(
                f"YOLO-NAS Pose ONNX input is named {input_tensor.name!r}, but pose_tensorrt.py's "
                "infer() call is hardcoded to the key 'input'. Either the published ONNX changed "
                "or the wrong file was downloaded -- refusing to build a mismatched engine."
            )
            return False

        if input_tensor.dtype != trt.DataType.UINT8:
            logger.error(
                f"YOLO-NAS Pose ONNX input dtype is {input_tensor.dtype}, but pose_tensorrt.py "
                "always sends a uint8 tensor. Refusing to build an engine that would only move "
                "today's frame-1 failure to a dtype mismatch instead of fixing it."
            )
            return False

        is_dynamic = any(d == -1 for d in input_shape)

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Using FP16 precision")

        if is_dynamic:
            logger.info(
                f"Input has dynamic dims {input_shape} -- adding a fixed optimization profile at {expected_shape}"
            )
            profile = builder.create_optimization_profile()
            profile.set_shape("input", expected_shape, expected_shape, expected_shape)
            config.add_optimization_profile(profile)
        else:
            if input_shape != expected_shape:
                logger.error(
                    f"YOLO-NAS Pose ONNX has a static input shape {input_shape}, but "
                    f"pose_tensorrt.py expects {expected_shape} (resolution={resolution}). Adjust "
                    "the preprocessor's detect_resolution to match, or use a different export."
                )
                return False
            logger.info(f"Input is static {input_shape} -- no optimization profile needed")

        logger.info("Building engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            logger.error("Failed to build TensorRT engine")
            return False

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        logger.info(f"TensorRT engine saved: {engine_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to build TensorRT engine: {e}")
        import traceback

        traceback.print_exc()
        return False


def compile_yolo_nas_pose(
    output_dir: str = "./engines/td/preprocessors",
    model_size: str = "l",
    resolution: int = 640,
    fp16: bool = True,
    keep_onnx: bool = False,
    device: str = "cuda",
):
    """
    Compile YOLO-NAS Pose to a TensorRT engine for the `pose_tensorrt` preprocessor.

    Args:
        output_dir: Directory to save the engine (should match the ControlNet config's
                     preprocessor_params.engine_path parent, e.g. engines/td/preprocessors)
        model_size: Published ONNX variant key -- see YOLO_NAS_POSE_VARIANTS
        resolution: Expected square input resolution; must match pose_tensorrt.py's
                    detect_resolution (default 640)
        fp16: Use FP16 precision
        keep_onnx: Keep the intermediate ONNX file
        device: Unused, kept for CLI/signature parity with compile_depth_anything
    """
    if not TENSORRT_AVAILABLE:
        logger.error("TensorRT is required. Install with: pip install tensorrt")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    engine_path = output_path / "yolonas_pose.engine"
    onnx_path = output_path / "yolonas_pose.onnx"

    if engine_path.exists():
        logger.info(f"Engine already exists: {engine_path}")
        overwrite = input("Overwrite? (y/N): ").lower().strip() == "y"
        if not overwrite:
            return

    logger.info("Step 1/2: Downloading YOLO-NAS Pose ONNX...")
    if not export_yolo_nas_pose_to_onnx(onnx_path, model_size, resolution, device):
        logger.error("ONNX download failed")
        return

    logger.info("Step 2/2: Building TensorRT engine...")
    if not build_tensorrt_engine(onnx_path, engine_path, resolution, fp16):
        logger.error("TensorRT build failed")
        return

    if not keep_onnx and onnx_path.exists():
        onnx_path.unlink()
        logger.info("Removed intermediate ONNX file")

    logger.info(f"\nSuccess! Engine saved to: {engine_path}")
    logger.info("\nTo use in config:")
    logger.info('  preprocessor: "pose_tensorrt"')
    logger.info("  preprocessor_params:")
    logger.info(f'    engine_path: "{engine_path}"')


if __name__ == "__main__":
    fire.Fire(compile_yolo_nas_pose)
