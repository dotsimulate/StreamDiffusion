from typing import Any

from .base import BasePreprocessor, PipelineAwareProcessor
from .blur import BlurPreprocessor
from .canny import CannyPreprocessor
from .depth import DepthPreprocessor
from .external import ExternalPreprocessor
from .faceid_embedding import FaceIDEmbeddingPreprocessor
from .feedback import FeedbackPreprocessor
from .hed import HEDPreprocessor
from .ipadapter_embedding import IPAdapterEmbeddingPreprocessor
from .latent_feedback import LatentFeedbackPreprocessor
from .lineart import LineartPreprocessor
from .openpose import OpenPosePreprocessor
from .passthrough import PassthroughPreprocessor
from .realesrgan_trt import RealESRGANProcessor
from .sharpen import SharpenPreprocessor
from .soft_edge import SoftEdgePreprocessor
from .standard_lineart import StandardLineartPreprocessor
from .upscale import UpscalePreprocessor


# Try to import TensorRT preprocessors - might not be available on all systems
try:
    from .depth_tensorrt import DepthAnythingTensorrtPreprocessor

    DEPTH_TENSORRT_AVAILABLE = True
except ImportError:
    DepthAnythingTensorrtPreprocessor = None
    DEPTH_TENSORRT_AVAILABLE = False

try:
    from .pose_tensorrt import YoloNasPoseTensorrtPreprocessor

    POSE_TENSORRT_AVAILABLE = True
except ImportError:
    YoloNasPoseTensorrtPreprocessor = None
    POSE_TENSORRT_AVAILABLE = False

try:
    from .temporal_net_tensorrt import TemporalNetTensorRTPreprocessor

    TEMPORAL_NET_TENSORRT_AVAILABLE = True
except ImportError:
    TemporalNetTensorRTPreprocessor = None
    TEMPORAL_NET_TENSORRT_AVAILABLE = False

try:
    from .mediapipe_pose import MediaPipePosePreprocessor

    MEDIAPIPE_POSE_AVAILABLE = True
except ImportError:
    MediaPipePosePreprocessor = None
    MEDIAPIPE_POSE_AVAILABLE = False

try:
    from .mediapipe_segmentation import MediaPipeSegmentationPreprocessor

    MEDIAPIPE_SEGMENTATION_AVAILABLE = True
except ImportError:
    MediaPipeSegmentationPreprocessor = None
    MEDIAPIPE_SEGMENTATION_AVAILABLE = False

# Registry for easy lookup
_preprocessor_registry = {
    "canny": CannyPreprocessor,
    "depth": DepthPreprocessor,
    "openpose": OpenPosePreprocessor,
    "lineart": LineartPreprocessor,
    "standard_lineart": StandardLineartPreprocessor,
    "passthrough": PassthroughPreprocessor,
    "external": ExternalPreprocessor,
    "soft_edge": SoftEdgePreprocessor,
    "hed": HEDPreprocessor,
    "feedback": FeedbackPreprocessor,
    "latent_feedback": LatentFeedbackPreprocessor,
    "sharpen": SharpenPreprocessor,
    "upscale": UpscalePreprocessor,
    "blur": BlurPreprocessor,
    "realesrgan_trt": RealESRGANProcessor,
}

# Add TensorRT preprocessors if available
if DEPTH_TENSORRT_AVAILABLE:
    _preprocessor_registry["depth_tensorrt"] = DepthAnythingTensorrtPreprocessor

if POSE_TENSORRT_AVAILABLE:
    _preprocessor_registry["pose_tensorrt"] = YoloNasPoseTensorrtPreprocessor

if TEMPORAL_NET_TENSORRT_AVAILABLE:
    _preprocessor_registry["temporal_net_tensorrt"] = TemporalNetTensorRTPreprocessor

# Add MediaPipe preprocessors if available
if MEDIAPIPE_POSE_AVAILABLE:
    _preprocessor_registry["mediapipe_pose"] = MediaPipePosePreprocessor

if MEDIAPIPE_SEGMENTATION_AVAILABLE:
    _preprocessor_registry["mediapipe_segmentation"] = MediaPipeSegmentationPreprocessor


def get_preprocessor_class(name: str) -> type:
    """
    Get a preprocessor class by name

    Args:
        name: Name of the preprocessor

    Returns:
        Preprocessor class

    Raises:
        ValueError: If preprocessor name is not found
    """
    if name not in _preprocessor_registry:
        available = ", ".join(_preprocessor_registry.keys())
        raise ValueError(f"Unknown preprocessor '{name}'. Available: {available}")

    return _preprocessor_registry[name]


def get_preprocessor(
    name: str, pipeline_ref: Any = None, normalization_context: str = "controlnet", params: Any = None
) -> BasePreprocessor:
    """
    Get a preprocessor by name

    Args:
        name: Name of the preprocessor
        pipeline_ref: Pipeline reference for pipeline-aware processors (required for some processors)
        normalization_context: Context for normalization handling
            - 'controlnet': Expects/produces [0,1] range for ControlNet conditioning
            - 'pipeline': Expects/produces [-1,1] range for pipeline image processing
            - 'latent': Works in latent space (no normalization needed)

    Returns:
        Preprocessor instance

    Raises:
        ValueError: If preprocessor name is not found or pipeline_ref missing for pipeline-aware processor
    """
    processor_class = get_preprocessor_class(name)

    # Check if this is a pipeline-aware processor
    if hasattr(processor_class, "requires_sync_processing") and processor_class.requires_sync_processing:
        if pipeline_ref is None:
            raise ValueError(f"Processor '{name}' requires a pipeline_ref")
        return processor_class(
            pipeline_ref=pipeline_ref,
            normalization_context=normalization_context,
            _registry_name=name,
            **(params or {}),
        )
    else:
        return processor_class(normalization_context=normalization_context, _registry_name=name, **(params or {}))


def register_preprocessor(name: str, preprocessor_class):
    """
    Register a new preprocessor

    Args:
        name: Name to register under
        preprocessor_class: Preprocessor class
    """
    _preprocessor_registry[name] = preprocessor_class


def list_preprocessors():
    """List all available preprocessors"""
    return list(_preprocessor_registry.keys())


__all__ = [
    "BasePreprocessor",
    "PipelineAwareProcessor",
    "CannyPreprocessor",
    "DepthPreprocessor",
    "OpenPosePreprocessor",
    "LineartPreprocessor",
    "StandardLineartPreprocessor",
    "PassthroughPreprocessor",
    "ExternalPreprocessor",
    "SoftEdgePreprocessor",
    "HEDPreprocessor",
    "IPAdapterEmbeddingPreprocessor",
    "FaceIDEmbeddingPreprocessor",
    "FeedbackPreprocessor",
    "LatentFeedbackPreprocessor",
    "get_preprocessor",
    "get_preprocessor_class",
    "register_preprocessor",
    "list_preprocessors",
]

if DEPTH_TENSORRT_AVAILABLE:
    __all__.append("DepthAnythingTensorrtPreprocessor")

if POSE_TENSORRT_AVAILABLE:
    __all__.append("YoloNasPoseTensorrtPreprocessor")

if TEMPORAL_NET_TENSORRT_AVAILABLE:
    __all__.append("TemporalNetTensorRTPreprocessor")

if MEDIAPIPE_POSE_AVAILABLE:
    __all__.append("MediaPipePosePreprocessor")

if MEDIAPIPE_SEGMENTATION_AVAILABLE:
    __all__.append("MediaPipeSegmentationPreprocessor")


# region Custom Processor Discovery
import importlib.util
import inspect
import logging
import os
from pathlib import Path


_logger = logging.getLogger(__name__)


def _discover_custom_processors():
    """Auto-discover custom processors from repo_root/custom_processors/ folder."""
    if os.getenv("STREAMDIFFUSION_DISABLE_CUSTOM_PROCESSORS") == "1":
        _logger.info("Custom processor discovery disabled via environment variable")
        return
    try:
        repo_root = Path(__file__).parent.parent.parent.parent.parent
        custom_dir = repo_root / "custom_processors"
        if not custom_dir.exists():
            _logger.debug("custom_processors/ folder not found, skipping discovery")
            return
        _logger.info("Scanning custom_processors/ for custom processors...")
        for item in custom_dir.iterdir():
            if not item.is_dir() or item.name.startswith((".", "_")):
                continue
            manifest_file = item / "processors.yaml"
            if manifest_file.exists():
                _load_processor_collection(item, manifest_file)
            else:
                _load_processor_folder_auto(item)
    except Exception as e:
        _logger.error(f"Custom processor discovery failed: {e}")


def _load_processor_collection(collection_dir, manifest_file):
    """Load processors from a collection with processors.yaml manifest."""
    import yaml

    try:
        with open(manifest_file, "r") as f:
            manifest = yaml.safe_load(f)
        processor_files = manifest.get("processors", [])
        if not processor_files:
            _logger.warning(f"Collection '{collection_dir.name}' has empty processors list")
            return
        _logger.info(f"Loading collection '{collection_dir.name}' ({len(processor_files)} processors)")
        for proc_file in processor_files:
            if isinstance(proc_file, dict):
                filename, enabled = proc_file.get("file"), proc_file.get("enabled", True)
                if not enabled:
                    continue
            else:
                filename = proc_file
            proc_path = collection_dir / filename
            if proc_path.exists():
                _load_processor_from_file(proc_path, proc_path.stem)
            else:
                _logger.warning(f"  Processor file not found: {filename}")
    except Exception as e:
        _logger.error(f"Failed to load collection {collection_dir.name}: {e}")


def _load_processor_folder_auto(folder):
    """Auto-discover processors by scanning for .py files (no manifest)."""
    _logger.info(f"Auto-scanning folder: {folder.name}")
    for py_file in folder.glob("*.py"):
        if py_file.name.startswith("_") or py_file.name in ["base.py", "setup.py"]:
            continue
        _load_processor_from_file(py_file, py_file.stem)


def _load_processor_from_file(file_path, proc_name):
    """Load and register a processor class from a Python file."""
    try:
        spec = importlib.util.spec_from_file_location(
            f"custom_processors.{file_path.parent.name}.{file_path.stem}", file_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        found_classes = [
            (name, obj)
            for name, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, (BasePreprocessor, PipelineAwareProcessor))
            and obj not in [BasePreprocessor, PipelineAwareProcessor]
        ]
        if not found_classes:
            _logger.warning(f"  No valid processor class in {file_path.name}")
            return
        if proc_name in _preprocessor_registry:
            _logger.error(f"  Name conflict: '{proc_name}' already exists, skipping")
            return
        register_preprocessor(proc_name, found_classes[0][1])
        _logger.info(f"  Registered: {proc_name} ({found_classes[0][0]})")
    except Exception as e:
        _logger.error(f"  Failed to load {file_path.name}: {e}")


_discover_custom_processors()
# endregion
