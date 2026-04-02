# StreamDiffusion Modules Package

from .controlnet_module import ControlNetModule
from .image_processing_module import ImagePostprocessingModule, ImagePreprocessingModule, ImageProcessingModule
from .ipadapter_module import IPAdapterModule
from .latent_processing_module import LatentPostprocessingModule, LatentPreprocessingModule, LatentProcessingModule


__all__ = [
    # Existing modules
    "ControlNetModule",
    "IPAdapterModule",
    # Pipeline processing base classes
    "ImageProcessingModule",
    "LatentProcessingModule",
    # Pipeline processing timing-specific modules
    "ImagePreprocessingModule",
    "ImagePostprocessingModule",
    "LatentPreprocessingModule",
    "LatentPostprocessingModule",
]
