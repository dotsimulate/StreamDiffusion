import logging
import time

from PIL import Image

from .base import BasePreprocessor


logger = logging.getLogger(__name__)

try:
    from controlnet_aux import LineartAnimeDetector, LineartDetector

    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False
    raise ImportError(
        "LineartPreprocessor: controlnet_aux is required for real-time optimization. Install with: pip install controlnet_aux"
    )


# TODO provide gpu native lineart detection
class LineartPreprocessor(BasePreprocessor):
    """
    Real-time optimized Lineart detection preprocessor for ControlNet

    Extracts line art from input images using controlnet_aux line art detection models.
    Supports both realistic and anime-style line art extraction.
    Optimized for real-time performance - no fallbacks.
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Line Art Detection",
            "description": "Detects line art and sketches from input images. Good for converting photos to line drawings.",
            "parameters": {
                "coarse": {
                    "type": "bool",
                    "default": True,
                    "description": "Whether to use coarse line art detection (faster but less detailed)",
                },
                "anime_style": {
                    "type": "bool",
                    "default": False,
                    "description": "Whether to use anime-style line art detection",
                },
            },
            "use_cases": ["Sketch to image", "Line art generation", "Clean line extraction"],
        }

    def __init__(
        self,
        detect_resolution: int = 512,
        image_resolution: int = 512,
        coarse: bool = True,
        anime_style: bool = False,
        **kwargs,
    ):
        """
        Initialize Lineart preprocessor

        Args:
            detect_resolution: Resolution for line art detection
            image_resolution: Output image resolution
            coarse: Whether to use coarse line art detection
            anime_style: Whether to use anime-style line art detection
            **kwargs: Additional parameters
        """
        super().__init__(
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            coarse=coarse,
            anime_style=anime_style,
            **kwargs,
        )

        self._detector = None

    @property
    def detector(self):
        """Lazy loading of the line art detector - controlnet_aux only"""
        if self._detector is None:
            start_time = time.time()
            anime_style = self.params.get("anime_style", False)

            if anime_style:
                self._detector = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
            else:
                self._detector = LineartDetector.from_pretrained("lllyasviel/Annotators")

            load_time = time.time() - start_time
            logger.info(f"Lineart detector loaded in {load_time:.3f}s")

        return self._detector

    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Apply line art detection to the input image
        """
        detect_resolution = self.params.get("detect_resolution", 512)
        coarse = self.params.get("coarse", False)

        if image.size != (detect_resolution, detect_resolution):
            image_resized = image.resize((detect_resolution, detect_resolution), Image.LANCZOS)
        else:
            image_resized = image

        lineart_image = self.detector(
            image_resized, detect_resolution=detect_resolution, image_resolution=detect_resolution, coarse=coarse
        )

        return lineart_image
