from PIL import Image

from .hed import HEDPreprocessor


class ScribblePreprocessor(HEDPreprocessor):
    """
    Scribble preprocessor for ControlNet conditioning

    Produces sketch-like scribble edge maps using the HED model with scribble mode enabled.
    Reuses the HED model cache so no extra model download is needed when HED is already loaded.
    Compatible with xinsir/controlnet-scribble-sdxl-1.0 and similar scribble ControlNets.
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Scribble (HED)",
            "description": "Produces scribble-style edge maps using HED in scribble mode. Compatible with scribble ControlNets.",
            "parameters": {
                "safe": {
                    "type": "bool",
                    "default": True,
                    "description": "Whether to use safe mode for edge detection",
                }
            },
            "use_cases": ["Scribble ControlNet conditioning", "Sketch-style edge maps"],
        }

    def _process_core(self, image: Image.Image) -> Image.Image:
        """Apply HED in scribble mode to produce sketch-like edge maps"""
        target_width, target_height = self.get_target_dimensions()

        result = self.model(image, output_type="pil", scribble=True)

        if not isinstance(result, Image.Image):
            import numpy as np

            if isinstance(result, np.ndarray):
                result = Image.fromarray(result)
            else:
                raise ValueError(f"ScribblePreprocessor: unexpected result type: {type(result)}")

        if result.size != (target_width, target_height):
            result = result.resize((target_width, target_height), Image.LANCZOS)

        return result

    # _process_tensor_core is inherited from HEDPreprocessor (PIL round-trip via tensor_to_pil /
    # _process_core / pil_to_tensor) — same GPU class as openpose/lineart/hed. Acceptable for v1.
