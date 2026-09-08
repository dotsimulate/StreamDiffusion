"""
HED TensorRT preprocessor — GPU-native edge detection via TRT engine.

The HED network (ControlNetHED_Apache2) is wrapped in HEDExportWrapper so that
ONNX export sees a single output tensor (full-resolution edge map) rather than
the native 5-output multi-scale tuple.  The wrapper reproduces exactly what
``controlnet_aux.HEDdetector.__call__`` does on the CPU:

    1. feed the network 0-255 pixels (its learned ``norm`` parameter is a
       per-channel mean of ~[122, 117, 104] and expects that range),
    2. bilinearly upsample the five side outputs to the input resolution,
    3. average them and apply a sigmoid.

Wrapper input/output contract:

    input  : float32 (B, 3, H, W) in [0, 1]   <- same as validate_tensor_input output
    output : float32 (B, 1, H, W) in [0, 1]   <- fused sigmoid edge map ("edge_map")

Engines built by the previous wrapper (block-1 side output only, 0-1 input) are
detected by their output tensor name and rebuilt automatically on load.
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from .category_params import EDGE_SMOOTHNESS_PARAM, apply_edge_smoothness
from .trt_base import SelfBuildingTRTPreprocessor, _first_output

logger = logging.getLogger(__name__)

try:
    from controlnet_aux import HEDdetector

    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False

#: Default post-process knobs, tuned on images/inputs/input.png at 640x384 against
#: SargeZT/controlnet-sd-xl-1.0-softedge-dexined: 0.3 drops the low-probability
#: texture haze (hair/cloth interiors) while keeping the soft outline values; a
#: light post-blur matches the DexiNed-style soft edges the ControlNet was trained on.
DEFAULT_EDGE_THRESHOLD = 0.3
DEFAULT_HED_SMOOTHNESS = 0.15


# ---------------------------------------------------------------------------
# ONNX export wrapper — fused multi-scale sigmoid edge map
# ---------------------------------------------------------------------------


class HEDExportWrapper(torch.nn.Module):
    """
    Thin wrapper around ControlNetHED_Apache2 for ONNX export.

    The native forward returns five side-output logit maps at (H, H/2, H/4,
    H/8, H/16).  Following ``HEDdetector.__call__`` the maps are upsampled to
    (H, W), averaged, and passed through a sigmoid, giving one (B, 1, H, W)
    edge-probability map.  Returning only ``outputs[0]`` (as an earlier version
    did) makes the exporter drop four of the five VGG blocks and yields a
    texture-sensitive first-layer response instead of HED.
    """

    #: Scale applied to the [0, 1] pipeline tensor before the network.
    INPUT_SCALE = 255.0

    def __init__(self, netNetwork: torch.nn.Module):
        super().__init__()
        self.netNetwork = netNetwork

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [0, 1]  ->  network expects 0-255 (mean-subtracted inside)
        outputs = self.netNetwork(x * self.INPUT_SCALE)
        if not isinstance(outputs, (list, tuple)):
            return torch.sigmoid(outputs)
        full = outputs[0]
        size = full.shape[-2:]
        acc = full
        for side in outputs[1:]:
            acc = acc + F.interpolate(side, size=size, mode="bilinear", align_corners=False)
        return torch.sigmoid(acc / float(len(outputs)))


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class HEDTensorrtPreprocessor(SelfBuildingTRTPreprocessor):
    """
    HED edge detection via a self-built TensorRT engine.

    GPU-native: no CPU / PIL round-trip on the tensor path.
    The engine is built on first use and cached in engines/preprocessors/hed.engine
    (or the path supplied via preprocessor_params.engine_path in the YAML config).
    """

    engine_filename = "hed.engine"
    onnx_filename = "hed.onnx"
    default_detect_resolution = 512
    #: Output tensor name of the current export format.  Engines whose output is
    #: still called ``output`` were built by the block-1-only wrapper and are stale.
    ENGINE_OUTPUT_NAME = "edge_map"

    def _get_engine_path(self):
        # HED and Scribble share this cache. Keep the old one-scale engine intact
        # so an older installation can still use it while the new export builds.
        path = super()._get_engine_path()
        if path.stem.endswith('.hed5scale'):
            return path
        return path.with_name(f'{path.stem}.hed5scale{path.suffix}')

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "HED Soft Edges (TensorRT)",
            "description": (
                "GPU-native HED (Holistically-Nested Edge Detection) via TensorRT: the soft "
                "white-on-black edge map expected by HED ControlNets. Self-builds hed.engine "
                "from the controlnet_aux model on first run; the Scribble preprocessor runs "
                "the same engine and only differs in post-processing. No CPU/PIL round-trips."
            ),
            "parameters": {
                "edge_threshold": {
                    "type": "float",
                    "default": DEFAULT_EDGE_THRESHOLD,
                    "range": [0.0, 1.0],
                    "description": (
                        "Zero out edge probabilities below this value (thins the map, keeps the "
                        "surviving values soft). 0 = raw HED output."
                    ),
                },
                "smoothness": {
                    **EDGE_SMOOTHNESS_PARAM["smoothness"],
                    "default": DEFAULT_HED_SMOOTHNESS,
                    "description": (
                        "Gaussian post-blur of the edge map, applied after the threshold "
                        "(0 = sharp; 1 = σ≈2, ~13×13 kernel)."
                    ),
                },
            },
            "use_cases": [
                "HED ControlNet conditioning",
                "Structured edge maps (real-time)",
            ],
        }

    def __init__(self, **kwargs):
        if not CONTROLNET_AUX_AVAILABLE:
            raise ImportError(
                "controlnet_aux is required for HEDTensorrtPreprocessor. Install with: pip install controlnet_aux"
            )
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Stale-engine detection
    # ------------------------------------------------------------------

    def _engine_is_current(self, trt_engine) -> bool:
        """True when the loaded engine exposes the fused ``edge_map`` output."""
        eng = trt_engine.engine
        names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
        if self.ENGINE_OUTPUT_NAME in names:
            return True
        logger.warning(
            "%s: engine outputs %s lack '%s' — built by the pre-fix wrapper (block-1 only, 0-1 input); rebuilding.",
            self.__class__.__name__,
            names,
            self.ENGINE_OUTPUT_NAME,
        )
        return False

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def _export_onnx(self, onnx_path: Path) -> None:
        """Load HEDdetector, wrap it, and export to ONNX."""
        logger.info("HEDTensorrtPreprocessor: loading HEDdetector for ONNX export…")
        detector = HEDdetector.from_pretrained("lllyasviel/Annotators")

        if not hasattr(detector, "netNetwork"):
            raise RuntimeError(
                "HEDTensorrtPreprocessor: HEDdetector has no 'netNetwork' attribute. "
                "controlnet_aux version may be incompatible."
            )

        wrapper = HEDExportWrapper(detector.netNetwork).to(self.device).eval()
        res = self.default_detect_resolution
        dummy = torch.zeros(1, 3, res, res, device=self.device)

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy,
                str(onnx_path),
                opset_version=17,
                input_names=["input"],
                output_names=[self.ENGINE_OUTPUT_NAME],
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    self.ENGINE_OUTPUT_NAME: {0: "batch", 2: "height", 3: "width"},
                },
                dynamo=False,
            )

        logger.info(f"HEDTensorrtPreprocessor: ONNX exported → {onnx_path}")
        # Free GPU memory used by the export model
        del wrapper, detector
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Post-process TRT output → CHW GPU tensor
    # ------------------------------------------------------------------

    def _postprocess(self, engine_outputs: dict) -> torch.Tensor:
        """
        Convert TRT output to a 3-channel [0, 1] edge map GPU tensor (CHW).

        Input  : engine_outputs["edge_map"]  shape (B, 1, H, W)  or (B, H, W)
        Output : (3, H, W) in [0, 1]

        The engine output is already a sigmoid probability, so no min/max
        normalisation is applied (it would amplify noise on flat frames and
        break the absolute threshold semantics of the scribble post-process).

        Optional knobs (``self.params``, live-updatable from TD):
          * ``edge_threshold`` — probabilities below it become 0, the rest keep
            their value (thins the map without binarising it).
          * ``smoothness``     — Gaussian post-blur of the result.
        """
        out = _first_output(engine_outputs).float()

        # Collapse batch + channel dims if present
        if out.dim() == 4:
            out = out.squeeze(1)  # (B, H, W) — B should be 1
        if out.dim() == 3:
            out = out.squeeze(0)  # (H, W)

        out = out.clamp(0.0, 1.0)

        threshold = float(self.params.get("edge_threshold", DEFAULT_EDGE_THRESHOLD))
        if threshold > 0.0:
            out = torch.where(out >= threshold, out, torch.zeros_like(out))

        smoothness = float(self.params.get("smoothness", DEFAULT_HED_SMOOTHNESS))
        if smoothness > 0.0:
            out = apply_edge_smoothness(out, smoothness)  # (H, W) in, (H, W) out

        # Expand to 3-channel RGB  →  (3, H, W)
        return out.unsqueeze(0).repeat(3, 1, 1)
