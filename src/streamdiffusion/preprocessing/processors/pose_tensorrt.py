# NOTE: ported from https://github.com/yuvraj108c/ComfyUI-YoloNasPose-Tensorrt

import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import BasePreprocessor
from .category_params import POSE_DRAW_PARAMS
from .trt_base import TENSORRT_AVAILABLE, TensorRTEngine  # shared engine wrapper


class PoseVisualization:
    """Pose drawing utilities.

    Rendering convention ported from controlnet_aux.open_pose.util.draw_bodypose (filled
    tapered-ellipse limbs at 60% color intensity, full-intensity color-coded joint dots) --
    the convention xinsir/controlnet-openpose-sdxl-1.0 was actually trained on. The previous
    cv2.line/hard-coded-green-circle rendering, plus a final cv2.addWeighted(overlay, 0.75,
    image, 0.25, 0) blend that silently dimmed every stroke to 75% intensity, produced an
    off-distribution, mislabeled skeleton (see plan Finding 3).
    """

    @staticmethod
    def draw_skeleton(
        image,
        keypoints,
        edge_links,
        edge_colors,
        joint_thickness=4,
        keypoint_radius=4,
        keypoint_threshold=0.5,
    ):
        """Draw one pose directly onto `image` (mutated in place, and returned).

        No alpha blend: `image` is drawn on at full intensity (limb fill color is 60% per the
        OpenPose convention above, not a canvas-wide blend).
        """
        # Draw limbs as filled, tapered ellipse polygons
        for (kp1, kp2), color in zip(edge_links, edge_colors):
            if kp1 >= len(keypoints) or kp2 >= len(keypoints):
                continue
            j1, j2 = keypoints[kp1], keypoints[kp2]
            if len(j1) < 3 or len(j2) < 3:
                continue
            if j1[2] <= keypoint_threshold or j2[2] <= keypoint_threshold:
                continue
            x1, y1 = float(j1[0]), float(j1[1])
            x2, y2 = float(j2[0]), float(j2[1])
            mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
            polygon = cv2.ellipse2Poly(
                (int(mid_x), int(mid_y)), (int(length / 2), joint_thickness), int(angle), 0, 360, 1
            )
            dimmed_color = [int(c * 0.6) for c in color]
            cv2.fillConvexPoly(image, polygon, dimmed_color, lineType=cv2.LINE_AA)

        # Draw keypoints as full-intensity dots, colour-coded per joint index (not
        # hard-coded green -- OpenPose ControlNets key limb/joint identity off colour).
        for keypoint, color in zip(keypoints, edge_colors):
            if len(keypoint) < 3 or keypoint[2] <= keypoint_threshold:
                continue
            x, y = int(keypoint[0]), int(keypoint[1])
            cv2.circle(image, (x, y), keypoint_radius, color, -1, cv2.LINE_AA)

        return image

    @staticmethod
    def draw_poses(
        image,
        poses,
        edge_links,
        edge_colors,
        joint_thickness=4,
        keypoint_radius=4,
        keypoint_threshold=0.5,
    ):
        """Draw multiple poses on image"""
        result = image.copy()

        for pose in poses:
            result = PoseVisualization.draw_skeleton(
                result,
                pose,
                edge_links,
                edge_colors,
                joint_thickness,
                keypoint_radius,
                keypoint_threshold,
            )

        return result


def iterate_over_batch_predictions(predictions, batch_size):
    """Process batch predictions from TensorRT output"""
    num_detections, batch_boxes, batch_scores, batch_joints = predictions

    for image_index in range(batch_size):
        num_detection_in_image = int(num_detections[image_index, 0])

        # Handle case where no detections are found
        if num_detection_in_image == 0:
            pred_scores = np.array([])
            pred_boxes = np.array([]).reshape(0, 4)
            pred_joints = np.array([]).reshape(0, 17, 3)
        else:
            pred_scores = batch_scores[image_index, :num_detection_in_image]
            pred_boxes = batch_boxes[image_index, :num_detection_in_image]
            pred_joints = batch_joints[image_index, :num_detection_in_image].reshape((num_detection_in_image, -1, 3))

        yield image_index, pred_boxes, pred_scores, pred_joints


def _letterbox_resize(image_tensor: torch.Tensor, target: int):
    """Aspect-ratio-preserving resize of a (1,3,H,W) [0,1] tensor into a (1,3,target,target)
    square, padded with black.

    Replaces the previous unconditional F.interpolate(..., size=(target, target)) square
    resize, which anisotropically stretched non-square frames (e.g. 640x384 -> 640x640 is a
    1.67x vertical stretch) before pose detection -- distorting the subject the detector sees
    and, after the compensating squash back to the output aspect ratio, turning skeleton
    circles/limb widths anisotropic (plan Finding 3e). Padding uses black (0) rather than the
    conventional YOLO grey (114/255): the engine's original training pad colour isn't known
    here, and an all-black region is the safer default against spurious detections.

    Returns:
        (padded_tensor, scale, pad_x, pad_y) -- callers map detection-space keypoints back to
        the original frame via `(x - pad_x) / scale`, `(y - pad_y) / scale`.
    """
    _, _, h, w = image_tensor.shape
    scale = min(target / h, target / w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = F.interpolate(image_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    padded = image_tensor.new_zeros((image_tensor.shape[0], image_tensor.shape[1], target, target))
    padded[:, :, pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return padded, scale, pad_x, pad_y


# YOLO-NAS Pose outputs COCO-17 keypoints, in the standard order: 0 nose, 1 left_eye,
# 2 right_eye, 3 left_ear, 4 right_ear, 5 left_shoulder, 6 right_shoulder, 7 left_elbow,
# 8 right_elbow, 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip, 13 left_knee,
# 14 right_knee, 15 left_ankle, 16 right_ankle.
#
# xinsir/controlnet-openpose-sdxl-1.0 was trained on OpenPose's COCO-18 body format, which
# reorders these and inserts a synthetic "neck" (shoulder midpoint) at index 1. This table maps
# OpenPose-18 slot -> COCO-17 index (None = no direct counterpart; synthesized separately in
# show_predictions_from_batch_format).
OPENPOSE18_FROM_COCO17 = [
    0,  # 0  nose
    None,  # 1  neck (synthetic: shoulder midpoint)
    6,  # 2  right_shoulder
    8,  # 3  right_elbow
    10,  # 4  right_wrist
    5,  # 5  left_shoulder
    7,  # 6  left_elbow
    9,  # 7  left_wrist
    12,  # 8  right_hip
    14,  # 9  right_knee
    16,  # 10 right_ankle
    11,  # 11 left_hip
    13,  # 12 left_knee
    15,  # 13 left_ankle
    2,  # 14 right_eye
    1,  # 15 left_eye
    4,  # 16 right_ear
    3,  # 17 left_ear
]

# Skeleton limb topology in OpenPose-18 index space, ported verbatim (converted from 1-indexed
# to 0-indexed) from controlnet_aux.open_pose.util.draw_bodypose's `limbSeq` -- the topology
# xinsir/controlnet-openpose-sdxl-1.0 was trained against.
edge_links = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
]

# Per-limb/joint RGB colours, ported verbatim from the same source (its `colors`). The first 17
# entries colour the limbs (zipped against edge_links above); all 18 colour the joints.
edge_colors = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
]


def show_predictions_from_batch_format(
    predictions,
    keypoint_threshold: float = 0.5,
    joint_thickness: int = 4,
    keypoint_radius: int = 4,
    canvas_width: int = 640,
    canvas_height: int = 640,
    letterbox_scale_x: float = 1.0,
    letterbox_scale_y: float = 1.0,
    letterbox_pad_x: int = 0,
    letterbox_pad_y: int = 0,
):
    """Convert predictions to an OpenPose-COCO18-format pose visualization.

    Args:
        predictions:         Raw TRT engine output list (num_dets, boxes, scores, joints).
        keypoint_threshold:  Confidence cutoff for drawing joints (category-standard param).
        joint_thickness:     Skeleton limb ellipse half-width, in pixels.
        keypoint_radius:     Keypoint dot radius in pixels.
        canvas_width:        Output canvas width, in pixels.
        canvas_height:       Output canvas height, in pixels.
        letterbox_scale_x:   Combined per-axis scale mapping detection-space (the letterboxed
                              detect_resolution square) directly to this canvas: undoes
                              `_letterbox_resize`'s scale, then rescales the original frame to
                              (canvas_width, canvas_height) -- the same squash the base class's
                              `_ensure_target_size` applies to every other preprocessor's output.
                              The original frame's own size is *not* necessarily canvas_width x
                              canvas_height (that's the pipeline's configured resolution, not the
                              source frame's), so this must not be conflated with the plain
                              `1/letterbox_scale` undo.
        letterbox_scale_y:   Combined per-axis scale, vertical. See letterbox_scale_x.
        letterbox_pad_x:     Horizontal letterbox padding (pixels) to undo, applied before scale.
        letterbox_pad_y:     Vertical letterbox padding (pixels) to undo, applied before scale.

    The canvas is allocated uint8, not the numpy float64 default: cv2.LINE_AA (used by
    PoseVisualization.draw_skeleton) is silently a no-op on float64 images -- measured 2 unique pixel
    values (hard edges) vs 30 on uint8 (real antialiasing) for the same drawing call. float64 also
    makes every image.copy()/cv2.fillConvexPoly in draw_poses/draw_skeleton move 8x the memory
    traffic per pose than necessary (9.8 MB vs 1.2 MB at 640x640).
    """
    try:
        image_index, pred_boxes, pred_scores, pred_joints = next(iter(iterate_over_batch_predictions(predictions, 1)))
    except Exception as e:
        raise RuntimeError(f"show_predictions_from_batch_format: Error in iterate_over_batch_predictions: {e}") from e

    # Handle case where no poses are detected
    if pred_joints.shape[0] == 0:
        return np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    try:
        pred_joints = pred_joints.astype(np.float32, copy=True)

        # Undo the letterbox directly into canvas space: joints are in the padded
        # detect_resolution-square's pixel space, not this canvas's -- see _letterbox_resize
        # and the letterbox_scale_x/y docstring above (this is NOT a plain 1/letterbox_scale
        # undo -- that would land joints in the original frame's own pixel space, which is a
        # different rectangle than (canvas_width, canvas_height) whenever the source frame's
        # resolution doesn't happen to match the pipeline's configured output resolution).
        if letterbox_scale_x != 1.0 or letterbox_scale_y != 1.0 or letterbox_pad_x or letterbox_pad_y:
            pred_joints[..., 0] = (pred_joints[..., 0] - letterbox_pad_x) * letterbox_scale_x
            pred_joints[..., 1] = (pred_joints[..., 1] - letterbox_pad_y) * letterbox_scale_y

        # Remap COCO-17 -> OpenPose-18 (see OPENPOSE18_FROM_COCO17). "neck" (index 1) is
        # synthesized as the shoulder midpoint, gated on the *lower* of the two shoulder
        # confidences -- not the average, which let a partially-visible pair falsely pass
        # keypoint_threshold and draw a false limb toward absent hips (plan Finding 3d).
        num_poses = pred_joints.shape[0]
        op_joints = np.zeros((num_poses, 18, 3), dtype=np.float32)
        for op_idx, coco_idx in enumerate(OPENPOSE18_FROM_COCO17):
            if coco_idx is not None:
                op_joints[:, op_idx] = pred_joints[:, coco_idx]
        left_shoulder, right_shoulder = pred_joints[:, 5], pred_joints[:, 6]
        op_joints[:, 1, 0] = (left_shoulder[:, 0] + right_shoulder[:, 0]) / 2.0
        op_joints[:, 1, 1] = (left_shoulder[:, 1] + right_shoulder[:, 1]) / 2.0
        op_joints[:, 1, 2] = np.minimum(left_shoulder[:, 2], right_shoulder[:, 2])
    except Exception as e:
        raise RuntimeError(f"show_predictions_from_batch_format: Error processing poses: {e}") from e

    # Create black background for pose visualization
    black_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    try:
        image = PoseVisualization.draw_poses(
            image=black_image,
            poses=op_joints,
            edge_links=edge_links,
            edge_colors=edge_colors,
            joint_thickness=joint_thickness,
            keypoint_radius=keypoint_radius,
            keypoint_threshold=keypoint_threshold,
        )
    except Exception as e:
        raise RuntimeError(f"show_predictions_from_batch_format: Error in pose drawing: {e}") from e

    return image


class YoloNasPoseTensorrtPreprocessor(BasePreprocessor):
    # TRT inference stays on GPU; keypoint-to-image rasterization has a tiny CPU hop
    # (~17 sparse keypoints → cv2 draw → re-upload).  Accepted by design (D5).
    gpu_native = True
    """
    YoloNas Pose TensorRT preprocessor for ControlNet

    Uses TensorRT-optimized YoloNas Pose model for fast pose estimation.
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Pose Detection (TensorRT)",
            "description": "Fast TensorRT-optimized pose detection using YOLO-NAS Pose model. Detects human pose keypoints with high performance.",
            "parameters": {
                **POSE_DRAW_PARAMS,
            },
            "use_cases": [
                "Human pose control",
                "Character animation",
                "Pose-guided generation",
                "Real-time pose detection",
            ],
        }

    def __init__(self, engine_path: str = None, detect_resolution: int = 640, image_resolution: int = 512, **kwargs):
        """
        Initialize TensorRT pose preprocessor

        Args:
            engine_path: Path to TensorRT engine file
            detect_resolution: Resolution for pose detection (should match engine input)
            image_resolution: Output image resolution
            **kwargs: Additional parameters
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT and polygraphy libraries are required for TensorRT pose preprocessing. "
                "Install them with: pip install tensorrt polygraphy"
            )

        super().__init__(
            engine_path=engine_path, detect_resolution=detect_resolution, image_resolution=image_resolution, **kwargs
        )

        self._engine = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_cuda_available = torch.cuda.is_available()

    @property
    def engine(self):
        """Lazy loading of the TensorRT engine"""
        if self._engine is None:
            engine_path = self.params.get("engine_path")
            if engine_path is None:
                raise ValueError(
                    "engine_path is required for TensorRT pose preprocessing. "
                    "Please provide it in the preprocessor_params config."
                )

            if not os.path.exists(engine_path):
                raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

            self._engine = TensorRTEngine(engine_path)
            self._engine.load()
            self._engine.activate()
            self._engine.allocate_buffers()

        return self._engine

    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Apply TensorRT pose estimation to the input image
        """
        detect_resolution = self.params.get("detect_resolution", 640)
        target_width, target_height = self.get_target_dimensions()

        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        _, _, orig_height, orig_width = image_tensor.shape

        image_resized, letterbox_scale, pad_x, pad_y = _letterbox_resize(image_tensor, detect_resolution)

        image_resized_uint8 = (image_resized * 255.0).type(torch.uint8)

        if self._is_cuda_available:
            image_resized_uint8 = image_resized_uint8.cuda()

        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = self.engine.infer({"input": image_resized_uint8}, cuda_stream)

        predictions = [result[key].cpu().numpy() for key in result.keys() if key != "input"]

        keypoint_threshold = float(self.params.get("keypoint_threshold", 0.5))
        joint_thickness = int(self.params.get("joint_thickness", 4))
        keypoint_radius = int(self.params.get("keypoint_radius", 4))

        # Compose the letterbox undo with the (possibly anisotropic) squash from the source
        # frame's own resolution to the pipeline's configured canvas resolution -- these are
        # independent quantities, see letterbox_scale_x/y's docstring.
        letterbox_scale_x = (target_width / orig_width) / letterbox_scale
        letterbox_scale_y = (target_height / orig_height) / letterbox_scale

        try:
            pose_image = show_predictions_from_batch_format(
                predictions,
                keypoint_threshold=keypoint_threshold,
                joint_thickness=joint_thickness,
                keypoint_radius=keypoint_radius,
                canvas_width=target_width,
                canvas_height=target_height,
                letterbox_scale_x=letterbox_scale_x,
                letterbox_scale_y=letterbox_scale_y,
                letterbox_pad_x=pad_x,
                letterbox_pad_y=pad_y,
            )
        except Exception:
            # Fallback to black image on error
            pose_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # show_predictions_from_batch_format and the fallback above are both already uint8 -- this
        # guard is defensive, not the hot path (it used to be an unconditional clip/astype on a
        # float64 canvas, which cost ~2.9ms/frame; see the docstring above).
        if pose_image.dtype != np.uint8:
            pose_image = pose_image.clip(0, 255).astype(np.uint8)

        # NOTE: no cv2.cvtColor(BGR2RGB) here -- the canvas is authored (edge_colors) and
        # consumed (Image.fromarray) as RGB throughout; the previous swap inverted every limb's
        # colour identity (plan Finding 3a).
        result = Image.fromarray(pose_image)

        return result

    def _process_tensor_core(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor directly on GPU to avoid CPU transfers
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if not image_tensor.is_cuda:
            image_tensor = image_tensor.cuda()

        detect_resolution = self.params.get("detect_resolution", 640)
        target_width, target_height = self.get_target_dimensions()
        _, _, orig_height, orig_width = image_tensor.shape

        image_resized, letterbox_scale, pad_x, pad_y = _letterbox_resize(image_tensor, detect_resolution)

        image_resized_uint8 = (image_resized * 255.0).type(torch.uint8)

        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = self.engine.infer({"input": image_resized_uint8}, cuda_stream)

        predictions = [result[key].cpu().numpy() for key in result.keys() if key != "input"]

        keypoint_threshold = float(self.params.get("keypoint_threshold", 0.5))
        joint_thickness = int(self.params.get("joint_thickness", 4))
        keypoint_radius = int(self.params.get("keypoint_radius", 4))

        # Compose the letterbox undo with the squash from the source frame's own resolution to
        # the pipeline's configured canvas resolution -- see letterbox_scale_x/y's docstring.
        letterbox_scale_x = (target_width / orig_width) / letterbox_scale
        letterbox_scale_y = (target_height / orig_height) / letterbox_scale

        try:
            pose_image = show_predictions_from_batch_format(
                predictions,
                keypoint_threshold=keypoint_threshold,
                joint_thickness=joint_thickness,
                keypoint_radius=keypoint_radius,
                canvas_width=target_width,
                canvas_height=target_height,
                letterbox_scale_x=letterbox_scale_x,
                letterbox_scale_y=letterbox_scale_y,
                letterbox_pad_x=pad_x,
                letterbox_pad_y=pad_y,
            )
            # Defensive guard, not the hot path -- see the CPU path's comment above.
            if pose_image.dtype != np.uint8:
                pose_image = pose_image.clip(0, 255).astype(np.uint8)

            # NOTE: no cv2.cvtColor(BGR2RGB) here -- see _process_core.
            pose_tensor = torch.from_numpy(pose_image).to(dtype=torch.float16) / 255.0
            pose_tensor = pose_tensor.permute(2, 0, 1).unsqueeze(0).cuda()

        except Exception:
            # Fallback to black tensor on error
            pose_tensor = torch.zeros(1, 3, target_height, target_width, dtype=torch.float16).cuda()

        return pose_tensor
