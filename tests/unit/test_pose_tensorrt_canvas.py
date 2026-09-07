"""
Regression test for the pose_tensorrt canvas dtype/size fix (plan
i-ve-noticed-that-in-graceful-kettle.md, "postprocess waste" changes), updated for the
OpenPose-COCO18 rendering + letterbox fix (plan in-the-previous-session-lively-snail.md).

Context: show_predictions_from_batch_format
(streamdiffusion.preprocessing.processors.pose_tensorrt) allocated its rasterization canvas with
np.zeros((640, 640, 3)) -- no dtype, so numpy defaults to float64. Two costs followed:

  1. cv2.LINE_AA (used by PoseVisualization.draw_skeleton for both the skeleton limbs and the
     keypoint circles) is silently a no-op on float64 images -- measured 2 unique pixel values
     (hard edges) on a single antialiased shape, vs 30 unique values (a real antialiased
     gradient) on the equivalent uint8 canvas.
  2. Every image.copy() / cv2.fillConvexPoly call in draw_poses/draw_skeleton moved 8x the
     memory traffic it needed to (9.8 MB float64 vs 1.2 MB uint8 at 640x640), measured at
     7.8-17.4 ms/frame wasted depending on pose count -- the actual perf motivation for this fix.

The canvas was also hardcoded to a square 640x640 regardless of the caller's target dimensions,
and previously to the *detect* resolution rather than the pipeline's output resolution -- both
silently corruptible: iterate_over_batch_predictions returns joints in detection-space pixel
coordinates, so a mismatched canvas size, or skipping the letterbox-undo, crops/misplaces the
skeleton without raising. The rendering was also switched from raw COCO-17 stick lines (wrong
topology, hard-coded green joints, inverted BGR/RGB colours, 75%-alpha-dimmed) to the OpenPose
COCO-18 filled-ellipse convention xinsir/controlnet-openpose-sdxl-1.0 was trained on -- see
show_predictions_from_batch_format's docstring and OPENPOSE18_FROM_COCO17.

This test exercises show_predictions_from_batch_format directly -- pure numpy/cv2, no TensorRT, no
GPU, no engine file -- so it runs in milliseconds and works on any machine.

Run with: pytest tests/unit/test_pose_tensorrt_canvas.py -v
"""

import numpy as np
import pytest

from streamdiffusion.preprocessing.processors.pose_tensorrt import show_predictions_from_batch_format


def _predictions_no_detections():
    """[num_detections, boxes, scores, joints] batch-format predictions with zero detections."""
    num_detections = np.array([[0]])
    batch_boxes = np.zeros((1, 1, 4))
    batch_scores = np.zeros((1, 1))
    batch_joints = np.zeros((1, 1, 51))  # 17 keypoints * 3 (x, y, conf)
    return [num_detections, batch_boxes, batch_scores, batch_joints]


def _predictions_one_pose(kp5=(100.0, 100.0, 0.9), kp6=(400.0, 300.0, 0.9)):
    """One detected pose with only COCO-17 keypoints 5 and 6 (left/right shoulder) confidently set.

    show_predictions_from_batch_format synthesizes OpenPose-18 "neck" (index 1) as the midpoint
    of the two shoulders, gated on min(conf5, conf6) -- and edge_links includes [1, 2]
    (neck->right_shoulder) and [1, 5] (neck->left_shoulder) directly -- so this is the minimal
    input that reaches PoseVisualization.draw_skeleton's cv2.fillConvexPoly/cv2.circle calls
    without needing a full 17-point pose.
    """
    num_detections = np.array([[1]])
    batch_boxes = np.zeros((1, 1, 4))
    batch_scores = np.array([[0.9]])

    joints = np.zeros((17, 3))
    joints[5] = kp5
    joints[6] = kp6
    batch_joints = joints.reshape(1, 1, 51)

    return [num_detections, batch_boxes, batch_scores, batch_joints]


class TestNoDetectionCanvas:
    """The early-return path at pose_tensorrt.py's `if pred_joints.shape[0] == 0` guard."""

    def test_dtype_is_uint8(self):
        image = show_predictions_from_batch_format(_predictions_no_detections())
        assert image.dtype == np.uint8

    def test_default_canvas_size_is_640(self):
        image = show_predictions_from_batch_format(_predictions_no_detections())
        assert image.shape == (640, 640, 3)

    def test_canvas_size_is_honoured(self):
        image = show_predictions_from_batch_format(_predictions_no_detections(), canvas_width=512, canvas_height=384)
        assert image.shape == (384, 512, 3)
        assert image.dtype == np.uint8


class TestWithDetectionCanvas:
    """The main drawing path -- the actual perf regression guard."""

    def test_dtype_is_uint8(self):
        image = show_predictions_from_batch_format(_predictions_one_pose())
        assert image.dtype == np.uint8

    def test_default_canvas_size_is_640(self):
        image = show_predictions_from_batch_format(_predictions_one_pose())
        assert image.shape == (640, 640, 3)

    def test_canvas_size_is_honoured(self):
        image = show_predictions_from_batch_format(_predictions_one_pose(), canvas_width=512, canvas_height=384)
        assert image.shape == (384, 512, 3)
        assert image.dtype == np.uint8

    def test_antialiasing_is_not_defeated_by_canvas_dtype(self):
        """cv2.LINE_AA is a no-op on float64 -- this is the signal that distinguishes the two
        dtypes. A regression back to a dtype-less np.zeros() canvas would collapse this to <=2
        unique nonzero pixel values (hard edges only)."""
        image = show_predictions_from_batch_format(_predictions_one_pose())
        nonzero = image[image.sum(axis=-1) > 0]
        assert nonzero.size > 0, "expected the skeleton limb/keypoints to draw something"
        unique_values = np.unique(nonzero)
        assert len(unique_values) > 2, (
            f"only {len(unique_values)} unique pixel value(s) in drawn region -- antialiasing "
            "looks defeated (a float64 canvas silently disables cv2.LINE_AA)"
        )

    def test_colours_are_not_bgr_swapped(self):
        """Regression guard for plan Finding 3a: a stray cv2.cvtColor(..., COLOR_BGR2RGB) used
        to invert every limb's authored colour. edge_colors[0] (nose->neck, RGB (255,0,0)) is
        drawn at 60% intensity as a filled ellipse -- the red channel must dominate, not blue."""
        image = show_predictions_from_batch_format(
            _predictions_one_pose(kp5=(100.0, 100.0, 0.9), kp6=(100.0, 100.0, 0.9))
        )
        # Set nose too, so edge_links[12] == [1, 0] (neck->nose, colour index 12, [0,0,255] blue)
        # is NOT what's tested here; instead confirm the neck->right_shoulder limb (edge_links[0]
        # == [1, 2], colour index 0, RGB (255,0,0) red) dominates red over blue somewhere on canvas.
        red_channel_max = int(image[..., 0].max())
        blue_channel_max = int(image[..., 2].max())
        assert red_channel_max > blue_channel_max, (
            f"red max {red_channel_max} <= blue max {blue_channel_max} -- looks BGR/RGB swapped"
        )

    def test_full_intensity_not_75_percent_dimmed(self):
        """Regression guard for plan Finding 3b: cv2.addWeighted(overlay, 0.75, image, 0.25, 0)
        used to cap every stroke's brightest channel at 255*0.75=192. Keypoint dots are drawn at
        full intensity (limb fills are intentionally 60% per the OpenPose convention -- see
        draw_skeleton -- so this checks a keypoint circle's peak, not a limb's)."""
        image = show_predictions_from_batch_format(_predictions_one_pose())
        assert image.max() == 255, f"brightest channel is {image.max()}, expected 255 (not 192)"


class TestSuccessFallbackReconciliation:
    """The success path (show_predictions_from_batch_format) and the CPU fallback
    (np.zeros((target_height, target_width, 3), dtype=np.uint8) in _process_core/
    _process_tensor_core) must agree on shape/dtype for the same target dimensions."""

    @pytest.mark.parametrize("canvas_width,canvas_height", [(384, 384), (512, 512), (640, 384)])
    def test_success_and_fallback_shapes_match(self, canvas_width, canvas_height):
        success_image = show_predictions_from_batch_format(
            _predictions_one_pose(), canvas_width=canvas_width, canvas_height=canvas_height
        )
        fallback_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        assert success_image.shape == fallback_image.shape
        assert success_image.dtype == fallback_image.dtype


class TestLetterboxUndo:
    """Regression guard for plan Finding 3e: joints returned by the TRT engine are in the
    padded detect-resolution-square's pixel space, not the original frame's -- the
    letterbox_scale/pad_x/pad_y params must be applied before drawing."""

    def test_pad_offset_shifts_the_drawn_pose(self):
        image_no_pad = show_predictions_from_batch_format(_predictions_one_pose(), canvas_width=640, canvas_height=640)
        image_with_pad = show_predictions_from_batch_format(
            _predictions_one_pose(),
            canvas_width=640,
            canvas_height=640,
            letterbox_scale=1.0,
            letterbox_pad_x=50,
            letterbox_pad_y=0,
        )
        assert not np.array_equal(image_no_pad, image_with_pad), (
            "letterbox_pad_x had no effect -- pad undo looks unapplied"
        )

    def test_scale_factor_rescales_the_drawn_pose(self):
        image_no_scale = show_predictions_from_batch_format(
            _predictions_one_pose(), canvas_width=640, canvas_height=640
        )
        image_scaled = show_predictions_from_batch_format(
            _predictions_one_pose(), canvas_width=640, canvas_height=640, letterbox_scale=2.0
        )
        assert not np.array_equal(image_no_scale, image_scaled), (
            "letterbox_scale had no effect -- scale undo looks unapplied"
        )
