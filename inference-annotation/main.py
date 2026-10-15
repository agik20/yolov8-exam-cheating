"""Offline video annotation tool.

Reads the input video via inference.config.paths, runs pose + object detection
with the existing YOLOv8 models, draws boxes/labels/keypoints, and writes an
annotated MP4 at input FPS. No real-time gating, no Flask, no evidence pipeline.
"""
import os
import sys
import time
import argparse

# Bootstrap: add project root so `inference` package is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inference.config import Config, paths as _paths
from inference.core.nms import apply_nms

import cv2
import numpy as np
from ultralytics import YOLO


def annotate_video(input_video=None, output_path=None):
    if input_video is None:
        input_video = _paths.resolve_input_video()
    pose_model_path = _paths.resolve_pose_model()
    object_model_path = _paths.resolve_object_model()
    if output_path is None:
        output_dir = os.path.join(_THIS_DIR, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "annotated.mp4")

    print(f"[INFO] Input video:  {input_video}")
    print(f"[INFO] Pose model:   {pose_model_path}")
    print(f"[INFO] Object model: {object_model_path}")
    print(f"[INFO] Output video: {output_path}")

    pose_model = YOLO(pose_model_path, task='pose')
    object_model = YOLO(object_model_path, task='detect')
    pose_class_map = {0: 'no-cheating', 1: 'provide-object', 2: 'see-friends-work'}
    object_class_map = {0: 'smartphone', 1: 'calculator'}

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {input_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        print("[ERROR] Could not open video writer. No .mp4 support via mp4v fourcc.")
        cap.release()
        return

    conf_pose = Config.CONF_THRESH_POSE
    conf_obj = Config.CONF_THRESH_OBJ
    nms_iou = Config.NMS_IOU_THRESHOLD
    box_thick = Config.POSE_BOX_THICKNESS
    text_thick = Config.TEXT_THICKNESS
    pose_colors = Config.POSE_COLORS
    obj_colors = Config.OBJECT_COLORS

    print(f"[INFO] Video: {w}x{h} @ {fps:.2f} FPS, {total_frames} frames total")
    print("[INFO] Processing started...")
    t0 = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        annotated = frame.copy()

        # --- Pose detection ---
        pose_results = pose_model(
            source=frame,
            conf=conf_pose,
            iou=nms_iou,
            imgsz=640,
            verbose=False,
        )
        _draw_pose(annotated, pose_results, pose_class_map, pose_colors, box_thick, text_thick)

        # --- Object detection ---
        obj_results = object_model(
            source=frame,
            conf=conf_obj,
            iou=nms_iou,
            imgsz=640,
            verbose=False,
        )
        _draw_objects(annotated, obj_results, object_class_map, obj_colors, box_thick, text_thick)

        # --- Overlay: frame counter ---
        cv2.putText(
            annotated, f"Frame {frame_idx}/{total_frames}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), text_thick + 1
        )

        writer.write(annotated)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t0
            rate = frame_idx / elapsed if elapsed > 0 else 0
            print(f"[INFO] Frame {frame_idx}/{total_frames}  ({rate:.1f} proc-FPS)")

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"[INFO] Done. {frame_idx} frames in {elapsed:.1f}s  "
          f"(avg {frame_idx/elapsed:.1f} proc-FPS)")
    print(f"[INFO] Annotated video: {output_path}")


def _draw_pose(annotated, results, class_map, colors, box_thick, text_thick):
    """Draw pose boxes + keypoints + labels on the annotated frame."""
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        all_boxes = []
        all_scores = []
        all_cls = []
        all_kps = []
        for xyxy, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            all_boxes.append(xyxy.cpu().numpy().astype(int).tolist())
            all_scores.append(float(conf))
            all_cls.append(int(cls_id))
            all_kps.append(getattr(result, "keypoints", None))

        if not all_boxes:
            continue
        keep = apply_nms(all_boxes, all_scores, 0.45)
        for i in keep:
            x1, y1, x2, y2 = all_boxes[i]
            cls_id = all_cls[i]
            conf = all_scores[i]
            class_name = class_map.get(cls_id, f"unknown_{cls_id}")
            color = colors.get(class_name, (255, 255, 255))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thick)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, text_thick)

            kps = all_kps[i]
            if kps is not None:
                _draw_keypoints(annotated, kps, color, person_idx=i)


def _draw_keypoints(annotated, keypoints, color, person_idx=0):
    """Draw pose keypoints: skeleton lines + joint dots for one person."""
    k = keypoints.data
    if k is None:
        return
    n_people = k.shape[0]
    idx = min(person_idx, n_people - 1)
    k = k[idx].cpu().numpy()
    n_joints = k.shape[0]
    # COCO-style skeleton: indices valid for this model's keypoint count
    skel = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10),
        (0, 8), (8, 12), (8, 11),
        (11, 12), (11, 13), (13, 14),
        (12, 15), (15, 16),
    ]
    joints = k
    for i, j in skel:
        if i >= n_joints or j >= n_joints:
            continue
        pi, pj = joints[i], joints[j]
        if pi[2] > 0.5 and pj[2] > 0.5:
            cv2.line(annotated,
                     (int(pi[0]), int(pi[1])),
                     (int(pj[0]), int(pj[1])),
                     color, 1)
    for joint in joints:
        x, y, conf = joint
        if conf > 0.5:
            cv2.circle(annotated, (int(x), int(y)), 2, (255, 255, 0), -1)


def _draw_objects(annotated, results, class_map, colors, box_thick, text_thick):
    """Draw object detection boxes + labels on the annotated frame."""
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        all_boxes = []
        all_scores = []
        all_cls = []
        for xyxy, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            all_boxes.append(xyxy.cpu().numpy().astype(int).tolist())
            all_scores.append(float(conf))
            all_cls.append(int(cls_id))
        if not all_boxes:
            continue
        keep = apply_nms(all_boxes, all_scores, 0.45)
        for i in keep:
            x1, y1, x2, y2 = all_boxes[i]
            cls_id = all_cls[i]
            conf = all_scores[i]
            class_name = class_map.get(cls_id, f"unknown_{cls_id}")
            color = colors.get(class_name, (255, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thick)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, text_thick)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 video annotation tool")
    parser.add_argument("--input", "-i", default=None,
                        help="Input video path (default: resolve from inference/input/)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output video path (default: inference-annotation/output/annotated.mp4)")
    args = parser.parse_args()
    annotate_video(args.input, args.output)
