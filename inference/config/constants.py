"""Configuration: thresholds, class lists, colors. Values are frozen for behavior-preservation."""
from . import paths


class Config:
    # Path configurations (root-anchored; input video + models resolved at runtime in main.py)
    OUTPUT_POSE_BASE = paths.OUTPUT_POSE_BASE
    OUTPUT_OBJ_BASE = paths.OUTPUT_OBJ_BASE
    OUTPUT_FULLROOM = paths.OUTPUT_FULLROOM
    OUTPUT_FRAMES = paths.OUTPUT_FRAMES

    # Detection parameters
    POSE_CLASSES = ["no-cheating", "provide-object", "see-friends-work"]
    OBJECT_CLASSES = ["smartphone", "calculator"]
    SAVE_POSE_CLASSES = ["provide-object", "see-friends-work"]
    SAVE_OBJECT_CLASSES = ["smartphone"]

    CONF_THRESH_POSE = 0.3
    CONF_THRESH_OBJ = 0.3
    DISPLAY_CROP_SCALE = 1.5
    POSE_SAVE_CROP_SCALE = 2.5
    OBJ_SAVE_CROP_SCALE = 4.5
    MIN_CROP_BRIGHTNESS = 30

    # NMS parameters
    NMS_IOU_THRESHOLD = 0.45

    # Display parameters
    POSE_BOX_THICKNESS = 1
    OBJECT_BOX_THICKNESS = 1
    TEXT_THICKNESS = 1

    # Cooldown parameters
    POSE_COOLDOWN = 90
    OBJECT_COOLDOWN = 90
    REQUIRED_DETECTION_TIME = 5

    FRAME_SAVE_INTERVAL = 15

    POSE_COLORS = {
        "no-cheating": (0, 255, 0),
        "provide-object": (255, 0, 0),
        "see-friends-work": (0, 255, 255)
    }
    OBJECT_COLORS = {
        "smartphone": (0, 0, 255),
        "calculator": (0, 255, 0)
    }
