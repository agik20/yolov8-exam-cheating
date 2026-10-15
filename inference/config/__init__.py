"""Config package: constants, thresholds, class definitions, and root-anchored paths."""
from .constants import Config
from .paths import (
    INFER_ROOT, INPUT_DIR, MODELS_DIR, OUTPUT_DIR,
    OUTPUT_POSE_BASE, OUTPUT_OBJ_BASE, OUTPUT_FULLROOM, OUTPUT_FRAMES,
    DETECTION_LOG_TXT,
    resolve_input_video, resolve_pose_model, resolve_object_model,
)
