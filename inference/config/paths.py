"""Central, root-anchored path resolution for the inference system.

Everything under inference/ is addressed relative to INFER_ROOT, which is the
directory containing this file (the inference/ root). This makes the system
independent of the current working directory from which it is launched.
"""
import os

# The inference root = the parent of the config/ directory (i.e. .../yolov8-exam-cheating/inference).
_INFER_PARENT = os.path.dirname(os.path.abspath(__file__))
INFER_ROOT = os.path.dirname(_INFER_PARENT)

# Inputs / models / outputs, all anchored to the root.
INPUT_DIR = os.path.join(INFER_ROOT, "input")
MODELS_DIR = os.path.join(INFER_ROOT, "models")
OUTPUT_DIR = os.path.join(INFER_ROOT, "output")

OUTPUT_POSE_BASE = os.path.join(OUTPUT_DIR, "pose")
OUTPUT_OBJ_BASE = os.path.join(OUTPUT_DIR, "object")
OUTPUT_FULLROOM = os.path.join(OUTPUT_DIR, "room-view")
OUTPUT_FRAMES = os.path.join(OUTPUT_DIR, "frames")

# Logs / CSVs written next to the code, not the CWD.
DETECTION_LOG_TXT = os.path.join(INFER_ROOT, "detection_log.txt")


def resolve_input_video() -> str:
    """Return the path to the single video file in input/."""
    if os.path.isdir(INPUT_DIR):
        for name in sorted(os.listdir(INPUT_DIR)):
            if name.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm")):
                return os.path.join(INPUT_DIR, name)
    # Fallback if no video file found yet: point at input/ so the caller errors clearly.
    return os.path.join(INPUT_DIR, "<missing-video>")


def resolve_pose_model() -> str:
    """Return the path to the pose .pt model in models/."""
    return _find_model("pose")


def resolve_object_model() -> str:
    """Return the path to the object-detection .pt model in models/."""
    return _find_model("object")


def _find_model(kind: str) -> str:
    """Pick a .pt model in models/. Prefer a file whose name includes the kind."""
    if not os.path.isdir(MODELS_DIR):
        return os.path.join(MODELS_DIR, f"<missing-{kind}-model>")
    preferred, fallback = None, None
    for name in sorted(os.listdir(MODELS_DIR)):
        if not name.lower().endswith(".pt"):
            continue
        if kind in name.lower():
            preferred = os.path.join(MODELS_DIR, name)
            break
        fallback = os.path.join(MODELS_DIR, name)
    if preferred:
        return preferred
    if fallback:
        return fallback
    return os.path.join(MODELS_DIR, f"<missing-{kind}-model>")
