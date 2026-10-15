"""Entry point / orchestrator for the YOLOv8 exam-cheating inference system.

Launch (from the inference/ root):  python main.py

This file is intentionally thin: it bootstraps the package so the sibling
modules load as a package, wires up root-anchored paths, owns the video-loop
orchestration and threading, and starts the Flask app. All detection/tracking/
threshold behavior lives in core/pipeline.py and is preserved unchanged.
"""
import os
import sys
import threading
import queue


# --- Package bootstrap -------------------------------------------------------
# When launched as "python main.py" from inside inference/, the directory this
# file lives in is already on sys.path (as ""), so sibling modules import only
# if we register the parent. Adding the parent directory makes the relative
# imports inside the modules resolve when the
# package is imported.
_INFER_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_INFER_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from inference.config import Config, paths as _paths
from inference.utils import log_info
from inference.utils.csv_log import init_csv_log
from inference.core import CheatingDetectionPipeline
from inference.web import build_app


def main():
    log_info("[INFO] Initializing cheating detection pipeline")

    config = Config()
    pipeline = CheatingDetectionPipeline(
        config=config,
        video_path=_paths.resolve_input_video(),
        pose_model_path=_paths.resolve_pose_model(),
        object_model_path=_paths.resolve_object_model(),
    )
    pipeline.frame_queue = queue.Queue(maxsize=1)
    pipeline.stop_event = threading.Event()
    init_csv_log()

    log_file = open(_paths.DETECTION_LOG_TXT, "a")

    def _run_pipeline():
        """Thread target: run process_video and log any uncaught exception."""
        try:
            pipeline.process_video(log_file)
        except Exception:
            import traceback
            traceback.print_exc()
            log_info("[ERROR] Pipeline thread crashed — stopping application", log_file)
            pipeline.stop_event.set()

    processing_thread = threading.Thread(target=_run_pipeline, daemon=True)
    processing_thread.start()

    app = build_app(pipeline)
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        log_info("[INFO] Shutting down...")
        pipeline.stop_event.set()
        processing_thread.join(timeout=10)
        if processing_thread.is_alive():
            log_info("[WARN] Processing thread did not stop within 10s")
        else:
            log_info("Pipeline execution completed gracefully.")


if __name__ == "__main__":
    main()
