"""Flask web interface: MJPEG stream, index page, and REST API.

The frame source (a generator) is injected from main.py so this module stays
decoupled from the pipeline.  The API endpoints expose read-only pipeline
state — detection tracks, shadow events, configuration, calibration
readiness, and environment info — without modifying any detection logic.

Endpoints:
  GET /                        → dashboard.html SPA (or the legacy index)
  GET /video_feed              → MJPEG stream (unchanged)
  GET /api/status              → pipeline running state
  GET /api/events              → all shadow events
  GET /api/events/<event_id>   → single shadow event
  GET /api/legacy-tracks       → active pose/object track counts
  GET /api/config              → legacy + shadow configuration
  GET /api/environment         → platform/GPU/torch info
  GET /api/calibration         → GT dataset readiness + R10 blockers
"""
import os
import json
import time
from typing import Any, Dict, List, Optional

import cv2
from flask import Flask, Response, jsonify, render_template_string, send_from_directory

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _pipeline_status(pipeline) -> Dict[str, Any]:
    """Build a JSON-safe status dict from the live pipeline object."""
    now = time.time()
    elapsed = now - pipeline.start_time
    frames = getattr(pipeline, "frame_idx", 0) or 0
    src_fps = getattr(pipeline, "_src_fps", None)
    processing_fps = (frames / elapsed) if elapsed > 0 else None

    shadow_lifecycle = getattr(pipeline, "_shadow_lifecycle", None)
    shadow_events = shadow_lifecycle.shadow_events if shadow_lifecycle else []

    return {
        "running": True,
        "frame_idx": frames,
        "src_fps": src_fps,
        "processing_fps": processing_fps,
        "elapsed_seconds": round(elapsed, 2),
        "pose_tracks": len(getattr(pipeline, "id_info_pose", {})),
        "object_tracks": len(getattr(pipeline, "id_info_obj", {})),
        "shadow_event_count": len(shadow_events),
    }


def _shadow_events_list(pipeline) -> List[Dict[str, Any]]:
    """Return a list of ShadowEventRecord.to_dict() for all shadow events."""
    shadow_lifecycle = getattr(pipeline, "_shadow_lifecycle", None)
    if shadow_lifecycle is None:
        return []
    return [ev.to_dict() for ev in shadow_lifecycle.shadow_events]


def _find_event(pipeline, event_id: str) -> Optional[Dict[str, Any]]:
    for ev in _shadow_events_list(pipeline):
        if ev.get("event_id") == event_id:
            return ev
    return None


def _legacy_tracks(pipeline) -> Dict[str, Any]:
    def _track_summary(id_info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            track_id: {
                "first_time": v.get("first_time"),
                "last_saved": v.get("last_saved"),
                "last_class": v.get("last_class"),
                "last_conf": v.get("last_conf"),
            }
            for track_id, v in id_info.items()
        }

    return {
        "pose": _track_summary(getattr(pipeline, "id_info_pose", {})),
        "object": _track_summary(getattr(pipeline, "id_info_obj", {})),
    }


def _config_payload(pipeline) -> Dict[str, Any]:
    """Return legacy and shadow config as a plain JSON dict."""
    from inference.config.constants import Config

    legacy = {
        "CONF_THRESH_POSE": Config.CONF_THRESH_POSE,
        "CONF_THRESH_OBJ": Config.CONF_THRESH_OBJ,
        "NMS_IOU_THRESHOLD": Config.NMS_IOU_THRESHOLD,
        "POSE_COOLDOWN": Config.POSE_COOLDOWN,
        "OBJECT_COOLDOWN": Config.OBJECT_COOLDOWN,
        "REQUIRED_DETECTION_TIME": Config.REQUIRED_DETECTION_TIME,
        "POSE_CLASSES": list(Config.POSE_CLASSES),
        "OBJECT_CLASSES": list(Config.OBJECT_CLASSES),
    }

    shadow_config = getattr(pipeline, "shadow_config", None)
    shadow = shadow_config.to_dict() if shadow_config else None

    return {"legacy": legacy, "shadow": shadow}


def _calibration_payload() -> Dict[str, Any]:
    """Return GT dataset readiness and R10 blockers (never fabricate values)."""
    from inference.evaluation.ground_truth import GroundTruthDataset
    from inference.evaluation.gt_readiness import assess_gt_dataset_readiness

    dataset = GroundTruthDataset()
    report = assess_gt_dataset_readiness(dataset)
    return {
        "readiness": report.to_dict(),
        "r10_blockers": list(report.r10_blockers),
    }


def build_app(pipeline):
    """Build the Flask app wired to the given pipeline instance."""
    app = Flask(__name__)

    @app.route("/")
    def index():
        dashboard = os.path.join(_STATIC_DIR, "dashboard.html")
        if os.path.isfile(dashboard):
            return send_from_directory(_STATIC_DIR, "dashboard.html")
        return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Streaming</title>
    </head>
    <body>
        <h1>Video Streaming</h1>
        <img src="{{ url_for('video_feed') }}">
    </body>
    </html>
    ''')

    @app.route("/video_feed")
    def video_feed():
        return Response(
            pipeline.generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/status")
    def api_status():
        return jsonify(_pipeline_status(pipeline))

    @app.route("/api/events")
    def api_events():
        return jsonify({"events": _shadow_events_list(pipeline)})

    @app.route("/api/events/<event_id>")
    def api_event_detail(event_id: str):
        ev = _find_event(pipeline, event_id)
        if ev is None:
            return jsonify({"error": "event not found", "event_id": event_id}), 404
        return jsonify({"event": ev})

    @app.route("/api/legacy-tracks")
    def api_legacy_tracks():
        return jsonify(_legacy_tracks(pipeline))

    @app.route("/api/config")
    def api_config():
        return jsonify(_config_payload(pipeline))

    @app.route("/api/environment")
    def api_environment():
        from inference.utils.telemetry import get_environment_info

        return jsonify(get_environment_info())

    @app.route("/api/calibration")
    def api_calibration():
        return jsonify(_calibration_payload())

    return app
