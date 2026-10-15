"""Phase 0A — Laptop Baseline Benchmark Runner.

Standalone benchmark: loads the pipeline headless, processes a fixed number
of seconds of the representative video, collects metrics, writes JSON and
prints a human-readable report.

Usage (from the yolov8-exam-cheating/ directory):
    python inference/benchmark.py
    python inference/benchmark.py --duration 120 --warmup 10
    python inference/benchmark.py --json-out inference/output/benchmark_results.json

Design principles:
  - Headless: no cv2.imshow, no cv2.waitKey.  The benchmark does NOT start
    the Flask app; it runs the processing thread directly.
  - Warm-up: the first N seconds of frames are processed but NOT counted
    in steady-state metrics.  The model is fully loaded before the warmup
    timer starts; only the per-frame timing samples from after warmup are
    kept in the JSON report.  Implementation: process warmup frames, then
    clear the metrics sample lists, then process steady-state frames.
  - No changes to detection semantics: confidence thresholds, NMS, tracker
    config, cooldown, class-switch logic are all inherited from
    CheatingDetectionPipeline and untouched.
  - Platform-agnostic: same code runs on laptop or Jetson.
"""
import argparse
import os
import queue
import sys
import threading
import time
from datetime import datetime
from typing import Dict, Optional

# Ensure the inference package is importable when launched as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from inference.config import Config, paths as _paths
from inference.core import CheatingDetectionPipeline
from inference.utils import log_info, MetricsCollector, flush_to_json
from inference.utils.telemetry import get_environment_info
from inference.utils.csv_log import init_csv_log
import cv2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _query_video_info(video_path: str) -> Dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return {"width": width, "height": height, "fps": fps,
            "total_frames": total_frames, "duration_s": duration_s}


def _run_warmup(pipeline, log_file, video_path: str, warmup_frames: int,
                metrics: MetricsCollector):
    """Process warmup_frames to let the model/allocator settle.

    During warmup, metrics sample lists are collected and then cleared so
    that steady-state statistics are not skewed by the first-iteration
    costs (CUDA context warmup, memory pool allocation, etc.).
    """
    log_info(f"[BENCH] Running {warmup_frames}-frame warmup ...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for warmup: {video_path}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    for i in range(warmup_frames):
        ok, frame = cap.read()
        if not ok:
            break
        pipeline.frame_idx += 1
        now = datetime.now()
        ts = now.strftime('%H%M%S')
        pipeline.process_frame(frame, now, ts, log_file, metrics)
        if (i + 1) % 100 == 0:
            log_info(f"[BENCH]   warmup frame {i + 1}/{warmup_frames} done")

    cap.release()

    # Discard all warmup samples and warmup evidence events so that
    # steady-state statistics and the evidence event log contain only
    # data collected after warmup.
    metrics.discard_warmup_samples()
    log_info("[BENCH] Warmup samples and events discarded; steady-state collection begins.")


def _run_steady_state(pipeline, log_file, video_path: str,
                      steady_frames: int, metrics: MetricsCollector,
                      input_fps: float, width: int, height: int):
    """Process steady_frames frames with full metrics collection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for benchmark: {video_path}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    metrics.mark_benchmark_start(
        input_fps=input_fps,
        width=width,
        height=height,
        duration_s=steady_frames / input_fps if input_fps > 0 else 0.0,
    )

    for i in range(steady_frames):
        ok, frame = cap.read()
        if not ok:
            log_info(f"[BENCH] End of video reached at frame {i + 1}/{steady_frames}")
            break

        metrics.on_frame_read()
        pipeline.frame_idx += 1
        now = datetime.now()
        ts = now.strftime('%H%M%S')

        with metrics.timer('frame'):
            pipeline.process_frame(frame, now, ts, log_file, metrics)

        metrics.on_frame_processed()

        # Simulate the same queue-insertion semantics as production
        if pipeline.frame_queue is not None:
            success_put = False
            if pipeline.frame_queue.empty():
                try:
                    pipeline.frame_queue.put(None, block=False)
                    success_put = True
                except queue.Full:
                    pass
            metrics.on_queue_insert(success_put)

        if (i + 1) % 100 == 0:
            n = len(metrics._frame_times)
            mean_ms = sum(metrics._frame_times) / max(1, n)
            log_info(f"[BENCH]   steady frame {i + 1}/{steady_frames}  "
                     f"mean frame={mean_ms:.1f}ms  processed={metrics.processed_frames}")

    metrics.mark_benchmark_end()
    cap.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 0A — Laptop Baseline Benchmark")
    parser.add_argument("--duration", type=int, default=60,
                        help="Steady-state benchmark duration in seconds (default: 60)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup duration in seconds (default: 5)")
    parser.add_argument("--fps-assume", type=float, default=None,
                        help="If the video FPS cannot be read, assume this value")
    parser.add_argument("--json-out", default=None,
                        help="Output JSON path (default: inference/output/benchmark_results.json)")
    args = parser.parse_args()

    json_out = args.json_out or os.path.join(
        _paths.INFER_ROOT, "output", "benchmark_results.json")

    log_file = open(os.path.join(_paths.INFER_ROOT, "detection_log.txt"), "a")
    init_csv_log()

    video_path = _paths.resolve_input_video()
    log_info(f"[BENCH] Video: {video_path}")

    info = _query_video_info(video_path)
    fps = info["fps"] or args.fps_assume or 30.0
    total_frames_available = info["total_frames"]
    steady_frames = int(fps * args.duration)
    warmup_frames = int(fps * args.warmup)

    if steady_frames + warmup_frames > total_frames_available:
        log_info(f"[BENCH] WARNING: requested {steady_frames + warmup_frames} frames, "
                 f"only {total_frames_available} available. Clamping.")
        steady_frames = max(1, total_frames_available - warmup_frames)

    log_info(f"[BENCH] Input: {info['width']}x{info['height']} @ {fps:.2f} fps, "
             f"{info['duration_s']:.1f} s total, {total_frames_available} frames")
    log_info(f"[BENCH] Benchmark: warmup={args.warmup}s ({warmup_frames} frames), "
             f"steady={args.duration}s ({steady_frames} frames)")

    # Build pipeline headless
    cfg = Config()
    pipeline = CheatingDetectionPipeline(
        config=cfg,
        video_path=video_path,
        pose_model_path=_paths.resolve_pose_model(),
        object_model_path=_paths.resolve_object_model(),
        headless=True,
    )
    pipeline.frame_queue = queue.Queue(maxsize=1)
    pipeline.stop_event = threading.Event()
    pipeline.start_time = time.time()

    metrics = MetricsCollector()
    pipeline.metrics = metrics

    log_info("[BENCH] Pipeline ready (headless, models loaded).")
    log_info(f"[BENCH] Python: {__import__('sys').version.split()[0]}")

    # Phase 1: Warm-up
    _run_warmup(pipeline, log_file, video_path, warmup_frames, metrics)

    # Phase 2: Steady-state
    log_info(f"[BENCH] Starting steady-state benchmark: {steady_frames} frames ...")
    _run_steady_state(pipeline, log_file, video_path, steady_frames,
                      metrics, fps, info["width"], info["height"])

    # Collect environment
    env = get_environment_info()
    env["benchmark"] = {
        "warmup_seconds": args.warmup,
        "duration_seconds": args.duration,
        "steady_frames": steady_frames,
        "warmup_frames": warmup_frames,
    }

    # Write JSON
    json_path = flush_to_json(metrics, env, json_out)
    log_info(f"[BENCH] JSON results written to: {json_path}")

    # Print human-readable report
    print(metrics.to_report(env))
    print(f"\nJSON output: {json_path}")

    log_file.close()


if __name__ == "__main__":
    main()
