"""Platform-agnostic performance metrics collector for the inference pipeline.

Uses time.perf_counter() (high-resolution monotonic).  All values are in
milliseconds.  The collector is a plain Python object — no external
dependencies, no import-time side-effects, zero cost when not attached.

Design:
  - One instance is created by the benchmark runner and injected into the
    pipeline via pipeline.metrics (attribute).
  - The pipeline checks `if self.metrics is not None:` before every timing
    call, so the production path pays ~0 overhead when metrics is None.
  - All timing data is in-memory; no disk I/O until to_json()/flush() is
    called explicitly by the benchmark runner.
"""
import json
import os
import time
from datetime import datetime
from statistics import median
from typing import Dict, List, Optional


def _percentile(samples: List[float], pct: int) -> float:
    """Return the pct-th percentile of a list of sample values (ms).

    Uses the 'nearest-rank' method (no numpy dependency).
    Returns 0.0 for an empty list.
    """
    if not samples:
        return 0.0
    n = len(samples)
    rank = max(0, min(n - 1, int(round(pct / 100 * (n - 1)))))
    return sorted(samples)[rank]


class _Timer:
    """Context-manager timer that records elapsed ms into a list."""

    def __init__(self, samples: List[float]):
        self.samples = samples
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.samples.append((time.perf_counter() - self._t0) * 1000.0)
        return False  # do not swallow exceptions

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0


class MetricsCollector:
    """Collects per-stage timing and frame counters for the pipeline.

    Create one instance, attach it to a pipeline, then call to_json()
    and to_report() after the benchmark run completes.
    """

    def __init__(self):
        # -- frame counters ------------------------------------------------
        self.input_frames: int = 0
        self.processed_frames: int = 0
        self.output_frames: int = 0
        self.dropped_frames: int = 0

        # -- queue instrumentation -----------------------------------------
        self.queue_insert_attempts: int = 0
        self.queue_insert_success: int = 0
        self.queue_full_events: int = 0

        # -- timing samples (ms) ------------------------------------------
        self._frame_times: List[float] = []
        self._pose_inference: List[float] = []
        self._pose_postprocess: List[float] = []
        self._pose_event: List[float] = []
        self._object_inference: List[float] = []
        self._object_postprocess: List[float] = []
        self._object_event: List[float] = []
        self._evidence_save: List[float] = []
        self._jpeg_encode: List[float] = []

        # -- evidence event log --------------------------------------------
        self.evidence_events: List[Dict] = []

        # -- bookkeeping ----------------------------------------------------
        self._benchmark_start: float = 0.0
        self._benchmark_end: float = 0.0
        self.input_fps: Optional[float] = None
        self.input_width: int = 0
        self.input_height: int = 0
        self.input_duration_s: float = 0.0

    # ------------------------------------------------------------------
    # Timing context managers — used inside the pipeline
    # ------------------------------------------------------------------
    _TIMER_MAP = {
        'frame': '_frame_times',
        'pose_inference': '_pose_inference',
        'pose_postprocess': '_pose_postprocess',
        'pose_event': '_pose_event',
        'object_inference': '_object_inference',
        'object_postprocess': '_object_postprocess',
        'object_event': '_object_event',
        'evidence': '_evidence_save',
        'jpeg_encode': '_jpeg_encode',
    }

    def timer(self, name: str) -> _Timer:
        """Return a context-manager timer that records into the named sample list."""
        attr = self._TIMER_MAP[name]
        return _Timer(getattr(self, attr))

    # ------------------------------------------------------------------
    # Frame counter helpers
    # ------------------------------------------------------------------
    def on_frame_read(self):
        self.input_frames += 1

    def on_frame_processed(self):
        self.processed_frames += 1

    def on_queue_insert(self, success: bool):
        self.queue_insert_attempts += 1
        if success:
            self.queue_insert_success += 1
            self.output_frames += 1
        else:
            self.dropped_frames += 1
            self.queue_full_events += 1

    # ------------------------------------------------------------------
    # Evidence event recording
    # ------------------------------------------------------------------
    def record_evidence_event(self, *,
                               frame_idx: int,
                               track_id: int,
                               class_name: str,
                               confidence: Optional[float],
                               save_duration_ms: float,
                               detection_type: str = "pose"):
        self.evidence_events.append({
            "timestamp": datetime.now().isoformat(),
            "frame_idx": frame_idx,
            "track_id": track_id,
            "class_name": class_name,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "detection_type": detection_type,
            "save_duration_ms": round(save_duration_ms, 3),
        })

    # ------------------------------------------------------------------
    # Benchmark boundary
    # ------------------------------------------------------------------
    def discard_warmup_samples(self):
        """Clear all timing sample lists and the evidence event log.

        Called by the benchmark runner after the warmup phase so that
        only steady-state data appears in the JSON report.  Does not
        modify any detection/pipeline state.
        """
        self._frame_times.clear()
        self._pose_inference.clear()
        self._pose_postprocess.clear()
        self._pose_event.clear()
        self._object_inference.clear()
        self._object_postprocess.clear()
        self._object_event.clear()
        self._evidence_save.clear()
        self._jpeg_encode.clear()
        self.evidence_events.clear()

    def mark_benchmark_start(self, input_fps=None, width=0, height=0, duration_s=0.0):
        self._benchmark_start = time.perf_counter()
        if input_fps is not None:
            self.input_fps = float(input_fps)
        self.input_width = int(width)
        self.input_height = int(height)
        self.input_duration_s = float(duration_s)

    def mark_benchmark_end(self):
        self._benchmark_end = time.perf_counter()

    @property
    def benchmark_elapsed_s(self) -> float:
        if self._benchmark_end == 0.0:
            return 0.0
        return (self._benchmark_end - self._benchmark_start)

    # ------------------------------------------------------------------
    # Statistical summary
    # ------------------------------------------------------------------
    @staticmethod
    def _summary(samples: List[float]) -> Dict:
        if not samples:
            return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        n = len(samples)
        mean = sum(samples) / n
        return {
            "count": n,
            "mean": round(mean, 3),
            "median": round(median(samples), 3),
            "p95": round(_percentile(samples, 95), 3),
            "p99": round(_percentile(samples, 99), 3),
            "max": round(max(samples), 3),
        }

    def _stage_summary(self, key: str) -> Dict:
        return self._summary(getattr(self, self._TIMER_MAP[key]))

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------
    def to_json(self, env: Dict) -> str:
        elapsed = self.benchmark_elapsed_s
        processed_fps = round(self.processed_frames / elapsed, 3) if elapsed > 0 else 0.0
        input_to_processing = round(self.processed_frames / self.input_frames, 4) if self.input_frames else 0.0

        # jpeg_encode: mark explicitly unavailable when no real MJPEG encoding occurred
        jpeg_count = len(self._jpeg_encode)
        if jpeg_count == 0:
            jpeg_entry: Dict = {
                "available": False,
                "reason": "benchmark does not execute generate_frames(); no JPEG encoding performed",
            }
        else:
            jpeg_entry = self._stage_summary("jpeg_encode")

        # output_fps is meaningful only when real frames were encoded through generate_frames()
        # The benchmark puts None (not real frames) into the queue, so output_frames reflects
        # queue-insertion success only, not actual MJPEG output.
        output_fps_value: Optional[float] = None
        output_fps_note: Optional[str] = None
        if jpeg_count == 0:
            output_fps_note = ("unavailable — benchmark puts None into frame_queue; "
                               "generate_frames() is not executed; no real MJPEG output is produced")

        # queue mode: benchmark runs with a producer-only queue (size 1, no consumer)
        queue_mode = "producer_only"
        consumer_active = False

        result = {
            "environment": env,
            "input": {
                "width": self.input_width,
                "height": self.input_height,
                "fps": self.input_fps,
                "duration_s": self.input_duration_s,
            },
            "frames": {
                "input": self.input_frames,
                "processed": self.processed_frames,
                "output": self.output_frames,
                "dropped": self.dropped_frames,
            },
            "timing_ms": {
                "frame": self._stage_summary("frame"),
                "pose_inference": self._stage_summary("pose_inference"),
                "pose_postprocess": self._stage_summary("pose_postprocess"),
                "pose_event": self._stage_summary("pose_event"),
                "object_inference": self._stage_summary("object_inference"),
                "object_postprocess": self._stage_summary("object_postprocess"),
                "object_event": self._stage_summary("object_event"),
                "evidence": self._stage_summary("evidence"),
                "jpeg_encode": jpeg_entry,
            },
            "throughput": {
                "processed_fps": processed_fps,
                "output_fps": output_fps_value,
                "output_fps_note": output_fps_note,
                "input_to_processing_ratio": input_to_processing,
            },
            "queue": {
                "mode": queue_mode,
                "consumer_active": consumer_active,
                "insert_attempts": self.queue_insert_attempts,
                "insert_success": self.queue_insert_success,
                "full_events": self.queue_full_events,
                "dropped": self.dropped_frames,
                "note": ("producer-only queue: no consumer drains the queue during benchmark; "
                         "dropped-frame count reflects queue-full events, not production drop rate"),
            },
            "evidence_events": self.evidence_events,
        }
        return json.dumps(result, indent=2)

    # ------------------------------------------------------------------
    # Human-readable terminal report
    # ------------------------------------------------------------------
    def to_report(self, env: Dict) -> str:
        lines = []
        w = 44
        lines.append("=" * w)
        lines.append("  BASELINE — LAPTOP")
        lines.append("=" * w)

        lines.append("\nPlatform:")
        lines.append(f"  OS:              {env.get('os', 'unknown')}")
        lines.append(f"  Python:          {env.get('python_version', 'unknown')}")
        lines.append(f"  PyTorch:         {env.get('torch_version', 'unavailable')}")
        lines.append(f"  OpenCV:          {env.get('opencv_version', 'unknown')}")
        lines.append(f"  Ultralytics:     {env.get('ultralytics_version', 'unavailable')}")
        if env.get("cuda_available"):
            lines.append(f"  CUDA:            available ({env.get('cuda_version', '?')})")
            if env.get("gpu_name"):
                lines.append(f"  GPU:             {env['gpu_name']}")
                if env.get("gpu_memory_mb"):
                    lines.append(f"  GPU memory:      {env['gpu_memory_mb']} MB total")
        else:
            lines.append("  CUDA:            unavailable")
            lines.append("  GPU:             CPU only / no GPU detected")

        lines.append("\nInput:")
        lines.append(f"  Resolution:      {self.input_width}x{self.input_height}")
        lines.append(f"  Source FPS:      {self.input_fps if self.input_fps is not None else 'unavailable'}")
        lines.append(f"  Duration:        {self.input_duration_s:.1f} s" if self.input_duration_s else "")

        lines.append("\nFrames:")
        lines.append(f"  Input:           {self.input_frames}")
        lines.append(f"  Processed:       {self.processed_frames}")
        lines.append(f"  Output:          {self.output_frames}")
        lines.append(f"  Dropped:         {self.dropped_frames}")

        elapsed = self.benchmark_elapsed_s
        processed_fps = round(self.processed_frames / elapsed, 2) if elapsed > 0 else 0.0
        lines.append("\nThroughput:")
        lines.append(f"  Processed FPS:   {processed_fps}")
        if self._jpeg_encode:
            output_fps = round(self.output_frames / elapsed, 2) if elapsed > 0 else 0.0
            lines.append(f"  Output FPS:      {output_fps}")
        else:
            lines.append("  Output FPS:      unavailable (generate_frames not executed)")

        # Queue mode
        lines.append("\nQueue (producer-only, no consumer):")
        lines.append(f"  Insert attempts: {self.queue_insert_attempts}")
        lines.append(f"  Success:         {self.queue_insert_success}")
        lines.append(f"  Full events:     {self.queue_full_events}")
        lines.append(f"  Dropped:         {self.dropped_frames}")
        lines.append("  Note:            producer-only mode; dropped count does NOT reflect production drop rate")

        for label, key in [
            ("Frame processing", "frame"),
            ("Pose inference", "pose_inference"),
            ("Pose postprocess", "pose_postprocess"),
            ("Object inference", "object_inference"),
            ("Object postprocess", "object_postprocess"),
        ]:
            s = self._stage_summary(key)
            if s["count"]:
                lines.append(f"\n{label}:")
                lines.append(f"  Count:           {s['count']}")
                lines.append(f"  Mean:            {s['mean']} ms")
                lines.append(f"  Median:          {s['median']} ms")
                lines.append(f"  P95:             {s['p95']} ms")
                lines.append(f"  P99:             {s['p99']} ms")
                lines.append(f"  Max:             {s['max']} ms")

        ev = self._stage_summary("evidence")
        lines.append("\nEvidence:")
        lines.append(f"  Events:          {len(self.evidence_events)}")
        if ev["count"]:
            lines.append(f"  Mean save time:  {ev['mean']} ms")
            lines.append(f"  P95 save time:   {ev['p95']} ms")
        else:
            lines.append("  Mean save time:  no evidence events in this run")

        lines.append("\n" + "=" * w)
        lines.append("  END")
        lines.append("=" * w)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: write the JSON report to a file
# ---------------------------------------------------------------------------
def flush_to_json(metrics: MetricsCollector, env: Dict, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(metrics.to_json(env))
    return output_path
