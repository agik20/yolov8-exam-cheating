"""Phase 2D — Hardware-agnostic deployment benchmark schema.

Defines JSON-serializable data structures for capturing benchmark results
on any platform (laptop, Jetson Orin Nano, or other embedded hardware).

What this module does:
- Define input/runtime configuration snapshots (BenchmarkInputConfig, BenchmarkRuntimeConfig).
- Define timing statistics (PercentileStats, ComponentTiming).
- Define resource/thermal snapshots (ResourceMetrics, ThermalMetrics).
- Define frame-accounting counters (FrameAccounting).
- Define event/evidence comparison results (EventComparison, EvidenceLatency).
- Define the full BenchmarkReport with validation status fields.

What this module does NOT do:
- No GPU-specific imports (no torch, no pynvml, no cv2).
- No wall-clock timestamps in metric values (timestamp is isolated).
- No assumption that Jetson is available — all hardware fields are Optional.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

BENCHMARK_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# PercentileStats
# ---------------------------------------------------------------------------

@dataclass
class PercentileStats:
    """Timing statistics for a list of sample values (milliseconds).

    All fields are Optional[float]; None when the sample count is 0
    (insufficient data).  p95/p99 require at least 2 samples.
    """
    count: int = 0
    mean: Optional[float] = None
    median: Optional[float] = None
    p50: Optional[float] = None   # alias of median, kept for clarity
    p95: Optional[float] = None
    p99: Optional[float] = None
    max: Optional[float] = None
    min: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "max": self.max,
            "min": self.min,
        }

    @classmethod
    def from_samples(cls, samples: List[float]) -> "PercentileStats":
        """Compute stats from a list of sample values.

        Empty list → all None except count=0.
        1 sample  → mean=max=min, median=p50, p95=p99=None.
        2+ samples → full stats.
        """
        if not samples:
            return cls(count=0)
        n = len(samples)
        mean = sum(samples) / n
        s = sorted(samples)
        min_v = s[0]
        max_v = s[-1]

        def _pct(p: int) -> Optional[float]:
            if n < 2:
                return None
            idx = min(n - 1, int(round(p / 100.0 * (n - 1))))
            return round(s[idx], 4)

        median_v = _pct(50)
        return cls(
            count=n,
            mean=round(mean, 4),
            median=median_v,
            p50=median_v,
            p95=_pct(95),
            p99=_pct(99),
            max=round(max_v, 4),
            min=round(min_v, 4),
        )


# ---------------------------------------------------------------------------
# ComponentTiming
# ---------------------------------------------------------------------------

@dataclass
class ComponentTiming:
    """Timing stats for one measurable pipeline component."""
    name: str
    stats: PercentileStats
    available: bool = True
    note: Optional[str] = None  # explanation when available=False

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "available": self.available,
            "stats": self.stats.to_dict(),
        }
        if self.note:
            d["note"] = self.note
        return d

    @classmethod
    def unavailable(cls, name: str, reason: str) -> "ComponentTiming":
        return cls(name=name, stats=PercentileStats(), available=False, note=reason)


# ---------------------------------------------------------------------------
# BenchmarkInputConfig
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkInputConfig:
    """Input video metadata snapshot for a benchmark run."""
    video_identifier: str = ""         # stable ID, NOT an absolute path
    source_fps: Optional[float] = None
    source_width: Optional[int] = None
    source_height: Optional[int] = None
    total_frames: Optional[int] = None
    warmup_frames: int = 0
    warmup_duration_s: Optional[float] = None
    steady_state_frames: int = 0
    steady_state_duration_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_identifier": self.video_identifier,
            "source_fps": self.source_fps,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "total_frames": self.total_frames,
            "warmup_frames": self.warmup_frames,
            "warmup_duration_s": self.warmup_duration_s,
            "steady_state_frames": self.steady_state_frames,
            "steady_state_duration_s": self.steady_state_duration_s,
        }


# ---------------------------------------------------------------------------
# BenchmarkRuntimeConfig
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRuntimeConfig:
    """Runtime/backend configuration snapshot for a benchmark run."""
    backend: str = "pytorch"          # "pytorch", "tensorrt", "onnx", "other"
    device: str = "cpu"               # "cuda", "cpu", "cuda:0", "other"
    model_names: List[str] = field(default_factory=list)
    model_image_size: Optional[int] = None
    confidence_threshold: Optional[float] = None
    nms_iou_threshold: Optional[float] = None
    tracking_enabled: bool = False
    tracking_config: Dict[str, Any] = field(default_factory=dict)
    batch_size: Optional[int] = None
    precision_mode: Optional[str] = None   # "fp32", "fp16", "int8", None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "model_names": self.model_names,
            "model_image_size": self.model_image_size,
            "confidence_threshold": self.confidence_threshold,
            "nms_iou_threshold": self.nms_iou_threshold,
            "tracking_enabled": self.tracking_enabled,
            "tracking_config": self.tracking_config,
            "batch_size": self.batch_size,
            "precision_mode": self.precision_mode,
        }


# ---------------------------------------------------------------------------
# FrameAccounting
# ---------------------------------------------------------------------------

@dataclass
class FrameAccounting:
    """Explicit frame counter for a benchmark run.

    Distinguishes intentional sampling from unintentional drops.
    """
    frames_read: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    frames_output: int = 0
    intentional_sampling: bool = False
    intentional_sampling_note: Optional[str] = None
    # When intentional_sampling=True, frames_dropped is interpreted as
    # intentional-skip count, not accidental drops.

    def processing_fps(self, elapsed_s: float) -> Optional[float]:
        if elapsed_s <= 0:
            return None
        return round(self.frames_processed / elapsed_s, 3)

    def real_time_sustainable(
        self,
        source_fps: Optional[float],
        processing_fps: Optional[float],
    ) -> Optional[bool]:
        """Return True/False/None.  None when either value is unavailable."""
        if source_fps is None or processing_fps is None:
            return None
        return processing_fps >= source_fps

    def to_dict(
        self,
        elapsed_s: float = 0.0,
        source_fps: Optional[float] = None,
    ) -> Dict[str, Any]:
        p_fps = self.processing_fps(elapsed_s)
        return {
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "frames_output": self.frames_output,
            "intentional_sampling": self.intentional_sampling,
            "intentional_sampling_note": self.intentional_sampling_note,
            "processing_fps": p_fps,
            "real_time_sustainable": self.real_time_sustainable(source_fps, p_fps),
        }


# ---------------------------------------------------------------------------
# ResourceMetrics
# ---------------------------------------------------------------------------

@dataclass
class ResourceMetrics:
    """Hardware resource snapshot.  All values are Optional.

    A missing/unavailable sensor produces None, NOT a fake zero.
    """
    # CPU
    cpu_utilization_pct: Optional[float] = None
    cpu_frequency_mhz: Optional[float] = None
    # Memory
    ram_used_mb: Optional[float] = None
    ram_total_mb: Optional[float] = None
    # GPU
    gpu_utilization_pct: Optional[float] = None
    gpu_frequency_mhz: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    # Thermal
    temperature_c: Optional[float] = None
    thermal_throttling: Optional[bool] = None
    # Power
    power_mode: Optional[str] = None
    power_consumption_w: Optional[float] = None
    # Process
    python_process_memory_mb: Optional[float] = None
    python_process_memory_peak_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResourceMetrics":
        valid_keys = set(cls.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# ThermalMetrics
# ---------------------------------------------------------------------------

@dataclass
class ThermalMetrics:
    """Thermal behaviour during a sustained run."""
    max_temperature_c: Optional[float] = None
    sustained_temperature_c: Optional[float] = None
    thermal_throttling_detected: Optional[bool] = None
    temperature_delta_c: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThermalMetrics":
        valid_keys = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ---------------------------------------------------------------------------
# EventComparison
# ---------------------------------------------------------------------------

@dataclass
class EventComparison:
    """Event-level comparison between reference and deployment backends."""
    reference_event_count: Optional[int] = None
    deployment_event_count: Optional[int] = None
    event_count_delta: Optional[int] = None
    reference_duplicate_count: Optional[int] = None
    deployment_duplicate_count: Optional[int] = None
    reference_tp: Optional[int] = None
    reference_fp: Optional[int] = None
    reference_fn: Optional[int] = None
    deployment_tp: Optional[int] = None
    deployment_fp: Optional[int] = None
    deployment_fn: Optional[int] = None
    reference_mean_confirmation_latency_s: Optional[float] = None
    deployment_mean_confirmation_latency_s: Optional[float] = None
    # Tolerance for event count difference (0 = exact match required)
    event_count_tolerance: int = 0

    def within_tolerance(self) -> Optional[bool]:
        if self.reference_event_count is None or self.deployment_event_count is None:
            return None
        return abs(self.deployment_event_count - self.reference_event_count) <= self.event_count_tolerance

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["within_tolerance"] = self.within_tolerance()
        return d


# ---------------------------------------------------------------------------
# EvidenceLatency
# ---------------------------------------------------------------------------

@dataclass
class EvidenceLatency:
    """Separate timing: event confirmation → evidence generation → available."""
    confirmation_to_evidence_ms: Optional[float] = None
    evidence_generation_ms: Optional[float] = None
    evidence_failure_count: int = 0
    evidence_total_count: int = 0
    asynchronous: bool = False   # True when evidence is generated off the main loop

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# BenchmarkReport
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    """Full benchmark report — JSON-serializable, hardware-agnostic.

    validation_status values:
      "PASS"               — all required criteria met on target hardware
      "PASS_WITH_LIMITATIONS" — passed with documented non-critical issues
      "FAIL"               — a critical validation criterion failed
      "NOT_RUN"            — hardware validation not executed (laptop-only run)
      "HARNESS_READY"      — harness complete, hardware validation pending
    """
    schema_version: str = BENCHMARK_SCHEMA_VERSION
    run_id: str = ""
    timestamp_utc: str = ""

    input_config: BenchmarkInputConfig = field(default_factory=BenchmarkInputConfig)
    runtime_config: BenchmarkRuntimeConfig = field(default_factory=BenchmarkRuntimeConfig)

    hardware_name: Optional[str] = None
    hardware_info: Dict[str, Any] = field(default_factory=dict)

    warmup_duration_s: Optional[float] = None
    steady_state_duration_s: Optional[float] = None

    timing_components: List[ComponentTiming] = field(default_factory=list)

    frame_accounting: FrameAccounting = field(default_factory=FrameAccounting)
    processing_fps: Optional[float] = None
    source_fps: Optional[float] = None
    target_fps: Optional[float] = None
    real_time_status: Optional[str] = None   # "PASS", "FAIL", "NOT_RUN"

    resources: ResourceMetrics = field(default_factory=ResourceMetrics)
    thermal: ThermalMetrics = field(default_factory=ThermalMetrics)

    event_comparison: EventComparison = field(default_factory=EventComparison)
    evidence_latency: EvidenceLatency = field(default_factory=EvidenceLatency)

    semantic_regression_status: Optional[str] = None   # "PASS", "FAIL", "NOT_RUN"
    thermal_status: Optional[str] = None               # "PASS", "FAIL", "NOT_RUN"
    validation_status: str = "NOT_RUN"
    limitations: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        elapsed = self.steady_state_duration_s or 0.0
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp_utc,
            "input_config": self.input_config.to_dict(),
            "runtime_config": self.runtime_config.to_dict(),
            "hardware_name": self.hardware_name,
            "hardware_info": self.hardware_info,
            "warmup_duration_s": self.warmup_duration_s,
            "steady_state_duration_s": self.steady_state_duration_s,
            "timing_components": [c.to_dict() for c in self.timing_components],
            "frame_accounting": self.frame_accounting.to_dict(
                elapsed_s=elapsed, source_fps=self.source_fps
            ),
            "processing_fps": self.processing_fps,
            "source_fps": self.source_fps,
            "target_fps": self.target_fps,
            "real_time_status": self.real_time_status,
            "resources": self.resources.to_dict(),
            "thermal": self.thermal.to_dict(),
            "event_comparison": self.event_comparison.to_dict(),
            "evidence_latency": self.evidence_latency.to_dict(),
            "semantic_regression_status": self.semantic_regression_status,
            "thermal_status": self.thermal_status,
            "validation_status": self.validation_status,
            "limitations": self.limitations,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkReport":
        fc = d.get("frame_accounting", {})
        frame_accounting = FrameAccounting(
            frames_read=fc.get("frames_read", 0),
            frames_processed=fc.get("frames_processed", 0),
            frames_dropped=fc.get("frames_dropped", 0),
            frames_output=fc.get("frames_output", 0),
            intentional_sampling=fc.get("intentional_sampling", False),
            intentional_sampling_note=fc.get("intentional_sampling_note"),
        )
        ic_data = d.get("input_config", {})
        input_config = BenchmarkInputConfig(
            video_identifier=ic_data.get("video_identifier", ""),
            source_fps=ic_data.get("source_fps"),
            source_width=ic_data.get("source_width"),
            source_height=ic_data.get("source_height"),
            total_frames=ic_data.get("total_frames"),
            warmup_frames=ic_data.get("warmup_frames", 0),
            warmup_duration_s=ic_data.get("warmup_duration_s"),
            steady_state_frames=ic_data.get("steady_state_frames", 0),
            steady_state_duration_s=ic_data.get("steady_state_duration_s"),
        )
        rc_data = d.get("runtime_config", {})
        runtime_config = BenchmarkRuntimeConfig(
            backend=rc_data.get("backend", "pytorch"),
            device=rc_data.get("device", "cpu"),
            model_names=rc_data.get("model_names", []),
            model_image_size=rc_data.get("model_image_size"),
            confidence_threshold=rc_data.get("confidence_threshold"),
            nms_iou_threshold=rc_data.get("nms_iou_threshold"),
            tracking_enabled=rc_data.get("tracking_enabled", False),
            tracking_config=rc_data.get("tracking_config", {}),
            batch_size=rc_data.get("batch_size"),
            precision_mode=rc_data.get("precision_mode"),
        )
        resources = ResourceMetrics.from_dict(d.get("resources", {}))
        thermal = ThermalMetrics.from_dict(d.get("thermal", {}))
        ev_data = d.get("event_comparison", {})
        event_comparison = EventComparison(
            reference_event_count=ev_data.get("reference_event_count"),
            deployment_event_count=ev_data.get("deployment_event_count"),
            event_count_delta=ev_data.get("event_count_delta"),
            reference_duplicate_count=ev_data.get("reference_duplicate_count"),
            deployment_duplicate_count=ev_data.get("deployment_duplicate_count"),
            reference_tp=ev_data.get("reference_tp"),
            reference_fp=ev_data.get("reference_fp"),
            reference_fn=ev_data.get("reference_fn"),
            deployment_tp=ev_data.get("deployment_tp"),
            deployment_fp=ev_data.get("deployment_fp"),
            deployment_fn=ev_data.get("deployment_fn"),
            reference_mean_confirmation_latency_s=ev_data.get(
                "reference_mean_confirmation_latency_s"),
            deployment_mean_confirmation_latency_s=ev_data.get(
                "deployment_mean_confirmation_latency_s"),
            event_count_tolerance=ev_data.get("event_count_tolerance", 0),
        )
        el_data = d.get("evidence_latency", {})
        evidence_latency = EvidenceLatency(
            confirmation_to_evidence_ms=el_data.get("confirmation_to_evidence_ms"),
            evidence_generation_ms=el_data.get("evidence_generation_ms"),
            evidence_failure_count=el_data.get("evidence_failure_count", 0),
            evidence_total_count=el_data.get("evidence_total_count", 0),
            asynchronous=el_data.get("asynchronous", False),
        )
        timing_components = []
        for c in d.get("timing_components", []):
            s = c.get("stats", {})
            stats = PercentileStats(
                count=s.get("count", 0),
                mean=s.get("mean"),
                median=s.get("median"),
                p50=s.get("p50"),
                p95=s.get("p95"),
                p99=s.get("p99"),
                max=s.get("max"),
                min=s.get("min"),
            )
            timing_components.append(ComponentTiming(
                name=c.get("name", ""),
                stats=stats,
                available=c.get("available", True),
                note=c.get("note"),
            ))
        return cls(
            schema_version=d.get("schema_version", BENCHMARK_SCHEMA_VERSION),
            run_id=d.get("run_id", ""),
            timestamp_utc=d.get("timestamp_utc", ""),
            input_config=input_config,
            runtime_config=runtime_config,
            hardware_name=d.get("hardware_name"),
            hardware_info=d.get("hardware_info", {}),
            warmup_duration_s=d.get("warmup_duration_s"),
            steady_state_duration_s=d.get("steady_state_duration_s"),
            timing_components=timing_components,
            frame_accounting=frame_accounting,
            processing_fps=d.get("processing_fps"),
            source_fps=d.get("source_fps"),
            target_fps=d.get("target_fps"),
            real_time_status=d.get("real_time_status"),
            resources=resources,
            thermal=thermal,
            event_comparison=event_comparison,
            evidence_latency=evidence_latency,
            semantic_regression_status=d.get("semantic_regression_status"),
            thermal_status=d.get("thermal_status"),
            validation_status=d.get("validation_status", "NOT_RUN"),
            limitations=d.get("limitations", []),
            notes=d.get("notes"),
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def compute_percentile_stats(samples: List[float]) -> PercentileStats:
    """Public wrapper for PercentileStats.from_samples()."""
    return PercentileStats.from_samples(samples)


def build_component_timing(
    name: str,
    samples: List[float],
    *,
    available: bool = True,
    note: Optional[str] = None,
) -> ComponentTiming:
    """Create a ComponentTiming from a list of sample values in ms."""
    if not available:
        return ComponentTiming.unavailable(name, note or "component unavailable")
    return ComponentTiming(name=name, stats=PercentileStats.from_samples(samples))


# ---------------------------------------------------------------------------
# ResourceMonitor abstraction
# ---------------------------------------------------------------------------

@dataclass
class ResourceMonitor:
    """Abstraction for hardware resource monitoring.

    `available` is False when the underlying sensor is not present on the
    current platform.  All read methods return None when the sensor is
    unavailable — never a fake zero.

    Subclass or wrap to provide real measurements on Jetson; on a laptop
    where no sensor exists, leave available=False so reports correctly
    show None for GPU/CPU utilisation fields.
    """
    available: bool = False
    _snapshot: Optional[ResourceMetrics] = None

    def snapshot(self) -> ResourceMetrics:
        """Return a ResourceMetrics snapshot.

        When available is False, returns a ResourceMetrics with all fields
        set to None (no fake values).
        """
        if self.available and self._snapshot is not None:
            return self._snapshot
        return ResourceMetrics()   # all fields None

    def set_snapshot(self, snapshot: ResourceMetrics) -> None:
        """Inject a pre-collected snapshot (used by callers that do their
        own measurement outside this abstraction)."""
        self._snapshot = snapshot
        self.available = True


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], unit: str = "") -> str:
    if v is None:
        return "NOT AVAILABLE"
    return f"{v}{unit}"


def build_human_summary(report: "BenchmarkReport") -> str:
    """Return a concise human-readable summary string for a BenchmarkReport.

    All NOT_RUN / NOT AVAILABLE states are shown explicitly; they are
    never hidden or replaced by zeros.
    """
    lines: List[str] = []
    w = 52
    lines.append("=" * w)
    lines.append("  PHASE 2D — BENCHMARK SUMMARY")
    lines.append("=" * w)

    # Hardware
    hw = report.hardware_name or "NOT IDENTIFIED (laptop fallback)"
    backend = report.runtime_config.backend
    device = report.runtime_config.device
    lines.append(f"\nHardware:   {hw}")
    lines.append(f"Backend:    {backend} / device={device}")

    # Input
    ic = report.input_config
    lines.append(f"\nInput:")
    lines.append(f"  Video ID:     {ic.video_identifier}")
    lines.append(f"  Source FPS:   {_fmt(ic.source_fps, ' fps')}")
    lines.append(f"  Resolution:   {_fmt(ic.source_width, 'x')}x{_fmt(ic.source_height, '')}")

    # FPS
    p_fps = report.processing_fps
    lines.append(f"\nFPS:")
    lines.append(f"  Source FPS:      {_fmt(report.source_fps, ' fps')}")
    lines.append(f"  Processing FPS:  {_fmt(p_fps, ' fps')}")
    lines.append(f"  Target FPS:      {_fmt(report.target_fps, ' fps')}")
    lines.append(f"  Real-time:       {report.real_time_status or 'NOT_RUN'}")

    # Steady-state latency
    lines.append(f"\nSteady-state frame latency:")
    for comp in report.timing_components:
        if comp.name == "frame" and comp.available:
            s = comp.stats
            lines.append(f"  p50:  {_fmt(s.p50, ' ms')}")
            lines.append(f"  p95:  {_fmt(s.p95, ' ms')}")
            lines.append(f"  p99:  {_fmt(s.p99, ' ms')}")
            break

    # Frames
    fa = report.frame_accounting
    lines.append(f"\nFrames:")
    lines.append(f"  Read:       {fa.frames_read}")
    lines.append(f"  Processed:  {fa.frames_processed}")
    lines.append(f"  Dropped:    {fa.frames_dropped}"
                 + ("  (intentional sampling)" if fa.intentional_sampling else ""))
    lines.append(f"  Output:     {fa.frames_output}")

    # Events
    ec = report.event_comparison
    lines.append(f"\nEvents:")
    lines.append(f"  Reference count:    {_fmt(ec.reference_event_count)}")
    lines.append(f"  Deployment count:   {_fmt(ec.deployment_event_count)}")
    lines.append(f"  Dup. (reference):   {_fmt(ec.reference_duplicate_count)}")
    lines.append(f"  Dup. (deployment):  {_fmt(ec.deployment_duplicate_count)}")
    lines.append(f"  Within tolerance:   {'NOT_RUN' if ec.within_tolerance() is None else ec.within_tolerance()}")

    # Evidence
    el = report.evidence_latency
    lines.append(f"\nEvidence:")
    lines.append(f"  Total:         {el.evidence_total_count}")
    lines.append(f"  Failures:      {el.evidence_failure_count}")
    lines.append(f"  Asynchronous:  {el.asynchronous}")

    # Resources
    rm = report.resources
    lines.append(f"\nResources:")
    lines.append(f"  CPU util:     {_fmt(rm.cpu_utilization_pct, '%')}")
    lines.append(f"  GPU util:     {_fmt(rm.gpu_utilization_pct, '%')}")
    lines.append(f"  RAM used:     {_fmt(rm.ram_used_mb, ' MB')}")
    lines.append(f"  GPU mem:      {_fmt(rm.gpu_memory_used_mb, ' MB')}")
    lines.append(f"  Temperature:  {_fmt(report.thermal.max_temperature_c, ' °C')}")

    # Validation status
    lines.append(f"\nValidation:")
    lines.append(f"  Semantic regression:  {report.semantic_regression_status or 'NOT_RUN'}")
    lines.append(f"  Thermal:              {report.thermal_status or 'NOT_RUN'}")
    lines.append(f"  Real-time:            {report.real_time_status or 'NOT_RUN'}")
    lines.append(f"  Overall:              {report.validation_status}")

    if report.limitations:
        lines.append(f"\nKnown limitations:")
        for lim in report.limitations:
            lines.append(f"  - {lim}")

    lines.append("\n" + "=" * w)
    return "\n".join(lines)
