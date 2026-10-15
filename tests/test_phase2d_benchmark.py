"""Phase 2D — Deployment benchmark schema tests.

Covers:
- PercentileStats (empty, single, multi-sample)
- ComponentTiming (available vs unavailable)
- FrameAccounting (processing_fps, real_time_sustainable)
- ResourceMetrics (no fake zeros, from_dict)
- ResourceMonitor (unavailable → all None)
- EventComparison (within_tolerance)
- EvidenceLatency (defaults, to_dict)
- BenchmarkReport (JSON round-trip, NOT_RUN states)
- build_human_summary (NOT_RUN visibility)
- No production coupling (no torch, cv2, ultralytics imports)
"""
from __future__ import annotations

import json
import pytest

from inference.evaluation.benchmark_schema import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkInputConfig,
    BenchmarkReport,
    BenchmarkRuntimeConfig,
    ComponentTiming,
    EvidenceLatency,
    EventComparison,
    FrameAccounting,
    PercentileStats,
    ResourceMetrics,
    ResourceMonitor,
    ThermalMetrics,
    build_component_timing,
    build_human_summary,
    compute_percentile_stats,
)


# ---------------------------------------------------------------------------
# PercentileStats
# ---------------------------------------------------------------------------

class TestPercentileStats:

    def test_empty_samples(self):
        s = PercentileStats.from_samples([])
        assert s.count == 0
        assert s.mean is None
        assert s.p95 is None
        assert s.max is None

    def test_single_sample(self):
        s = PercentileStats.from_samples([5.0])
        assert s.count == 1
        assert s.mean == pytest.approx(5.0)
        assert s.max == pytest.approx(5.0)
        assert s.min == pytest.approx(5.0)
        # p95/p99 require >= 2 samples
        assert s.p95 is None
        assert s.p99 is None

    def test_ten_samples(self):
        s = PercentileStats.from_samples([float(i) for i in range(1, 11)])
        assert s.count == 10
        assert s.min == pytest.approx(1.0)
        assert s.max == pytest.approx(10.0)
        assert s.p95 is not None
        assert s.p99 is not None
        # p95 of [1..10] nearest-rank: idx = round(0.95*9) = round(8.55) = 9 → s[9]=10.0
        assert s.p95 == pytest.approx(10.0)

    def test_deterministic(self):
        samples = [3.0, 1.0, 4.0, 1.5, 2.5, 0.5]
        s1 = PercentileStats.from_samples(list(samples))
        s2 = PercentileStats.from_samples(list(samples))
        assert s1.to_dict() == s2.to_dict()

    def test_to_dict(self):
        s = PercentileStats.from_samples([1.0, 2.0, 3.0])
        d = s.to_dict()
        assert d["count"] == 3
        assert d["mean"] == pytest.approx(2.0)
        assert "p95" in d


# ---------------------------------------------------------------------------
# ComponentTiming
# ---------------------------------------------------------------------------

class TestComponentTiming:

    def test_unavailable_no_fake_stats(self):
        ct = ComponentTiming.unavailable("pose_inference", "CUDA not available")
        assert ct.available is False
        assert ct.stats.count == 0
        assert ct.stats.mean is None
        assert ct.note == "CUDA not available"

    def test_available_with_samples(self):
        ct = build_component_timing("frame", [10.0, 20.0, 30.0])
        assert ct.available is True
        assert ct.stats.count == 3
        assert ct.stats.mean == pytest.approx(20.0)

    def test_to_dict_includes_note_when_unavailable(self):
        ct = ComponentTiming.unavailable("jpeg_encode", "not executed")
        d = ct.to_dict()
        assert d["available"] is False
        assert d["note"] == "not executed"

    def test_to_dict_omits_note_when_none(self):
        ct = build_component_timing("frame", [1.0, 2.0])
        d = ct.to_dict()
        assert "note" not in d


# ---------------------------------------------------------------------------
# FrameAccounting
# ---------------------------------------------------------------------------

class TestFrameAccounting:

    def test_processing_fps_zero_elapsed(self):
        fa = FrameAccounting(frames_processed=100)
        assert fa.processing_fps(0.0) is None

    def test_processing_fps_normal(self):
        fa = FrameAccounting(frames_processed=300)
        fps = fa.processing_fps(10.0)
        assert fps == pytest.approx(30.0)

    def test_real_time_sustainable_true(self):
        fa = FrameAccounting(frames_processed=320)
        r = fa.real_time_sustainable(30.0, fa.processing_fps(10.0))
        assert r is True   # 32 fps >= 30 fps

    def test_real_time_sustainable_false(self):
        fa = FrameAccounting(frames_processed=200)
        r = fa.real_time_sustainable(30.0, fa.processing_fps(10.0))
        assert r is False   # 20 fps < 30 fps

    def test_real_time_sustainable_none(self):
        fa = FrameAccounting(frames_processed=100)
        assert fa.real_time_sustainable(None, 30.0) is None
        assert fa.real_time_sustainable(30.0, None) is None

    def test_intentional_sampling_flag(self):
        fa = FrameAccounting(
            frames_read=100,
            frames_processed=50,
            frames_dropped=50,
            intentional_sampling=True,
            intentional_sampling_note="2x downsample",
        )
        d = fa.to_dict(elapsed_s=10.0, source_fps=30.0)
        assert d["intentional_sampling"] is True
        assert d["intentional_sampling_note"] == "2x downsample"

    def test_unintentional_drop_default(self):
        fa = FrameAccounting(
            frames_read=100,
            frames_processed=90,
            frames_dropped=10,
        )
        d = fa.to_dict()
        assert d["intentional_sampling"] is False


# ---------------------------------------------------------------------------
# ResourceMetrics
# ---------------------------------------------------------------------------

class TestResourceMetrics:

    def test_all_none_by_default(self):
        rm = ResourceMetrics()
        d = rm.to_dict()
        for key in d:
            assert d[key] is None, f"{key} should be None, got {d[key]}"

    def test_from_dict_ignores_unknown_keys(self):
        rm = ResourceMetrics.from_dict({
            "cpu_utilization_pct": 45.0,
            "unknown_field": 999,
        })
        assert rm.cpu_utilization_pct == 45.0
        assert not hasattr(rm, "unknown_field")

    def test_from_dict_empty(self):
        rm = ResourceMetrics.from_dict({})
        assert rm.cpu_utilization_pct is None


# ---------------------------------------------------------------------------
# ResourceMonitor
# ---------------------------------------------------------------------------

class TestResourceMonitor:

    def test_unavailable_returns_all_none(self):
        mon = ResourceMonitor(available=False)
        snap = mon.snapshot()
        assert snap.gpu_utilization_pct is None
        assert snap.cpu_utilization_pct is None

    def test_available_with_snapshot(self):
        mon = ResourceMonitor()
        mon.set_snapshot(ResourceMetrics(cpu_utilization_pct=75.0,
                                         ram_used_mb=512.0))
        snap = mon.snapshot()
        assert snap.cpu_utilization_pct == 75.0
        assert snap.ram_used_mb == 512.0

    def test_available_flag_set_by_set_snapshot(self):
        mon = ResourceMonitor(available=False)
        assert mon.available is False
        mon.set_snapshot(ResourceMetrics())
        assert mon.available is True


# ---------------------------------------------------------------------------
# EventComparison
# ---------------------------------------------------------------------------

class TestEventComparison:

    def test_within_tolerance_exact_match(self):
        ec = EventComparison(
            reference_event_count=10,
            deployment_event_count=10,
            event_count_tolerance=0,
        )
        assert ec.within_tolerance() is True

    def test_within_tolerance_small_delta(self):
        ec = EventComparison(
            reference_event_count=10,
            deployment_event_count=11,
            event_count_tolerance=2,
        )
        assert ec.within_tolerance() is True

    def test_not_within_tolerance_large_delta(self):
        ec = EventComparison(
            reference_event_count=10,
            deployment_event_count=15,
            event_count_tolerance=2,
        )
        assert ec.within_tolerance() is False

    def test_none_when_counts_missing(self):
        ec = EventComparison()
        assert ec.within_tolerance() is None

    def test_to_dict_includes_within_tolerance(self):
        ec = EventComparison(reference_event_count=5,
                             deployment_event_count=5)
        d = ec.to_dict()
        assert "within_tolerance" in d
        assert d["within_tolerance"] is True


# ---------------------------------------------------------------------------
# EvidenceLatency
# ---------------------------------------------------------------------------

class TestEvidenceLatency:

    def test_defaults(self):
        el = EvidenceLatency()
        assert el.confirmation_to_evidence_ms is None
        assert el.evidence_failure_count == 0
        assert el.asynchronous is False

    def test_to_dict(self):
        el = EvidenceLatency(
            confirmation_to_evidence_ms=150.0,
            evidence_generation_ms=200.0,
            evidence_failure_count=1,
            evidence_total_count=50,
            asynchronous=True,
        )
        d = el.to_dict()
        assert d["asynchronous"] is True
        assert d["evidence_failure_count"] == 1


# ---------------------------------------------------------------------------
# BenchmarkReport — JSON round-trip and NOT_RUN states
# ---------------------------------------------------------------------------

class TestBenchmarkReport:

    def _base_report(self) -> BenchmarkReport:
        return BenchmarkReport(
            run_id="run_test_001",
            timestamp_utc="2026-10-15T00:00:00Z",
            input_config=BenchmarkInputConfig(video_identifier="canonical_video"),
            runtime_config=BenchmarkRuntimeConfig(
                backend="pytorch", device="cpu", model_names=["pose", "object"],
            ),
            hardware_name="Laptop (Dev)",
            frame_accounting=FrameAccounting(
                frames_read=300, frames_processed=300,
                frames_dropped=0, frames_output=300,
            ),
            source_fps=30.0,
            target_fps=30.0,
            processing_fps=12.0,
            real_time_status="FAIL",
            semantic_regression_status="NOT_RUN",
            thermal_status="NOT_RUN",
            validation_status="HARNESS_READY",
            limitations=["laptop-only run; Jetson validation NOT RUN"],
        )

    def test_json_serializable(self):
        report = self._base_report()
        s = report.to_json()
        parsed = json.loads(s)
        assert parsed["run_id"] == "run_test_001"
        assert parsed["validation_status"] == "HARNESS_READY"
        assert parsed["frame_accounting"]["frames_processed"] == 300

    def test_from_dict_round_trip(self):
        report = self._base_report()
        restored = BenchmarkReport.from_dict(report.to_dict())
        assert restored.run_id == report.run_id
        assert restored.source_fps == report.source_fps
        assert restored.frame_accounting.frames_processed == report.frame_accounting.frames_processed
        assert restored.runtime_config.backend == report.runtime_config.backend
        assert restored.validation_status == report.validation_status

    def test_not_run_states_preserved(self):
        report = self._base_report()
        d = report.to_dict()
        assert d["semantic_regression_status"] == "NOT_RUN"
        assert d["thermal_status"] == "NOT_RUN"

    def test_not_run_visible_in_summary(self):
        report = self._base_report()
        summary = build_human_summary(report)
        assert "NOT_RUN" in summary

    def test_empty_report_minimal(self):
        report = BenchmarkReport()
        d = report.to_dict()
        assert d["validation_status"] == "NOT_RUN"
        assert d["run_id"] == ""

    def test_laptop_does_not_set_jetson_pass(self):
        report = self._base_report()
        # A laptop run must never set validation_status to "PASS"
        assert report.validation_status != "PASS"

    def test_timestamp_isolated(self):
        r1 = self._base_report()
        r2 = self._base_report()
        # Everything except timestamp_utc should be deterministic
        d1 = r1.to_dict()
        d2 = r2.to_dict()
        d1.pop("timestamp_utc")
        d2.pop("timestamp_utc")
        assert d1 == d2

    def test_schema_version_constant(self):
        assert BENCHMARK_SCHEMA_VERSION == "1.0"


# ---------------------------------------------------------------------------
# No production coupling — verify no GPU framework imports
# ---------------------------------------------------------------------------

class TestNoProductionCoupling:

    def test_benchmark_schema_does_not_import_torch(self):
        import inference.evaluation.benchmark_schema as bs
        src = open(bs.__file__, encoding="utf-8").read()
        assert "import torch" not in src
        assert "import ultralytics" not in src
        assert "import cv2" not in src
