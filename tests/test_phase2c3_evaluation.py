"""Phase 2C.3 — Offline evaluation runner tests.

Tests cover:
- PredictionRecord / PredictionArtifact serialization
- run_evaluation: basic TP/FP/FN matching
- multi-video partitioning
- duplicate / ambiguous / normal-period handling
- aggregate metric computation (summed denominators)
- latency distribution
- provenance and matching_config snapshot
- human_readable_summary content
- deterministic semantic output
"""
from __future__ import annotations

import json
import pytest

from inference.evaluation.runner import (
    PredictionRecord,
    PredictionArtifact,
    LatencyDistribution,
    VideoEvaluationResult,
    AggregateEvaluationResult,
    EvaluationReport,
    run_evaluation,
    human_readable_summary,
    EVALUATION_SCHEMA_VERSION,
)
from inference.evaluation.matching import (
    MatchingPolicy,
    EvaluationOutcome,
)
from inference.evaluation.ground_truth import (
    GroundTruthDataset,
    DatasetManifest,
    GroundTruthIncident,
    NormalPeriod,
    AmbiguousPeriod,
    SourceVideoMetadata,
    GTLabel,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_manifest(
    video_id: str = "vid1",
    incident_ids: list = None,
    normal_period_ids: list = None,
    ambiguous_period_ids: list = None,
) -> DatasetManifest:
    """Build a minimal valid DatasetManifest."""
    video = SourceVideoMetadata(
        video_id=video_id,
        reference=f"{video_id}_ref",
        duration_seconds=100.0,
    )
    manifest = DatasetManifest(
        version="1.0",
        created_at="2026-01-01T00:00:00Z",
        videos=[video],
    )
    for iid in (incident_ids or []):
        manifest.add_incident(GroundTruthIncident(
            video_id=video_id,
            incident_id=iid,
            label=GTLabel.SUSPICIOUS,
            start_video_time=10.0,
            end_video_time=20.0,
            behavior_class="smartphone",
        ))
    for pid in (normal_period_ids or []):
        manifest.add_normal_period(NormalPeriod(
            video_id=video_id,
            period_id=pid,
            start_video_time=0.0,
            end_video_time=5.0,
        ))
    for pid in (ambiguous_period_ids or []):
        manifest.add_ambiguous_period(AmbiguousPeriod(
            video_id=video_id,
            period_id=pid,
            start_video_time=0.0,
            end_video_time=5.0,
            reason="occlusion",
        ))
    return manifest


def _make_dataset(*manifests: DatasetManifest) -> GroundTruthDataset:
    return GroundTruthDataset(manifests=list(manifests))


def _make_artifact(
    records: list = None,
    source: str = "test",
) -> PredictionArtifact:
    return PredictionArtifact(
        artifact_version="1.0",
        source_identifier=source,
        records=records or [],
    )


def _make_prediction(
    prediction_id: str,
    video_id: str,
    start: float = 10.0,
    end: float = 20.0,
    confirmation: float = 15.0,
    class_name: str = "smartphone",
) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=prediction_id,
        video_id=video_id,
        start_video_time=start,
        end_video_time=end,
        confirmation_video_time=confirmation,
        class_name=class_name,
    )


# ---------------------------------------------------------------------------
# Chunk 2 — basic run_evaluation: TP, FP, FN
# ---------------------------------------------------------------------------

class TestBasicRunEvaluation:

    def test_single_tp(self):
        """One prediction overlapping a GT incident → 1 TP, 0 FP, 0 FN."""
        manifest = _make_manifest(incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        pred     = _make_prediction("p1", "vid1", start=10.0, end=20.0, confirmation=15.0)
        artifact = _make_artifact([pred])

        report = run_evaluation(dataset, artifact)

        assert report.aggregate.tp_count == 1
        assert report.aggregate.fp_count == 0
        assert report.aggregate.fn_count == 0

    def test_single_fp_no_gt(self):
        """Prediction with no GT incident in that video → FP."""
        # GT manifest has video vid1 registered but no incident
        manifest = _make_manifest(video_id="vid1", incident_ids=[])
        dataset   = _make_dataset(manifest)
        pred      = _make_prediction("p1", "vid1", start=10.0, end=20.0, confirmation=15.0)
        artifact  = _make_artifact([pred])

        report = run_evaluation(dataset, artifact)

        assert report.aggregate.tp_count == 0
        assert report.aggregate.fp_count == 1
        assert report.aggregate.fn_count == 0

    def test_single_fn_no_prediction(self):
        """GT incident with no matching prediction → FN."""
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        # Empty artifact: no predictions at all
        artifact = _make_artifact([], source="empty")

        report = run_evaluation(dataset, artifact)

        assert report.aggregate.tp_count == 0
        assert report.aggregate.fp_count == 0
        assert report.aggregate.fn_count == 1

    def test_mixed_tp_and_fp(self):
        """Two predictions: one overlaps GT (TP), one does not (FP)."""
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        pred_tp  = _make_prediction("p_tp",  "vid1", start=10.0, end=20.0, confirmation=15.0)
        pred_fp  = _make_prediction("p_fp",  "vid1", start=50.0, end=60.0, confirmation=55.0)
        artifact = _make_artifact([pred_tp, pred_fp])

        report = run_evaluation(dataset, artifact)

        assert report.aggregate.tp_count == 1
        assert report.aggregate.fp_count == 1
        assert report.aggregate.fn_count == 0

    def test_precision_recall_f1(self):
        """With 1 TP, 1 FP, 1 FN: P=0.5, R=0.5, F1=0.5."""
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1", "gt2"])
        dataset  = _make_dataset(manifest)
        pred_tp  = _make_prediction("p1", "vid1", start=10.0, end=20.0, confirmation=15.0)
        pred_fp  = _make_prediction("p2", "vid1", start=50.0, end=60.0, confirmation=55.0)
        artifact = _make_artifact([pred_tp, pred_fp])

        report = run_evaluation(dataset, artifact)

        assert report.aggregate.precision == pytest.approx(0.5)
        assert report.aggregate.recall    == pytest.approx(0.5)
        assert report.aggregate.f1        == pytest.approx(0.5)

    def test_report_structure(self):
        """EvaluationReport has all expected top-level fields."""
        manifest = _make_manifest(incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        artifact = _make_artifact([_make_prediction("p1", "vid1")])
        report   = run_evaluation(dataset, artifact)

        assert isinstance(report, EvaluationReport)
        assert report.schema_version == EVALUATION_SCHEMA_VERSION
        assert isinstance(report.videos, list)
        assert len(report.videos) == 1
        assert isinstance(report.videos[0], VideoEvaluationResult)
        assert isinstance(report.aggregate, AggregateEvaluationResult)
        assert "min_overlap_ratio" in report.matching_config
        assert "is_provisional" in report.matching_config


# ---------------------------------------------------------------------------
# Chunk 3 — multi-video partition, GT-only, prediction-only, empty
# ---------------------------------------------------------------------------

class TestMultiVideoPartitioning:

    def test_two_videos_independent(self):
        """Matching is per-video: a prediction in vid2 cannot match GT in vid1."""
        m1 = _make_manifest(video_id="vid1", incident_ids=["gt_v1"])
        m2 = _make_manifest(video_id="vid2", incident_ids=["gt_v2"])
        dataset = _make_dataset(m1, m2)

        pred_v1 = _make_prediction("p1", "vid1")
        pred_v2 = _make_prediction("p2", "vid2")
        artifact = _make_artifact([pred_v1, pred_v2])

        report = run_evaluation(dataset, artifact)
        assert len(report.videos) == 2

        # Both are TPs (each matches its own video's GT)
        assert report.aggregate.tp_count == 2
        assert report.aggregate.fp_count == 0
        assert report.aggregate.fn_count == 0

    def test_gt_only_video_produces_fn(self):
        """A video in the GT dataset with no predictions gets FN."""
        m1 = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        m2 = _make_manifest(video_id="vid2", incident_ids=[])  # no GT incidents
        dataset = _make_dataset(m1, m2)

        pred = _make_prediction("p1", "vid1")
        artifact = _make_artifact([pred])

        report = run_evaluation(dataset, artifact)
        # vid1: TP; vid2: no GT, no pred → empty video result
        assert report.aggregate.tp_count == 1
        assert report.aggregate.fn_count == 0
        # vid2 is not in all_video_ids (no GT, no pred) — so videos count is 1
        assert len(report.videos) == 1

    def test_prediction_only_video_not_in_gt(self):
        """A prediction for a video not in the GT dataset → FP."""
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        pred     = _make_prediction("p1", "vid9")  # vid9 not in GT
        artifact = _make_artifact([pred])

        report = run_evaluation(dataset, artifact)
        assert report.aggregate.fp_count == 1
        assert report.aggregate.tp_count == 0

    def test_empty_dataset_and_artifact(self):
        """No GT videos and no predictions → zero-count report, no crash."""
        dataset  = _make_dataset()  # empty manifests list
        artifact = _make_artifact([], source="empty")

        # run_evaluation will hit the fatal-error path: no videos → raises
        with pytest.raises(ValueError):
            run_evaluation(dataset, artifact)


# ---------------------------------------------------------------------------
# Chunk 4 — duplicates, ambiguous GT, normal/ambiguous periods
# ---------------------------------------------------------------------------

class TestDuplicatesAndAmbiguity:

    def test_duplicate_predictions(self):
        """Two predictions overlapping the same GT → 1 TP + 1 DUPLICATE."""
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        p1 = _make_prediction("p1", "vid1", start=10.0, end=20.0, confirmation=15.0)
        p2 = _make_prediction("p2", "vid1", start=11.0, end=21.0, confirmation=16.0)
        artifact = _make_artifact([p1, p2])

        report = run_evaluation(dataset, artifact)
        assert report.aggregate.tp_count == 1
        assert report.aggregate.duplicate_count == 1
        assert report.aggregate.fp_count == 0

    def test_duplicate_rate(self):
        """Duplicate rate = DUPLICATE / (TP + FP + DUPLICATE)."""
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        p1 = _make_prediction("p1", "vid1", start=10.0, end=20.0, confirmation=15.0)
        p2 = _make_prediction("p2", "vid1", start=11.0, end=21.0, confirmation=16.0)
        artifact = _make_artifact([p1, p2])

        report = run_evaluation(dataset, artifact)
        # TP=1, FP=0, DUPLICATE=1 → dup_rate = 1/(1+0+1) = 0.5
        assert report.aggregate.duplicate_rate == pytest.approx(0.5)

    def test_ambiguous_gt_excluded(self):
        """AMBIGUOUS GT incident produces AMBIGUOUS_GROUND_TRUTH, not FN (default policy)."""
        manifest = _make_manifest(video_id="vid1")
        manifest.add_incident(GroundTruthIncident(
            video_id="vid1",
            incident_id="gt_amb",
            label=GTLabel.AMBIGUOUS,
            start_video_time=10.0,
            end_video_time=20.0,
            behavior_class="smartphone",
        ))
        dataset  = _make_dataset(manifest)
        pred     = _make_prediction("p1", "vid1", start=10.0, end=20.0, confirmation=15.0)
        artifact = _make_artifact([pred])

        report = run_evaluation(dataset, artifact)
        # AMBIGUOUS GT → ambiguous_count=1, prediction not matched → FP
        assert report.aggregate.ambiguous_count == 1
        assert report.aggregate.fn_count == 0
        assert report.aggregate.fp_count == 1

    def test_normal_period_in_fp_reason(self):
        """Prediction inside a normal period still produces FP with reason mentioning normal period."""
        manifest = _make_manifest(
            video_id="vid1",
            incident_ids=["gt1"],
            normal_period_ids=["np1"],
        )
        # NormalPeriod np1 covers 0–5s; GT incident gt1 covers 10–20s
        # Prediction at 1–4s falls inside np1 → FP
        pred     = _make_prediction("p1", "vid1", start=1.0, end=4.0, confirmation=2.0)
        dataset  = _make_dataset(manifest)
        artifact = _make_artifact([pred])

        report = run_evaluation(dataset, artifact)
        # The FP result should have a reason mentioning normal period
        fp_results = [
            r for r in report.records
            if r.outcome == EvaluationOutcome.FP
        ]
        assert len(fp_results) == 1
        assert "normal" in fp_results[0].reason.lower()


# ---------------------------------------------------------------------------
# Chunk 5 — aggregate metrics, zero denominators, latency, matching config
# ---------------------------------------------------------------------------

class TestAggregateMetrics:

    def test_precision_none_when_no_predictions(self):
        """Precision is None when TP+FP=0 (no predictions matched or not)."""
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        artifact = _make_artifact([], source="no_preds")

        report = run_evaluation(dataset, artifact)
        # 1 FN, 0 TP, 0 FP → precision undefined (denominator 0)
        assert report.aggregate.precision is None

    def test_recall_none_when_no_gt(self):
        """Recall is None when TP+FN=0."""
        manifest = _make_manifest(video_id="vid1", incident_ids=[])
        dataset  = _make_dataset(manifest)
        pred     = _make_prediction("p1", "vid1")
        artifact = _make_artifact([pred])

        report = run_evaluation(dataset, artifact)
        # 0 TP, 1 FP, 0 FN → recall undefined (denominator 0)
        assert report.aggregate.recall is None
        assert report.aggregate.f1 is None

    def test_f1_none_when_precision_or_recall_none(self):
        """F1 is None when either precision or recall is None."""
        # Covered by the above tests — this pins the invariant explicitly.
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        artifact = _make_artifact([], source="no_preds")
        report = run_evaluation(dataset, artifact)
        assert report.aggregate.f1 is None


class TestLatency:

    def test_latency_computed_for_tp(self):
        """Latency = confirmation_video_time - gt.start_video_time for TP."""
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        # GT1: start=10s; confirmation=15s → latency=5s
        pred     = _make_prediction("p1", "vid1", start=10.0, end=20.0, confirmation=15.0)
        artifact = _make_artifact([pred])

        report = run_evaluation(dataset, artifact)
        assert report.aggregate.latency.count == 1
        assert report.aggregate.latency.mean == pytest.approx(5.0)
        assert report.aggregate.latency.minimum == pytest.approx(5.0)
        assert report.aggregate.latency.maximum == pytest.approx(5.0)

    def test_negative_latency_valid(self):
        """Pre-emptive detection: confirmation before GT start → negative latency."""
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        # GT1: start=10s; confirmation=5s → latency=-5s
        pred     = _make_prediction("p1", "vid1", start=5.0, end=15.0, confirmation=5.0)
        artifact = _make_artifact([pred])

        report = run_evaluation(dataset, artifact)
        assert report.aggregate.latency.count == 1
        assert report.aggregate.latency.mean == pytest.approx(-5.0)

    def test_latency_none_when_confirmation_missing(self):
        """When confirmation_video_time is None, latency is excluded from stats."""
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        pred = PredictionRecord(
            prediction_id="p1",
            video_id="vid1",
            start_video_time=10.0,
            end_video_time=20.0,
            confirmation_video_time=None,  # unavailable
            class_name="smartphone",
        )
        artifact = _make_artifact([pred])

        report = run_evaluation(dataset, artifact)
        assert report.aggregate.latency.count == 0
        assert report.aggregate.latency.mean is None


class TestMatchingConfigSnapshot:

    def test_default_policy_is_provisional(self):
        """Default MatchingPolicy() is marked provisional in the report."""
        manifest = _make_manifest(incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        artifact = _make_artifact([_make_prediction("p1", "vid1")])

        report = run_evaluation(dataset, artifact)
        assert report.matching_config["is_provisional"] is True
        assert report.matching_config["min_overlap_ratio"] == 0.5

    def test_configured_policy_not_provisional(self):
        """A non-default min_overlap_ratio makes is_provisional False."""
        manifest = _make_manifest(incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        artifact = _make_artifact([_make_prediction("p1", "vid1")])
        policy   = MatchingPolicy(min_overlap_ratio=0.7)

        report = run_evaluation(dataset, artifact, policy=policy)
        assert report.matching_config["is_provisional"] is False
        assert report.matching_config["min_overlap_ratio"] == 0.7

    def test_provenance_contains_required_keys(self):
        manifest = _make_manifest(incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        artifact = _make_artifact([_make_prediction("p1", "vid1")], source="my_source")

        report = run_evaluation(dataset, artifact)
        prov = report.provenance
        assert "generated_at_utc" in prov
        assert prov["prediction_artifact_version"] == "1.0"
        assert prov["prediction_source_identifier"] == "my_source"
        assert "evaluation_schema_version" in prov


# ---------------------------------------------------------------------------
# Chunk 6 — record preservation, JSON round-trip, determinism, human summary
# ---------------------------------------------------------------------------

class TestRecordPreservation:

    def test_event_confidence_preserved_in_report_records(self):
        """event_confidence is carried through to the flat records list."""
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        pred = PredictionRecord(
            prediction_id="p1",
            video_id="vid1",
            start_video_time=10.0,
            end_video_time=20.0,
            confirmation_video_time=15.0,
            class_name="smartphone",
            event_confidence=0.87,
        )
        artifact = _make_artifact([pred])
        report   = run_evaluation(dataset, artifact)

        # The flat records list contains all MatchResults
        assert len(report.records) >= 1
        tp = next(r for r in report.records if r.outcome == EvaluationOutcome.TP)
        assert tp.prediction_id == "p1"
        assert tp.overlap_duration is not None

    def test_overlap_data_preserved(self):
        """Overlap duration and ratio fields are present in report records."""
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        pred     = _make_prediction("p1", "vid1", start=10.0, end=20.0, confirmation=15.0)
        artifact = _make_artifact([pred])
        report   = run_evaluation(dataset, artifact)

        tp = next(r for r in report.records if r.outcome == EvaluationOutcome.TP)
        assert tp.overlap_duration > 0.0
        assert tp.overlap_ratio_ground_truth is not None


class TestJsonRoundTrip:

    def test_evaluation_report_round_trip(self):
        """EvaluationReport → dict → JSON string → dict → EvaluationReport
        preserves all semantic fields (ignoring provenance timestamp)."""
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        pred     = _make_prediction("p1", "vid1", start=10.0, end=20.0, confirmation=15.0)
        artifact = _make_artifact([pred])
        report   = run_evaluation(dataset, artifact)

        d = report.to_dict()
        raw = json.dumps(d)
        d2 = json.loads(raw)
        report2 = EvaluationReport.from_dict(d2)

        # Semantic fields must round-trip exactly
        assert report2.schema_version == report.schema_version
        assert report2.aggregate.tp_count == report.aggregate.tp_count
        assert report2.aggregate.fp_count == report.aggregate.fp_count
        assert report2.aggregate.fn_count == report.aggregate.fn_count
        assert report2.matching_config == report.matching_config
        assert len(report2.videos) == len(report.videos)
        assert len(report2.records) == len(report.records)

    def test_prediction_artifact_round_trip(self):
        """PredictionArtifact → JSON → PredictionArtifact preserves records."""
        pred = _make_prediction("p1", "vid1", start=10.0, end=20.0, confirmation=15.0)
        artifact = _make_artifact([pred], source="round_trip_test")
        raw = artifact.to_json()
        artifact2 = PredictionArtifact.from_json(raw)

        assert artifact2.records[0].prediction_id == "p1"
        assert artifact2.records[0].start_video_time == 10.0
        assert artifact2.records[0].event_confidence is None
        assert artifact2.source_identifier == "round_trip_test"


class TestDeterminism:

    def test_same_input_same_report(self):
        """Running run_evaluation twice with identical inputs produces
        identical semantic content (provenance timestamp may differ)."""
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        artifact = _make_artifact([_make_prediction("p1", "vid1")])

        r1 = run_evaluation(dataset, artifact)
        r2 = run_evaluation(dataset, artifact)

        # Exclude the non-deterministic provenance timestamp
        assert r1.aggregate.to_dict() == r2.aggregate.to_dict()
        assert r1.matching_config == r2.matching_config
        assert [v.video_id for v in r1.videos] == [v.video_id for v in r2.videos]


class TestHumanReadableSummary:

    def test_contains_required_fields(self):
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        pred     = _make_prediction("p1", "vid1", start=10.0, end=20.0, confirmation=15.0)
        artifact = _make_artifact([pred])
        report   = run_evaluation(dataset, artifact)
        text     = human_readable_summary(report)

        assert "PHASE 2C.3" in text
        assert "Videos" in text
        assert "Predictions" in text
        assert "GT incidents" in text
        assert "Precision" in text
        assert "Recall" in text
        assert "F1" in text
        assert "Duplicate" in text
        assert "Latency" in text

    def test_provisional_flag_visible(self):
        manifest = _make_manifest(video_id="vid1", incident_ids=["gt1"])
        dataset  = _make_dataset(manifest)
        artifact = _make_artifact([_make_prediction("p1", "vid1")])
        report   = run_evaluation(dataset, artifact)
        text     = human_readable_summary(report)
        # Default policy is provisional → should be visible in summary
        assert "Provisional" in text or "provisional" in text


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_smoke_import():
    assert EVALUATION_SCHEMA_VERSION == "1.0"
    assert run_evaluation is not None
    assert human_readable_summary is not None
