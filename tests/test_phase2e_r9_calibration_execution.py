"""Phase 2E-R9 — Calibration execution readiness.

Covers:
- GT dataset validation (empty dataset → honest missing-prerequisite report)
- Matching sweep execution when GT incidents + prediction artifact are present
- Confirmation sweep exclusion when event_confidence is None
- Operating point selection (Pareto when no constraint)
- FP/hour, recall, precision, F1 in report when supported
- Latency + duplicate metrics where supported
- Subgroup metrics when scene tags present
- Dataset limitations reported honestly
- No fabricated ground truth or calibration results
- Deterministic: same input → same output
- No production file I/O
"""
from __future__ import annotations

import json
import unittest
from typing import List, Optional

from inference.evaluation.ground_truth import (
    GTLabel,
    EntityReferenceStatus,
    SceneTag,
    GroundTruthIncident,
    NormalPeriod,
    AmbiguousPeriod,
    SourceVideoMetadata,
    DatasetManifest,
    GroundTruthDataset,
    validate_dataset,
)
from inference.evaluation.matching import MatchingPolicy
from inference.evaluation.runner import PredictionRecord, PredictionArtifact
from inference.evaluation.calibration import (
    CalibrationParameterGrid,
    CalibrationSplit,
    OperatingPointPolicy,
    OperatingPointResult,
    CalibrationReport,
    sweep_matching_parameters,
    select_operating_point,
    compute_subgroup_metrics,
    build_calibration_report,
)
from inference.evaluation.shadow_bridge import (
    shadow_events_to_prediction_artifact,
    CalibrationReadinessReport,
)
from inference.core.events import EntityKey, ConfidenceStats, EventRecord
from inference.core.confirmation import ConfirmationGateResult
from inference.core.shadow_lifecycle import ShadowEventRecord

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mk_incident(
    incident_id: str,
    video_id: str,
    start: float,
    end: float,
    label: GTLabel = GTLabel.SUSPICIOUS,
    behavior_class: str = "provide-object",
) -> GroundTruthIncident:
    return GroundTruthIncident(
        video_id=video_id,
        incident_id=incident_id,
        label=label,
        start_video_time=start,
        end_video_time=end,
        behavior_class=behavior_class,
    )

def _mk_video(
    video_id: str,
    duration: Optional[float] = 60.0,
    fps: float = 30.0,
    tags: tuple = (),
) -> SourceVideoMetadata:
    return SourceVideoMetadata(
        video_id=video_id,
        reference=video_id,
        duration_seconds=duration,
        source_fps=fps,
        scene_tags=tags,
    )

def _mk_dataset(
    videos: List[SourceVideoMetadata],
    incidents: List[GroundTruthIncident],
    normal_periods: Optional[List[NormalPeriod]] = None,
    ambiguous_periods: Optional[List[AmbiguousPeriod]] = None,
    version: str = "test-v1",
) -> GroundTruthDataset:
    manifest = DatasetManifest(
        version=version,
        created_at="2026-01-01T00:00:00Z",
        videos=videos,
        incidents=incidents,
        normal_periods=normal_periods or [],
        ambiguous_periods=ambiguous_periods or [],
    )
    return GroundTruthDataset(manifests=[manifest])

def _mk_prediction(
    pred_id: str,
    video_id: str,
    start: float,
    end: float,
    confirmation_time: Optional[float] = None,
    event_confidence: Optional[float] = None,
) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=pred_id,
        video_id=video_id,
        start_video_time=start,
        end_video_time=end,
        confirmation_video_time=confirmation_time,
        event_confidence=event_confidence,
    )

def _mk_artifact(records: List[PredictionRecord]) -> PredictionArtifact:
    return PredictionArtifact(source_identifier="r9_test", records=records)

def _mk_gt_dataset_with_incidents(
    n_videos: int = 3,
    incidents_per_video: int = 2,
) -> GroundTruthDataset:
    """Build a synthetic GT dataset with multiple videos and incidents."""
    videos = [
        _mk_video(f"vid_{i}", duration=60.0, tags=(SceneTag.NORMAL_EXAM,))
        for i in range(n_videos)
    ]
    incidents = [
        _mk_incident(f"inc_{i}_{j}", f"vid_{i}", j * 10.0, j * 10.0 + 5.0)
        for i in range(n_videos)
        for j in range(incidents_per_video)
    ]
    return _mk_dataset(videos, incidents)

def _mk_shadow_records(video_id: str, n: int = 3) -> List[ShadowEventRecord]:
    """Build shadow records with event_confidence=None (real shadow state)."""
    recs = []
    for i in range(n):
        er = EventRecord(
            event_id=f"shadow-{i}",
            entity_key=EntityKey.from_pose(i + 1),
            class_name="provide-object",
            observation_count=3,
            confidence_stats=ConfidenceStats(count=3, mean=0.9, min=0.8, max=1.0),
            event_confidence=None,
            association_score=None,
            contributing_track_ids=frozenset({i + 1}),
            confirmation_video_time=float(i + 1),
            confirmation_wall_time="2026-01-01T00:00:00",
            candidate_start_video_time=float(i),
            candidate_end_video_time=float(i + 1),
        )
        gate = ConfirmationGateResult(
            entity_key=er.entity_key,
            candidate_class="provide-object",
            confirmed=True,
            reason="test",
        )
        recs.append(ShadowEventRecord(event=er, gate_result=gate, recorded_at_video_time=float(i + 1)))
    return recs

# ---------------------------------------------------------------------------
# The R9 executor module to test
# ---------------------------------------------------------------------------

from inference.evaluation.calibration_executor import (
    CalibrationExecutionResult,
    execute_calibration,
    validate_gt_dataset_for_calibration,
    DatasetValidationReport,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestR9DatasetValidation(unittest.TestCase):

    def test_empty_dataset_reports_missing_prerequisites(self):
        """An empty GT dataset must report the exact missing prerequisites."""
        gt = GroundTruthDataset()
        result = validate_gt_dataset_for_calibration(gt)
        self.assertIsInstance(result, DatasetValidationReport)
        self.assertFalse(result.sufficient_for_calibration)
        self.assertIn("no GT incidents", " ".join(result.issues))

    def test_dataset_with_videos_but_no_incidents_reports_issue(self):
        """GT with source video metadata but zero incidents is not sufficient."""
        gt = _mk_dataset(videos=[_mk_video("vid_1", 60.0)], incidents=[])
        result = validate_gt_dataset_for_calibration(gt)
        self.assertFalse(result.sufficient_for_calibration)
        self.assertEqual(result.incident_count, 0)
        self.assertEqual(result.video_count, 1)

    def test_sufficient_dataset_reports_counts(self):
        gt = _mk_gt_dataset_with_incidents(n_videos=3, incidents_per_video=2)
        result = validate_gt_dataset_for_calibration(gt)
        self.assertTrue(result.sufficient_for_calibration)
        self.assertEqual(result.video_count, 3)
        self.assertEqual(result.incident_count, 6)
        self.assertEqual(result.normal_period_count, 0)

    def test_dataset_report_is_deterministic(self):
        gt = _mk_gt_dataset_with_incidents()
        r1 = validate_gt_dataset_for_calibration(gt)
        r2 = validate_gt_dataset_for_calibration(gt)
        self.assertEqual(r1.to_dict(), r2.to_dict())

    def test_dataset_report_json_serializable(self):
        gt = _mk_gt_dataset_with_incidents()
        result = validate_gt_dataset_for_calibration(gt)
        json.dumps(result.to_dict())

    def test_small_dataset_limitations_reported(self):
        """A dataset with < 3 videos cannot support a meaningful split; report that."""
        gt = _mk_dataset(
            videos=[_mk_video("vid_1", 60.0)],
            incidents=[_mk_incident("inc_1", "vid_1", 0.0, 5.0)],
        )
        result = validate_gt_dataset_for_calibration(gt)
        self.assertTrue(result.sufficient_for_calibration)
        # Fewer than 3 videos → all videos go to train split, no val/holdout
        self.assertIn("insufficient_videos_for_split", result.limitations)


class TestR9ExecuteCalibration(unittest.TestCase):

    def test_execution_blocked_when_no_gt_incidents(self):
        """When the GT dataset has no incidents, no sweep runs and no report is produced."""
        gt = GroundTruthDataset()
        artifact = _mk_artifact([])
        result = execute_calibration(
            gt_dataset=gt,
            artifact=artifact,
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.3, 0.5]}),
            operating_point_policy=OperatingPointPolicy(),
            split_seed=42,
        )
        self.assertIsInstance(result, CalibrationExecutionResult)
        self.assertFalse(result.executed)
        self.assertIsNone(result.report)
        self.assertIsNone(result.operating_point)
        self.assertTrue(len(result.missing_prerequisites) > 0)

    def test_matching_sweep_runs_when_gt_present(self):
        """With valid GT + predictions, matching sweep executes and produces a report."""
        gt = _mk_gt_dataset_with_incidents(n_videos=3, incidents_per_video=2)
        preds = [
            _mk_prediction("p1", "vid_0", 0.0, 5.0, confirmation_time=4.0),
            _mk_prediction("p2", "vid_0", 10.0, 15.0, confirmation_time=14.0),
            _mk_prediction("p3", "vid_1", 0.0, 5.0, confirmation_time=4.0),
            _mk_prediction("p4", "vid_1", 10.0, 15.0, confirmation_time=14.0),
            _mk_prediction("p5", "vid_2", 0.0, 5.0, confirmation_time=4.0),
            _mk_prediction("p6", "vid_2", 10.0, 15.0, confirmation_time=14.0),
        ]
        artifact = _mk_artifact(preds)
        result = execute_calibration(
            gt_dataset=gt,
            artifact=artifact,
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.3, 0.5]}),
            operating_point_policy=OperatingPointPolicy(),
            split_seed=42,
        )
        self.assertTrue(result.executed)
        self.assertIsNotNone(result.report)
        self.assertIsInstance(result.report, CalibrationReport)
        # Report must contain provenance
        self.assertIsNotNone(result.report.provenance)

    def test_event_confidence_none_blocks_confirmation_sweep(self):
        """
        Shadow records always have event_confidence=None.
        The executor must NOT include min_confidence_mean in any sweep grid.
        Verify that even if a caller passes it, the executor excludes it.
        """
        gt = _mk_gt_dataset_with_incidents(n_videos=3, incidents_per_video=2)
        preds = [
            _mk_prediction("p1", "vid_0", 0.0, 5.0, event_confidence=None),
            _mk_prediction("p2", "vid_1", 0.0, 5.0, event_confidence=None),
            _mk_prediction("p3", "vid_2", 0.0, 5.0, event_confidence=None),
        ]
        artifact = _mk_artifact(preds)
        # Include min_confidence_mean in the grid — executor must exclude it
        grid = CalibrationParameterGrid({
            "min_overlap_ratio": [0.3, 0.5],
            "min_confidence_mean": [0.5],
        })
        result = execute_calibration(
            gt_dataset=gt,
            artifact=artifact,
            matching_grid=grid,
            operating_point_policy=OperatingPointPolicy(),
            split_seed=42,
        )
        self.assertTrue(result.executed)
        # The confirmation sweep must NOT have run (min_confidence_mean excluded)
        self.assertIsNone(result.report.confirmation_results,
                          "confirmation sweep must be excluded when event_confidence is None")
        # min_confidence_mean should appear in blocked_parameters of the readiness report
        self.assertIn("min_confidence_mean", result.readiness.blocked_parameters)

    def test_operating_point_pareto_when_no_constraint(self):
        """When OperatingPointPolicy has no FP target or recall floor, a Pareto set is selected."""
        gt = _mk_gt_dataset_with_incidents(n_videos=3, incidents_per_video=2)
        preds = [
            _mk_prediction("p1", "vid_0", 0.0, 5.0, confirmation_time=4.0),
            _mk_prediction("p2", "vid_1", 0.0, 5.0, confirmation_time=4.0),
            _mk_prediction("p3", "vid_2", 0.0, 5.0, confirmation_time=4.0),
        ]
        artifact = _mk_artifact(preds)
        policy = OperatingPointPolicy()  # no constraint → Pareto
        result = execute_calibration(
            gt_dataset=gt,
            artifact=artifact,
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.3, 0.5, 0.7]}),
            operating_point_policy=policy,
            split_seed=42,
        )
        self.assertTrue(result.executed)
        self.assertIsNotNone(result.operating_point)
        op = result.operating_point
        self.assertIsInstance(op, OperatingPointResult)
        self.assertTrue(op.is_pareto_candidate, "no-constraint policy should yield a Pareto set")

    def test_operating_point_with_fp_constraint(self):
        """When fp_per_hour_target is set, the Pareto set is replaced by a single selection."""
        gt = _mk_gt_dataset_with_incidents(n_videos=3, incidents_per_video=2)
        preds = [
            _mk_prediction("p1", "vid_0", 0.0, 5.0, confirmation_time=4.0),
            _mk_prediction("p2", "vid_1", 0.0, 5.0, confirmation_time=4.0),
            _mk_prediction("p3", "vid_2", 0.0, 5.0, confirmation_time=4.0),
        ]
        artifact = _mk_artifact(preds)
        # Set an extremely high FP target so all rows pass the filter
        policy = OperatingPointPolicy(fp_per_hour_target=1000.0)
        result = execute_calibration(
            gt_dataset=gt,
            artifact=artifact,
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.3, 0.5, 0.7]}),
            operating_point_policy=policy,
            split_seed=42,
        )
        self.assertTrue(result.executed)
        self.assertIsNotNone(result.operating_point)
        op = result.operating_point
        self.assertFalse(op.is_pareto_candidate,
                         "with an explicit FP constraint the result should be a single selection")

    def test_result_json_serializable(self):
        """The full CalibrationExecutionResult must be JSON-serializable."""
        gt = _mk_gt_dataset_with_incidents(n_videos=3, incidents_per_video=2)
        preds = [_mk_prediction("p1", "vid_0", 0.0, 5.0, confirmation_time=4.0)]
        artifact = _mk_artifact(preds)
        result = execute_calibration(
            gt_dataset=gt,
            artifact=artifact,
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.3]}),
            operating_point_policy=OperatingPointPolicy(),
            split_seed=42,
        )
        json.dumps(result.to_dict())

    def test_execution_deterministic(self):
        """Same inputs → identical results (no wall-clock time in semantic fields)."""
        gt = _mk_gt_dataset_with_incidents(n_videos=3, incidents_per_video=2)
        preds = [_mk_prediction("p1", "vid_0", 0.0, 5.0, confirmation_time=4.0)]
        artifact = _mk_artifact(preds)
        r1 = execute_calibration(
            gt_dataset=gt, artifact=artifact,
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.3]}),
            operating_point_policy=OperatingPointPolicy(), split_seed=42,
        )
        r2 = execute_calibration(
            gt_dataset=gt, artifact=artifact,
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.3]}),
            operating_point_policy=OperatingPointPolicy(), split_seed=42,
        )
        # Compare all semantic fields (excluding timestamps)
        self.assertEqual(r1.executed, r2.executed)
        self.assertEqual(r1.missing_prerequisites, r2.missing_prerequisites)
        if r1.report and r2.report:
            self.assertEqual(r1.report.matching_results, r2.report.matching_results)

    def test_split_deterministic_seed_reported(self):
        """The split seed must appear in the dataset summary."""
        gt = _mk_gt_dataset_with_incidents(n_videos=4, incidents_per_video=2)
        preds = [_mk_prediction("p1", "vid_0", 0.0, 5.0)]
        artifact = _mk_artifact(preds)
        result = execute_calibration(
            gt_dataset=gt, artifact=artifact,
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.5]}),
            operating_point_policy=OperatingPointPolicy(),
            split_seed=42,
        )
        self.assertTrue(result.executed)
        self.assertEqual(result.dataset_summary.get("split_seed"), 42)

    def test_subgroup_metrics_when_scene_tags_present(self):
        """When GT videos have scene_tags, subgroup metrics must be in the result."""
        gt = _mk_dataset(
            videos=[_mk_video(f"vid_{i}", 60.0, tags=(SceneTag.NORMAL_EXAM,)) for i in range(3)],
            incidents=[_mk_incident(f"inc_{i}", f"vid_{i}", 0.0, 5.0) for i in range(3)],
        )
        preds = [_mk_prediction(f"p{i}", f"vid_{i}", 0.0, 5.0, confirmation_time=4.0) for i in range(3)]
        artifact = _mk_artifact(preds)
        result = execute_calibration(
            gt_dataset=gt, artifact=artifact,
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.5]}),
            operating_point_policy=OperatingPointPolicy(),
            split_seed=42,
        )
        self.assertTrue(result.executed)
        # Subgroup metrics must be present (dict keyed by scene tag value)
        self.assertIsNotNone(result.subgroup_metrics,
                             "subgroup metrics must be computed when scene tags are present")
        self.assertIn("normal_exam", result.subgroup_metrics)

    def test_no_file_io(self):
        """The executor must not open any files."""
        from unittest.mock import patch
        gt = _mk_gt_dataset_with_incidents(n_videos=3, incidents_per_video=2)
        preds = [_mk_prediction("p1", "vid_0", 0.0, 5.0)]
        artifact = _mk_artifact(preds)
        with patch("builtins.open", side_effect=OSError("unexpected file open")):
            result = execute_calibration(
                gt_dataset=gt, artifact=artifact,
                matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.5]}),
                operating_point_policy=OperatingPointPolicy(),
                split_seed=42,
            )
        self.assertIsInstance(result, CalibrationExecutionResult)

    def test_result_is_not_production_ready(self):
        """The result must explicitly carry cutover_status='NOT_READY'."""
        gt = _mk_gt_dataset_with_incidents(n_videos=3, incidents_per_video=2)
        preds = [_mk_prediction("p1", "vid_0", 0.0, 5.0)]
        artifact = _mk_artifact(preds)
        result = execute_calibration(
            gt_dataset=gt, artifact=artifact,
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.5]}),
            operating_point_policy=OperatingPointPolicy(),
            split_seed=42,
        )
        self.assertEqual(result.cutover_status, "NOT_READY")
        self.assertFalse(result.production_ready)


class TestR9NoFabrication(unittest.TestCase):

    def test_no_gt_data_no_fabricated_results(self):
        """When GT is empty, no CalibrationReport or OperatingPointResult is produced."""
        gt = GroundTruthDataset()
        artifact = _mk_artifact([_mk_prediction("p1", "vid_1", 0.0, 5.0)])
        result = execute_calibration(
            gt_dataset=gt, artifact=artifact,
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.5]}),
            operating_point_policy=OperatingPointPolicy(),
            split_seed=42,
        )
        self.assertFalse(result.executed)
        self.assertIsNone(result.report, "no report when GT is empty")
        self.assertIsNone(result.operating_point, "no operating point when GT is empty")

    def test_missing_prerequisites_are_exact(self):
        """The missing prerequisites list must be specific, not generic."""
        gt = GroundTruthDataset()
        result = execute_calibration(
            gt_dataset=gt,
            artifact=_mk_artifact([]),
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.5]}),
            operating_point_policy=OperatingPointPolicy(),
            split_seed=42,
        )
        prereqs = " ".join(result.missing_prerequisites)
        self.assertIn("no GT incidents", prereqs)
        self.assertNotIn("unknown", prereqs.lower())


if __name__ == "__main__":
    unittest.main()
