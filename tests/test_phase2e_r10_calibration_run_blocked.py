"""Phase 2E-R10 — Calibration Run and Operating-Point Audit.

R10 is blocked: no human-annotated GroundTruthDataset exists in this repo.
The three data-adjacent files that do exist are all inference output:
- 2026-10-01_detection_log.csv, inference/2026-10-02_detection_log.csv
  → per-crop detection logs (class + confidence + timestamp), no GT schema
- inference/output/benchmark_results.json → model throughput benchmark

None can be loaded via DatasetManifest.from_file (JSONDecodeError /
KeyError: 'version'). R10 therefore documents the BLOCKED state with
exact missing-prerequisite provenance and pins the no-fabrication
behavior so a future session cannot accidentally substitute a synthetic
dataset and report it as a real calibration result.
"""
from __future__ import annotations

import json
import unittest

from inference.evaluation.ground_truth import (
    GroundTruthDataset,
    DatasetManifest,
    GTLabel,
    SourceVideoMetadata,
    GroundTruthIncident,
)
from inference.evaluation.runner import PredictionRecord, PredictionArtifact
from inference.evaluation.calibration import (
    CalibrationParameterGrid,
    OperatingPointPolicy,
)
from inference.evaluation.calibration_executor import (
    CalibrationExecutionResult,
    execute_calibration,
    validate_gt_dataset_for_calibration,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mk_prediction(pred_id: str, video_id: str, conf=None) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=pred_id,
        video_id=video_id,
        start_video_time=0.0,
        end_video_time=5.0,
        event_confidence=conf,
    )


def _mk_artifact(records) -> PredictionArtifact:
    return PredictionArtifact(source_identifier="r10_test", records=records)


# ---------------------------------------------------------------------------
# Blockage pin: real repo files are not loadable as GT manifests
# ---------------------------------------------------------------------------

class TestR10RepoFilesAreNotGT(unittest.TestCase):
    """Each of the three data files in the repo fails DatasetManifest.from_file.

    This is the direct evidence that no real GroundTruthDataset is present.
    """

    def test_detection_log_csv_fails_manifest_load(self):
        """The per-crop CSV detection log cannot be parsed as a DatasetManifest."""
        for path in ("2026-10-01_detection_log.csv", "inference/2026-10-02_detection_log.csv"):
            try:
                DatasetManifest.from_file(path)
                self.fail(f"{path} unexpectedly loaded as DatasetManifest")
            except (json.JSONDecodeError, KeyError, ValueError):
                pass  # expected — CSV is not JSON, benchmark JSON has no 'version' key

    def test_benchmark_json_fails_manifest_load(self):
        """benchmark_results.json is a throughput benchmark, not a GT manifest."""
        path = "inference/output/benchmark_results.json"
        try:
            DatasetManifest.from_file(path)
            self.fail(f"{path} unexpectedly loaded as DatasetManifest")
        except KeyError:
            pass  # expected — no 'version' key


# ---------------------------------------------------------------------------
# No-fabrication: empty GT dataset → honest blockage, no results
# ---------------------------------------------------------------------------

class TestR10NoFabricationWhenNoGT(unittest.TestCase):
    """When no GT dataset exists, the executor must produce no report or
    operating point, and must say so explicitly."""

    def test_empty_gt_yields_executed_false_and_no_report(self):
        gt = GroundTruthDataset()
        preds = [_mk_prediction("p1", "vid_1")]
        artifact = _mk_artifact(preds)
        result = execute_calibration(
            gt_dataset=gt,
            artifact=artifact,
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.3, 0.5, 0.7]}),
            operating_point_policy=OperatingPointPolicy(),
            split_seed=42,
        )
        self.assertFalse(result.executed, "no GT → executed must be False")
        self.assertIsNone(result.report, "no GT → no CalibrationReport")
        self.assertIsNone(result.operating_point, "no GT → no OperatingPointResult")
        self.assertIsNone(result.subgroup_metrics, "no GT → no subgroup metrics")

    def test_empty_gt_missing_prereqs_are_specific(self):
        """The prerequisites list names the exact gaps, not a generic message."""
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
        self.assertIn("no source video metadata", prereqs)
        self.assertNotIn("unknown", prereqs.lower(),
                         "missing prerequisites must be specific, not 'unknown'")

    def test_empty_gt_cutover_status_not_ready(self):
        gt = GroundTruthDataset()
        result = execute_calibration(
            gt_dataset=gt,
            artifact=_mk_artifact([]),
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.5]}),
            operating_point_policy=OperatingPointPolicy(),
            split_seed=42,
        )
        self.assertEqual(result.cutover_status, "NOT_READY")
        self.assertFalse(result.production_ready)

    def test_validation_report_is_empty(self):
        """validate_gt_dataset_for_calibration on an empty dataset reports zero counts."""
        gt = GroundTruthDataset()
        dv = validate_gt_dataset_for_calibration(gt)
        self.assertEqual(dv.video_count, 0)
        self.assertEqual(dv.incident_count, 0)
        self.assertFalse(dv.sufficient_for_calibration)
        self.assertIn("no GT incidents", " ".join(dv.issues))


# ---------------------------------------------------------------------------
# No-fabrication: shadow records with event_confidence=None are flagged
# even in a populated GT scenario — no min_confidence_mean may enter the grid
# ---------------------------------------------------------------------------

class TestR10ShadowEventConfidenceBlocked(unittest.TestCase):
    """Even when a valid GT dataset is supplied, all shadow records have
    event_confidence=None; the executor must strip min_confidence_mean from
    any grid the caller passes and flag it in the readiness report."""

    def test_min_confidence_mean_stripped_from_grid_when_all_none_conf(self):
        gt = GroundTruthDataset(manifests=[
            DatasetManifest(
                version="r10", created_at="2026-01-01T00:00:00Z",
                videos=[SourceVideoMetadata(video_id="v1", reference="v1",
                                            duration_seconds=60.0, source_fps=30.0,
                                            scene_tags=())],
                incidents=[GroundTruthIncident(video_id="v1", incident_id="i1",
                                               label=GTLabel.SUSPICIOUS,
                                               start_video_time=0.0,
                                               end_video_time=5.0,
                                               behavior_class="provide-object")],
                normal_periods=[], ambiguous_periods=[],
            )
        ])
        # All shadow records: event_confidence=None
        preds = [_mk_prediction(f"p{i}", "v1", conf=None) for i in range(3)]
        artifact = _mk_artifact(preds)

        # Caller passes a grid that includes min_confidence_mean — executor must exclude it
        grid = CalibrationParameterGrid({
            "min_overlap_ratio": [0.3, 0.5],
            "min_confidence_mean": [0.5, 0.7],
        })
        result = execute_calibration(
            gt_dataset=gt,
            artifact=artifact,
            matching_grid=grid,
            operating_point_policy=OperatingPointPolicy(),
            split_seed=42,
        )
        self.assertTrue(result.executed, "GT is present so matching sweep should run")
        self.assertIn("min_confidence_mean", result.readiness.blocked_parameters,
                      "min_confidence_mean must be flagged as blocked")
        # The matching sweep must NOT have iterated over min_confidence_mean values.
        # The only valid sweep results will have one row per min_overlap_ratio value,
        # not rows per confidence value.
        if result.report and result.report.matching_results:
            for combo in result.report.matching_results:
                self.assertNotIn("min_confidence_mean", combo,
                                 "min_confidence_mean must not appear in any matching sweep row")


# ---------------------------------------------------------------------------
# Determinism: two identical blocked runs produce identical output
# ---------------------------------------------------------------------------

class TestR10BlockedRunIsDeterministic(unittest.TestCase):

    def test_same_blocked_input_twice_gives_same_dict(self):
        gt = GroundTruthDataset()
        preds = [_mk_prediction("p1", "vid_1")]
        artifact = _mk_artifact(preds)
        kwargs = dict(
            gt_dataset=gt, artifact=artifact,
            matching_grid=CalibrationParameterGrid({"min_overlap_ratio": [0.3, 0.5]}),
            operating_point_policy=OperatingPointPolicy(),
            split_seed=42,
        )
        r1 = execute_calibration(**kwargs)
        r2 = execute_calibration(**kwargs)
        self.assertEqual(r1.to_dict(), r2.to_dict(),
                         "two identical blocked runs must produce identical output")


if __name__ == "__main__":
    unittest.main()
