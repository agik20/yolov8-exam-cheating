"""Phase 2E-R8 — Calibration integration readiness.

Covers:
- Conversion from ShadowEventRecord to PredictionRecord (calibration input)
- event_confidence passes through as None (no fabricated score)
- association_score passes through as None (no fabrication)
- timing fields (start/end/confirmation video time) are preserved exactly
- deterministic conversion (same input → same output, independent of call order)
- no fabricated values in source_metadata
- backward compatibility: ShadowEventRecord with evidence_bundle=None still converts
- CalibrationReadinessReport: which fields are ready, which are blocked
- matching parameters always sweepable; confirmation blocked with None confidence
- unavailable association data honestly reported
- no production file I/O
"""
from __future__ import annotations

import json
import unittest
from typing import List, Optional

from inference.core.events import (
    EntityKey,
    ModelSource,
    Observation,
    ConfidenceStats,
    EventRecord,
)
from inference.core.confirmation import ConfirmationGateResult
from inference.core.shadow_lifecycle import ShadowEventRecord
from inference.evaluation.runner import PredictionRecord, PredictionArtifact
from inference.evaluation.shadow_bridge import (
    shadow_events_to_prediction_records,
    shadow_events_to_prediction_artifact,
    CalibrationReadinessReport,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_shadow_record(
    event_id: str = "uuid-1",
    track_id: int = 1,
    class_name: str = "provide-object",
    start_video_time: float = 0.0,
    end_video_time: float = 0.5,
    confirmation_video_time: float = 0.5,
    obs_count: int = 3,
    conf_mean: float = 0.9,
    event_confidence: Optional[float] = None,
    association_score: Optional[float] = None,
) -> ShadowEventRecord:
    er = EventRecord(
        event_id=event_id,
        entity_key=EntityKey.from_pose(track_id),
        class_name=class_name,
        confirmation_video_time=confirmation_video_time,
        confirmation_wall_time="2026-01-01T00:00:00",
        candidate_start_video_time=start_video_time,
        candidate_end_video_time=end_video_time,
        observation_count=obs_count,
        confidence_stats=ConfidenceStats(
            count=obs_count, mean=conf_mean,
            min=conf_mean - 0.1, max=conf_mean + 0.1,
        ),
        event_confidence=event_confidence,
        association_score=association_score,
        contributing_track_ids=frozenset({track_id}),
    )
    gate = ConfirmationGateResult(
        entity_key=er.entity_key,
        candidate_class=class_name,
        confirmed=True,
        reason="all_gates_satisfied",
        observation_count=obs_count,
        video_time_span=end_video_time - start_video_time,
        confidence_stats=er.confidence_stats,
        track_ids=frozenset({track_id}),
    )
    return ShadowEventRecord(
        event=er,
        gate_result=gate,
        recorded_at_video_time=confirmation_video_time,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestR8ShadowToPredictionAdapter(unittest.TestCase):

    def test_basic_conversion(self):
        rec = _make_shadow_record()
        records = shadow_events_to_prediction_records([rec], video_id="vid-1")
        self.assertEqual(len(records), 1)
        pr = records[0]
        self.assertIsInstance(pr, PredictionRecord)
        self.assertEqual(pr.prediction_id, rec.event.event_id)
        self.assertEqual(pr.video_id, "vid-1")
        self.assertEqual(pr.class_name, "provide-object")
        self.assertEqual(pr.identity, "pose:1")

    def test_event_confidence_none_passes_through(self):
        rec = _make_shadow_record(event_confidence=None)
        records = shadow_events_to_prediction_records([rec], video_id="vid-1")
        self.assertIsNone(records[0].event_confidence)

    def test_event_confidence_finite_passes_through(self):
        rec = _make_shadow_record(event_confidence=0.85)
        records = shadow_events_to_prediction_records([rec], video_id="vid-1")
        self.assertAlmostEqual(records[0].event_confidence, 0.85)

    def test_association_score_none_not_fabricated(self):
        rec = _make_shadow_record(association_score=None)
        records = shadow_events_to_prediction_records([rec], video_id="vid-1")
        sm = records[0].source_metadata or {}
        self.assertIsNone(sm.get("association_score"))

    def test_association_score_populated_preserved(self):
        rec = _make_shadow_record(association_score=0.7)
        records = shadow_events_to_prediction_records([rec], video_id="vid-1")
        sm = records[0].source_metadata or {}
        self.assertAlmostEqual(sm["association_score"], 0.7)

    def test_timing_fields_preserved_exactly(self):
        rec = _make_shadow_record(
            start_video_time=1.25,
            end_video_time=3.75,
            confirmation_video_time=3.75,
        )
        records = shadow_events_to_prediction_records([rec], video_id="vid-2")
        pr = records[0]
        self.assertEqual(pr.start_video_time, 1.25)
        self.assertEqual(pr.end_video_time, 3.75)
        self.assertEqual(pr.confirmation_video_time, 3.75)

    def test_deterministic_conversion(self):
        recs = [_make_shadow_record(event_id=f"uuid-{i}", track_id=i) for i in range(3)]
        r1 = shadow_events_to_prediction_records(recs, video_id="vid-1")
        r2 = shadow_events_to_prediction_records(recs, video_id="vid-1")
        self.assertEqual(len(r1), len(r2))
        for a, b in zip(r1, r2):
            self.assertEqual(a.prediction_id, b.prediction_id)
            self.assertEqual(a.start_video_time, b.start_video_time)
            self.assertEqual(a.end_video_time, b.end_video_time)
            self.assertEqual(a.class_name, b.class_name)

    def test_no_fabricated_values_in_source_metadata(self):
        rec = _make_shadow_record(obs_count=4, conf_mean=0.72)
        records = shadow_events_to_prediction_records([rec], video_id="vid-1")
        sm = records[0].source_metadata
        self.assertIsNotNone(sm)
        self.assertEqual(sm.get("observation_count"), 4)
        cs = sm.get("confidence_stats")
        self.assertIsNotNone(cs)
        self.assertEqual(cs["count"], 4)
        self.assertAlmostEqual(cs["mean"], 0.72, places=6)

    def test_multiple_records_independent(self):
        recs = [
            _make_shadow_record(event_id="a", track_id=1, class_name="provide-object"),
            _make_shadow_record(event_id="b", track_id=2, class_name="provide-object"),
            _make_shadow_record(event_id="c", track_id=3, class_name="hide-object"),
        ]
        records = shadow_events_to_prediction_records(recs, video_id="vid-1")
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].prediction_id, "a")
        self.assertEqual(records[2].prediction_id, "c")
        self.assertEqual(records[2].class_name, "hide-object")

    def test_empty_list_produces_empty_artifact(self):
        artifact = shadow_events_to_prediction_artifact([], video_id="vid-1")
        self.assertEqual(len(artifact.records), 0)

    def test_artifact_construction_and_json(self):
        recs = [_make_shadow_record(event_id="uuid-1")]
        artifact = shadow_events_to_prediction_artifact(
            recs, video_id="vid-1", source_identifier="shadow-test"
        )
        self.assertIsInstance(artifact, PredictionArtifact)
        self.assertEqual(len(artifact.records), 1)
        self.assertEqual(artifact.source_identifier, "shadow-test")
        json.dumps(artifact.to_dict())

    def test_artifact_validate_no_fatal_errors(self):
        recs = [_make_shadow_record(event_id="uuid-1")]
        artifact = shadow_events_to_prediction_artifact(recs, video_id="vid-1")
        errors = artifact.validate()
        fatal = [e for e in errors if e.startswith("MISSING") or e.startswith("DUPLICATE")]
        self.assertEqual(fatal, [])

    def test_evidence_bundle_none_backward_compat(self):
        rec = _make_shadow_record()
        self.assertIsNone(rec.evidence_bundle)
        records = shadow_events_to_prediction_records([rec], video_id="vid-1")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].prediction_id, rec.event.event_id)

    def test_no_file_io(self):
        """Adapter performs no file I/O (confirmed by mock patching builtins.open)."""
        from unittest.mock import patch
        recs = [_make_shadow_record()]
        with patch("builtins.open", side_effect=OSError("unexpected file open")):
            records = shadow_events_to_prediction_records(recs, video_id="vid-1")
            artifact = shadow_events_to_prediction_artifact(recs, video_id="vid-1")
        self.assertEqual(len(records), 1)
        self.assertEqual(len(artifact.records), 1)


class TestR8CalibrationReadinessReport(unittest.TestCase):

    def _readiness(self, event_confidence: Optional[float] = None,
                   association_score: Optional[float] = None) -> CalibrationReadinessReport:
        rec = _make_shadow_record(event_confidence=event_confidence,
                                  association_score=association_score)
        records = shadow_events_to_prediction_records([rec], video_id="vid-1")
        return CalibrationReadinessReport.from_prediction_records(records)

    def test_matching_params_always_sweepable(self):
        rep = self._readiness(event_confidence=None)
        self.assertTrue(rep.matching_parameters_sweepable)
        for p in ("min_overlap_ratio", "overlap_metric",
                  "require_class_match", "identity_mode", "ambiguous_handling"):
            self.assertIn(p, rep.sweepable_parameters)

    def test_confirmation_blocked_when_event_confidence_none(self):
        rep = self._readiness(event_confidence=None)
        self.assertNotIn("min_confidence_mean", rep.sweepable_parameters)
        self.assertIn("min_confidence_mean", rep.blocked_parameters)
        self.assertIn("min_confidence_mean", rep.blocked_reasons)

    def test_confirmation_sweepable_when_confidence_populated(self):
        rep = self._readiness(event_confidence=0.85)
        self.assertIn("min_confidence_mean", rep.sweepable_parameters)
        self.assertNotIn("min_confidence_mean", rep.blocked_parameters)

    def test_association_unavailable_honestly_reported(self):
        rep = self._readiness(association_score=None)
        self.assertEqual(rep.association_evidence_status, "unavailable")

    def test_association_available_when_score_populated(self):
        rep = self._readiness(association_score=0.7)
        self.assertEqual(rep.association_evidence_status, "available")

    def test_report_deterministic(self):
        rec = _make_shadow_record()
        records = shadow_events_to_prediction_records([rec], video_id="vid-1")
        r1 = CalibrationReadinessReport.from_prediction_records(records)
        r2 = CalibrationReadinessReport.from_prediction_records(records)
        self.assertEqual(r1.sweepable_parameters, r2.sweepable_parameters)
        self.assertEqual(r1.blocked_parameters, r2.blocked_parameters)
        self.assertEqual(r1.blocked_reasons, r2.blocked_reasons)

    def test_report_to_dict_json_serializable(self):
        rep = self._readiness(event_confidence=None)
        json.dumps(rep.to_dict())

    def test_total_predictions_reflects_input(self):
        recs = [_make_shadow_record(event_id=f"u{i}") for i in range(5)]
        records = shadow_events_to_prediction_records(recs, video_id="vid-1")
        rep = CalibrationReadinessReport.from_prediction_records(records)
        self.assertEqual(rep.total_predictions, 5)

    def test_confirmation_sweep_all_none_excludes_all_records(self):
        """Document the blocker: when ALL records have event_confidence=None,
        sweep_confirmation_parameters with min_confidence_mean excludes all records."""
        from inference.evaluation.calibration import (
            CalibrationParameterGrid,
            CalibrationSplit,
            sweep_confirmation_parameters,
        )
        from inference.evaluation.ground_truth import GroundTruthDataset

        recs = [_make_shadow_record(event_id=f"u{i}", track_id=i + 1) for i in range(2)]
        artifact = shadow_events_to_prediction_artifact(recs, video_id="vid-1")
        # Use a minimal GT dataset (no incidents — validation errors are non-blocking)
        gt = GroundTruthDataset()
        split = CalibrationSplit(split_name="all", videos=["vid-1"])
        grid = CalibrationParameterGrid({"min_confidence_mean": [0.5]})
        rows = sweep_confirmation_parameters(gt, artifact, grid, split)
        # All records have event_confidence=None → filtered out → 0 predictions
        self.assertEqual(len(rows), 1)
        m = rows[0]["metrics"]
        self.assertEqual(m.total_predictions, 0,
                         "min_confidence_mean sweep must exclude all None-confidence records")


if __name__ == "__main__":
    unittest.main()
