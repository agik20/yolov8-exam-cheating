"""Phase 2C.2 — Prediction-vs-ground-truth matching tests.

Run:  python -m pytest tests/test_phase2c2_matching.py -v

Pins the evaluation matching layer:
- NormalizedPrediction adapter (EventRecord → NormalizedPrediction)
- Temporal matching: interval overlap, ratios, configurable threshold
- Class/behavior matching: CLASS_MATCH / MISMATCH / UNAVAILABLE / AMBIGUOUS
- One-to-one assignment with deterministic priority
- Duplicate detection
- FN / FP / DUPLICATE / AMBIGUOUS outcome taxonomy
- Evaluation summary: precision, recall, F1, duplicate rate
- Confirmation latency
- Determinism: repeated inputs → identical output
"""
import os
import sys
import json
import math
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from inference.core.events import (
    EntityKey,
    ModelSource,
    EventRecord,
    ConfidenceStats,
)
from inference.evaluation.ground_truth import (
    GTLabel,
    EntityReferenceStatus,
    GroundTruthIncident,
    NormalPeriod,
    AmbiguousPeriod,
    SourceVideoMetadata,
    DatasetManifest,
    GroundTruthDataset,
    SCHEMA_VERSION,
    VALID_BEHAVIOR_CLASSES,
)

from inference.evaluation.matching import (
    NormalizedPrediction,
    PredictionBoundary,
    EvaluationOutcome,
    ClassMatchStatus,
    IdentityMatchStatus,
    TemporalMatchStatus,
    TemporalRelationship,
    MatchResult,
    MatchingPolicy,
    EvaluationSummary,
    PredictionNormalizer,
    compute_temporal_relationship,
    compute_class_match_status,
    match_predictions_to_ground_truth,
    compute_evaluation_summary,
    compute_confirmation_latency,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_video(video_id="vid1", fps=30.0, duration=100.0):
    return SourceVideoMetadata(
        video_id=video_id, reference=video_id,
        source_fps=fps, duration_seconds=duration,
    )


def _make_gt_incident(
    video_id="vid1", incident_id="inc1",
    label=GTLabel.SUSPICIOUS, start=10.0, end=20.0,
    behavior_class="provide-object",
    person_ref_status=EntityReferenceStatus.UNKNOWN,
    object_ref_status=EntityReferenceStatus.NOT_APPLICABLE,
    person_reference=None, object_reference=None,
):
    return GroundTruthIncident(
        video_id=video_id, incident_id=incident_id, label=label,
        start_video_time=start, end_video_time=end,
        behavior_class=behavior_class,
        person_ref_status=person_ref_status,
        object_ref_status=object_ref_status,
        person_reference=person_reference,
        object_reference=object_reference,
    )


def _make_event_record(
    event_id="evt1", class_name="provide-object",
    confirmation_time=15.0, candidate_start=10.0, candidate_end=20.0,
    entity_key=None,
):
    if entity_key is None:
        entity_key = EntityKey.from_pose(1)
    return EventRecord(
        event_id=event_id, entity_key=entity_key,
        class_name=class_name,
        confirmation_video_time=confirmation_time,
        confirmation_wall_time="2026-01-01T00:00:00Z",
        candidate_start_video_time=candidate_start,
        candidate_end_video_time=candidate_end,
        observation_count=3,
        confidence_stats=ConfidenceStats(count=3, mean=0.8, min=0.7, max=0.9),
        event_confidence=None, association_score=None,
        contributing_track_ids=frozenset({1}),
    )


def _make_pred(
    prediction_id="p1", video_id="vid1",
    start_time=10.0, end_time=20.0,
    class_name="provide-object",
    identity="pose:1", confirmation_time=15.0,
    event_confidence=None,
    boundary_type=PredictionBoundary.EXACT_INTERVAL,
):
    return NormalizedPrediction(
        prediction_id=prediction_id, video_id=video_id,
        start_video_time=start_time, end_video_time=end_time,
        class_name=class_name, identity=identity,
        confirmation_video_time=confirmation_time,
        event_confidence=event_confidence,
        boundary_type=boundary_type,
    )


def _make_policy(
    min_overlap_ratio=0.5, overlap_metric="gt_ratio",
    require_class_match=False, identity_mode="ignore",
    ambiguous_handling="exclude",
):
    return MatchingPolicy(
        min_overlap_ratio=min_overlap_ratio,
        overlap_metric=overlap_metric,
        require_class_match=require_class_match,
        identity_mode=identity_mode,
        ambiguous_handling=ambiguous_handling,
    )


# ---------------------------------------------------------------------------
# 1. NormalizedPrediction
# ---------------------------------------------------------------------------

class TestNormalizedPrediction(unittest.TestCase):

    def test_exact_interval(self):
        p = _make_pred()
        self.assertIs(p.boundary_type, PredictionBoundary.EXACT_INTERVAL)
        self.assertIsNotNone(p.start_video_time)
        self.assertIsNotNone(p.end_video_time)

    def test_point_event(self):
        p = _make_pred(
            boundary_type=PredictionBoundary.POINT_EVENT,
            start_time=None, end_time=15.0,
        )
        self.assertIsNone(p.start_video_time)
        self.assertEqual(p.end_video_time, 15.0)

    def test_confirmation_unavailable(self):
        p = _make_pred(confirmation_time=None)
        self.assertIsNone(p.confirmation_video_time)

    def test_identity_unavailable(self):
        p = _make_pred(identity=None)
        self.assertIsNone(p.identity)

    def test_prediction_id_required(self):
        with self.assertRaises(TypeError):
            NormalizedPrediction(
                video_id="vid1", start_video_time=0.0,
                end_video_time=1.0, class_name="smartphone",
            )


# ---------------------------------------------------------------------------
# 2. PredictionNormalizer
# ---------------------------------------------------------------------------

class TestPredictionNormalizer(unittest.TestCase):

    def test_basic_normalization(self):
        evt = _make_event_record(event_id="evt1")
        p = PredictionNormalizer.event_record_to_normalized(
            evt, video_id="vid1", prediction_id="p1",
        )
        self.assertEqual(p.prediction_id, "p1")
        self.assertEqual(p.video_id, "vid1")
        self.assertEqual(p.start_video_time, 10.0)
        self.assertEqual(p.end_video_time, 20.0)
        self.assertEqual(p.class_name, "provide-object")
        self.assertEqual(p.confirmation_video_time, 15.0)
        self.assertEqual(p.identity, "pose:1")
        self.assertIsNone(p.event_confidence)
        self.assertIs(p.boundary_type, PredictionBoundary.EXACT_INTERVAL)

    def test_entity_key_identity(self):
        evt = _make_event_record(
            event_id="evt3", entity_key=EntityKey.from_object(7),
        )
        p = PredictionNormalizer.event_record_to_normalized(
            evt, video_id="vid1", prediction_id="p3",
        )
        self.assertEqual(p.identity, "object:7")

    def test_event_confidence_carried(self):
        evt = _make_event_record(event_id="evt4")
        evt.event_confidence = 0.92
        p = PredictionNormalizer.event_record_to_normalized(
            evt, video_id="vid1", prediction_id="p4",
        )
        self.assertEqual(p.event_confidence, 0.92)


# ---------------------------------------------------------------------------
# 3. Temporal relationship
# ---------------------------------------------------------------------------

class TestTemporalRelationship(unittest.TestCase):

    def test_exact_overlap(self):
        pred = _make_pred(start_time=10.0, end_time=20.0)
        gt   = _make_gt_incident(start=10.0, end=20.0)
        rel = compute_temporal_relationship(pred, gt)
        self.assertAlmostEqual(rel.overlap_duration, 10.0)
        self.assertAlmostEqual(rel.overlap_ratio_prediction, 1.0)
        self.assertAlmostEqual(rel.overlap_ratio_ground_truth, 1.0)

    def test_partial_overlap(self):
        pred = _make_pred(start_time=10.0, end_time=20.0)
        gt   = _make_gt_incident(start=15.0, end=25.0)
        rel = compute_temporal_relationship(pred, gt)
        self.assertAlmostEqual(rel.overlap_duration, 5.0)
        self.assertAlmostEqual(rel.overlap_ratio_prediction, 0.5)
        self.assertAlmostEqual(rel.overlap_ratio_ground_truth, 0.5)

    def test_no_overlap(self):
        pred = _make_pred(start_time=0.0, end_time=5.0)
        gt   = _make_gt_incident(start=10.0, end=20.0)
        rel = compute_temporal_relationship(pred, gt)
        self.assertAlmostEqual(rel.overlap_duration, 0.0)

    def test_prediction_contained_in_gt(self):
        pred = _make_pred(start_time=12.0, end_time=18.0)
        gt   = _make_gt_incident(start=10.0, end=20.0)
        rel = compute_temporal_relationship(pred, gt)
        self.assertAlmostEqual(rel.overlap_duration, 6.0)
        self.assertAlmostEqual(rel.overlap_ratio_prediction, 1.0)
        self.assertAlmostEqual(rel.overlap_ratio_ground_truth, 0.6)

    def test_gt_contained_in_prediction(self):
        pred = _make_pred(start_time=8.0, end_time=22.0)
        gt   = _make_gt_incident(start=10.0, end=20.0)
        rel = compute_temporal_relationship(pred, gt)
        self.assertAlmostEqual(rel.overlap_duration, 10.0)
        self.assertAlmostEqual(rel.overlap_ratio_prediction, 10.0 / 14.0)
        self.assertAlmostEqual(rel.overlap_ratio_ground_truth, 1.0)

    def test_zero_duration_prediction(self):
        pred = _make_pred(
            boundary_type=PredictionBoundary.POINT_EVENT,
            start_time=None, end_time=15.0,
        )
        gt = _make_gt_incident(start=10.0, end=20.0)
        rel = compute_temporal_relationship(pred, gt)
        self.assertIsNone(rel.overlap_ratio_prediction)

    def test_zero_duration_gt(self):
        pred = _make_pred(start_time=10.0, end_time=20.0)
        gt   = _make_gt_incident(start=15.0, end=15.0)
        rel = compute_temporal_relationship(pred, gt)
        self.assertIsNone(rel.overlap_ratio_ground_truth)

    def test_boundary_touching_no_overlap(self):
        pred = _make_pred(start_time=0.0, end_time=10.0)
        gt   = _make_gt_incident(start=10.0, end=20.0)
        rel = compute_temporal_relationship(pred, gt)
        self.assertAlmostEqual(rel.overlap_duration, 0.0)

    def test_prediction_before_gt(self):
        pred = _make_pred(start_time=0.0, end_time=5.0)
        gt   = _make_gt_incident(start=10.0, end=20.0)
        rel = compute_temporal_relationship(pred, gt)
        self.assertAlmostEqual(rel.overlap_duration, 0.0)

    def test_prediction_after_gt(self):
        pred = _make_pred(start_time=25.0, end_time=35.0)
        gt   = _make_gt_incident(start=10.0, end=20.0)
        rel = compute_temporal_relationship(pred, gt)
        self.assertAlmostEqual(rel.overlap_duration, 0.0)

    def test_identical_intervals(self):
        pred = _make_pred(start_time=10.0, end_time=20.0)
        gt   = _make_gt_incident(start=10.0, end=20.0)
        rel = compute_temporal_relationship(pred, gt)
        self.assertAlmostEqual(rel.overlap_ratio_prediction, 1.0)
        self.assertAlmostEqual(rel.overlap_ratio_ground_truth, 1.0)


# ---------------------------------------------------------------------------
# 4. Class matching
# ---------------------------------------------------------------------------

class TestClassMatching(unittest.TestCase):

    def test_same_class_is_match(self):
        pred = _make_pred(class_name="provide-object")
        gt   = _make_gt_incident(behavior_class="provide-object")
        self.assertIs(compute_class_match_status(pred, gt),
                      ClassMatchStatus.CLASS_MATCH)

    def test_different_class_is_mismatch(self):
        pred = _make_pred(class_name="smartphone")
        gt   = _make_gt_incident(behavior_class="provide-object")
        self.assertIs(compute_class_match_status(pred, gt),
                      ClassMatchStatus.CLASS_MISMATCH)

    def test_unavailable_prediction_class(self):
        pred = _make_pred(class_name=None)
        gt   = _make_gt_incident(behavior_class="provide-object")
        self.assertIs(compute_class_match_status(pred, gt),
                      ClassMatchStatus.CLASS_UNAVAILABLE)

    def test_ambiguous_gt_label(self):
        pred = _make_pred(class_name="smartphone")
        gt   = _make_gt_incident(label=GTLabel.AMBIGUOUS,
                                  behavior_class="smartphone")
        self.assertIs(compute_class_match_status(pred, gt),
                      ClassMatchStatus.CLASS_AMBIGUOUS)


# ---------------------------------------------------------------------------
# 5. One-to-one matching
# ---------------------------------------------------------------------------

class TestOneToOneMatching(unittest.TestCase):

    def test_one_prediction_one_gt_match(self):
        pred = _make_pred(prediction_id="p1", start_time=10.0, end_time=20.0)
        gt   = _make_gt_incident(incident_id="inc1", start=10.0, end=20.0)
        policy = _make_policy()
        results = match_predictions_to_ground_truth([pred], [gt], policy)
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].outcome, EvaluationOutcome.TP)
        self.assertEqual(results[0].prediction_id, "p1")
        self.assertEqual(results[0].ground_truth_id, "inc1")

    def test_multiple_predictions_one_gt_produces_duplicate(self):
        p1 = _make_pred(prediction_id="p1", start_time=10.0, end_time=18.0)
        p2 = _make_pred(prediction_id="p2", start_time=11.0, end_time=19.0)
        gt = _make_gt_incident(incident_id="inc1", start=10.0, end=20.0)
        policy = _make_policy()
        results = match_predictions_to_ground_truth([p1, p2], [gt], policy)
        outcomes = [r.outcome for r in results]
        self.assertEqual(outcomes.count(EvaluationOutcome.TP), 1)
        self.assertEqual(outcomes.count(EvaluationOutcome.DUPLICATE), 1)

    def test_one_prediction_multiple_gt(self):
        pred = _make_pred(
            prediction_id="p1",
            start_time=10.0, end_time=30.0,
        )
        gt1 = _make_gt_incident(incident_id="inc1", start=10.0, end=20.0)
        gt2 = _make_gt_incident(incident_id="inc2", start=15.0, end=25.0)
        policy = _make_policy()
        results = match_predictions_to_ground_truth([pred], [gt1, gt2], policy)
        tp_results = [r for r in results if r.outcome == EvaluationOutcome.TP]
        self.assertEqual(len(tp_results), 1)

    def test_multiple_predictions_multiple_gt(self):
        p1 = _make_pred(prediction_id="p1", start_time=10.0, end_time=18.0)
        p2 = _make_pred(prediction_id="p2", start_time=20.0, end_time=28.0)
        gt1 = _make_gt_incident(incident_id="inc1", start=10.0, end=20.0)
        gt2 = _make_gt_incident(incident_id="inc2", start=20.0, end=30.0)
        policy = _make_policy()
        results = match_predictions_to_ground_truth([p1, p2], [gt1, gt2], policy)
        outcomes = [r.outcome for r in results]
        self.assertEqual(outcomes.count(EvaluationOutcome.TP), 2)


# ---------------------------------------------------------------------------
# 6. Duplicate predictions
# ---------------------------------------------------------------------------

class TestDuplicates(unittest.TestCase):

    def test_duplicate_distinct_from_fp(self):
        # p_primary and p_dup both overlap GT; p_primary (earlier start_time)
        # is the primary TP, p_dup is the DUPLICATE.
        # p_fp is far away (no GT overlap) and must be FP, not DUPLICATE.
        p_primary = _make_pred(prediction_id="p_primary", start_time=10.0, end_time=18.0)
        p_dup     = _make_pred(prediction_id="p_dup",     start_time=11.0, end_time=19.0)
        p_fp      = _make_pred(prediction_id="p_fp",      start_time=50.0, end_time=60.0)
        gt        = _make_gt_incident(incident_id="inc1", start=10.0, end=20.0)
        policy    = _make_policy()
        results   = match_predictions_to_ground_truth([p_primary, p_dup, p_fp], [gt], policy)
        by_id     = {r.prediction_id: r for r in results if r.prediction_id is not None}
        self.assertIs(by_id["p_primary"].outcome, EvaluationOutcome.TP)
        self.assertIs(by_id["p_dup"].outcome,     EvaluationOutcome.DUPLICATE)
        self.assertIs(by_id["p_fp"].outcome,       EvaluationOutcome.FP)

    def test_duplicate_rate_in_summary(self):
        p1 = _make_pred(prediction_id="p1", start_time=10.0, end_time=18.0)
        p2 = _make_pred(prediction_id="p2", start_time=10.0, end_time=18.0)
        gt = _make_gt_incident(incident_id="inc1", start=10.0, end=20.0)
        policy = _make_policy()
        results = match_predictions_to_ground_truth([p1, p2], [gt], policy)
        summary = compute_evaluation_summary(results)
        self.assertEqual(summary.duplicate_count, 1)
        self.assertAlmostEqual(summary.duplicate_rate, 1.0 / 2.0)


# ---------------------------------------------------------------------------
# 7. False negatives
# ---------------------------------------------------------------------------

class TestFalseNegatives(unittest.TestCase):

    def test_unmatched_gt_is_fn(self):
        pred = _make_pred(prediction_id="p1", start_time=50.0, end_time=60.0)
        gt   = _make_gt_incident(incident_id="inc1", start=10.0, end=20.0)
        policy = _make_policy()
        results = match_predictions_to_ground_truth([pred], [gt], policy)
        fn_results = [r for r in results if r.outcome == EvaluationOutcome.FN]
        self.assertEqual(len(fn_results), 1)
        self.assertEqual(fn_results[0].ground_truth_id, "inc1")

    def test_ambiguous_gt_not_counted_as_fn_by_default(self):
        pred = _make_pred(prediction_id="p1", start_time=10.0, end_time=20.0)
        gt   = _make_gt_incident(
            incident_id="inc_amb", label=GTLabel.AMBIGUOUS,
            start=10.0, end=20.0, behavior_class="smartphone",
        )
        policy = _make_policy(ambiguous_handling="exclude")
        results = match_predictions_to_ground_truth([pred], [gt], policy)
        outcomes = [r.outcome for r in results]
        self.assertNotIn(EvaluationOutcome.FN, outcomes)
        self.assertIn(EvaluationOutcome.AMBIGUOUS_GROUND_TRUTH, outcomes)


# ---------------------------------------------------------------------------
# 8. False positives
# ---------------------------------------------------------------------------

class TestFalsePositives(unittest.TestCase):

    def test_unmatched_prediction_is_fp(self):
        pred = _make_pred(prediction_id="p1", start_time=50.0, end_time=60.0)
        gt   = _make_gt_incident(incident_id="inc1", start=10.0, end=20.0)
        policy = _make_policy()
        results = match_predictions_to_ground_truth([pred], [gt], policy)
        fp_results = [r for r in results if r.outcome == EvaluationOutcome.FP]
        self.assertEqual(len(fp_results), 1)

    def test_prediction_inside_normal_period_is_fp(self):
        pred = _make_pred(prediction_id="p1", start_time=0.0, end_time=5.0)
        np_  = NormalPeriod(
            video_id="vid1", period_id="norm1",
            start_video_time=0.0, end_video_time=10.0,
        )
        gt_suspicious = []  # no suspicious GT incidents
        policy = _make_policy()
        results = match_predictions_to_ground_truth(
            [pred], gt_suspicious, policy,
            normal_periods=[np_],
        )
        fp_results = [r for r in results if r.outcome == EvaluationOutcome.FP]
        self.assertEqual(len(fp_results), 1)


# ---------------------------------------------------------------------------
# 9. Identity matching
# ---------------------------------------------------------------------------

class TestIdentityMatching(unittest.TestCase):

    def test_known_identity_match(self):
        pred = _make_pred(
            prediction_id="p1",
            identity="student_A",
            start_time=10.0, end_time=20.0,
        )
        gt = _make_gt_incident(
            incident_id="inc1",
            person_ref_status=EntityReferenceStatus.KNOWN,
            person_reference="student_A",
            start=10.0, end=20.0,
        )
        policy = _make_policy(identity_mode="strict")
        results = match_predictions_to_ground_truth([pred], [gt], policy)
        self.assertIs(results[0].outcome, EvaluationOutcome.TP)
        self.assertIs(results[0].identity_match, IdentityMatchStatus.IDENTITY_MATCH)

    def test_known_identity_mismatch(self):
        pred = _make_pred(
            prediction_id="p1",
            identity="student_A",
            start_time=10.0, end_time=20.0,
        )
        gt = _make_gt_incident(
            incident_id="inc1",
            person_ref_status=EntityReferenceStatus.KNOWN,
            person_reference="student_B",
            start=10.0, end=20.0,
        )
        policy = _make_policy(identity_mode="strict")
        results = match_predictions_to_ground_truth([pred], [gt], policy)
        self.assertIs(results[0].identity_match, IdentityMatchStatus.IDENTITY_MISMATCH)

    def test_unknown_identity_gt_not_mismatch(self):
        pred = _make_pred(
            prediction_id="p1",
            identity="student_A",
            start_time=10.0, end_time=20.0,
        )
        gt = _make_gt_incident(
            incident_id="inc1",
            person_ref_status=EntityReferenceStatus.UNKNOWN,
            person_reference=None,
            start=10.0, end=20.0,
        )
        policy = _make_policy(identity_mode="strict")
        results = match_predictions_to_ground_truth([pred], [gt], policy)
        self.assertIs(results[0].identity_match, IdentityMatchStatus.IDENTITY_UNAVAILABLE)

    def test_identity_unavailable_on_both_sides(self):
        pred = _make_pred(
            prediction_id="p1",
            identity=None,
            start_time=10.0, end_time=20.0,
        )
        gt = _make_gt_incident(
            incident_id="inc1",
            person_ref_status=EntityReferenceStatus.UNKNOWN,
            start=10.0, end=20.0,
        )
        policy = _make_policy(identity_mode="strict")
        results = match_predictions_to_ground_truth([pred], [gt], policy)
        self.assertIs(results[0].identity_match, IdentityMatchStatus.IDENTITY_UNAVAILABLE)


# ---------------------------------------------------------------------------
# 10. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):

    def test_repeated_inputs_same_output(self):
        preds = [
            _make_pred(prediction_id="p1", start_time=10.0, end_time=18.0),
            _make_pred(prediction_id="p2", start_time=10.0, end_time=18.0),
            _make_pred(prediction_id="p3", start_time=50.0, end_time=60.0),
        ]
        gts = [
            _make_gt_incident(incident_id="inc1", start=10.0, end=20.0),
        ]
        policy = _make_policy()
        r1 = match_predictions_to_ground_truth(preds, gts, policy)
        r2 = match_predictions_to_ground_truth(preds, gts, policy)
        self.assertEqual(
            [(x.prediction_id, x.outcome) for x in r1],
            [(x.prediction_id, x.outcome) for x in r2],
        )

    def test_order_invariance_of_gt_list(self):
        preds = [
            _make_pred(prediction_id="p1", start_time=10.0, end_time=18.0),
        ]
        gt_a = _make_gt_incident(incident_id="inc_a", start=10.0, end=20.0)
        gt_b = _make_gt_incident(incident_id="inc_b", start=30.0, end=40.0)
        policy = _make_policy()
        r1 = match_predictions_to_ground_truth([preds[0]], [gt_a, gt_b], policy)
        r2 = match_predictions_to_ground_truth([preds[0]], [gt_b, gt_a], policy)
        self.assertEqual(
            [(x.prediction_id, x.outcome, x.ground_truth_id) for x in r1],
            [(x.prediction_id, x.outcome, x.ground_truth_id) for x in r2],
        )


# ---------------------------------------------------------------------------
# 11. Metrics / EvaluationSummary
# ---------------------------------------------------------------------------

class TestEvaluationSummary(unittest.TestCase):

    def _make_results(self, outcomes):
        """Build a list of MatchResults with the given outcomes."""
        results = []
        for i, oc in enumerate(outcomes):
            results.append(MatchResult(
                prediction_id=f"p{i}",
                ground_truth_id=None,
                outcome=oc,
                class_match=ClassMatchStatus.CLASS_MATCH,
                identity_match=IdentityMatchStatus.IDENTITY_UNAVAILABLE,
                temporal_status=TemporalMatchStatus.MATCHED,
                overlap_duration=0.0,
                overlap_ratio_prediction=0.0,
                overlap_ratio_ground_truth=0.0,
                reason="test",
            ))
        return results

    def test_precision_recall_f1(self):
        results = self._make_results([
            EvaluationOutcome.TP,
            EvaluationOutcome.FP,
            EvaluationOutcome.FN,
            EvaluationOutcome.TP,
        ])
        s = compute_evaluation_summary(results)
        self.assertEqual(s.tp_count, 2)
        self.assertEqual(s.fp_count, 1)
        self.assertEqual(s.fn_count, 1)
        # precision = TP / (TP + FP) = 2/3
        self.assertAlmostEqual(s.precision, 2.0 / 3.0)
        # recall = TP / (TP + FN) = 2/3
        self.assertAlmostEqual(s.recall, 2.0 / 3.0)
        # F1 = 2*p*r/(p+r) = 2*(2/3)*(2/3) / (2/3+2/3) = (8/9)/(4/3) = 2/3
        self.assertAlmostEqual(s.f1, 2.0 / 3.0)

    def test_zero_denominator_precision(self):
        results = self._make_results([EvaluationOutcome.FN])
        s = compute_evaluation_summary(results)
        self.assertIsNone(s.precision)

    def test_zero_denominator_recall(self):
        results = self._make_results([EvaluationOutcome.FP])
        s = compute_evaluation_summary(results)
        self.assertIsNone(s.recall)

    def test_f1_zero_denominators(self):
        results = self._make_results([EvaluationOutcome.DUPLICATE])
        s = compute_evaluation_summary(results)
        self.assertIsNone(s.f1)

    def test_duplicate_rate(self):
        results = self._make_results([
            EvaluationOutcome.TP,
            EvaluationOutcome.DUPLICATE,
            EvaluationOutcome.DUPLICATE,
        ])
        s = compute_evaluation_summary(results)
        self.assertEqual(s.duplicate_count, 2)
        self.assertAlmostEqual(s.duplicate_rate, 2.0 / 3.0)

    def test_total_counts(self):
        results = self._make_results([
            EvaluationOutcome.TP,
            EvaluationOutcome.FP,
            EvaluationOutcome.FN,
            EvaluationOutcome.DUPLICATE,
            EvaluationOutcome.AMBIGUOUS_GROUND_TRUTH,
        ])
        s = compute_evaluation_summary(results)
        self.assertEqual(s.total_predictions, 2)   # TP + FP
        self.assertEqual(s.total_gt_incidents, 2)  # TP + FN


# ---------------------------------------------------------------------------
# 12. Confirmation latency
# ---------------------------------------------------------------------------

class TestConfirmationLatency(unittest.TestCase):

    def test_valid_latency(self):
        pred = _make_pred(confirmation_time=15.0)
        gt   = _make_gt_incident(start=10.0)
        latency = compute_confirmation_latency(pred, gt)
        self.assertAlmostEqual(latency, 5.0)

    def test_missing_confirmation_time(self):
        pred = _make_pred(confirmation_time=None)
        gt   = _make_gt_incident(start=10.0)
        latency = compute_confirmation_latency(pred, gt)
        self.assertIsNone(latency)

    def test_negative_latency(self):
        pred = _make_pred(confirmation_time=5.0)
        gt   = _make_gt_incident(start=10.0)
        latency = compute_confirmation_latency(pred, gt)
        self.assertAlmostEqual(latency, -5.0)


# ---------------------------------------------------------------------------
# 13. MatchResult structure
# ---------------------------------------------------------------------------

class TestMatchResultStructure(unittest.TestCase):

    def test_reason_populated(self):
        pred = _make_pred(prediction_id="p1", start_time=10.0, end_time=20.0)
        gt   = _make_gt_incident(incident_id="inc1", start=10.0, end=20.0)
        policy = _make_policy()
        results = match_predictions_to_ground_truth([pred], [gt], policy)
        for r in results:
            self.assertIsInstance(r.reason, str)
            self.assertGreater(len(r.reason), 0)

    def test_match_result_fields(self):
        pred = _make_pred(prediction_id="p1", start_time=10.0, end_time=20.0)
        gt   = _make_gt_incident(incident_id="inc1", start=10.0, end=20.0)
        policy = _make_policy()
        results = match_predictions_to_ground_truth([pred], [gt], policy)
        r = results[0]
        self.assertEqual(r.prediction_id, "p1")
        self.assertEqual(r.ground_truth_id, "inc1")
        self.assertIs(r.outcome, EvaluationOutcome.TP)
        self.assertIsInstance(r.overlap_duration, float)


# ---------------------------------------------------------------------------
# 14. Empty sets
# ---------------------------------------------------------------------------

class TestEmptySets(unittest.TestCase):

    def test_empty_predictions(self):
        gt = _make_gt_incident(incident_id="inc1", start=10.0, end=20.0)
        policy = _make_policy()
        results = match_predictions_to_ground_truth([], [gt], policy)
        fn_results = [r for r in results if r.outcome == EvaluationOutcome.FN]
        self.assertEqual(len(fn_results), 1)

    def test_empty_gt(self):
        pred = _make_pred(prediction_id="p1", start_time=10.0, end_time=20.0)
        policy = _make_policy()
        results = match_predictions_to_ground_truth([pred], [], policy)
        fp_results = [r for r in results if r.outcome == EvaluationOutcome.FP]
        self.assertEqual(len(fp_results), 1)

    def test_both_empty(self):
        policy = _make_policy()
        results = match_predictions_to_ground_truth([], [], policy)
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# 15. Ambiguous handling modes
# ---------------------------------------------------------------------------

class TestAmbiguousHandling(unittest.TestCase):

    def test_exclude_mode(self):
        pred = _make_pred(prediction_id="p1", start_time=10.0, end_time=20.0)
        gt   = _make_gt_incident(
            incident_id="inc1", label=GTLabel.AMBIGUOUS,
            start=10.0, end=20.0,
        )
        policy = _make_policy(ambiguous_handling="exclude")
        results = match_predictions_to_ground_truth([pred], [gt], policy)
        self.assertIn(EvaluationOutcome.AMBIGUOUS_GROUND_TRUTH,
                      [r.outcome for r in results])

    def test_report_mode(self):
        pred = _make_pred(prediction_id="p1", start_time=10.0, end_time=20.0)
        gt   = _make_gt_incident(
            incident_id="inc1", label=GTLabel.AMBIGUOUS,
            start=10.0, end=20.0,
        )
        policy = _make_policy(ambiguous_handling="report")
        results = match_predictions_to_ground_truth([pred], [gt], policy)
        outcomes = [r.outcome for r in results]
        # In report mode, ambiguous GT is treated as eligible for matching
        self.assertIn(EvaluationOutcome.TP, outcomes)


# ---------------------------------------------------------------------------
# 16. Adjacent / nested intervals
# ---------------------------------------------------------------------------

class TestAdjacentAndNested(unittest.TestCase):

    def test_adjacent_incidents_no_overlap(self):
        pred = _make_pred(prediction_id="p1", start_time=0.0, end_time=10.0)
        gt_a = _make_gt_incident(incident_id="inc_a", start=0.0, end=10.0)
        gt_b = _make_gt_incident(incident_id="inc_b", start=10.0, end=20.0)
        policy = _make_policy()
        results = match_predictions_to_ground_truth([pred], [gt_a, gt_b], policy)
        tp_results = [r for r in results if r.outcome == EvaluationOutcome.TP]
        self.assertEqual(len(tp_results), 1)

    def test_nested_intervals(self):
        pred = _make_pred(prediction_id="p1", start_time=5.0, end_time=15.0)
        gt   = _make_gt_incident(incident_id="inc1", start=0.0, end=20.0)
        policy = _make_policy()
        results = match_predictions_to_ground_truth([pred], [gt], policy)
        tp_results = [r for r in results if r.outcome == EvaluationOutcome.TP]
        self.assertEqual(len(tp_results), 1)

    def test_identical_timestamps_multiple_predictions(self):
        p1 = _make_pred(prediction_id="p1", start_time=10.0, end_time=20.0)
        p2 = _make_pred(prediction_id="p2", start_time=10.0, end_time=20.0)
        gt = _make_gt_incident(incident_id="inc1", start=10.0, end=20.0)
        policy = _make_policy()
        results = match_predictions_to_ground_truth([p1, p2], [gt], policy)
        outcomes = [r.outcome for r in results]
        self.assertEqual(outcomes.count(EvaluationOutcome.TP), 1)
        self.assertEqual(outcomes.count(EvaluationOutcome.DUPLICATE), 1)


# ---------------------------------------------------------------------------
# 17. Configurable threshold
# ---------------------------------------------------------------------------

class TestConfigurableThreshold(unittest.TestCase):

    def test_higher_threshold_reduces_matches(self):
        pred = _make_pred(prediction_id="p1", start_time=10.0, end_time=15.0)
        gt   = _make_gt_incident(incident_id="inc1", start=10.0, end=20.0)
        # With gt_ratio=0.1: overlap=5s, gt_dur=10s → ratio=0.5 → match
        p_low  = _make_policy(min_overlap_ratio=0.1)
        r_low  = match_predictions_to_ground_truth([pred], [gt], p_low)
        self.assertIn(EvaluationOutcome.TP, [r.outcome for r in r_low])
        # With gt_ratio=0.6: ratio=0.5 < 0.6 → no match → FP
        p_high = _make_policy(min_overlap_ratio=0.6)
        r_high = match_predictions_to_ground_truth([pred], [gt], p_high)
        self.assertIn(EvaluationOutcome.FP, [r.outcome for r in r_high])

    def test_overlap_metric_configurable(self):
        pred = _make_pred(prediction_id="p1", start_time=10.0, end_time=18.0)
        gt   = _make_gt_incident(incident_id="inc1", start=10.0, end=20.0)
        # pred_ratio: overlap=8s, pred_dur=8s → 1.0
        p_pred = _make_policy(overlap_metric="pred_ratio", min_overlap_ratio=0.9)
        results = match_predictions_to_ground_truth([pred], [gt], p_pred)
        self.assertIn(EvaluationOutcome.TP, [r.outcome for r in results])

