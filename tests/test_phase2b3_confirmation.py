"""Phase 2B.3 — Unit tests for the shadow confirmation decision layer.

Covers:
- ConfirmationGateResult structure and JSON safety
- UnconfiguredConfirmationPolicy: always returns confirmed=False with a
  clear "confirmation_policy_unconfigured" reason
- CalibratedConfirmationPolicy: each gate dimension (observation count,
  temporal span, mean confidence, max gap) can independently confirm or
  reject; all-met case yields confirmed=True
- Evidence reporting: observation_count, video_time_span, confidence_stats,
  gap stats, track IDs are all populated in the result
- Separation: shadow confirmation does not call save_crop, does not call
  log_detection_to_csv, does not modify legacy id_info state
- Static check: process_detection source does not reference confirmation

Run:  python -m pytest tests/test_phase2b3_confirmation.py -v
"""
import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from inference.core.events import (
    ModelSource,
    EntityKey,
    Observation,
    BoundedObservationBuffer,
    CandidateState,
    Candidate,
    ConfidenceStats,
)
from inference.core.confirmation import (
    ConfirmationGateResult,
    UnconfiguredConfirmationPolicy,
    CalibratedConfirmationPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(frame_index=10, video_time=0.333, class_name="provide-object",
              confidence=0.72, bbox=(100, 200, 400, 500),
              source=ModelSource.POSE, track_id=3) -> Observation:
    return Observation(
        frame_index=frame_index,
        video_time=video_time,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        source=source,
        track_id=track_id,
    )


def _make_candidate(
    track_id=3,
    class_name="provide-object",
    source=ModelSource.POSE,
    n_observations=5,
    conf_min=0.55,
    conf_max=0.91,
    span_seconds=2.0,
    gap_count=0,
    max_gap_seconds=None,
) -> Candidate:
    """Build a candidate with deterministic observation data.

    Observations are spaced evenly across span_seconds.  Alternates
    between conf_min and conf_max so the mean is predictable.
    """
    key = EntityKey(source=source, track_id=track_id)
    buf = BoundedObservationBuffer(maxlen=64)
    start_time = 10.0
    for i in range(n_observations):
        if n_observations > 1:
            t = start_time + (span_seconds * i / (n_observations - 1))
        else:
            t = start_time
        conf = conf_min if i % 2 == 0 else conf_max
        obs = _make_obs(frame_index=i, video_time=t, track_id=track_id,
                        class_name=class_name, confidence=conf)
        buf.append(obs)
    cand = Candidate(
        entity_key=key,
        candidate_class=class_name,
        observations=buf,
        gap_count=gap_count,
        max_gap_seconds=max_gap_seconds,
    )
    cand.start_video_time = start_time
    cand.latest_video_time = start_time + span_seconds
    cand.observed_track_ids = frozenset([track_id])
    return cand


def _make_calibrated_policy(
    min_observations=5,
    min_duration_seconds=2.0,
    min_confidence_mean=0.65,
    max_gap_seconds=None,
) -> CalibratedConfirmationPolicy:
    return CalibratedConfirmationPolicy(
        min_observations=min_observations,
        min_duration_seconds=min_duration_seconds,
        min_confidence_mean=min_confidence_mean,
        max_gap_seconds=max_gap_seconds,
    )


# ---------------------------------------------------------------------------
# ConfirmationGateResult structure
# ---------------------------------------------------------------------------

class TestConfirmationGateResult(unittest.TestCase):

    def test_result_is_json_serializable(self):
        cand = _make_candidate()
        policy = UnconfiguredConfirmationPolicy()
        result = policy.evaluate(cand)
        json.dumps(result.to_dict())

    def test_unconfigured_policy_always_rejects(self):
        """UnconfiguredConfirmationPolicy must never return confirmed=True,
        regardless of how much evidence the candidate holds."""
        cand = _make_candidate(n_observations=100, span_seconds=60.0,
                               conf_min=0.9, conf_max=0.95)
        policy = UnconfiguredConfirmationPolicy()
        result = policy.evaluate(cand)
        self.assertFalse(result.confirmed,
                         "unconfigured policy must never confirm")
        self.assertEqual(result.reason, "confirmation_policy_unconfigured")

    def test_result_contains_evidence_fields(self):
        cand = _make_candidate(
            n_observations=4,
            conf_min=0.5,
            conf_max=0.8,
            span_seconds=1.5,
            gap_count=2,
            max_gap_seconds=0.7,
        )
        policy = UnconfiguredConfirmationPolicy()
        result = policy.evaluate(cand)
        self.assertEqual(result.observation_count, 4)
        self.assertAlmostEqual(result.video_time_span, 1.5, places=4)
        self.assertEqual(result.confidence_stats.count, 4)
        self.assertAlmostEqual(result.confidence_stats.min, 0.5, places=4)
        self.assertAlmostEqual(result.confidence_stats.max, 0.8, places=4)
        self.assertEqual(result.gap_count, 2)
        self.assertAlmostEqual(result.max_gap_seconds, 0.7, places=4)
        self.assertIn(3, result.track_ids)

    def test_association_score_is_none_in_2b3(self):
        """Person-object association is Phase 2B.5; must be None in 2B.3."""
        cand = _make_candidate()
        result = UnconfiguredConfirmationPolicy().evaluate(cand)
        self.assertIsNone(result.association_score,
                          "association_score must be None in Phase 2B.3")

    def test_event_confidence_is_none_in_2b3(self):
        """No calibrated event-confidence formula exists yet; must be None."""
        cand = _make_candidate()
        result = UnconfiguredConfirmationPolicy().evaluate(cand)
        self.assertIsNone(result.event_confidence,
                          "event_confidence must be None until calibrated")


# ---------------------------------------------------------------------------
# CalibratedConfirmationPolicy — each gate dimension
# ---------------------------------------------------------------------------

class TestCalibratedPolicyGates(unittest.TestCase):

    def test_insufficient_observations_rejects(self):
        """Candidate with 2 observations vs min_observations=5 → not confirmed."""
        cand = _make_candidate(n_observations=2, span_seconds=5.0,
                               conf_min=0.8, conf_max=0.9)
        policy = _make_calibrated_policy(min_observations=5)
        result = policy.evaluate(cand)
        self.assertFalse(result.confirmed)
        self.assertIn("min_observations", result.unmet_requirements)

    def test_insufficient_duration_rejects(self):
        """Candidate span 0.5s vs min_duration_seconds=2.0 → not confirmed."""
        cand = _make_candidate(n_observations=10, span_seconds=0.5,
                               conf_min=0.8, conf_max=0.9)
        policy = _make_calibrated_policy(min_duration_seconds=2.0)
        result = policy.evaluate(cand)
        self.assertFalse(result.confirmed)
        self.assertIn("min_duration", result.unmet_requirements)

    def test_insufficient_confidence_rejects(self):
        """Low-confidence candidate vs min_confidence_mean=0.65 → not confirmed."""
        cand = _make_candidate(n_observations=10, span_seconds=5.0,
                               conf_min=0.2, conf_max=0.3)
        policy = _make_calibrated_policy(min_confidence_mean=0.65)
        result = policy.evaluate(cand)
        self.assertFalse(result.confirmed)
        self.assertIn("min_confidence", result.unmet_requirements)

    def test_excessive_gap_rejects(self):
        """Candidate with a large gap vs max_gap_seconds=1.0 → not confirmed."""
        cand = _make_candidate(n_observations=10, span_seconds=5.0,
                               conf_min=0.8, conf_max=0.9,
                               gap_count=1, max_gap_seconds=3.0)
        policy = _make_calibrated_policy(max_gap_seconds=1.0)
        result = policy.evaluate(cand)
        self.assertFalse(result.confirmed)
        self.assertIn("max_gap", result.unmet_requirements)

    def test_all_gates_satisfied_confirms(self):
        """All four dimensions met → confirmed=True.

        obs=10 ≥ 5, span=5.0 ≥ 2.0, mean conf ≈ 0.75 ≥ 0.65,
        max_gap=0.5 ≤ 1.0
        """
        cand = _make_candidate(
            n_observations=10,
            span_seconds=5.0,
            conf_min=0.70,
            conf_max=0.80,
            gap_count=1,
            max_gap_seconds=0.5,
        )
        policy = _make_calibrated_policy(
            min_observations=5,
            min_duration_seconds=2.0,
            min_confidence_mean=0.65,
            max_gap_seconds=1.0,
        )
        result = policy.evaluate(cand)
        self.assertTrue(result.confirmed,
                        f"all gates met but rejected: unmet={result.unmet_requirements}")

    def test_single_high_confidence_does_not_confirm(self):
        """A single high-confidence observation must not be sufficient —
        the gate requires accumulated evidence."""
        cand = _make_candidate(
            n_observations=1,
            span_seconds=0.0,
            conf_min=0.99,
            conf_max=0.99,
        )
        policy = _make_calibrated_policy(
            min_observations=3,
            min_duration_seconds=1.0,
            min_confidence_mean=0.5,
        )
        result = policy.evaluate(cand)
        self.assertFalse(result.confirmed,
                         "single high-confidence observation must not confirm")

    def test_none_parameters_act_as_unenforced_gates(self):
        """When a gate parameter is None, that gate is not enforced.
        A candidate with no gap data and max_gap_seconds=None should not
        be rejected on the gap gate."""
        cand = _make_candidate(n_observations=10, span_seconds=5.0,
                               conf_min=0.7, conf_max=0.8,
                               gap_count=0, max_gap_seconds=None)
        policy = CalibratedConfirmationPolicy(
            min_observations=5,
            min_duration_seconds=None,
            min_confidence_mean=None,
            max_gap_seconds=None,
        )
        result = policy.evaluate(cand)
        self.assertTrue(result.confirmed,
                        "only min_observations is enforced; candidate has 10 obs")

    def test_all_none_parameters_behaves_as_unconfigured(self):
        """If all parameters are None the policy has no calibrated gates
        and must behave like the unconfigured case: confirmed=False."""
        cand = _make_candidate(n_observations=10, span_seconds=5.0)
        policy = CalibratedConfirmationPolicy(
            min_observations=None,
            min_duration_seconds=None,
            min_confidence_mean=None,
            max_gap_seconds=None,
        )
        result = policy.evaluate(cand)
        self.assertFalse(result.confirmed)
        self.assertEqual(result.reason, "confirmation_policy_unconfigured")


# ---------------------------------------------------------------------------
# Evidence reporting completeness
# ---------------------------------------------------------------------------

class TestEvidenceReporting(unittest.TestCase):

    def test_result_has_all_required_fields(self):
        cand = _make_candidate(
            n_observations=6,
            conf_min=0.6,
            conf_max=0.85,
            span_seconds=3.0,
            gap_count=1,
            max_gap_seconds=0.4,
        )
        result = UnconfiguredConfirmationPolicy().evaluate(cand)
        self.assertEqual(result.observation_count, 6)
        self.assertAlmostEqual(result.video_time_span, 3.0, places=4)
        self.assertEqual(result.confidence_stats.count, 6)
        self.assertGreater(result.confidence_stats.mean, 0.6)
        self.assertLess(result.confidence_stats.mean, 0.86)
        self.assertEqual(result.gap_count, 1)
        self.assertAlmostEqual(result.max_gap_seconds, 0.4, places=4)
        self.assertEqual(set(result.track_ids), {3})
        self.assertEqual(result.entity_key.as_string(), "pose:3")
        self.assertEqual(result.candidate_class, "provide-object")

    def test_unmet_requirements_listed_when_gates_fail(self):
        cand = _make_candidate(n_observations=2, span_seconds=0.5)
        policy = _make_calibrated_policy(
            min_observations=5,
            min_duration_seconds=2.0,
        )
        result = policy.evaluate(cand)
        self.assertFalse(result.confirmed)
        self.assertIn("min_observations", result.unmet_requirements)
        self.assertIn("min_duration", result.unmet_requirements)

    def test_confidence_stats_computed_not_stale(self):
        """The result must use ConfidenceStats (computed from the buffer),
        NOT the legacy stale last_conf value."""
        cand = _make_candidate(n_observations=4, conf_min=0.5, conf_max=0.9)
        result = UnconfiguredConfirmationPolicy().evaluate(cand)
        # Alternating pattern: obs at even indices = 0.5, odd = 0.9
        expected_mean = (0.5 + 0.9 + 0.5 + 0.9) / 4
        self.assertAlmostEqual(result.confidence_stats.mean, expected_mean,
                               places=6)


# ---------------------------------------------------------------------------
# Separation from legacy event path
# ---------------------------------------------------------------------------

class TestSeparation(unittest.TestCase):

    def test_confirmation_does_not_call_save_crop(self):
        import inference.core.confirmation as conf_mod
        with open(conf_mod.__file__) as f:
            src = f.read()
        self.assertNotIn("save_crop", src)
        self.assertNotIn("log_detection_to_csv", src)
        self.assertNotIn("cv2.imwrite", src)

    def test_process_detection_does_not_reference_confirmation(self):
        import inspect
        import inference.core.pipeline as pipeline_mod
        src = inspect.getsource(pipeline_mod.CheatingDetectionPipeline.process_detection)
        for forbidden in ("_shadow_confirmed", "confirmation",
                          "ConfirmationGateResult", "CalibratedConfirmationPolicy",
                          "UnconfiguredConfirmationPolicy"):
            self.assertNotIn(forbidden, src,
                             f"process_detection must not reference {forbidden!r}")


# ---------------------------------------------------------------------------
# Shadow state bookkeeping
# ---------------------------------------------------------------------------

class TestShadowStateBookkeeping(unittest.TestCase):

    def test_repeated_evaluations_do_not_reconfirm(self):
        """Once a candidate is shadow-confirmed, subsequent evaluations
        for the same candidate should report already_confirmed=True."""
        cand = _make_candidate(
            n_observations=10,
            span_seconds=5.0,
            conf_min=0.70,
            conf_max=0.80,
        )
        policy = _make_calibrated_policy(
            min_observations=5,
            min_duration_seconds=2.0,
            min_confidence_mean=0.65,
        )
        r1 = policy.evaluate(cand)
        self.assertTrue(r1.confirmed)
        self.assertFalse(r1.already_confirmed,
                         "first confirmation must not be marked already_confirmed")

        # Simulate: candidate has now transitioned to CONFIRMED
        cand.state = CandidateState.CONFIRMED
        r2 = policy.evaluate(cand)
        self.assertTrue(r2.confirmed)
        self.assertTrue(r2.already_confirmed,
                         "re-evaluating a CONFIRMED candidate must set already_confirmed=True")


if __name__ == "__main__":
    unittest.main()
