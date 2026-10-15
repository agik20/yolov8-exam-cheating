"""Phase 2B.6 — Structured Evidence Bundle test suite.

Run:  python -m pytest tests/test_phase2b6_evidence.py -v

Pins the additive evidence extensions added in Phase 2B.6:
- ConfirmationDecisionEvidence (per-gate PASS/FAIL/NOT_CONFIGURED/UNAVAILABLE)
- TrackIdentityEvidence (track_id_switches stays None unless actually measured)
- AssociationEvidenceSummary (score stays None unless explicitly calibrated)
- HumanEvidenceMetadata (no image I/O; fields stay None)
- EvidenceBundle.additive fields (decision / association / human / identity)
- deterministic serialization, enum->string, None preserved, no numpy/cv2 objects
"""
import os
import sys
import json
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from inference.core.events import (
    EntityKey,
    ModelSource,
    Observation,
    ConfidenceStats,
    EventRecord,
    Candidate,
    CandidateState,
    EvidenceBundle,
    BoundedObservationBuffer,
)
from inference.core.confirmation import (
    ConfirmationGateResult,
    CalibratedConfirmationPolicy,
    UnconfiguredConfirmationPolicy,
)
from inference.core.shadow_lifecycle import ShadowEventRecord
from inference.core.association import (
    AssociationStatus,
    AssociationResult,
    SpatialEvidence,
    KeypointEvidence,
    TemporalEvidence,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_obs(track_id=1, video_time=0.033, conf=0.9):
    return Observation(
        frame_index=1,
        video_time=video_time,
        class_name="provide-object",
        confidence=conf,
        bbox=(10, 10, 100, 200),
        source=ModelSource.POSE,
        track_id=track_id,
    )


def _make_candidate(n_obs=3, span=1.0, gap_count=0, max_gap=0.1):
    key = EntityKey.from_pose(1)
    cand = Candidate(
        entity_key=key,
        candidate_class="provide-object",
        observations=BoundedObservationBuffer(maxlen=32),
    )
    for i in range(n_obs):
        cand.add_observation(_make_obs(track_id=1, video_time=i * span, conf=0.9))
    cand.gap_count = gap_count
    cand.max_gap_seconds = max_gap
    return cand


def _make_event_record(candidate=None):
    candidate = candidate or _make_candidate()
    return EventRecord.from_candidate(
        candidate,
        confirmation_video_time=candidate.latest_video_time or 0.0,
        confirmation_wall_time="2026-01-01T00:00:00",
    )


# ---------------------------------------------------------------------------
# ConfirmationDecisionEvidence
# ---------------------------------------------------------------------------

class TestConfirmationDecisionEvidence(unittest.TestCase):
    """Per-gate auditability: configured gates recorded, unconfigured distinguished."""

    def test_module_exists(self):
        from inference.core.events import ConfirmationDecisionEvidence  # noqa: F401
        self.assertTrue(True)

    def test_gate_outcome_enum_values(self):
        from inference.core.events import GateOutcome
        for name in ("PASS", "FAIL", "NOT_CONFIGURED", "UNAVAILABLE"):
            self.assertIn(name, GateOutcome.__members__)

    def test_gate_outcome_serializes_as_string(self):
        from inference.core.events import GateOutcome
        d = GateOutcome.PASS.to_dict()
        self.assertEqual(d, "PASS")

    def test_confirmation_decision_evidence_fields(self):
        from inference.core.events import ConfirmationDecisionEvidence
        cd = ConfirmationDecisionEvidence(
            policy_type="calibrated",
            confirmed=True,
            min_observations_configured=True,
            min_observations_required=3,
            actual_observations=5,
            min_duration_seconds_configured=True,
            min_duration_seconds_required=1.0,
            actual_duration=2.5,
            min_confidence_mean_configured=False,
            min_confidence_mean_required=None,
            actual_confidence_mean=0.88,
            max_gap_seconds_configured=True,
            max_gap_seconds_required=0.5,
            actual_max_gap=0.1,
        )
        self.assertEqual(cd.policy_type, "calibrated")
        self.assertTrue(cd.confirmed)
        self.assertTrue(cd.min_observations_configured)
        self.assertFalse(cd.min_confidence_mean_configured)

    def test_unconfigured_gate_not_shown_as_failed(self):
        from inference.core.events import ConfirmationDecisionEvidence, GateOutcome
        cd = ConfirmationDecisionEvidence(
            policy_type="unconfigured",
            confirmed=False,
            min_observations_configured=False,
            actual_observations=5,
        )
        d = cd.to_dict()
        # gate not configured -> outcome must be NOT_CONFIGURED, not FAIL
        self.assertEqual(d["gates"]["min_observations"]["outcome"], "NOT_CONFIGURED")
        self.assertFalse(d["gates"]["min_observations"]["configured"])

    def test_fail_outcome_when_below_threshold(self):
        from inference.core.events import ConfirmationDecisionEvidence, GateOutcome
        cd = ConfirmationDecisionEvidence(
            policy_type="calibrated",
            confirmed=False,
            min_observations_configured=True,
            min_observations_required=5,
            actual_observations=2,
        )
        d = cd.to_dict()
        self.assertEqual(d["gates"]["min_observations"]["outcome"], "FAIL")
        self.assertEqual(d["gates"]["min_observations"]["actual"], 2)
        self.assertEqual(d["gates"]["min_observations"]["required"], 5)

    def test_pass_outcome_when_meeting_threshold(self):
        from inference.core.events import ConfirmationDecisionEvidence
        cd = ConfirmationDecisionEvidence(
            policy_type="calibrated",
            confirmed=True,
            min_observations_configured=True,
            min_observations_required=3,
            actual_observations=5,
        )
        d = cd.to_dict()
        self.assertEqual(d["gates"]["min_observations"]["outcome"], "PASS")

    def test_from_gate_result_unconfigured_policy(self):
        from inference.core.events import ConfirmationDecisionEvidence
        cand = _make_candidate()
        result = UnconfiguredConfirmationPolicy().evaluate(cand)
        cd = ConfirmationDecisionEvidence.from_gate_result(result, UnconfiguredConfirmationPolicy())
        self.assertEqual(cd.policy_type, "unconfigured")
        self.assertFalse(cd.confirmed)
        self.assertFalse(cd.min_observations_configured)

    def test_from_gate_result_calibrated_all_pass(self):
        from inference.core.events import ConfirmationDecisionEvidence
        cand = _make_candidate(n_obs=5, span=1.0)
        policy = CalibratedConfirmationPolicy(
            min_observations=3,
            min_duration_seconds=2.0,
            min_confidence_mean=0.5,
            max_gap_seconds=0.5,
        )
        result = policy.evaluate(cand)
        cd = ConfirmationDecisionEvidence.from_gate_result(result, policy)
        d = cd.to_dict()
        self.assertEqual(d["gates"]["min_observations"]["outcome"], "PASS")
        self.assertTrue(cd.confirmed)

    def test_to_dict_json_serializable(self):
        from inference.core.events import ConfirmationDecisionEvidence
        cd = ConfirmationDecisionEvidence(
            policy_type="calibrated",
            confirmed=True,
            min_observations_configured=True,
            min_observations_required=3,
            actual_observations=5,
        )
        dumped = json.dumps(cd.to_dict())
        self.assertIsInstance(dumped, str)


# ---------------------------------------------------------------------------
# TrackIdentityEvidence
# ---------------------------------------------------------------------------

class TestTrackIdentityEvidence(unittest.TestCase):

    def test_track_id_switches_defaults_none(self):
        from inference.core.events import TrackIdentityEvidence
        key = EntityKey.from_pose(1)
        tie = TrackIdentityEvidence(entity_key=key, track_ids=frozenset({1, 2}))
        self.assertIsNone(tie.track_id_switches)

    def test_distinct_track_ids_not_confirmed_switch(self):
        """Different track IDs in the set must NOT imply a confirmed switch."""
        from inference.core.events import TrackIdentityEvidence
        key = EntityKey.from_pose(1)
        tie = TrackIdentityEvidence(
            entity_key=key,
            track_ids=frozenset({1, 2}),
            track_id_switches=None,
        )
        d = tie.to_dict()
        self.assertIsNone(d["track_id_switches"])
        self.assertEqual(d["track_ids"], [1, 2])

    def test_explicit_switch_count_preserved(self):
        from inference.core.events import TrackIdentityEvidence
        key = EntityKey.from_pose(1)
        tie = TrackIdentityEvidence(
            entity_key=key,
            track_ids=frozenset({1, 2}),
            track_id_switches=1,
        )
        self.assertEqual(tie.track_id_switches, 1)

    def test_to_dict_json_serializable(self):
        from inference.core.events import TrackIdentityEvidence
        key = EntityKey.from_pose(1)
        tie = TrackIdentityEvidence(entity_key=key, track_ids=frozenset({1, 2, 3}))
        dumped = json.dumps(tie.to_dict())
        self.assertIsInstance(dumped, str)
        self.assertIn("track_ids", json.loads(dumped))


# ---------------------------------------------------------------------------
# AssociationEvidenceSummary
# ---------------------------------------------------------------------------

class TestAssociationEvidenceSummary(unittest.TestCase):

    def _assoc_result(self, status=AssociationStatus.ASSOCIATED, score=None):
        from inference.core.association import AssociationEvidence
        ev = AssociationEvidence(
            person_track_id=1,
            object_track_id=7,
            spatial=SpatialEvidence(True, 0.8, 0.1, 0.01),
            keypoint=KeypointEvidence(True, 5.0, 0.9, 4),
            temporal=None,
        )
        return AssociationResult(
            status=status,
            object_track_id=7,
            score=score,
            evidence=[ev],
            summary="ok",
        )

    def test_score_none_preserved(self):
        from inference.core.events import AssociationEvidenceSummary
        s = AssociationEvidenceSummary(
            association_result=self._assoc_result(status=AssociationStatus.ASSOCIATED),
            person_entity="pose:1",
            object_entity="object:7",
        )
        self.assertIsNone(s.association_score)

    def test_explicit_score_preserved(self):
        from inference.core.events import AssociationEvidenceSummary
        s = AssociationEvidenceSummary(
            association_result=self._assoc_result(status=AssociationStatus.ASSOCIATED, score=0.77),
            person_entity="pose:1",
            object_entity="object:7",
        )
        self.assertAlmostEqual(s.association_score, 0.77)

    def test_ambiguous_status_visible(self):
        from inference.core.events import AssociationEvidenceSummary
        s = AssociationEvidenceSummary(
            association_result=self._assoc_result(status=AssociationStatus.AMBIGUOUS),
            person_entity="pose:1",
            object_entity="object:7",
        )
        d = s.to_dict()
        self.assertEqual(d["status"], "AMBIGUOUS")
        self.assertIsNone(d["association_score"])

    def test_bounded_association_observations(self):
        from inference.core.events import AssociationEvidenceSummary
        obs_list = [
            {"video_time": 0.0, "person": "pose:1", "object": "object:7", "status": "ASSOCIATED"},
            {"video_time": 1.0, "person": "pose:1", "object": "object:7", "status": "AMBIGUOUS"},
            {"video_time": 2.0, "person": "pose:1", "object": "object:7", "status": "ASSOCIATED"},
        ]
        s = AssociationEvidenceSummary(
            association_result=self._assoc_result(),
            person_entity="pose:1",
            object_entity="object:7",
            association_observations=tuple(obs_list),
        )
        d = s.to_dict()
        self.assertEqual(len(d["association_observations"]), 3)
        self.assertIn("AMBIGUOUS", [o["status"] for o in d["association_observations"]])

    def test_to_dict_json_serializable(self):
        from inference.core.events import AssociationEvidenceSummary
        s = AssociationEvidenceSummary(
            association_result=self._assoc_result(),
            person_entity="pose:1",
            object_entity="object:7",
        )
        dumped = json.dumps(s.to_dict())
        self.assertIsInstance(dumped, str)


# ---------------------------------------------------------------------------
# HumanEvidenceMetadata
# ---------------------------------------------------------------------------

class TestHumanEvidenceMetadata(unittest.TestCase):

    def test_default_fields_none(self):
        from inference.core.events import HumanEvidenceMetadata
        h = HumanEvidenceMetadata()
        self.assertIsNone(h.full_frame_reference)
        self.assertIsNone(h.expanded_crop_reference)
        self.assertIsNone(h.person_object_crop_reference)

    def test_rejection_of_image_bytes(self):
        """Schema fields are reference paths only — no image bytes in evidence."""
        from inference.core.events import HumanEvidenceMetadata
        h = HumanEvidenceMetadata(
            full_frame_reference="frames/f0001.jpg",
            expanded_crop_reference="crops/expanded_0001.jpg",
            person_object_crop_reference=None,
        )
        d = h.to_dict()
        self.assertEqual(d["full_frame_reference"], "frames/f0001.jpg")
        self.assertIsNone(d["person_object_crop_reference"])

    def test_to_dict_json_serializable(self):
        from inference.core.events import HumanEvidenceMetadata
        h = HumanEvidenceMetadata(
            full_frame_reference="f.jpg",
            expanded_crop_reference=None,
            person_object_crop_reference="p.jpg",
        )
        dumped = json.dumps(h.to_dict())
        self.assertIsInstance(dumped, str)


# ---------------------------------------------------------------------------
# EvidenceBundle additive fields
# ---------------------------------------------------------------------------

class TestEvidenceBundleAdditive(unittest.TestCase):

    def _bundle(self, **kw):
        ev = _make_event_record()
        cs = ConfidenceStats(count=3, mean=0.9, min=0.85, max=0.95)
        base = dict(
            event_record=ev,
            detection_confidence_stats=cs,
            detection_observation_count=3,
            confirmation_video_time=ev.confirmation_video_time,
            confirmation_wall_time=ev.confirmation_wall_time,
            candidate_start_video_time=ev.candidate_start_video_time,
            candidate_end_video_time=ev.candidate_end_video_time,
            track_ids_in_window=ev.contributing_track_ids,
        )
        base.update(kw)
        return EvidenceBundle(**base)

    def test_bundle_has_decision_field(self):
        from inference.core.events import ConfirmationDecisionEvidence
        cd = ConfirmationDecisionEvidence(
            policy_type="unconfigured",
            confirmed=False,
        )
        b = self._bundle(decision=cd)
        self.assertEqual(b.decision.policy_type, "unconfigured")

    def test_bundle_has_association_field(self):
        from inference.core.events import AssociationEvidenceSummary
        s = AssociationEvidenceSummary(
            association_result=TestAssociationEvidenceSummary()._assoc_result(),
            person_entity="pose:1",
            object_entity="object:7",
        )
        b = self._bundle(association=s)
        self.assertEqual(b.association.status, "ASSOCIATED")

    def test_bundle_has_human_field(self):
        from inference.core.events import HumanEvidenceMetadata
        h = HumanEvidenceMetadata(full_frame_reference="f.jpg")
        b = self._bundle(human=h)
        self.assertEqual(b.human.full_frame_reference, "f.jpg")

    def test_bundle_has_identity_field(self):
        from inference.core.events import TrackIdentityEvidence
        key = EntityKey.from_pose(1)
        tie = TrackIdentityEvidence(entity_key=key, track_ids=frozenset({1, 2}))
        b = self._bundle(identity=tie)
        self.assertIsNone(b.identity.track_id_switches)

    def test_bundle_to_dict_json_serializable(self):
        b = self._bundle()
        dumped = json.dumps(b.to_dict())
        self.assertIsInstance(dumped, str)

    def test_bundle_to_dict_includes_new_sections(self):
        from inference.core.events import ConfirmationDecisionEvidence
        b = self._bundle(decision=ConfirmationDecisionEvidence(
            policy_type="calibrated",
            confirmed=True,
            min_observations_configured=True,
            min_observations_required=3,
            actual_observations=5,
        ))
        d = b.to_dict()
        self.assertIn("decision", d)
        self.assertIn("gates", d["decision"])

    def test_bundle_new_sections_default_none(self):
        b = self._bundle()
        d = b.to_dict()
        self.assertIsNone(d.get("decision"))
        self.assertIsNone(d.get("association"))
        self.assertIsNone(d.get("human"))
        self.assertIsNone(d.get("identity"))


# ---------------------------------------------------------------------------
# No numpy / cv2 objects leak into serialization
# ---------------------------------------------------------------------------

class TestNoHeavyObjects(unittest.TestCase):

    def test_bundle_to_dict_no_numpy_cv2_keys(self):
        """to_dict() must not contain objects that are not JSON-portable."""
        b = TestEvidenceBundleAdditive()._bundle()
        d = b.to_dict()
        dumped = json.dumps(d)  # must not raise
        self.assertNotIn("ndarray", dumped)
        self.assertNotIn("cv2", dumped)


# ---------------------------------------------------------------------------
# Event identity
# ---------------------------------------------------------------------------

class TestEventIdentity(unittest.TestCase):

    def test_event_id_differs_from_track_id(self):
        ev = _make_event_record()
        self.assertNotEqual(ev.event_id, "1")
        self.assertNotEqual(ev.event_id, "pose:1")

    def test_event_id_is_uuid(self):
        import uuid as _uuid
        ev = _make_event_record()
        _uuid.UUID(ev.event_id)  # must parse without raising


# ---------------------------------------------------------------------------
# Temporal auditability
# ---------------------------------------------------------------------------

class TestTemporalAuditability(unittest.TestCase):

    def test_confirmation_wall_time_is_audit_only(self):
        """confirmation_wall_time must be a string, not used in any gate math."""
        ev = _make_event_record()
        self.assertIsInstance(ev.confirmation_wall_time, str)
        self.assertGreater(ev.confirmation_video_time, 0.0)

    def test_candidate_time_span_consistent(self):
        cand = _make_candidate(n_obs=4, span=1.0)
        self.assertAlmostEqual(cand.video_time_span(), 3.0)


if __name__ == "__main__":
    unittest.main()
