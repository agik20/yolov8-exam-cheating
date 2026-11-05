"""Phase 2E-R6 — EventConfidenceEvidence (evidence-only, no score).

Covers:
- EventConfidenceEvidence fields are preserved
- missing/unavailable components remain explicit (None)
- no score is fabricated (event_confidence stays None)
- scoring_status is always "not_calibrated"
- serialization is backward compatible
- existing confirmation behavior is unchanged
"""
from __future__ import annotations

import json
import pytest

from inference.core.events import (
    Candidate,
    CandidateState,
    EntityKey,
    ModelSource,
    Observation,
    BoundedObservationBuffer,
    ConfidenceStats,
    TrackContinuity,
    ConfirmationDecisionEvidence,
)
from inference.core.candidate_layer import CandidateLayer, _compute_continuity
from inference.core.confirmation import (
    UnconfiguredConfirmationPolicy,
    CalibratedConfirmationPolicy,
)
from inference.core.association import AssociationStatus, AssociationResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_obs(frame: int, track_id: int, conf: float = 0.9, fps: float = 30.0) -> Observation:
    return Observation(
        frame_index=frame,
        video_time=frame / fps,
        class_name="provide-object",
        confidence=conf,
        bbox=(0, 0, 100, 100),
        source=ModelSource.POSE,
        track_id=track_id,
    )


def _make_candidate(n_obs: int = 5) -> Candidate:
    layer = CandidateLayer(src_fps=30.0, observation_buffer_maxlen=8)
    for f in range(n_obs):
        layer.record_observation(_make_obs(f, 1))
    return list(layer.all_candidates().values())[0]


# ---------------------------------------------------------------------------
# TestR6EventConfidenceEvidence
# ---------------------------------------------------------------------------

class TestR6EventConfidenceEvidence:

    def test_module_export(self):
        from inference.core.events import EventConfidenceEvidence  # noqa: F401
        assert True

    # -- evidence fields preserved --

    def test_observation_count_preserved(self):
        from inference.core.events import EventConfidenceEvidence
        cand = _make_candidate(n_obs=5)
        ev = EventConfidenceEvidence.from_candidate(cand)
        assert ev.observation_count == 5

    def test_confidence_stats_preserved(self):
        from inference.core.events import EventConfidenceEvidence
        cand = _make_candidate(n_obs=3)
        ev = EventConfidenceEvidence.from_candidate(cand)
        assert ev.confidence_stats.mean == pytest.approx(0.9)
        assert ev.confidence_stats.count == 3

    def test_video_time_span_preserved(self):
        from inference.core.events import EventConfidenceEvidence
        cand = _make_candidate(n_obs=5)  # frames 0-4 at 30fps → 4/30 s
        ev = EventConfidenceEvidence.from_candidate(cand)
        assert ev.video_time_span == pytest.approx(4 / 30.0)

    def test_continuity_preserved(self):
        from inference.core.events import EventConfidenceEvidence
        cand = _make_candidate(n_obs=5)
        ev = EventConfidenceEvidence.from_candidate(cand)
        assert isinstance(ev.continuity, TrackContinuity)
        assert ev.continuity.observation_count == 5
        assert ev.continuity.primary_track_id == 1

    # -- missing components remain explicit --

    def test_association_status_none_when_not_supplied(self):
        from inference.core.events import EventConfidenceEvidence
        cand = _make_candidate()
        ev = EventConfidenceEvidence.from_candidate(cand)
        assert ev.association_status is None

    def test_association_score_none_when_not_supplied(self):
        from inference.core.events import EventConfidenceEvidence
        cand = _make_candidate()
        ev = EventConfidenceEvidence.from_candidate(cand)
        assert ev.association_score is None

    def test_confirmation_decision_none_when_not_supplied(self):
        from inference.core.events import EventConfidenceEvidence
        cand = _make_candidate()
        ev = EventConfidenceEvidence.from_candidate(cand)
        assert ev.confirmation_decision is None

    # -- no score fabricated --

    def test_event_confidence_always_none_without_policy(self):
        from inference.core.events import EventConfidenceEvidence
        cand = _make_candidate()
        ev = EventConfidenceEvidence.from_candidate(cand)
        assert ev.event_confidence is None

    def test_scoring_status_is_not_calibrated(self):
        from inference.core.events import EventConfidenceEvidence
        cand = _make_candidate()
        ev = EventConfidenceEvidence.from_candidate(cand)
        assert ev.scoring_status == "not_calibrated"

    # -- with association result supplied --

    def test_association_status_populated_when_result_supplied(self):
        from inference.core.events import EventConfidenceEvidence
        cand = _make_candidate()
        assoc = AssociationResult(
            status=AssociationStatus.ASSOCIATED,
            object_track_id=1,
        )
        ev = EventConfidenceEvidence.from_candidate(cand, association_result=assoc)
        assert ev.association_status == "ASSOCIATED"
        assert ev.association_score is None  # score not fabricated

    def test_association_ambiguous_status_preserved(self):
        from inference.core.events import EventConfidenceEvidence
        cand = _make_candidate()
        assoc = AssociationResult(
            status=AssociationStatus.AMBIGUOUS,
            object_track_id=1,
        )
        ev = EventConfidenceEvidence.from_candidate(cand, association_result=assoc)
        assert ev.association_status == "AMBIGUOUS"

    # -- with confirmation gate result supplied --

    def test_confirmation_decision_populated_when_result_supplied(self):
        from inference.core.events import EventConfidenceEvidence, ConfirmationDecisionEvidence
        cand = _make_candidate()
        policy = CalibratedConfirmationPolicy(min_observations=3)
        gate_result = policy.evaluate(cand)
        cd = ConfirmationDecisionEvidence.from_gate_result(gate_result, policy)
        ev = EventConfidenceEvidence.from_candidate(cand, confirmation_decision=cd)
        assert isinstance(ev.confirmation_decision, ConfirmationDecisionEvidence)
        assert ev.confirmation_decision.confirmed is True
        assert ev.event_confidence is None  # still None even with calibrated gate

    # -- serialization --

    def test_to_dict_json_serializable(self):
        from inference.core.events import EventConfidenceEvidence
        cand = _make_candidate(n_obs=4)
        ev = EventConfidenceEvidence.from_candidate(cand)
        d = ev.to_dict()
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["event_confidence"] is None
        assert parsed["scoring_status"] == "not_calibrated"
        assert parsed["observation_count"] == 4
        assert parsed["continuity"]["possible_id_switch_count"] == 0

    def test_to_dict_includes_continuity_block(self):
        from inference.core.events import EventConfidenceEvidence
        cand = _make_candidate(n_obs=3)
        ev = EventConfidenceEvidence.from_candidate(cand)
        d = ev.to_dict()
        assert "continuity" in d
        assert d["continuity"]["track_ids"] == [1]

    def test_to_dict_association_fields_present_even_when_none(self):
        from inference.core.events import EventConfidenceEvidence
        cand = _make_candidate()
        ev = EventConfidenceEvidence.from_candidate(cand)
        d = ev.to_dict()
        assert "association_status" in d
        assert "association_score" in d
        assert d["association_status"] is None
        assert d["association_score"] is None

    # -- backward compatibility --

    def test_evidence_bundle_without_r6_field_still_works(self):
        """EvidenceBundle constructed without event_confidence_evidence has it as None."""
        from inference.core.events import EvidenceBundle, EventRecord
        cand = _make_candidate()
        er = EventRecord.from_candidate(cand, 0.1, "2026-01-01T00:00:00")
        bundle = EvidenceBundle(
            event_record=er,
            detection_confidence_stats=ConfidenceStats(count=5, mean=0.9, min=0.9, max=0.9),
            detection_observation_count=5,
            confirmation_video_time=0.1,
            confirmation_wall_time="2026-01-01T00:00:00",
            candidate_start_video_time=0.0,
            candidate_end_video_time=0.1,
            track_ids_in_window=frozenset({1}),
        )
        assert bundle.event_confidence_evidence is None

    def test_evidence_bundle_with_r6_field_serialized(self):
        from inference.core.events import EvidenceBundle, EventRecord, EventConfidenceEvidence
        cand = _make_candidate(n_obs=4)
        er = EventRecord.from_candidate(cand, 0.1, "2026-01-01T00:00:00")
        ev = EventConfidenceEvidence.from_candidate(cand)
        bundle = EvidenceBundle(
            event_record=er,
            detection_confidence_stats=ConfidenceStats(count=4, mean=0.9, min=0.9, max=0.9),
            detection_observation_count=4,
            confirmation_video_time=0.1,
            confirmation_wall_time="2026-01-01T00:00:00",
            candidate_start_video_time=0.0,
            candidate_end_video_time=0.1,
            track_ids_in_window=frozenset({1}),
            event_confidence_evidence=ev,
        )
        d = bundle.to_dict()
        assert "event_confidence_evidence" in d
        assert d["event_confidence_evidence"]["event_confidence"] is None
        assert d["event_confidence_evidence"]["scoring_status"] == "not_calibrated"

    # -- confirmation behavior unchanged --

    def test_unconfigured_policy_confirmation_unchanged(self):
        cand = _make_candidate(n_obs=10)
        result = UnconfiguredConfirmationPolicy().evaluate(cand)
        assert result.confirmed is False
        assert result.reason == "confirmation_policy_unconfigured"

    def test_calibrated_policy_confirmation_unchanged(self):
        cand = _make_candidate(n_obs=5)
        policy = CalibratedConfirmationPolicy(min_observations=3)
        result = policy.evaluate(cand)
        assert result.confirmed is True
        assert result.event_confidence is None  # still None
