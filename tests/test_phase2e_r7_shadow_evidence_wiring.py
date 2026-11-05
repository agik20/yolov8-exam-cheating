"""Phase 2E-R7 — Shadow evidence bundle wiring into ShadowLifecycleManager.

Covers:
- ShadowEventRecord.evidence_bundle is an Optional[EvidenceBundle] field
- build_evidence_bundle is called on confirmation; attaches to the record
- bundle entity_key / class_name match the confirming candidate
- ConfirmationDecisionEvidence present when a calibrated policy confirms
- EventConfidenceEvidence present, carries observation_count, continuity
- event_confidence is None (no fabricated score)
- association is None when no associator was supplied to the manager
- to_dict() includes evidence_bundle (nested dict when populated, None otherwise)
- snapshot() serializes evidence bundles
- two independent confirmations produce two independent bundles
- no production file I/O occurs during evidence construction
- ShadowEventRecord without a bundle (backward compat) still serializes
- cooldown transition does NOT mutate an already-recorded bundle
- unconfirmed candidates produce no ShadowEventRecord (and no bundle)
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from inference.core.events import (
    EntityKey,
    ModelSource,
    Observation,
    ConfidenceStats,
    EventRecord,
    Candidate,
    CandidateState,
    BoundedObservationBuffer,
    EvidenceBundle,
    ConfirmationDecisionEvidence,
    EventConfidenceEvidence,
)
from inference.core.candidate_layer import CandidateLayer
from inference.core.confirmation import (
    UnconfiguredConfirmationPolicy,
    CalibratedConfirmationPolicy,
    ConfirmationGateResult,
)
from inference.core.cooldown import CooldownPolicy, CalibratedCooldownPolicy
from inference.core.shadow_lifecycle import (
    ShadowLifecycleManager,
    ShadowEventRecord,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_obs(frame: int, track_id: int = 1, conf: float = 0.9, fps: float = 30.0) -> Observation:
    return Observation(
        frame_index=frame,
        video_time=frame / fps,
        class_name="provide-object",
        confidence=conf,
        bbox=(0, 0, 100, 100),
        source=ModelSource.POSE,
        track_id=track_id,
    )


def _make_layer(src_fps: float = 30.0, maxlen: int = 8) -> CandidateLayer:
    return CandidateLayer(src_fps=src_fps, observation_buffer_maxlen=maxlen)


def _fill_candidates(layer: CandidateLayer, n_obs: int = 5) -> None:
    """Add n_obs observations for one entity so a calibrated gate confirms."""
    for f in range(n_obs):
        layer.record_observation(_make_obs(f))


def _make_manager(
    layer: CandidateLayer | None = None,
    policy: CalibratedConfirmationPolicy | UnconfiguredConfirmationPolicy | None = None,
    cooldown: CooldownPolicy | None = None,
) -> ShadowLifecycleManager:
    layer = layer or _make_layer()
    policy = policy or CalibratedConfirmationPolicy(min_observations=3)
    cooldown = cooldown or CalibratedCooldownPolicy(cooldown_seconds=10.0)
    return ShadowLifecycleManager(layer=layer, confirmation_policy=policy, cooldown_policy=cooldown)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestR7ShadowEvidenceWiring(unittest.TestCase):

    # -- evidence_bundle field exists on ShadowEventRecord --

    def test_shadow_event_record_has_evidence_bundle_field(self):
        """ShadowEventRecord must expose an evidence_bundle attribute defaulting to None."""
        er = EventRecord(
            event_id="test-uuid",
            entity_key=EntityKey.from_pose(1),
            class_name="provide-object",
            observation_count=3,
            confidence_stats=ConfidenceStats(count=3, mean=0.9, min=0.9, max=0.9),
            event_confidence=None,
            association_score=None,
            contributing_track_ids=frozenset({1}),
            confirmation_video_time=0.1,
            confirmation_wall_time="2026-01-01T00:00:00",
            candidate_start_video_time=0.0,
            candidate_end_video_time=0.1,
        )
        gate = ConfirmationGateResult(
            entity_key=er.entity_key,
            candidate_class=er.class_name,
            confirmed=True,
            reason="test",
        )
        rec = ShadowEventRecord(event=er, gate_result=gate, recorded_at_video_time=0.1)
        self.assertIsNone(rec.evidence_bundle)

    # -- bundle is attached when a calibrated policy confirms --

    def test_confirmed_shadow_event_has_bundle(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=5)
        result = mgr.process_observation(_make_obs(5))
        self.assertIsNotNone(result, "expected a confirmed shadow event")
        self.assertIsNotNone(result.evidence_bundle, "expected an evidence bundle on a confirmed event")
        self.assertIsInstance(result.evidence_bundle, EvidenceBundle)

    # -- bundle entity_key and class_name match the confirming candidate --

    def test_bundle_entity_key_matches_candidate(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=5)
        result = mgr.process_observation(_make_obs(5))
        self.assertIsNotNone(result)
        bundle = result.evidence_bundle
        self.assertEqual(bundle.event_record.entity_key, EntityKey.from_pose(1))
        self.assertEqual(bundle.event_record.class_name, "provide-object")

    # -- ConfirmationDecisionEvidence present in the bundle when calibrated policy used --

    def test_confirmation_decision_evidence_present_with_calibrated_policy(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=5)
        result = mgr.process_observation(_make_obs(5))
        self.assertIsNotNone(result)
        bundle = result.evidence_bundle
        self.assertIsNotNone(bundle.decision, "decision evidence missing for a calibrated confirmation")
        self.assertIsInstance(bundle.decision, ConfirmationDecisionEvidence)
        self.assertTrue(bundle.decision.confirmed)

    # -- ConfirmationDecisionEvidence reflects correct gate outcomes --

    def test_confirmation_decision_evidence_gate_outcomes(self):
        policy = CalibratedConfirmationPolicy(
            min_observations=3,
            min_duration_seconds=0.01,
            min_confidence_mean=0.0,
            max_gap_seconds=999.0,
        )
        mgr = _make_manager(policy=policy)
        layer = mgr._layer
        _fill_candidates(layer, n_obs=5)
        result = mgr.process_observation(_make_obs(5))
        self.assertIsNotNone(result)
        decision = result.evidence_bundle.decision
        self.assertIsNotNone(decision)
        d = decision.to_dict()
        self.assertEqual(d["gates"]["min_observations"]["outcome"], "PASS")
        self.assertEqual(d["gates"]["min_duration"]["outcome"], "PASS")
        self.assertEqual(d["gates"]["min_confidence"]["outcome"], "PASS")
        # candidate.max_gap_seconds is None (no gaps recorded) → UNAVAILABLE, not PASS
        self.assertEqual(d["gates"]["max_gap"]["outcome"], "UNAVAILABLE")

    # -- EventConfidenceEvidence is present and carries observation_count --

    def test_event_confidence_evidence_present(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=4)
        result = mgr.process_observation(_make_obs(4))
        self.assertIsNotNone(result)
        ev = result.evidence_bundle.event_confidence_evidence
        self.assertIsNotNone(ev, "event_confidence_evidence missing from bundle")
        self.assertIsInstance(ev, EventConfidenceEvidence)
        # _fill_candidates adds 4 obs; process_observation adds a 5th → total 5
        self.assertEqual(ev.observation_count, 5)

    # -- EventConfidenceEvidence.continuity is populated from candidate R4 state --

    def test_event_confidence_evidence_continuity_populated(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=5)
        result = mgr.process_observation(_make_obs(5))
        self.assertIsNotNone(result)
        ev = result.evidence_bundle.event_confidence_evidence
        self.assertIsNotNone(ev)
        self.assertIsNotNone(ev.continuity)
        # _fill_candidates adds 5 obs; process_observation adds a 6th → total 6
        self.assertEqual(ev.continuity.observation_count, 6)

    # -- event_confidence is None (no fabricated score) --

    def test_event_confidence_is_none(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=5)
        result = mgr.process_observation(_make_obs(5))
        self.assertIsNotNone(result)
        ev = result.evidence_bundle.event_confidence_evidence
        self.assertIsNone(ev.event_confidence, "event_confidence must not be fabricated")
        self.assertEqual(ev.scoring_status, "not_calibrated")

    # -- association is None when no associator was supplied --

    def test_association_is_none_when_no_associator(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=5)
        result = mgr.process_observation(_make_obs(5))
        self.assertIsNotNone(result)
        self.assertIsNone(
            result.evidence_bundle.association,
            "association must be None when no association_policy was supplied",
        )

    # -- bundle detection_confidence_stats matches candidate confidence_stats --

    def test_bundle_detection_confidence_stats(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=3)
        result = mgr.process_observation(_make_obs(3))
        self.assertIsNotNone(result)
        bundle = result.evidence_bundle
        cs = bundle.detection_confidence_stats
        # _fill_candidates adds 3 obs; process_observation adds a 4th → total 4
        self.assertEqual(cs.count, 4)
        self.assertAlmostEqual(cs.mean, 0.9, places=6)

    # -- bundle detection_observation_count matches candidate observation count --

    def test_bundle_detection_observation_count(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=7)
        result = mgr.process_observation(_make_obs(7))
        self.assertIsNotNone(result)
        # _fill_candidates adds 7 obs; process_observation adds an 8th → total 8
        self.assertEqual(result.evidence_bundle.detection_observation_count, 8)

    # -- bundle candidate_start/end video times match candidate times --

    def test_bundle_candidate_video_times(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=5)
        result = mgr.process_observation(_make_obs(5))
        self.assertIsNotNone(result)
        bundle = result.evidence_bundle
        cand = layer.get_candidate(EntityKey.from_pose(1), "provide-object")
        self.assertEqual(bundle.candidate_start_video_time, cand.start_video_time)
        self.assertEqual(bundle.candidate_end_video_time, cand.latest_video_time)

    # -- bundle track_ids_in_window equals candidate observed_track_ids --

    def test_bundle_track_ids_in_window(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=5)
        result = mgr.process_observation(_make_obs(5))
        self.assertIsNotNone(result)
        bundle = result.evidence_bundle
        cand = layer.get_candidate(EntityKey.from_pose(1), "provide-object")
        self.assertEqual(bundle.track_ids_in_window, cand.observed_track_ids)

    # -- cooldown transition does NOT mutate an already-recorded bundle --

    def test_cooldown_does_not_mutate_recorded_bundle(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=5)
        result = mgr.process_observation(_make_obs(5))
        self.assertIsNotNone(result)
        snapshot = json.dumps(result.evidence_bundle.to_dict())

        # Advance video time past the cooldown window to trigger structural reset
        for f in range(5, 5 + 300):
            mgr.process_observation(_make_obs(f))

        # The already-recorded bundle must be unchanged
        self.assertEqual(json.dumps(result.evidence_bundle.to_dict()), snapshot)

    # -- unconfirmed candidates produce no ShadowEventRecord and no bundle --

    def test_unconfirmed_produces_no_bundle(self):
        # Unconfigured policy never confirms
        mgr = _make_manager(policy=UnconfiguredConfirmationPolicy())
        layer = mgr._layer
        _fill_candidates(layer, n_obs=50)
        result = mgr.process_observation(_make_obs(50))
        self.assertIsNone(result, "UnconfiguredConfirmationPolicy must not confirm")

    # -- to_dict() includes evidence_bundle key --

    def test_to_dict_includes_evidence_bundle_key(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=5)
        result = mgr.process_observation(_make_obs(5))
        self.assertIsNotNone(result)
        d = result.to_dict()
        self.assertIn("evidence_bundle", d)
        self.assertIsNotNone(d["evidence_bundle"])
        self.assertIsInstance(d["evidence_bundle"], dict)

    # -- to_dict() with evidence_bundle=None serializes cleanly --

    def test_to_dict_bundle_none_serializes(self):
        er = EventRecord(
            event_id="uuid-1",
            entity_key=EntityKey.from_pose(1),
            class_name="provide-object",
            observation_count=1,
            confidence_stats=ConfidenceStats(count=1, mean=0.9, min=0.9, max=0.9),
            event_confidence=None,
            association_score=None,
            contributing_track_ids=frozenset({1}),
            confirmation_video_time=0.0,
            confirmation_wall_time="2026-01-01T00:00:00",
            candidate_start_video_time=0.0,
            candidate_end_video_time=0.0,
        )
        gate = ConfirmationGateResult(
            entity_key=er.entity_key, candidate_class="provide-object",
            confirmed=True, reason="test",
        )
        rec = ShadowEventRecord(event=er, gate_result=gate, recorded_at_video_time=0.0)
        d = rec.to_dict()
        self.assertIsNone(d.get("evidence_bundle"))
        # Must be JSON-serializable
        json.dumps(d)

    # -- snapshot() serializes evidence bundles in shadow_events --

    def test_snapshot_serializes_evidence_bundles(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=5)
        mgr.process_observation(_make_obs(5))
        snap = mgr.snapshot()
        events = snap["shadow_events"]
        self.assertEqual(len(events), 1)
        self.assertIn("evidence_bundle", events[0])
        self.assertIsNotNone(events[0]["evidence_bundle"])
        # The bundle in the snapshot must be JSON-serializable
        json.dumps(events[0])

    # -- no production file I/O during evidence construction --

    @patch("builtins.open", side_effect=OSError("unexpected file open"))
    def test_no_file_io_during_evidence_construction(self, _mock_open):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=5)
        # Must not raise; no file opens
        result = mgr.process_observation(_make_obs(5))
        self.assertIsNotNone(result)

    # -- backward compatibility: ShadowEventRecord without bundle still works --

    def test_backward_compat_no_bundle(self):
        er = EventRecord(
            event_id="uuid-2",
            entity_key=EntityKey.from_pose(2),
            class_name="provide-object",
            observation_count=1,
            confidence_stats=ConfidenceStats(count=1, mean=0.8, min=0.8, max=0.8),
            event_confidence=None,
            association_score=None,
            contributing_track_ids=frozenset({2}),
            confirmation_video_time=0.1,
            confirmation_wall_time="2026-01-01T00:00:00",
            candidate_start_video_time=0.0,
            candidate_end_video_time=0.1,
        )
        gate = ConfirmationGateResult(
            entity_key=er.entity_key, candidate_class="provide-object",
            confirmed=True, reason="test",
        )
        rec = ShadowEventRecord(event=er, gate_result=gate, recorded_at_video_time=0.1)
        self.assertIsNone(rec.evidence_bundle)
        d = rec.to_dict()
        self.assertIn("evidence_bundle", d)
        self.assertIsNone(d["evidence_bundle"])

    # -- two independent confirmations produce two independent bundles --

    def test_two_independent_bundles_no_cross_contamination(self):
        mgr = _make_manager()
        layer = mgr._layer
        _fill_candidates(layer, n_obs=5)
        r1 = mgr.process_observation(_make_obs(5))
        self.assertIsNotNone(r1)

        # Second entity to get an independent candidate
        for f in range(5, 10):
            layer.record_observation(_make_obs(f, track_id=2))
        r2 = mgr.process_observation(_make_obs(10, track_id=2))
        self.assertIsNotNone(r2)

        self.assertIsNotNone(r1.evidence_bundle)
        self.assertIsNotNone(r2.evidence_bundle)
        # Different entity keys
        self.assertEqual(r1.evidence_bundle.event_record.entity_key, EntityKey.from_pose(1))
        self.assertEqual(r2.evidence_bundle.event_record.entity_key, EntityKey.from_pose(2))


if __name__ == "__main__":
    unittest.main()
