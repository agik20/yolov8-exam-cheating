"""Phase 2B.1 — Unit tests for event data structures.

Covers:
- EntityKey uniqueness across model namespaces
- Candidate construction and observation accumulation
- Observation field retention and serializability
- EventRecord construction (direct + from_candidate)
- EvidenceBundle three-layer separation
- CandidateState representation (all five states)
- BoundedObservationBuffer FIFO behavior and capacity enforcement

Run:  python -m pytest tests/test_phase2b1_event_structures.py -v
"""
import json
import os
import sys
import unittest
import uuid
from datetime import datetime, timezone

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
    ConfidenceStats,
    Candidate,
    EventRecord,
    EvidenceBundle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_observation(
    frame_index=10,
    video_time=0.333,
    class_name="provide-object",
    confidence=0.72,
    bbox=(100, 200, 400, 500),
    source=ModelSource.POSE,
    track_id=3,
) -> Observation:
    return Observation(
        frame_index=frame_index,
        video_time=video_time,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        source=source,
        track_id=track_id,
    )


# ---------------------------------------------------------------------------
# EntityKey uniqueness
# ---------------------------------------------------------------------------

class TestEntityKeyUniqueness(unittest.TestCase):
    """pose:3 must NOT equal object:3 — two independent ID spaces."""

    def test_pose_3_neq_object_3(self):
        pose = EntityKey.from_pose(3)
        obj = EntityKey.from_object(3)
        self.assertNotEqual(pose, obj,
                            "pose track_id 3 and object track_id 3 are distinct entities")
        self.assertNotEqual(pose.as_string(), obj.as_string())

    def test_same_namespace_same_id_equality(self):
        self.assertEqual(EntityKey.from_pose(3), EntityKey.from_pose(3))
        self.assertEqual(EntityKey.from_object(17), EntityKey.from_object(17))

    def test_different_track_id_different_key(self):
        self.assertNotEqual(EntityKey.from_pose(3), EntityKey.from_pose(4))
        self.assertNotEqual(EntityKey.from_object(3), EntityKey.from_object(4))

    def test_as_string_format(self):
        self.assertEqual(EntityKey.from_pose(3).as_string(), "pose:3")
        self.assertEqual(EntityKey.from_object(17).as_string(), "object:17")

    def test_hashable_and_deterministic(self):
        k1 = EntityKey.from_pose(3)
        k2 = EntityKey.from_pose(3)
        self.assertEqual(hash(k1), hash(k2))
        self.assertEqual({k1, k2}, {k1})

    def test_type_validation(self):
        with self.assertRaises(TypeError):
            EntityKey(source="not_an_enum", track_id=1)
        with self.assertRaises(TypeError):
            EntityKey(source=ModelSource.POSE, track_id=3.14)


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------

class TestCandidateConstruction(unittest.TestCase):

    def test_minimal_candidate(self):
        key = EntityKey.from_pose(3)
        c = Candidate(entity_key=key, candidate_class="provide-object")
        self.assertEqual(c.entity_key, key)
        self.assertEqual(c.candidate_class, "provide-object")
        self.assertEqual(c.state, CandidateState.CANDIDATE)
        self.assertIsNone(c.start_video_time,
                          "start_video_time must be None until first observation")
        self.assertIsNone(c.latest_video_time)
        self.assertIsNone(c.event_confidence)
        self.assertIsNone(c.association_score)
        self.assertIsNone(c.class_consistency)
        self.assertIsNone(c.gap_count)
        self.assertIsNone(c.max_gap_seconds)
        self.assertEqual(c.observation_count(), 0)

    def test_candidate_with_custom_buffer_maxlen(self):
        key = EntityKey.from_object(5)
        c = Candidate(entity_key=key, candidate_class="smartphone",
                     observations=BoundedObservationBuffer(maxlen=8))
        self.assertEqual(c.observations.maxlen, 8)

    def test_add_observation_populates_times_and_track_ids(self):
        key = EntityKey.from_pose(3)
        c = Candidate(entity_key=key, candidate_class="provide-object")
        obs = _make_observation(frame_index=10, video_time=0.333, track_id=3)
        c.add_observation(obs)

        self.assertEqual(c.start_video_time, 0.333)
        self.assertEqual(c.latest_video_time, 0.333)
        self.assertIn(3, c.observed_track_ids)
        self.assertEqual(c.observation_count(), 1)

    def test_add_observation_id_switch_records_both_ids(self):
        key = EntityKey.from_pose(3)
        c = Candidate(entity_key=key, candidate_class="provide-object")
        c.add_observation(_make_observation(frame_index=10, video_time=0.333, track_id=3))
        c.add_observation(_make_observation(frame_index=20, video_time=0.667, track_id=47))
        self.assertEqual(c.observed_track_ids, frozenset({3, 47}),
                         "an ID switch inside the window must record both IDs")

    def test_candidate_to_dict_is_json_serializable(self):
        key = EntityKey.from_pose(3)
        c = Candidate(entity_key=key, candidate_class="provide-object")
        c.add_observation(_make_observation())
        dumped = json.dumps(c.to_dict())
        self.assertIsInstance(dumped, str)


# ---------------------------------------------------------------------------
# Observation representation
# ---------------------------------------------------------------------------

class TestObservationRepresentation(unittest.TestCase):

    def test_all_fields_retained(self):
        obs = _make_observation()
        self.assertEqual(obs.frame_index, 10)
        self.assertEqual(obs.video_time, 0.333)
        self.assertEqual(obs.class_name, "provide-object")
        self.assertEqual(obs.confidence, 0.72)
        self.assertEqual(obs.bbox, (100, 200, 400, 500))
        self.assertEqual(obs.source, ModelSource.POSE)
        self.assertEqual(obs.track_id, 3)

    def test_to_dict_keys_and_json(self):
        obs = _make_observation()
        d = obs.to_dict()
        for field_name in ("frame_index", "video_time", "class_name",
                           "confidence", "bbox", "source", "track_id"):
            self.assertIn(field_name, d)
        json.dumps(d)  # must not raise

    def test_observation_is_immutable(self):
        obs = _make_observation()
        with self.assertRaises(AttributeError):
            obs.frame_index = 999  # frozen dataclass: assignment raises AttributeError


# ---------------------------------------------------------------------------
# EventRecord construction
# ---------------------------------------------------------------------------

class TestEventRecordConstruction(unittest.TestCase):

    @staticmethod
    def _make_populated_candidate() -> Candidate:
        key = EntityKey.from_pose(3)
        c = Candidate(entity_key=key, candidate_class="provide-object")
        c.add_observation(_make_observation(frame_index=10, video_time=0.333, track_id=3))
        c.add_observation(_make_observation(frame_index=20, video_time=0.667, track_id=3))
        return c

    def test_direct_construction(self):
        key = EntityKey.from_pose(3)
        stats = ConfidenceStats(count=2, mean=0.72, min=0.65, max=0.79)
        rec = EventRecord(
            event_id="evt-001",
            entity_key=key,
            class_name="provide-object",
            confirmation_video_time=1.0,
            confirmation_wall_time="2026-10-10T12:00:00+00:00",
            candidate_start_video_time=0.333,
            candidate_end_video_time=0.667,
            observation_count=2,
            confidence_stats=stats,
            event_confidence=None,
            association_score=None,
            contributing_track_ids=frozenset({3}),
        )
        self.assertEqual(rec.event_id, "evt-001")
        self.assertEqual(rec.entity_key, key)
        self.assertIsInstance(rec.contributing_track_ids, frozenset)

    def test_from_candidate_factory(self):
        c = self._make_populated_candidate()
        wall_time = datetime.now(timezone.utc).isoformat()
        rec = EventRecord.from_candidate(c, confirmation_video_time=0.667,
                                         confirmation_wall_time=wall_time)
        self.assertEqual(rec.entity_key, c.entity_key)
        self.assertEqual(rec.class_name, "provide-object")
        self.assertEqual(rec.observation_count, 2)
        self.assertEqual(rec.contributing_track_ids, c.observed_track_ids)
        self.assertEqual(rec.confidence_stats.count, 2)
        # event_id must be a valid UUID
        uuid.UUID(rec.event_id)  # raises ValueError if not valid

    def test_event_record_to_dict_is_json_serializable(self):
        c = self._make_populated_candidate()
        rec = EventRecord.from_candidate(c, 0.667, "2026-10-10T12:00:00+00:00")
        json.dumps(rec.to_dict())


# ---------------------------------------------------------------------------
# EvidenceBundle structure
# ---------------------------------------------------------------------------

class TestEvidenceBundleStructure(unittest.TestCase):

    @staticmethod
    def _make_bundle() -> EvidenceBundle:
        key = EntityKey.from_pose(3)
        stats = ConfidenceStats(count=5, mean=0.75, min=0.55, max=0.91)
        rec = EventRecord(
            event_id="evt-bundle-test",
            entity_key=key,
            class_name="smartphone",
            confirmation_video_time=2.0,
            confirmation_wall_time="2026-10-10T12:00:00+00:00",
            candidate_start_video_time=0.5,
            candidate_end_video_time=2.0,
            observation_count=5,
            confidence_stats=stats,
            event_confidence=None,
            association_score=None,
            contributing_track_ids=frozenset({3, 47}),
        )
        return EvidenceBundle(
            event_record=rec,
            detection_confidence_stats=stats,
            detection_observation_count=5,
            confirmation_video_time=2.0,
            confirmation_wall_time="2026-10-10T12:00:00+00:00",
            candidate_start_video_time=0.5,
            candidate_end_video_time=2.0,
            track_ids_in_window=frozenset({3, 47}),
        )

    def test_three_layers_structurally_separate(self):
        b = self._make_bundle()
        # detection layer
        self.assertEqual(b.detection_observation_count, 5)
        self.assertEqual(b.detection_confidence_stats.count, 5)
        # decision layer
        self.assertEqual(b.track_ids_in_window, frozenset({3, 47}))
        self.assertEqual(b.confirmation_video_time, 2.0)
        self.assertEqual(b.confirmation_terms_met, (),
                          "no confirmation terms populated in 2B.1")
        self.assertIsNone(b.id_switch_count)
        # human layer — all not-yet-generated
        self.assertIsNone(b.annotated_frame_path)
        self.assertIsNone(b.crop_path)
        self.assertIsNone(b.person_object_pair_crop_path)
        self.assertIsNone(b.temporal_summary)
        self.assertIsNone(b.human_verdict_line)

    def test_to_dict_includes_all_three_layers(self):
        b = self._make_bundle()
        d = b.to_dict()
        # detection
        self.assertIn("detection_confidence_stats", d)
        self.assertIn("detection_observation_count", d)
        # decision
        self.assertIn("confirmation_video_time", d)
        self.assertIn("track_ids_in_window", d)
        self.assertIn("confirmation_terms_met", d)
        # human
        self.assertIn("annotated_frame_path", d)
        self.assertIn("crop_path", d)
        self.assertIn("person_object_pair_crop_path", d)
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# CandidateState representation
# ---------------------------------------------------------------------------

class TestCandidateStateRepresentation(unittest.TestCase):

    def test_all_five_states_exist(self):
        expected = {"IDLE", "CANDIDATE", "CONFIRMED", "COOLDOWN", "LOST"}
        actual = {s.value for s in CandidateState}
        self.assertEqual(actual, expected)

    def test_candidate_can_hold_each_state(self):
        key = EntityKey.from_pose(1)
        for state in CandidateState:
            c = Candidate(entity_key=key, candidate_class="x", state=state)
            self.assertEqual(c.state, state)

    def test_state_values_are_strings(self):
        for state in CandidateState:
            self.assertIsInstance(state.value, str)


# ---------------------------------------------------------------------------
# BoundedObservationBuffer
# ---------------------------------------------------------------------------

class TestBoundedObservationBuffer(unittest.TestCase):

    def _make_obs(self, track_id: int, video_time: float) -> Observation:
        return Observation(
            frame_index=track_id,
            video_time=video_time,
            class_name="smartphone",
            confidence=0.8,
            bbox=(0, 0, 10, 10),
            source=ModelSource.OBJECT,
            track_id=track_id,
        )

    def test_fifo_eviction(self):
        buf = BoundedObservationBuffer(maxlen=3)
        for i in range(5):
            buf.append(self._make_obs(track_id=i, video_time=float(i)))
        self.assertEqual(len(buf), 3)
        remaining = buf.all()
        # oldest two (track_id 0, 1) must be evicted
        self.assertEqual([o.track_id for o in remaining], [2, 3, 4])

    def test_maxlen_property(self):
        buf = BoundedObservationBuffer(maxlen=8)
        self.assertEqual(buf.maxlen, 8)

    def test_is_full(self):
        buf = BoundedObservationBuffer(maxlen=2)
        self.assertFalse(buf.is_full())
        buf.append(self._make_obs(1, 0.1))
        self.assertFalse(buf.is_full())
        buf.append(self._make_obs(2, 0.2))
        self.assertTrue(buf.is_full())

    def test_iterable_and_len(self):
        buf = BoundedObservationBuffer(maxlen=4)
        for i in range(4):
            buf.append(self._make_obs(i, float(i)))
        self.assertEqual(len(buf), 4)
        self.assertEqual([o.track_id for o in buf], [0, 1, 2, 3])

    def test_invalid_maxlen_raises(self):
        with self.assertRaises(ValueError):
            BoundedObservationBuffer(maxlen=0)
        with self.assertRaises(ValueError):
            BoundedObservationBuffer(maxlen=-1)


# ---------------------------------------------------------------------------
# ConfidenceStats
# ---------------------------------------------------------------------------

class TestConfidenceStats(unittest.TestCase):

    def test_from_observations(self):
        obs = [
            _make_observation(confidence=0.55),
            _make_observation(confidence=0.79),
            _make_observation(confidence=0.91),
        ]
        stats = ConfidenceStats.from_observations(obs)
        self.assertEqual(stats.count, 3)
        self.assertAlmostEqual(stats.min, 0.55)
        self.assertAlmostEqual(stats.max, 0.91)
        self.assertAlmostEqual(stats.mean, (0.55 + 0.79 + 0.91) / 3)

    def test_empty_list(self):
        stats = ConfidenceStats.from_observations([])
        self.assertEqual(stats.count, 0)
        self.assertEqual(stats.mean, 0.0)

    def test_to_dict_json(self):
        stats = ConfidenceStats(count=1, mean=0.8, min=0.8, max=0.8)
        json.dumps(stats.to_dict())


if __name__ == "__main__":
    unittest.main()
