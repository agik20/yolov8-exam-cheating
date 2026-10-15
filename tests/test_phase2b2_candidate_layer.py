"""Phase 2B.2 — Unit tests for the shadow candidate observation layer.

Covers:
- VideoTimeSource: frame index + FPS → video time; invalid FPS handling
- CandidateLayer: candidate accumulation, namespace isolation, gap bookkeeping,
  LOST representation, bounded history, stale-candidate eviction
- Explicit non-goals: no confirmation, no event output, no policy numbers

Run:  python -m pytest tests/test_phase2b2_candidate_layer.py -v
"""
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(
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
# VideoTimeSource
# ---------------------------------------------------------------------------

class TestVideoTimeSource(unittest.TestCase):
    """frame_index + src_fps → video_time.  Invalid FPS must fail explicitly."""

    def test_video_time_computation(self):
        from inference.core.candidate_layer import VideoTimeSource
        vts = VideoTimeSource(src_fps=30.0)
        # frame 90 at 30 fps = 3.0 s of source video
        self.assertAlmostEqual(vts.video_time_for(90), 3.0, places=6)

    def test_video_time_is_monotonic_with_frame_index(self):
        from inference.core.candidate_layer import VideoTimeSource
        vts = VideoTimeSource(src_fps=30.0)
        t0 = vts.video_time_for(0)
        t1 = vts.video_time_for(30)
        t2 = vts.video_time_for(60)
        self.assertLess(t0, t1)
        self.assertLess(t1, t2)

    def test_zero_fps_raises(self):
        from inference.core.candidate_layer import VideoTimeSource, InvalidSourceFps
        with self.assertRaises(InvalidSourceFps):
            VideoTimeSource(src_fps=0.0)

    def test_negative_fps_raises(self):
        from inference.core.candidate_layer import VideoTimeSource, InvalidSourceFps
        with self.assertRaises(InvalidSourceFps):
            VideoTimeSource(src_fps=-30.0)

    def test_none_fps_raises(self):
        from inference.core.candidate_layer import VideoTimeSource, InvalidSourceFps
        with self.assertRaises(InvalidSourceFps):
            VideoTimeSource(src_fps=None)

    def test_is_available_property(self):
        from inference.core.candidate_layer import VideoTimeSource
        self.assertTrue(VideoTimeSource(30.0).is_available)
        self.assertTrue(VideoTimeSource(25.0).is_available)


# ---------------------------------------------------------------------------
# CandidateKey
# ---------------------------------------------------------------------------

class TestCandidateKeyUniqueness(unittest.TestCase):

    def test_pose_3_and_object_3_different_keys(self):
        from inference.core.candidate_layer import CandidateKey
        k1 = CandidateKey(EntityKey.from_pose(3), "smartphone")
        k2 = CandidateKey(EntityKey.from_object(3), "smartphone")
        self.assertNotEqual(k1, k2)

    def test_same_entity_same_class_equal(self):
        from inference.core.candidate_layer import CandidateKey
        k1 = CandidateKey(EntityKey.from_pose(3), "smartphone")
        k2 = CandidateKey(EntityKey.from_pose(3), "smartphone")
        self.assertEqual(k1, k2)

    def test_same_entity_different_class_different_key(self):
        from inference.core.candidate_layer import CandidateKey
        k1 = CandidateKey(EntityKey.from_pose(3), "smartphone")
        k2 = CandidateKey(EntityKey.from_pose(3), "provide-object")
        self.assertNotEqual(k1, k2)


# ---------------------------------------------------------------------------
# CandidateLayer — accumulation
# ---------------------------------------------------------------------------

class TestCandidateLayerAccumulation(unittest.TestCase):

    def _layer(self, src_fps=30.0, maxlen=16) -> "object":
        from inference.core.candidate_layer import CandidateLayer
        return CandidateLayer(src_fps=src_fps, observation_buffer_maxlen=maxlen)

    def test_first_observation_creates_candidate(self):
        layer = self._layer()
        obs = _obs(frame_index=0, video_time=0.0, track_id=3)
        layer.record_observation(obs)
        self.assertTrue(layer.has_candidate(EntityKey.from_pose(3), "provide-object"))

    def test_second_observation_updates_latest_time(self):
        layer = self._layer()
        layer.record_observation(_obs(frame_index=0,  video_time=0.0, track_id=3))
        layer.record_observation(_obs(frame_index=30, video_time=1.0, track_id=3))
        cand = layer.get_candidate(EntityKey.from_pose(3), "provide-object")
        self.assertIsNotNone(cand)
        self.assertAlmostEqual(cand.latest_video_time, 1.0)
        self.assertAlmostEqual(cand.start_video_time, 0.0)

    def test_observation_count_increases(self):
        layer = self._layer()
        for i in range(5):
            layer.record_observation(_obs(frame_index=i, video_time=i / 30.0, track_id=3))
        cand = layer.get_candidate(EntityKey.from_pose(3), "provide-object")
        self.assertEqual(cand.observation_count(), 5)

    def test_video_time_span(self):
        layer = self._layer()
        layer.record_observation(_obs(frame_index=0,  video_time=0.0, track_id=3))
        layer.record_observation(_obs(frame_index=90, video_time=3.0, track_id=3))
        cand = layer.get_candidate(EntityKey.from_pose(3), "provide-object")
        self.assertAlmostEqual(cand.video_time_span(), 3.0)

    def test_observed_track_ids_contains_own_track_id(self):
        """Each candidate's observed_track_ids contains the track_id used to
        key the candidate.  Cross-track_id linking (ID-switch consolidation)
        is Phase 2B.5+ behavior, not implemented here."""
        layer = self._layer()
        layer.record_observation(_obs(frame_index=0,  video_time=0.0, track_id=3))
        layer.record_observation(_obs(frame_index=30, video_time=1.0, track_id=3))
        cand = layer.get_candidate(EntityKey.from_pose(3), "provide-object")
        self.assertEqual(cand.observed_track_ids, frozenset({3}))

    def test_different_track_ids_produce_separate_candidates(self):
        """In Phase 2B.2 each (EntityKey, class) pair is a distinct candidate.
        pose:3 and pose:47 are two separate candidates; linking them is
        Phase 2B.5+ (ID-switch consolidation)."""
        layer = self._layer()
        layer.record_observation(_obs(frame_index=0,  video_time=0.0, track_id=3))
        layer.record_observation(_obs(frame_index=30, video_time=1.0, track_id=47))
        self.assertTrue(layer.has_candidate(EntityKey.from_pose(3),  "provide-object"))
        self.assertTrue(layer.has_candidate(EntityKey.from_pose(47), "provide-object"))
        self.assertEqual(len(layer.active_candidates()), 2)

    def test_class_specific_candidates_separate(self):
        layer = self._layer()
        layer.record_observation(_obs(class_name="provide-object",  track_id=3))
        layer.record_observation(_obs(class_name="see-friends-work", track_id=3))
        self.assertTrue(layer.has_candidate(EntityKey.from_pose(3), "provide-object"))
        self.assertTrue(layer.has_candidate(EntityKey.from_pose(3), "see-friends-work"))
        c1 = layer.get_candidate(EntityKey.from_pose(3), "provide-object")
        c2 = layer.get_candidate(EntityKey.from_pose(3), "see-friends-work")
        self.assertEqual(c1.observation_count(), 1)
        self.assertEqual(c2.observation_count(), 1)


# ---------------------------------------------------------------------------
# CandidateLayer — namespace isolation
# ---------------------------------------------------------------------------

class TestCandidateLayerNamespaceIsolation(unittest.TestCase):

    def test_pose_3_and_object_3_never_share_state(self):
        from inference.core.candidate_layer import CandidateLayer
        layer = CandidateLayer(src_fps=30.0, observation_buffer_maxlen=16)
        layer.record_observation(_obs(source=ModelSource.POSE,   track_id=3, video_time=0.0, class_name="smartphone"))
        layer.record_observation(_obs(source=ModelSource.OBJECT, track_id=3, video_time=0.1, class_name="smartphone"))
        pose_cand   = layer.get_candidate(EntityKey.from_pose(3),   "smartphone")
        object_cand = layer.get_candidate(EntityKey.from_object(3), "smartphone")
        self.assertIsNotNone(pose_cand)
        self.assertIsNotNone(object_cand)
        self.assertEqual(pose_cand.observation_count(), 1)
        self.assertEqual(object_cand.observation_count(), 1)

    def test_pose_and_object_different_classes_independent(self):
        from inference.core.candidate_layer import CandidateLayer
        layer = CandidateLayer(src_fps=30.0, observation_buffer_maxlen=16)
        layer.record_observation(_obs(source=ModelSource.POSE,   track_id=3, class_name="provide-object"))
        layer.record_observation(_obs(source=ModelSource.OBJECT, track_id=3, class_name="smartphone"))
        self.assertEqual(len(layer.active_candidates()), 2)


# ---------------------------------------------------------------------------
# CandidateLayer — gap bookkeeping
# ---------------------------------------------------------------------------

class TestCandidateLayerGapBookkeeping(unittest.TestCase):

    def _layer(self):
        from inference.core.candidate_layer import CandidateLayer
        return CandidateLayer(src_fps=30.0, observation_buffer_maxlen=64)

    def test_continuous_observations_no_gap(self):
        layer = self._layer()
        for i in range(30):  # 1 s of continuous observation at 30 fps
            layer.record_observation(_obs(frame_index=i, video_time=i / 30.0, track_id=3))
        cand = layer.get_candidate(EntityKey.from_pose(3), "provide-object")
        self.assertEqual(cand.gap_count, 0)
        self.assertIsNone(cand.max_gap_seconds)

    def test_observation_gap_increments_gap_count(self):
        layer = self._layer()
        for i in range(30):
            layer.record_observation(_obs(frame_index=i, video_time=i / 30.0, track_id=3))
        # 30 frames at 30 fps = 1.0 s of video time (30/30 = 1.0)
        layer.note_gap(EntityKey.from_pose(3), "provide-object", gap_seconds=1.0)
        for i in range(30, 61):
            layer.record_observation(_obs(frame_index=i, video_time=i / 30.0, track_id=3))
        cand = layer.get_candidate(EntityKey.from_pose(3), "provide-object")
        self.assertEqual(cand.gap_count, 1)
        self.assertAlmostEqual(cand.max_gap_seconds, 1.0)

    def test_lost_state_representation(self):
        layer = self._layer()
        layer.record_observation(_obs(frame_index=0, video_time=0.0, track_id=3))
        layer.note_gap(EntityKey.from_pose(3), "provide-object")
        cand = layer.get_candidate(EntityKey.from_pose(3), "provide-object")
        # After a gap, candidate is in LOST state (representation only, no policy)
        self.assertEqual(cand.state, CandidateState.LOST)

    def test_recovery_from_lost_back_to_candidate(self):
        layer = self._layer()
        layer.record_observation(_obs(frame_index=0, video_time=0.0, track_id=3))
        layer.note_gap(EntityKey.from_pose(3), "provide-object")
        # Next observation recovers the candidate back to CANDIDATE
        layer.record_observation(_obs(frame_index=60, video_time=2.0, track_id=3))
        cand = layer.get_candidate(EntityKey.from_pose(3), "provide-object")
        self.assertEqual(cand.state, CandidateState.CANDIDATE)

    def test_gap_does_not_require_policy_decision(self):
        """note_gap records a measured gap; it does not dissolve the candidate
        or make any confirmation/cooldown decision."""
        layer = self._layer()
        layer.record_observation(_obs(frame_index=0, video_time=0.0, track_id=3))
        layer.note_gap(EntityKey.from_pose(3), "provide-object")
        # Candidate still exists (LOST), not removed from the layer
        self.assertTrue(layer.has_candidate(EntityKey.from_pose(3), "provide-object"))


# ---------------------------------------------------------------------------
# CandidateLayer — bounded history
# ---------------------------------------------------------------------------

class TestCandidateLayerBoundedHistory(unittest.TestCase):

    def test_observation_buffer_respects_configured_maxlen(self):
        from inference.core.candidate_layer import CandidateLayer
        layer = CandidateLayer(src_fps=30.0, observation_buffer_maxlen=4)
        for i in range(10):
            layer.record_observation(_obs(frame_index=i, video_time=i / 30.0, track_id=3))
        cand = layer.get_candidate(EntityKey.from_pose(3), "provide-object")
        # Only the last 4 observations should be in the buffer
        self.assertEqual(cand.observation_count(), 4)
        self.assertEqual(cand.observations.maxlen, 4)

    def test_active_candidate_store_is_bounded(self):
        """The layer caps the number of active candidates.

        The cap is a structural parameter, not an event-policy value:
        it controls memory, not which events are confirmed.
        """
        from inference.core.candidate_layer import CandidateLayer
        layer = CandidateLayer(src_fps=30.0, observation_buffer_maxlen=16,
                               max_active_candidates=2)
        # Create 3 distinct candidates — the oldest should be evicted
        layer.record_observation(_obs(track_id=1, class_name="provide-object"))
        layer.record_observation(_obs(track_id=2, class_name="provide-object"))
        layer.record_observation(_obs(track_id=3, class_name="provide-object"))
        active = layer.active_candidates()
        self.assertLessEqual(len(active), 2,
                             "active candidate store must not exceed max_active_candidates")
        # The most-recently-touched candidate must still be present
        self.assertTrue(layer.has_candidate(EntityKey.from_pose(3), "provide-object"))


# ---------------------------------------------------------------------------
# CandidateLayer — stale candidate eviction interface
# ---------------------------------------------------------------------------

class TestCandidateLayerEviction(unittest.TestCase):

    def test_evict_stale_candidates_is_explicit(self):
        """Eviction is caller-driven (not automatic on a hidden timer)."""
        from inference.core.candidate_layer import CandidateLayer
        layer = CandidateLayer(src_fps=30.0, observation_buffer_maxlen=16,
                               max_active_candidates=8)
        layer.record_observation(_obs(track_id=1, class_name="provide-object"))
        layer.record_observation(_obs(track_id=2, class_name="smartphone"))
        self.assertEqual(len(layer.active_candidates()), 2)

        layer.evict_candidate(EntityKey.from_pose(1), "provide-object")
        self.assertFalse(layer.has_candidate(EntityKey.from_pose(1), "provide-object"))
        self.assertTrue(layer.has_candidate(EntityKey.from_pose(2), "smartphone"))

    def test_evict_nonexistent_is_noop(self):
        from inference.core.candidate_layer import CandidateLayer
        layer = CandidateLayer(src_fps=30.0, observation_buffer_maxlen=16)
        # Should not raise
        layer.evict_candidate(EntityKey.from_pose(99), "nonexistent-class")


# ---------------------------------------------------------------------------
# CandidateLayer — diagnostic access
# ---------------------------------------------------------------------------

class TestCandidateLayerDiagnostics(unittest.TestCase):

    def test_snapshot_for_diagnostics_is_json_safe(self):
        import json
        from inference.core.candidate_layer import CandidateLayer
        layer = CandidateLayer(src_fps=30.0, observation_buffer_maxlen=16)
        layer.record_observation(_obs(frame_index=0,  video_time=0.0, track_id=3))
        layer.record_observation(_obs(frame_index=30, video_time=1.0, track_id=3))
        snap = layer.snapshot()
        json.dumps(snap)
        self.assertGreaterEqual(len(snap), 1)

    def test_temporal_data_unavailable_on_invalid_fps(self):
        """When src_fps is invalid, CandidateLayer construction must fail
        visibly — not silently fall back to wall-clock time."""
        from inference.core.candidate_layer import CandidateLayer, InvalidSourceFps
        with self.assertRaises(InvalidSourceFps):
            CandidateLayer(src_fps=0.0, observation_buffer_maxlen=16)


# ---------------------------------------------------------------------------
# process_detection must NOT call candidate layer
# ---------------------------------------------------------------------------

class TestProcessDetectionDecoupled(unittest.TestCase):

    def test_process_detection_does_not_reference_candidate_layer(self):
        """Static check: the legacy process_detection method body must not
        reference _candidate_layer, record_observation, note_gap, or
        evict_candidate."""
        import inspect
        import inference.core.pipeline as pipeline_mod
        src = inspect.getsource(pipeline_mod.CheatingDetectionPipeline.process_detection)
        for forbidden in ("_candidate_layer", "record_observation",
                          "note_gap", "evict_candidate"):
            self.assertNotIn(forbidden, src,
                             f"process_detection must not reference {forbidden!r} — "
                             f"legacy event path must remain decoupled from the shadow layer")


if __name__ == "__main__":
    unittest.main()
