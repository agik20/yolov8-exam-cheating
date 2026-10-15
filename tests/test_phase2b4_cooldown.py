"""Phase 2B.4 - Post-Confirmation Cooldown and Shadow Lifecycle tests.

Covers: CooldownGateResult, UnconfiguredCooldownPolicy, CalibratedCooldownPolicy,
CandidateLayer.advance_cooldown, ShadowEventRecord, ShadowComparisonRecord,
ShadowLifecycleManager full lifecycle, video-time semantics, identity isolation,
and pipeline decoupling.

Run:  python -m pytest tests/test_phase2b4_cooldown.py -v
"""
import json, os, sys, unittest
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from inference.core.events import (
    ModelSource, EntityKey, Observation,
    BoundedObservationBuffer, CandidateState, Candidate,
    ConfidenceStats, EventRecord,
)
from inference.core.candidate_layer import CandidateLayer, CandidateKey
from inference.core.confirmation import (
    CalibratedConfirmationPolicy, UnconfiguredConfirmationPolicy,
)
from inference.core.cooldown import (
    CooldownGateResult, UnconfiguredCooldownPolicy, CalibratedCooldownPolicy,
)
from inference.core.shadow_lifecycle import (
    ShadowEventRecord, ShadowComparisonRecord, ShadowLifecycleManager,
)


def _obs(frame_index=0, video_time=0.0, class_name="provide-object",
         confidence=0.72, bbox=(100, 200, 400, 500),
         source=ModelSource.POSE, track_id=3):
    return Observation(frame_index=frame_index, video_time=video_time,
                       class_name=class_name, confidence=confidence,
                       bbox=bbox, source=source, track_id=track_id)


def _cand(track_id=3, cls="provide-object", src=ModelSource.POSE,
          n=5, span=2.0):
    key = EntityKey(source=src, track_id=track_id)
    buf = BoundedObservationBuffer(maxlen=64)
    t0 = 10.0
    for i in range(n):
        t = t0 + (span * i / (n - 1)) if n > 1 else t0
        buf.append(_obs(frame_index=i, video_time=t, track_id=track_id,
                     class_name=cls))
    c = Candidate(entity_key=key, candidate_class=cls,
                  observations=buf, gap_count=0, max_gap_seconds=None)
    c.start_video_time = t0
    c.latest_video_time = t0 + span
    c.observed_track_ids = frozenset([track_id])
    return c


def _conf():
    return CalibratedConfirmationPolicy(
        min_observations=3, min_duration_seconds=1.0,
        min_confidence_mean=0.5)


def _cd(sec=30.0):
    return CalibratedCooldownPolicy(cooldown_seconds=sec)


class TestCooldownGateResult(unittest.TestCase):

    def test_result_json(self):
        r = CooldownGateResult(EntityKey.from_pose(3), "x", True, 15.0, 42.0,
                              "cooldown_active")
        json.dumps(r.to_dict())

    def test_fields(self):
        r = CooldownGateResult(EntityKey.from_pose(3), "s", False, 0.0, None,
                              "cooldown_elapsed")
        self.assertFalse(r.in_cooldown)
        self.assertEqual(r.reason, "cooldown_elapsed")
        self.assertEqual(r.entity_key.as_string(), "pose:3")


class TestUnconfiguredCooldownPolicy(unittest.TestCase):

    def test_enter_immediately_eligible(self):
        c = _cand()
        r = UnconfiguredCooldownPolicy().enter_cooldown(c, 10.0)
        self.assertFalse(r.in_cooldown)
        self.assertEqual(c.state, CandidateState.CANDIDATE)

    def test_check_always_eligible(self):
        c = _cand()
        c.state = CandidateState.COOLDOWN
        c.cooldown_expiry_video_time = 999.0
        r = UnconfiguredCooldownPolicy().check_eligibility(c, 0.0)
        self.assertFalse(r.in_cooldown)

    def test_reason(self):
        r = UnconfiguredCooldownPolicy().enter_cooldown(_cand(), 10.0)
        self.assertEqual(r.reason, "cooldown_policy_unconfigured")


class TestCalibratedCooldownPolicy(unittest.TestCase):

    def test_zero_rejected(self):
        with self.assertRaises(ValueError):
            CalibratedCooldownPolicy(0.0)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            CalibratedCooldownPolicy(-1.0)

    def test_none_rejected(self):
        with self.assertRaises(ValueError):
            CalibratedCooldownPolicy(None)

    def test_enter_sets_expiry(self):
        c = _cand()
        r = _cd(30.0).enter_cooldown(c, 10.0)
        self.assertEqual(c.state, CandidateState.COOLDOWN)
        self.assertAlmostEqual(c.cooldown_expiry_video_time, 40.0)
        self.assertTrue(r.in_cooldown)

    def test_active_before_expiry(self):
        c = _cand()
        _cd(30.0).enter_cooldown(c, 10.0)
        r = _cd(30.0).check_eligibility(c, 25.0)
        self.assertTrue(r.in_cooldown)
        self.assertAlmostEqual(r.cooldown_remaining_seconds, 15.0)

    def test_elapsed_exact(self):
        c = _cand()
        _cd(30.0).enter_cooldown(c, 10.0)
        r = _cd(30.0).check_eligibility(c, 40.0)
        self.assertFalse(r.in_cooldown)
        self.assertEqual(r.reason, "cooldown_elapsed")

    def test_elapsed_past(self):
        c = _cand()
        _cd(30.0).enter_cooldown(c, 10.0)
        r = _cd(30.0).check_eligibility(c, 45.0)
        self.assertFalse(r.in_cooldown)

    def test_not_in_cooldown(self):
        c = _cand()
        r = _cd(30.0).check_eligibility(c, 10.0)
        self.assertFalse(r.in_cooldown)
        self.assertEqual(r.reason, "not_in_cooldown")

    def test_linear_remaining(self):
        c = _cand()
        p = _cd(20.0)
        p.enter_cooldown(c, 0.0)
        self.assertAlmostEqual(p.check_eligibility(c, 5.0).cooldown_remaining_seconds, 15.0)
        self.assertAlmostEqual(p.check_eligibility(c, 10.0).cooldown_remaining_seconds, 10.0)
        self.assertAlmostEqual(p.check_eligibility(c, 15.0).cooldown_remaining_seconds, 5.0)


class TestVideoTimeSemantics(unittest.TestCase):

    def test_video_time_not_wall(self):
        c = _cand()
        p = _cd(10.0)
        p.enter_cooldown(c, 5.0)
        self.assertAlmostEqual(c.cooldown_expiry_video_time, 15.0)
        self.assertTrue(p.check_eligibility(c, 14.0).in_cooldown)
        self.assertFalse(p.check_eligibility(c, 15.0).in_cooldown)
        self.assertFalse(p.check_eligibility(c, 16.0).in_cooldown)

    def test_pure_function_of_video_time(self):
        p = _conf()
        m1 = ShadowLifecycleManager(CandidateLayer(30.0, 64), p, _cd(1.0))
        m2 = ShadowLifecycleManager(CandidateLayer(30.0, 64), p, _cd(1.0))
        for i in range(120):
            m1.process_observation(_obs(i, i / 30.0, track_id=3))
        for i in range(120):
            m2.process_observation(_obs(i, i / 30.0, track_id=3))
        self.assertEqual(len(m1.shadow_events), len(m2.shadow_events))


class TestCandidateLayerAdvanceCooldown(unittest.TestCase):

    def _L(self):
        return CandidateLayer(30.0, 16)

    def test_reset_on_elapsed(self):
        L = self._L()
        L.record_observation(_obs(0, 0.0))
        c = L.get_candidate(EntityKey.from_pose(3), "provide-object")
        c.state = CandidateState.COOLDOWN
        c.cooldown_expiry_video_time = 5.0
        L.advance_cooldown(EntityKey.from_pose(3), "provide-object", 6.0, _cd(5.0))
        c2 = L.get_candidate(EntityKey.from_pose(3), "provide-object")
        self.assertEqual(c2.state, CandidateState.CANDIDATE)
        self.assertIsNone(c2.cooldown_expiry_video_time)
        self.assertEqual(c2.observation_count(), 0)

    def test_no_reset_during_cooldown(self):
        L = self._L()
        L.record_observation(_obs(0, 0.0))
        c = L.get_candidate(EntityKey.from_pose(3), "provide-object")
        c.state = CandidateState.COOLDOWN
        c.cooldown_expiry_video_time = 5.0
        L.advance_cooldown(EntityKey.from_pose(3), "provide-object", 3.0, _cd(5.0))
        c2 = L.get_candidate(EntityKey.from_pose(3), "provide-object")
        self.assertEqual(c2.state, CandidateState.COOLDOWN)
        self.assertEqual(c2.observation_count(), 1)

    def test_noop_when_missing(self):
        L = self._L()
        self.assertIsNone(L.advance_cooldown(EntityKey.from_pose(99), "x", 10.0, _cd(5.0)))


class TestIdentityIsolation(unittest.TestCase):

    def test_pose_vs_object_independent(self):
        L = CandidateLayer(30.0, 16)
        L.record_observation(_obs(0, 0.0, source=ModelSource.POSE, track_id=3))
        L.get_candidate(EntityKey.from_pose(3), "provide-object").state = CandidateState.COOLDOWN
        L.record_observation(_obs(0, 0.0, source=ModelSource.OBJECT, track_id=3))
        o = L.get_candidate(EntityKey.from_object(3), "provide-object")
        self.assertEqual(o.state, CandidateState.CANDIDATE)
        self.assertIsNone(o.cooldown_expiry_video_time)

    def test_class_independence(self):
        L = CandidateLayer(30.0, 16)
        L.record_observation(_obs(0, 0.0, class_name="provide-object"))
        L.get_candidate(EntityKey.from_pose(3), "provide-object").state = CandidateState.COOLDOWN
        L.record_observation(_obs(0, 0.0, class_name="see-friends-work"))
        c = L.get_candidate(EntityKey.from_pose(3), "see-friends-work")
        self.assertEqual(c.state, CandidateState.CANDIDATE)


class TestShadowEventRecord(unittest.TestCase):

    def _se(self):
        c = _cand()
        g = UnconfiguredConfirmationPolicy().evaluate(c)
        e = EventRecord.from_candidate(c, 12.0, datetime.now().isoformat())
        return ShadowEventRecord(event=e, gate_result=g, recorded_at_video_time=12.0)

    def test_json(self):
        json.dumps(self._se().to_dict())

    def test_event_id(self):
        s = self._se()
        self.assertEqual(s.event_id, s.event.event_id)
        self.assertIsInstance(s.event_id, str)

    def test_entity_key(self):
        s = self._se()
        self.assertEqual(s.entity_key, s.event.entity_key)

    def test_video_time(self):
        self.assertAlmostEqual(self._se().recorded_at_video_time, 12.0)

    def test_dict_fields(self):
        d = self._se().to_dict()
        for f in ("event_id", "entity_key", "class_name",
                 "recorded_at_video_time", "confirmation_video_time",
                 "observation_count", "confidence_stats",
                 "contributing_track_ids", "gate_reason", "gate_confirmed"):
            self.assertIn(f, d)


class TestShadowComparisonRecord(unittest.TestCase):

    def test_no_fp_fn_fields(self):
        r = ShadowComparisonRecord(EntityKey.from_pose(3), "provide-object", "CONFIRMED",
                                   "abc", True, 10.0)
        d = r.to_dict()
        for f in ("false_positive", "false_negative", "is_fp", "is_fn",
                 "fp", "fn", "error_type", "classification"):
            self.assertNotIn(f, d)
        for f in ("entity_key", "class_name", "shadow_state",
                 "shadow_event_id", "legacy_event_observed", "comparison_video_time"):
            self.assertIn(f, d)

    def test_json(self):
        r = ShadowComparisonRecord(EntityKey.from_pose(3), "provide-object", "CANDIDATE",
                                   None, False, 5.0)
        json.dumps(r.to_dict())


class TestShadowLifecycleManager(unittest.TestCase):

    def _M(self, L=None, cd=1.0, cf=None):
        L = L or CandidateLayer(30.0, 64)
        cf = cf or _conf()
        return ShadowLifecycleManager(L, cf, _cd(cd))

    def test_confirmation_creates_event(self):
        L = CandidateLayer(30.0, 64)
        m = self._M(L, 1.0)
        for i in range(31):
            m.process_observation(_obs(i, i / 30.0))
        c = L.get_candidate(EntityKey.from_pose(3), "provide-object")
        self.assertEqual(c.state, CandidateState.COOLDOWN)
        self.assertGreaterEqual(len(m.shadow_events), 1)

    def test_reaccumulation_after_expiry(self):
        L = CandidateLayer(30.0, 64)
        m = self._M(L, 1.0)
        for i in range(31):
            m.process_observation(_obs(i, i / 30.0))
        n1 = len(m.shadow_events)
        for i in range(31, 91):
            m.process_observation(_obs(i, i / 30.0))
        self.assertGreater(len(m.shadow_events), n1)

    def test_no_event_when_gate_rejected(self):
        L = CandidateLayer(30.0, 64)
        m = ShadowLifecycleManager(L, CalibratedConfirmationPolicy(100, 1.0, 0.5), _cd(30.0))
        for i in range(5):
            m.process_observation(_obs(i, i / 30.0))
        self.assertEqual(len(m.shadow_events), 0)

    def test_comparison_populated(self):
        L = CandidateLayer(30.0, 64)
        m = self._M(L, 1.0)
        for i in range(31):
            m.process_observation(_obs(i, i / 30.0), legacy_event_observed=(i == 30))
        self.assertGreaterEqual(len(m.comparison_records), 1)

    def test_no_file_io(self):
        import inference.core.shadow_lifecycle as sm
        src = open(sm.__file__).read()
        for f in ("save_crop", "log_detection_to_csv", "cv2.imwrite"):
            self.assertNotIn(f, src)

    def test_unconfigured_allows_reaccumulation(self):
        L = CandidateLayer(30.0, 64)
        m = ShadowLifecycleManager(L, _conf(), UnconfiguredCooldownPolicy())
        for i in range(31):
            m.process_observation(_obs(i, i / 30.0))
        n1 = len(m.shadow_events)
        self.assertGreaterEqual(n1, 1)
        for i in range(31, 61):
            m.process_observation(_obs(i, i / 30.0))
        self.assertGreater(len(m.shadow_events), n1)

    def test_snapshot_json(self):
        L = CandidateLayer(30.0, 64)
        m = self._M(L, 1.0)
        for i in range(31):
            m.process_observation(_obs(i, i / 30.0))
        snap = m.snapshot()
        json.dumps(snap)
        self.assertIn("shadow_events", snap)


class TestPipelineDecoupling(unittest.TestCase):

    def test_no_2b4_in_process_detection(self):
        import inspect
        import inference.core.pipeline as P
        src = inspect.getsource(P.CheatingDetectionPipeline.process_detection)
        for f in ("CooldownPolicy", "CalibratedCooldownPolicy",
                 "UnconfiguredCooldownPolicy", "ShadowLifecycleManager",
                 "ShadowEventRecord", "ShadowComparisonRecord",
                 "_cooldown_policy", "_shadow_lifecycle",
                 "advance_cooldown", "enter_cooldown", "check_eligibility"):
            self.assertNotIn(f, src)

    def test_no_shadow_event_record_in_process_detection(self):
        import inspect
        import inference.core.pipeline as P
        src = inspect.getsource(P.CheatingDetectionPipeline.process_detection)
        for f in ("_shadow_results", "_cooldown_policy", "_shadow_lifecycle",
                 "EventRecord"):
            self.assertNotIn(f, src)


if __name__ == "__main__":
    unittest.main()
