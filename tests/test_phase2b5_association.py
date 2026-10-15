"""Phase 2B.5 — Person-object association test suite.

Run:  python -m pytest tests/test_phase2b5_association.py -v
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

from inference.core.association import (
    AssociationStatus,
    SpatialEvidence,
    KeypointEvidence,
    TemporalEvidence,
    AssociationEvidence,
    AssociationResult,
    PersonKeypoints,
    PersonObservation,
    ObjectObservation,
    PersonObjectAssociator,
    HAND_KEYPOINT_INDICES,
    center_inside,
    containment_ratio,
    normalized_center_distance,
    iou,
)
from inference.core.events import (
    EntityKey,
    ModelSource,
    Observation,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

class TestGeometryHelpers(unittest.TestCase):
    """Pure-geometry helper tests — no model objects involved."""

    def test_center_inside_true(self):
        person = (0.0, 0.0, 200.0, 400.0)
        phone = (80.0, 180.0, 110.0, 200.0)
        self.assertTrue(center_inside(person, phone))

    def test_center_inside_false_no_overlap(self):
        person = (0.0, 0.0, 100.0, 200.0)
        phone = (500.0, 50.0, 550.0, 100.0)
        self.assertFalse(center_inside(person, phone))

    def test_center_inside_boundary(self):
        person = (0.0, 0.0, 100.0, 100.0)
        phone = (50.0, 50.0, 150.0, 150.0)  # center = (100, 100) -> on boundary
        self.assertTrue(center_inside(person, phone))

    def test_degenerate_person_bbox_raises(self):
        with self.assertRaises(ValueError):
            center_inside((100.0, 0.0, 0.0, 200.0), (0.0, 0.0, 50.0, 50.0))

    def test_degenerate_object_bbox_raises(self):
        with self.assertRaises(ValueError):
            center_inside((0.0, 0.0, 100.0, 100.0), (50.0, 0.0, 0.0, 50.0))

    def test_containment_ratio_full_containment(self):
        person = (0.0, 0.0, 200.0, 400.0)
        phone = (50.0, 100.0, 100.0, 200.0)
        self.assertAlmostEqual(containment_ratio(person, phone), 1.0)

    def test_containment_ratio_no_overlap(self):
        person = (0.0, 0.0, 100.0, 100.0)
        phone = (200.0, 200.0, 300.0, 300.0)
        self.assertEqual(containment_ratio(person, phone), 0.0)

    def test_containment_ratio_partial(self):
        person = (0.0, 0.0, 100.0, 100.0)
        phone = (50.0, 50.0, 150.0, 150.0)
        # overlap = 50x50=2500, phone area=100x100=10000, ratio=0.25
        self.assertAlmostEqual(containment_ratio(person, phone), 0.25)

    def test_normalized_center_distance_zero_when_centers_coincide(self):
        person = (0.0, 0.0, 100.0, 100.0)  # center (50, 50)
        phone = (0.0, 0.0, 100.0, 100.0)   # center (50, 50)
        self.assertAlmostEqual(normalized_center_distance(person, phone, 100.0), 0.0)

    def test_normalized_center_distance_value(self):
        person = (0.0, 0.0, 100.0, 100.0)  # center (50, 50)
        phone = (100.0, 0.0, 200.0, 100.0)  # center (150, 50)
        self.assertAlmostEqual(normalized_center_distance(person, phone, 100.0), 1.0)

    def test_normalized_center_distance_zero_scale_raises(self):
        with self.assertRaises(ValueError):
            normalized_center_distance((0,0,10,10), (0,0,10,10), 0.0)

    def test_iou_no_overlap(self):
        self.assertEqual(iou((0,0,10,10), (20,20,30,30)), 0.0)

    def test_iou_identical_boxes(self):
        self.assertAlmostEqual(iou((0,0,10,10), (0,0,10,10)), 1.0)

    def test_iou_half_overlap(self):
        # box a: (0,0,10,10), box b: (5,0,15,10)
        # inter = 5x10=50, union=100+100-50=150, iou=50/150=1/3
        self.assertAlmostEqual(iou((0,0,10,10), (5,0,15,10)), 1.0/3.0)

    def test_iou_not_used_as_gate(self):
        """IoU is recorded as diagnostic but must not be a gate in _is_sufficient."""
        import inspect
        src = inspect.getsource(PersonObjectAssociator._is_sufficient)
        self.assertNotIn("spatial.iou", src, "IoU must not be used as an association gate")


# ---------------------------------------------------------------------------
# Identity / namespace disambiguation
# ---------------------------------------------------------------------------

class TestIdentityDisambiguation(unittest.TestCase):

    def test_pose_and_object_same_track_id_not_equal(self):
        ek_pose = EntityKey.from_pose(3)
        ek_obj  = EntityKey.from_object(3)
        self.assertNotEqual(ek_pose, ek_obj)

    def test_pose_track_ids_distinct(self):
        self.assertNotEqual(EntityKey.from_pose(3), EntityKey.from_pose(4))

    def test_as_string_distinct(self):
        self.assertNotEqual(
            EntityKey.from_pose(3).as_string(),
            EntityKey.from_object(3).as_string(),
        )


# ---------------------------------------------------------------------------
# Keypoint evidence
# ---------------------------------------------------------------------------

class TestKeypointEvidence(unittest.TestCase):

    def _make_17_kp(self):
        x = [0.0]*17
        y = [0.0]*17
        vis = [0.0]*17
        for idx, hx, hy, hv in [
            (9, 85.0, 180.0, 0.9),
            (10, 90.0, 185.0, 0.9),
            (11, 80.0, 190.0, 0.85),
            (12, 190.0, 195.0, 0.9),
            (13, 195.0, 200.0, 0.9),
            (14, 200.0, 205.0, 0.85),
        ]:
            x[idx] = hx
            y[idx] = hy
            vis[idx] = hv
        return PersonKeypoints(x=tuple(x), y=tuple(y), visibility=tuple(vis))

    def test_kp_available(self):
        dist, conf, count = self._make_17_kp().hand_distances_to_point(
            95.0, 190.0, visibility_cutoff=0.3
        )
        self.assertIsNotNone(dist)
        self.assertEqual(count, 6)
        self.assertAlmostEqual(conf, 0.85)

    def test_kp_unavailable_all_hidden(self):
        kp = PersonKeypoints(
            x=tuple([0.0]*17),
            y=tuple([0.0]*17),
            visibility=tuple([0.0]*17),
        )
        dist, conf, count = kp.hand_distances_to_point(95.0, 190.0, visibility_cutoff=0.3)
        self.assertIsNone(dist)
        self.assertEqual(count, 0)

    def test_kp_out_of_range_raises(self):
        kp = PersonKeypoints(
            x=tuple([0.0]*5),
            y=tuple([0.0]*5),
            visibility=tuple([1.0]*5),
        )
        with self.assertRaises(ValueError):
            kp.hand_distances_to_point(0.0, 0.0, hand_indices=(9, 10, 11))

    def test_kp_empty(self):
        kp = PersonKeypoints(x=(), y=(), visibility=())
        dist, conf, count = kp.hand_distances_to_point(0.0, 0.0)
        self.assertIsNone(dist)
        self.assertEqual(count, 0)

    def test_kp_evidence_to_dict(self):
        ev = KeypointEvidence(True, 5.0, 0.9, 4)
        self.assertTrue(ev.to_dict()["available"])
        self.assertEqual(ev.to_dict()["hand_keypoint_count"], 4)
        ev2 = KeypointEvidence(False, None, None, 0)
        self.assertFalse(ev2.to_dict()["available"])


# ---------------------------------------------------------------------------
# Spatial and temporal evidence
# ---------------------------------------------------------------------------

class TestSpatialAndTemporalEvidence(unittest.TestCase):

    def test_spatial_to_dict(self):
        ev = SpatialEvidence(True, 0.75, 0.1, 0.02)
        d = ev.to_dict()
        self.assertTrue(d["center_inside"])
        self.assertAlmostEqual(d["containment_ratio"], 0.75)
        self.assertAlmostEqual(d["iou"], 0.02)

    def test_spatial_null_distance(self):
        ev = SpatialEvidence(False, 0.0, None, 0.0)
        self.assertIsNone(ev.to_dict()["normalized_center_distance"])

    def test_temporal_overlap(self):
        ev = TemporalEvidence(True, 8.0, 240)
        self.assertTrue(ev.to_dict()["overlap"])
        self.assertEqual(ev.to_dict()["observation_count"], 240)

    def test_temporal_no_overlap(self):
        self.assertFalse(TemporalEvidence(False, 0.0, 0).overlap)


# ---------------------------------------------------------------------------
# AssociationResult
# ---------------------------------------------------------------------------

class TestAssociationResult(unittest.TestCase):

    def test_json_serializable(self):
        ev = AssociationEvidence(
            1, 5,
            SpatialEvidence(True, 0.8, 0.1, 0.01),
            KeypointEvidence(True, 5.0, 0.9, 4),
            TemporalEvidence(True, 3.0, 90),
        )
        r = AssociationResult(
            status=AssociationStatus.ASSOCIATED,
            object_track_id=5,
            evidence=[ev],
            summary="ok",
        )
        self.assertIn("ASSOCIATED", json.dumps(r.to_dict()))

    def test_score_none_default(self):
        self.assertIsNone(AssociationResult(
            status=AssociationStatus.UNASSIGNED,
            object_track_id=1,
        ).score)


# ---------------------------------------------------------------------------
# PersonObjectAssociator
# ---------------------------------------------------------------------------

class TestPersonObjectAssociator(unittest.TestCase):

    def _assoc(self, **kw):
        return PersonObjectAssociator(**kw)

    def test_single_person_object_inside(self):
        p = PersonObservation(1, "no-cheating", (0.0, 0.0, 200.0, 400.0))
        o = ObjectObservation(5, "smartphone", (80.0, 180.0, 110.0, 200.0))
        r = self._assoc().associate([p], [o])[0]
        self.assertEqual(r.status, AssociationStatus.ASSOCIATED)
        self.assertTrue(r.evidence[0].spatial.center_inside)

    def test_no_persons_unavailable(self):
        o = ObjectObservation(5, "smartphone", (80.0, 180.0, 110.0, 200.0))
        self.assertEqual(
            self._assoc().associate([], [o])[0].status,
            AssociationStatus.UNAVAILABLE,
        )

    def test_empty_objects_empty_results(self):
        p = PersonObservation(1, "no-cheating", (0.0, 0.0, 200.0, 400.0))
        self.assertEqual(self._assoc().associate([p], []), [])

    def test_no_overlap_unavailable(self):
        p = PersonObservation(9, "no-cheating", (0.0, 0.0, 100.0, 200.0))
        o = ObjectObservation(11, "calculator", (500.0, 50.0, 550.0, 100.0))
        self.assertEqual(
            self._assoc().associate([p], [o])[0].status,
            AssociationStatus.UNAVAILABLE,
        )

    def test_below_threshold_unassigned(self):
        p = PersonObservation(10, "no-cheating", (0.0, 0.0, 200.0, 200.0))
        o = ObjectObservation(12, "smartphone", (150.0, 150.0, 250.0, 250.0))
        self.assertEqual(
            self._assoc(containment_threshold=0.9).associate([p], [o])[0].status,
            AssociationStatus.UNASSIGNED,
        )

    def test_between_two_persons_ambiguous(self):
        p1 = PersonObservation(1, "no-cheating", (0.0, 0.0, 200.0, 400.0))
        p2 = PersonObservation(2, "no-cheating", (150.0, 0.0, 350.0, 400.0))
        o = ObjectObservation(7, "smartphone", (140.0, 180.0, 170.0, 200.0))
        r = self._assoc().associate([p1, p2], [o])[0]
        self.assertEqual(r.status, AssociationStatus.AMBIGUOUS)
        self.assertEqual(len(r.evidence), 2)
        self.assertEqual({ev.person_track_id for ev in r.evidence}, {1, 2})

    def test_multi_object_results(self):
        p = PersonObservation(1, "no-cheating", (0.0, 0.0, 400.0, 800.0))
        o1 = ObjectObservation(5, "smartphone", (80.0, 180.0, 110.0, 200.0))
        o2 = ObjectObservation(6, "calculator", (200.0, 300.0, 250.0, 350.0))
        rs = self._assoc().associate([p], [o1, o2])
        self.assertEqual(len(rs), 2)
        self.assertEqual(rs[0].object_track_id, 5)
        self.assertEqual(rs[1].object_track_id, 6)
        for r in rs:
            self.assertEqual(r.status, AssociationStatus.ASSOCIATED)

    def test_duplicate_track_ids_distinct_namespaces(self):
        p = PersonObservation(3, "no-cheating", (0.0, 0.0, 200.0, 400.0))
        o = ObjectObservation(3, "smartphone", (80.0, 180.0, 110.0, 200.0))
        ev = self._assoc().associate([p], [o])[0].evidence[0]
        self.assertEqual(ev.person_track_id, 3)
        self.assertEqual(ev.object_track_id, 3)
        self.assertNotEqual(
            EntityKey.from_pose(3),
            EntityKey.from_object(3),
        )

    def test_score_none_default(self):
        p = PersonObservation(1, "no-cheating", (0.0, 0.0, 200.0, 400.0))
        o = ObjectObservation(5, "smartphone", (80.0, 180.0, 110.0, 200.0))
        self.assertIsNone(self._assoc().associate([p], [o])[0].score)

    def test_kp_evidence_populated(self):
        vis = tuple([0.0]*9 + [0.9, 0.9, 0.85, 0.9, 0.9, 0.85, 0.0, 0.0])
        x   = tuple([0.0]*9 + [85.0, 190.0, 195.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        y   = tuple([50.0]*9 + [180.0, 195.0, 210.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        p = PersonObservation(
            1, "no-cheating", (0.0, 0.0, 200.0, 400.0),
            keypoints=PersonKeypoints(x=x, y=y, visibility=vis),
        )
        o = ObjectObservation(5, "smartphone", (80.0, 180.0, 110.0, 200.0))
        ev = self._assoc().associate([p], [o])[0].evidence[0]
        self.assertTrue(ev.keypoint.available)
        self.assertEqual(ev.keypoint.hand_keypoint_count, 6)
        self.assertIsNotNone(ev.keypoint.nearest_hand_distance)

    def test_kp_unavailable_no_keypoints(self):
        p = PersonObservation(1, "no-cheating", (0.0, 0.0, 200.0, 400.0))
        o = ObjectObservation(5, "smartphone", (80.0, 180.0, 110.0, 200.0))
        ev = self._assoc().associate([p], [o])[0].evidence[0]
        self.assertFalse(ev.keypoint.available)
        self.assertIsNone(ev.keypoint.nearest_hand_distance)
        self.assertEqual(ev.keypoint.hand_keypoint_count, 0)

    def test_kp_low_confidence_unavailable(self):
        vis = tuple([0.0]*9 + [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0])
        p = PersonObservation(
            1, "no-cheating", (0.0, 0.0, 200.0, 400.0),
            keypoints=PersonKeypoints(
                x=tuple([0.0]*17),
                y=tuple([0.0]*17),
                visibility=vis,
            ),
        )
        o = ObjectObservation(5, "smartphone", (80.0, 180.0, 110.0, 200.0))
        ev = self._assoc().associate([p], [o])[0].evidence[0]
        self.assertFalse(ev.keypoint.available)

    def test_invalid_bbox_raises(self):
        p = PersonObservation(1, "no-cheating", (200.0, 0.0, 0.0, 400.0))
        o = ObjectObservation(5, "smartphone", (80.0, 180.0, 110.0, 200.0))
        with self.assertRaises(ValueError):
            self._assoc().associate([p], [o])

    def test_non_list_persons_raises(self):
        o = ObjectObservation(5, "smartphone", (80.0, 180.0, 110.0, 200.0))
        with self.assertRaises(ValueError):
            self._assoc().associate("not a list", [o])

    def test_containment_threshold_range(self):
        with self.assertRaises(ValueError):
            PersonObjectAssociator(containment_threshold=1.5)

    def test_visibility_cutoff_range(self):
        with self.assertRaises(ValueError):
            PersonObjectAssociator(visibility_cutoff=1.5)

    def test_temporal_passed_through(self):
        p = PersonObservation(1, "no-cheating", (0.0, 0.0, 200.0, 400.0))
        o = ObjectObservation(5, "smartphone", (80.0, 180.0, 110.0, 200.0))
        ev = self._assoc().associate(
            [p], [o],
            temporal={5: TemporalEvidence(True, 8.0, 240)},
        )[0].evidence[0]
        self.assertIsNotNone(ev.temporal)
        self.assertTrue(ev.temporal.overlap)
        self.assertEqual(ev.temporal.observation_count, 240)

    def test_temporal_none_default(self):
        p = PersonObservation(1, "no-cheating", (0.0, 0.0, 200.0, 400.0))
        o = ObjectObservation(5, "smartphone", (80.0, 180.0, 110.0, 200.0))
        ev = self._assoc().associate([p], [o])[0].evidence[0]
        self.assertIsNone(ev.temporal)


# ---------------------------------------------------------------------------
# Observation extension (Phase 2B.5 additive)
# ---------------------------------------------------------------------------

class TestObservationKeypoints(unittest.TestCase):

    def test_default_keypoints_none(self):
        obs = Observation(
            frame_index=1, video_time=0.033,
            class_name="no-cheating", confidence=0.9,
            bbox=(10, 10, 100, 200),
            source=ModelSource.POSE, track_id=1,
        )
        self.assertIsNone(obs.keypoints)

    def test_with_keypoints_to_dict(self):
        obs = Observation(
            frame_index=1, video_time=0.033,
            class_name="no-cheating", confidence=0.9,
            bbox=(10, 10, 100, 200),
            source=ModelSource.POSE, track_id=1,
            keypoints=((85.0, 180.0, 0.9), (190.0, 195.0, 0.85)),
        )
        d = obs.to_dict()
        self.assertIn("keypoints", d)
        self.assertEqual(len(d["keypoints"]), 2)

    def test_without_keypoints_no_key(self):
        obs = Observation(
            frame_index=1, video_time=0.033,
            class_name="smartphone", confidence=0.8,
            bbox=(50, 60, 100, 110),
            source=ModelSource.OBJECT, track_id=5,
        )
        self.assertNotIn("keypoints", obs.to_dict())


if __name__ == "__main__":
    unittest.main()
