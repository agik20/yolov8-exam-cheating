"""Phase 2C.1 — Ground-truth schema and dataset manifest tests.

Run:  python -m pytest tests/test_phase2c1_ground_truth.py -v

Pins the ground-truth dataset and annotation schema:
- GTLabel enum (NORMAL / SUSPICIOUS / AMBIGUOUS)
- EntityReferenceStatus enum (KNOWN / UNKNOWN / NOT_APPLICABLE / AMBIGUOUS)
- GroundTruthIncident (typed, serializable, validated)
- NormalPeriod / AmbiguousPeriod
- AnnotationProvenance
- SceneTag enum + SourceVideoMetadata + DatasetManifest
- GroundTruthDataset aggregate
- Serialization: json.dumps(...) succeeds, no numpy/cv2/datetime leakage
- Independence: no model output required
- Validation: interval bounds, NaN/inf, duplicate IDs, unknown labels, overlaps
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

from inference.evaluation.ground_truth import (
    GTLabel,
    EntityReferenceStatus,
    SceneTag,
    ReviewStatus,
    AnnotationProvenance,
    GroundTruthIncident,
    NormalPeriod,
    AmbiguousPeriod,
    SourceVideoMetadata,
    DatasetManifest,
    GroundTruthDataset,
    validate_incident,
    validate_dataset,
    find_overlapping_incidents,
    VALID_BEHAVIOR_CLASSES,
    VALID_POSE_CLASSES,
    VALID_OBJECT_CLASSES,
    SCHEMA_VERSION,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_video(video_id="vid1", fps=30.0, duration=100.0):
    return SourceVideoMetadata(
        video_id=video_id,
        reference="vid1",
        source_fps=fps,
        duration_seconds=duration,
    )


def _make_incident(
    video_id="vid1",
    incident_id="inc1",
    label=GTLabel.SUSPICIOUS,
    start=10.0,
    end=20.0,
    behavior_class="provide-object",
    person_ref_status=EntityReferenceStatus.UNKNOWN,
    object_ref_status=EntityReferenceStatus.NOT_APPLICABLE,
    person_reference=None,
    object_reference=None,
    start_frame=None,
    end_frame=None,
    notes=None,
):
    return GroundTruthIncident(
        video_id=video_id,
        incident_id=incident_id,
        label=label,
        start_video_time=start,
        end_video_time=end,
        behavior_class=behavior_class,
        person_ref_status=person_ref_status,
        object_ref_status=object_ref_status,
        person_reference=person_reference,
        object_reference=object_reference,
        start_frame=start_frame,
        end_frame=end_frame,
        notes=notes,
    )


def _make_manifest(incidents=None, videos=None):
    return DatasetManifest(
        version=SCHEMA_VERSION,
        created_at="2026-01-01T00:00:00Z",
        videos=videos or [],
        incidents=incidents or [],
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums(unittest.TestCase):

    def test_gtlabel_values(self):
        self.assertEqual(GTLabel.NORMAL.value, "NORMAL")
        self.assertEqual(GTLabel.SUSPICIOUS.value, "SUSPICIOUS")
        self.assertEqual(GTLabel.AMBIGUOUS.value, "AMBIGUOUS")

    def test_gtlabel_from_string(self):
        self.assertIs(GTLabel("NORMAL"), GTLabel.NORMAL)
        self.assertIs(GTLabel("AMBIGUOUS"), GTLabel.AMBIGUOUS)

    def test_entity_reference_status_values(self):
        self.assertEqual(EntityReferenceStatus.KNOWN.value, "KNOWN")
        self.assertEqual(EntityReferenceStatus.UNKNOWN.value, "UNKNOWN")
        self.assertEqual(EntityReferenceStatus.NOT_APPLICABLE.value, "NOT_APPLICABLE")
        self.assertEqual(EntityReferenceStatus.AMBIGUOUS.value, "AMBIGUOUS")

    def test_scene_tag_values(self):
        self.assertEqual(SceneTag.NORMAL_EXAM.value, "normal_exam")
        self.assertEqual(SceneTag.OCCLUSION.value, "occlusion")
        self.assertEqual(SceneTag.MULTI_PERSON.value, "multi_person")
        self.assertEqual(SceneTag.CROWDED.value, "crowded")

    def test_review_status_values(self):
        self.assertEqual(ReviewStatus.DRAFT.value, "DRAFT")
        self.assertEqual(ReviewStatus.APPROVED.value, "APPROVED")
        self.assertEqual(ReviewStatus.ADJUDICATED.value, "ADJUDICATED")

    def test_valid_behavior_classes_match_model(self):
        for c in VALID_POSE_CLASSES:
            self.assertIn(c, VALID_BEHAVIOR_CLASSES)
        for c in VALID_OBJECT_CLASSES:
            self.assertIn(c, VALID_BEHAVIOR_CLASSES)
        # spot-check against the actual model class names
        self.assertIn("no-cheating", VALID_BEHAVIOR_CLASSES)
        self.assertIn("provide-object", VALID_BEHAVIOR_CLASSES)
        self.assertIn("see-friends-work", VALID_BEHAVIOR_CLASSES)
        self.assertIn("smartphone", VALID_BEHAVIOR_CLASSES)
        self.assertIn("calculator", VALID_BEHAVIOR_CLASSES)


# ---------------------------------------------------------------------------
# Schema: valid incident construction
# ---------------------------------------------------------------------------

class TestIncidentSchema(unittest.TestCase):

    def test_valid_suspicious_incident(self):
        inc = _make_incident()
        self.assertEqual(inc.video_id, "vid1")
        self.assertEqual(inc.incident_id, "inc1")
        self.assertIs(inc.label, GTLabel.SUSPICIOUS)
        self.assertEqual(inc.behavior_class, "provide-object")
        self.assertEqual(inc.duration_seconds, 10.0)

    def test_zero_duration_incident_valid(self):
        inc = _make_incident(start=5.0, end=5.0)
        self.assertEqual(inc.duration_seconds, 0.0)
        errors = validate_incident(inc)
        self.assertEqual(errors, [])

    def test_positive_duration_incident(self):
        inc = _make_incident(start=1.0, end=30.0)
        self.assertEqual(inc.duration_seconds, 29.0)

    def test_normal_label_incident(self):
        inc = _make_incident(label=GTLabel.NORMAL, behavior_class="no-cheating")
        self.assertIs(inc.label, GTLabel.NORMAL)
        self.assertEqual(inc.behavior_class, "no-cheating")

    def test_ambiguous_label_incident(self):
        inc = _make_incident(label=GTLabel.AMBIGUOUS, behavior_class="smartphone")
        self.assertIs(inc.label, GTLabel.AMBIGUOUS)

    def test_person_ref_known(self):
        inc = _make_incident(
            person_ref_status=EntityReferenceStatus.KNOWN,
            person_reference="student_A",
        )
        self.assertEqual(inc.person_ref_status, EntityReferenceStatus.KNOWN)
        self.assertEqual(inc.person_reference, "student_A")

    def test_person_ref_unknown(self):
        inc = _make_incident(
            person_ref_status=EntityReferenceStatus.UNKNOWN,
            person_reference=None,
        )
        self.assertIsNone(inc.person_reference)

    def test_object_ref_not_applicable(self):
        inc = _make_incident(object_ref_status=EntityReferenceStatus.NOT_APPLICABLE)
        self.assertIs(inc.object_ref_status, EntityReferenceStatus.NOT_APPLICABLE)
        self.assertIsNone(inc.object_reference)

    def test_object_ref_ambiguous(self):
        inc = _make_incident(
            object_ref_status=EntityReferenceStatus.AMBIGUOUS,
            object_reference=None,
        )
        self.assertIs(inc.object_ref_status, EntityReferenceStatus.AMBIGUOUS)
        self.assertIsNone(inc.object_reference)

    def test_with_frames(self):
        inc = _make_incident(start_frame=300, end_frame=600, start=10.0, end=20.0)
        self.assertEqual(inc.start_frame, 300)
        self.assertEqual(inc.end_frame, 600)

    def test_provenance_default(self):
        inc = _make_incident()
        self.assertEqual(inc.provenance.schema_version, SCHEMA_VERSION)
        self.assertIsNone(inc.provenance.annotator_id)
        self.assertIsNone(inc.provenance.review_status)

    def test_provenance_populated(self):
        prov = AnnotationProvenance(
            schema_version=SCHEMA_VERSION,
            annotator_id="ann-001",
            annotation_id="ann_vid1_0001",
            created_at="2026-01-15T10:30:00Z",
            review_status=ReviewStatus.APPROVED,
        )
        inc = GroundTruthIncident(
            video_id="vid1",
            incident_id="inc1",
            label=GTLabel.SUSPICIOUS,
            start_video_time=0.0,
            end_video_time=5.0,
            behavior_class="see-friends-work",
            provenance=prov,
        )
        self.assertEqual(inc.provenance.annotator_id, "ann-001")
        self.assertIs(inc.provenance.review_status, ReviewStatus.APPROVED)

    def test_incident_is_frozen(self):
        inc = _make_incident()
        with self.assertRaises(AttributeError):
            inc.label = GTLabel.NORMAL  # type: ignore


# ---------------------------------------------------------------------------
# Temporal validation
# ---------------------------------------------------------------------------

class TestTemporalValidation(unittest.TestCase):

    def test_valid_interval_passes(self):
        inc = _make_incident(start=0.0, end=60.0)
        self.assertEqual(validate_incident(inc), [])

    def test_start_after_end_fails(self):
        inc = _make_incident(start=30.0, end=10.0)
        errors = validate_incident(inc)
        self.assertTrue(any("must be <=" in e for e in errors))

    def test_negative_start_fails(self):
        inc = _make_incident(start=-1.0, end=10.0)
        errors = validate_incident(inc)
        self.assertTrue(any(">= 0" in e for e in errors))

    def test_negative_end_fails(self):
        inc = _make_incident(start=0.0, end=-5.0)
        errors = validate_incident(inc)
        self.assertTrue(any(">= 0" in e or "must be <=" in e for e in errors))

    def test_nan_start_fails(self):
        inc = _make_incident(start=float('nan'), end=10.0)
        errors = validate_incident(inc)
        self.assertTrue(any("finite" in e for e in errors))

    def test_inf_end_fails(self):
        inc = _make_incident(start=0.0, end=float('inf'))
        errors = validate_incident(inc)
        self.assertTrue(any("finite" in e for e in errors))

    def test_zero_duration_passes(self):
        inc = _make_incident(start=5.0, end=5.0)
        self.assertEqual(validate_incident(inc), [])

    def test_start_frame_consistency_with_fps(self):
        # frame 300 at 30 fps => 10.0 s
        inc = _make_incident(start=10.0, end=20.0, start_frame=300, end_frame=600)
        vid = _make_video(fps=30.0)
        errors = validate_incident(inc, vid)
        self.assertEqual(errors, [])

    def test_start_frame_inconsistency_with_fps(self):
        # frame 300 at 30 fps => 10.0 s, but we claim 5.0 s
        inc = _make_incident(start=5.0, end=20.0, start_frame=300, end_frame=600)
        vid = _make_video(fps=30.0)
        errors = validate_incident(inc, vid)
        self.assertTrue(any("implies video_time" in e for e in errors))

    def test_incident_exceeds_video_duration(self):
        inc = _make_incident(start=90.0, end=120.0)
        vid = _make_video(fps=30.0, duration=100.0)
        errors = validate_incident(inc, vid)
        self.assertTrue(any("exceeds" in e for e in errors))

    def test_negative_frame_fails(self):
        inc = _make_incident(start=0.0, end=10.0, start_frame=-1, end_frame=300)
        errors = validate_incident(inc)
        self.assertTrue(any("start_frame must be >= 0" in e for e in errors))

    def test_start_frame_gt_end_frame_fails(self):
        inc = _make_incident(start=0.0, end=10.0, start_frame=600, end_frame=300)
        errors = validate_incident(inc)
        self.assertTrue(any("start_frame" in e and "end_frame" in e for e in errors))


# ---------------------------------------------------------------------------
# Validation: invalid behavior_class
# ---------------------------------------------------------------------------

class TestBehaviorClassValidation(unittest.TestCase):

    def test_valid_behavior_classes_pass(self):
        for cls_name in ["provide-object", "see-friends-work", "smartphone", "calculator", "no-cheating"]:
            inc = _make_incident(behavior_class=cls_name)
            errors = validate_incident(inc)
            self.assertEqual(errors, [], f"expected no errors for {cls_name!r}")

    def test_invalid_behavior_class_fails(self):
        inc = _make_incident(behavior_class="using-laptop")
        errors = validate_incident(inc)
        self.assertTrue(any("behavior_class" in e and "not in" in e for e in errors))

    def test_empty_behavior_class_fails(self):
        inc = _make_incident(behavior_class="")
        errors = validate_incident(inc)
        self.assertTrue(any("behavior_class" in e for e in errors))


# ---------------------------------------------------------------------------
# Serialization: JSON round-trip
# ---------------------------------------------------------------------------

class TestSerialization(unittest.TestCase):

    def test_incident_json_dumps_no_numpy_or_datetime(self):
        inc = _make_incident(
            start=10.0, end=20.0,
            start_frame=300, end_frame=600,
            person_reference="student_A",
            notes="difficult negative: hand near face",
        )
        raw = json.dumps(inc.to_dict())
        # json.dumps would raise on numpy arrays or datetime objects
        self.assertIsInstance(raw, str)
        self.assertIn("student_A", raw)

    def test_incident_round_trip(self):
        inc = _make_incident(
            start=10.0, end=20.0,
            start_frame=300, end_frame=600,
            person_reference="student_A",
            object_reference="phone_blue",
            notes="test",
        )
        raw = json.dumps(inc.to_dict())
        restored = GroundTruthIncident.from_dict(json.loads(raw))
        self.assertEqual(restored.incident_id, inc.incident_id)
        self.assertEqual(restored.video_id, inc.video_id)
        self.assertIs(restored.label, inc.label)
        self.assertAlmostEqual(restored.start_video_time, inc.start_video_time)
        self.assertAlmostEqual(restored.end_video_time, inc.end_video_time)
        self.assertEqual(restored.behavior_class, inc.behavior_class)
        self.assertEqual(restored.start_frame, inc.start_frame)
        self.assertEqual(restored.end_frame, inc.end_frame)
        self.assertEqual(restored.person_reference, inc.person_reference)
        self.assertEqual(restored.object_reference, inc.object_reference)
        self.assertEqual(restored.notes, inc.notes)
        self.assertIs(restored.person_ref_status, inc.person_ref_status)
        self.assertIs(restored.object_ref_status, inc.object_ref_status)

    def test_incident_round_trip_defaults(self):
        inc = _make_incident()
        raw = json.dumps(inc.to_dict())
        restored = GroundTruthIncident.from_dict(json.loads(raw))
        self.assertIs(restored.person_ref_status, EntityReferenceStatus.UNKNOWN)
        self.assertIs(restored.object_ref_status, EntityReferenceStatus.NOT_APPLICABLE)
        self.assertIsNone(restored.start_frame)
        self.assertIsNone(restored.notes)

    def test_manifest_to_json(self):
        inc = _make_incident()
        vid = _make_video()
        manifest = _make_manifest(incidents=[inc], videos=[vid])
        raw = manifest.to_json()
        self.assertIsInstance(raw, str)
        parsed = json.loads(raw)
        self.assertEqual(parsed["version"], SCHEMA_VERSION)
        self.assertEqual(len(parsed["incidents"]), 1)
        self.assertEqual(len(parsed["videos"]), 1)

    def test_manifest_round_trip(self):
        inc = _make_incident()
        vid = _make_video()
        manifest = _make_manifest(incidents=[inc], videos=[vid])
        raw = manifest.to_json()
        restored = DatasetManifest.from_json(raw)
        self.assertEqual(restored.version, manifest.version)
        self.assertEqual(len(restored.incidents), 1)
        self.assertEqual(len(restored.videos), 1)
        self.assertEqual(
            restored.incidents[0].incident_id,
            manifest.incidents[0].incident_id,
        )
        self.assertEqual(
            restored.videos[0].video_id,
            manifest.videos[0].video_id,
        )

    def test_normal_period_round_trip(self):
        p = NormalPeriod(
            video_id="vid1",
            period_id="norm1",
            start_video_time=0.0,
            end_video_time=5.0,
        )
        raw = json.dumps(p.to_dict())
        restored = NormalPeriod.from_dict(json.loads(raw))
        self.assertEqual(restored.period_id, p.period_id)
        self.assertAlmostEqual(restored.start_video_time, p.start_video_time)

    def test_ambiguous_period_round_trip(self):
        p = AmbiguousPeriod(
            video_id="vid1",
            period_id="amb1",
            start_video_time=10.0,
            end_video_time=15.0,
            reason="occlusion by another student",
        )
        raw = json.dumps(p.to_dict())
        restored = AmbiguousPeriod.from_dict(json.loads(raw))
        self.assertEqual(restored.reason, p.reason)
        self.assertEqual(restored.period_id, p.period_id)

    def test_source_video_metadata_round_trip(self):
        v = SourceVideoMetadata(
            video_id="vid1",
            reference="sk2",
            duration_seconds=100.0,
            source_fps=30.0,
            scene_tags=(SceneTag.MULTI_PERSON, SceneTag.OCCLUSION),
        )
        raw = json.dumps(v.to_dict())
        restored = SourceVideoMetadata.from_dict(json.loads(raw))
        self.assertEqual(restored.video_id, v.video_id)
        self.assertEqual(restored.source_fps, v.source_fps)
        self.assertIn(SceneTag.MULTI_PERSON, restored.scene_tags)
        self.assertIn(SceneTag.OCCLUSION, restored.scene_tags)


# ---------------------------------------------------------------------------
# Independence: no model output required
# ---------------------------------------------------------------------------

class TestIndependence(unittest.TestCase):

    def test_no_model_classes_in_imports(self):
        """The ground-truth module must not import any inference pipeline modules."""
        import inference.evaluation.ground_truth as gt_module
        import inspect
        source = inspect.getsource(gt_module)
        self.assertNotIn("from inference.core", source)
        self.assertNotIn("from inference.config", source)
        self.assertNotIn("import cv2", source)
        self.assertNotIn("import numpy", source)
        self.assertNotIn("import ultralytics", source)

    def test_incident_does_not_require_tracker_ids(self):
        """Tracker IDs are NOT part of the ground-truth identity model."""
        inc = _make_incident(person_reference="student_A")
        d = inc.to_dict()
        self.assertNotIn("track_id", d)
        self.assertNotIn("entity_key", d)
        self.assertNotIn("event_id", d)

# ---------------------------------------------------------------------------
# Overlapping incidents
# ---------------------------------------------------------------------------

class TestOverlappingIncidents(unittest.TestCase):

    def test_disjoint_intervals_do_not_overlap(self):
        a = _make_incident(incident_id="a", start=0.0,  end=10.0)
        b = _make_incident(incident_id="b", start=10.0, end=20.0)
        overlaps = find_overlapping_incidents([a, b])
        self.assertEqual(overlaps, [])

    def test_overlapping_intervals_detected(self):
        a = _make_incident(incident_id="a", start=10.0, end=20.0)
        b = _make_incident(incident_id="b", start=15.0, end=25.0)
        overlaps = find_overlapping_incidents([a, b])
        self.assertEqual(len(overlaps), 1)
        self.assertIs(overlaps[0][0], a)
        self.assertIs(overlaps[0][1], b)

    def test_contained_interval_detected(self):
        a = _make_incident(incident_id="a", start=0.0,  end=100.0)
        b = _make_incident(incident_id="b", start=30.0, end=70.0)
        overlaps = find_overlapping_incidents([a, b])
        self.assertEqual(len(overlaps), 1)

    def test_same_interval_detected(self):
        a = _make_incident(incident_id="a", start=10.0, end=20.0)
        b = _make_incident(incident_id="b", start=10.0, end=20.0)
        overlaps = find_overlapping_incidents([a, b])
        self.assertEqual(len(overlaps), 1)

    def test_incidents_in_different_videos_not_compared(self):
        a = _make_incident(video_id="vid1", incident_id="a", start=0.0, end=10.0)
        b = _make_incident(video_id="vid2", incident_id="b", start=0.0, end=10.0)
        overlaps = find_overlapping_incidents([a, b])
        self.assertEqual(overlaps, [])

    def test_overlapping_incidents_remain_distinct(self):
        a = _make_incident(incident_id="a", start=10.0, end=20.0)
        b = _make_incident(incident_id="b", start=15.0, end=25.0)
        _ = find_overlapping_incidents([a, b])
        self.assertEqual(a.incident_id, "a")
        self.assertEqual(b.incident_id, "b")
        self.assertEqual(a.end_video_time, 20.0)
        self.assertEqual(b.end_video_time, 25.0)

    def test_touching_intervals_not_overlapping(self):
        a = _make_incident(incident_id="a", start=0.0,  end=10.0)
        b = _make_incident(incident_id="b", start=10.0, end=20.0)
        overlaps = find_overlapping_incidents([a, b])
        self.assertEqual(overlaps, [])

# ---------------------------------------------------------------------------
# Dataset-level validation
# ---------------------------------------------------------------------------

class TestDatasetValidation(unittest.TestCase):

    def test_valid_dataset_passes(self):
        vid = _make_video()
        inc = _make_incident()
        manifest = _make_manifest(incidents=[inc], videos=[vid])
        dataset = GroundTruthDataset(manifests=[manifest])
        issues = validate_dataset(dataset)
        self.assertEqual(issues, [])

    def test_duplicate_incident_ids_detected(self):
        vid = _make_video()
        inc1 = _make_incident(incident_id="same-id")
        inc2 = _make_incident(incident_id="same-id")
        manifest = _make_manifest(incidents=[inc1, inc2], videos=[vid])
        dataset = GroundTruthDataset(manifests=[manifest])
        issues = validate_dataset(dataset)
        self.assertTrue(any("DUPLICATE incident_id" in i for i in issues))

    def test_duplicate_video_ids_detected(self):
        v1 = _make_video(video_id="vid1")
        v2 = _make_video(video_id="vid1")
        manifest = _make_manifest(videos=[v1, v2])
        dataset = GroundTruthDataset(manifests=[manifest])
        issues = validate_dataset(dataset)
        self.assertTrue(any("DUPLICATE video_id" in i for i in issues))

    def test_incident_references_missing_video(self):
        inc = _make_incident(video_id="vid1")
        manifest = _make_manifest(incidents=[inc])
        dataset = GroundTruthDataset(manifests=[manifest])
        issues = validate_dataset(dataset)
        self.assertTrue(any("MISSING VIDEO" in i for i in issues))

    def test_empty_dataset_flagged(self):
        dataset = GroundTruthDataset(manifests=[_make_manifest()])
        issues = validate_dataset(dataset)
        self.assertTrue(any("EMPTY DATASET" in i for i in issues))

    def test_invalid_interval_in_dataset_detected(self):
        vid = _make_video()
        inc = _make_incident(start=30.0, end=10.0)
        manifest = _make_manifest(incidents=[inc], videos=[vid])
        dataset = GroundTruthDataset(manifests=[manifest])
        issues = validate_dataset(dataset)
        self.assertTrue(any("must be <=" in i for i in issues))

    def test_unknown_behavior_class_in_dataset_detected(self):
        vid = _make_video()
        inc = _make_incident(behavior_class="unknown-class")
        manifest = _make_manifest(incidents=[inc], videos=[vid])
        dataset = GroundTruthDataset(manifests=[manifest])
        issues = validate_dataset(dataset)
        self.assertTrue(any("behavior_class" in i for i in issues))


# ---------------------------------------------------------------------------
# Normal / Ambiguous period validation
# ---------------------------------------------------------------------------

class TestPeriodValidation(unittest.TestCase):

    def test_normal_period_valid(self):
        p = NormalPeriod(video_id="vid1", period_id="n1",
                         start_video_time=0.0, end_video_time=10.0)
        vid = _make_video()
        manifest = DatasetManifest(
            version=SCHEMA_VERSION,
            created_at="2026-01-01T00:00:00Z",
            videos=[vid],
            normal_periods=[p],
        )
        dataset = GroundTruthDataset(manifests=[manifest])
        issues = validate_dataset(dataset)
        self.assertEqual(issues, [])

    def test_normal_period_start_after_end(self):
        p = NormalPeriod(video_id="vid1", period_id="n1",
                         start_video_time=20.0, end_video_time=10.0)
        vid = _make_video()
        manifest = DatasetManifest(
            version=SCHEMA_VERSION,
            created_at="2026-01-01T00:00:00Z",
            videos=[vid],
            normal_periods=[p],
        )
        dataset = GroundTruthDataset(manifests=[manifest])
        issues = validate_dataset(dataset)
        self.assertTrue(any("must be <=" in i for i in issues))

    def test_ambiguous_period_valid(self):
        p = AmbiguousPeriod(
            video_id="vid1", period_id="a1",
            start_video_time=0.0, end_video_time=10.0,
            reason="occlusion",
        )
        vid = _make_video()
        manifest = DatasetManifest(
            version=SCHEMA_VERSION,
            created_at="2026-01-01T00:00:00Z",
            videos=[vid],
            ambiguous_periods=[p],
        )
        dataset = GroundTruthDataset(manifests=[manifest])
        issues = validate_dataset(dataset)
        self.assertEqual(issues, [])


# ---------------------------------------------------------------------------
# Ambiguity: AMBIGUOUS remains explicit
# ---------------------------------------------------------------------------

class TestAmbiguityExplicitness(unittest.TestCase):

    def test_ambiguous_label_not_coerced(self):
        inc = _make_incident(label=GTLabel.AMBIGUOUS, behavior_class="smartphone")
        d = inc.to_dict()
        self.assertEqual(d["label"], "AMBIGUOUS")
        restored = GroundTruthIncident.from_dict(d)
        self.assertIs(restored.label, GTLabel.AMBIGUOUS)

    def test_ambiguous_period_distinct_from_normal(self):
        ap = AmbiguousPeriod(
            video_id="vid1", period_id="amb1",
            start_video_time=0.0, end_video_time=5.0,
            reason="blur",
        )
        np_ = NormalPeriod(
            video_id="vid1", period_id="norm1",
            start_video_time=0.0, end_video_time=5.0,
        )
        manifest = DatasetManifest(
            version=SCHEMA_VERSION,
            created_at="2026-01-01T00:00:00Z",
            normal_periods=[np_],
            ambiguous_periods=[ap],
            videos=[_make_video()],
        )
        dataset = GroundTruthDataset(manifests=[manifest])
        self.assertEqual(len(dataset.all_incidents()), 0)
        self.assertEqual(len(manifest.ambiguous_periods), 1)
        self.assertEqual(len(manifest.normal_periods), 1)


# ---------------------------------------------------------------------------
# Regression: Phase 2B files must be unchanged
# ---------------------------------------------------------------------------

class TestPhase2BRegression(unittest.TestCase):

    def test_events_module_still_importable(self):
        import inference.core.events as events
        self.assertTrue(hasattr(events, "EvidenceBundle"))
        self.assertTrue(hasattr(events, "EntityKey"))
        self.assertTrue(hasattr(events, "GateOutcome"))

    def test_shadow_config_still_importable(self):
        import inference.config.shadow_config as sc
        self.assertTrue(hasattr(sc, "ShadowConfig"))

    def test_pipeline_still_importable(self):
        import inference.core.pipeline
        self.assertTrue(hasattr(inference.core.pipeline, "CheatingDetectionPipeline"))
