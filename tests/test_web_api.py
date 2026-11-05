"""Tests for the new REST API endpoints in inference/web/app.py.

Covers:
- GET /api/status — correct shape with a mock pipeline
- GET /api/events — empty list when no shadow events
- GET /api/events/<id> — 404 for unknown id
- GET /api/events/<id> — full evidence bundle (None-safe serialization)
- GET /api/legacy-tracks — pose + object sections
- GET /api/config — legacy and shadow keys present
- GET /api/calibration — NOT READY with empty GT dataset
- GET /api/environment — 200 with platform key
- GET / — serves dashboard.html when static file exists, falls back to HTML
- GET /video_feed — mimetype unchanged
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from inference.core.confirmation import ConfirmationGateResult
from inference.core.events import EntityKey, EventRecord, ModelSource, ConfidenceStats
from inference.core.shadow_lifecycle import ShadowEventRecord
from inference.web.app import build_app


def _mk_mock_pipeline(
    frame_idx: int = 42,
    src_fps: float = 30.0,
    start_time: float = 1000.0,
    pose_tracks: int = 2,
    object_tracks: int = 1,
    shadow_events: list = None,
    shadow_config=None,
) -> MagicMock:
    """Build a lightweight mock that mimics the pipeline attributes
    the web endpoints access — no real model loading required.
    """
    from datetime import datetime
    import time

    mock = MagicMock()
    mock.frame_idx = frame_idx
    mock._src_fps = src_fps
    mock.start_time = start_time
    mock.id_info_pose = {i: {"first_time": 0, "last_saved": 10,
                             "last_class": "no-cheating", "last_conf": 0.5}
                         for i in range(pose_tracks)}
    mock.id_info_obj = {i: {"first_time": 0, "last_saved": 10,
                            "last_class": "smartphone", "last_conf": 0.6}
                        for i in range(object_tracks)}
    mock.shadow_config = shadow_config

    if shadow_events is None:
        mock._shadow_lifecycle = None
    else:
        mock._shadow_lifecycle = SimpleNamespace(shadow_events=shadow_events)

    # generate_frames must yield a generator for the Response
    def _gen():
        yield b""
    mock.generate_frames = _gen
    return mock


# ---------------------------------------------------------------------------
# /api/status
# ---------------------------------------------------------------------------


class TestApiStatus(unittest.TestCase):

    def test_status_empty_pipeline(self):
        from inference.web.app import build_app
        pipeline = _mk_mock_pipeline()
        client = build_app(pipeline).test_client()
        res = client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("frame_idx", data)
        self.assertIn("src_fps", data)
        self.assertIn("processing_fps", data)
        self.assertIn("pose_tracks", data)
        self.assertIn("object_tracks", data)
        self.assertIn("shadow_event_count", data)
        self.assertEqual(data["frame_idx"], 42)
        self.assertEqual(data["pose_tracks"], 2)
        self.assertEqual(data["object_tracks"], 1)
        self.assertEqual(data["shadow_event_count"], 0)

    def test_status_with_shadow_events(self):
        from inference.web.app import build_app
        ev = MagicMock()
        ev.to_dict = lambda: {"event_id": "abc"}
        pipeline = _mk_mock_pipeline(shadow_events=[ev])
        client = build_app(pipeline).test_client()
        data = client.get("/api/status").get_json()
        self.assertEqual(data["shadow_event_count"], 1)


# ---------------------------------------------------------------------------
# /api/events
# ---------------------------------------------------------------------------


class TestApiEvents(unittest.TestCase):

    def _mk_event(self, event_id, source, track_id, class_name, confirmed,
                  reason, unmet=None):
        entity_key = EntityKey(source=source, track_id=track_id)
        ev = ShadowEventRecord(
            event=EventRecord(
                event_id=event_id,
                entity_key=entity_key,
                class_name=class_name,
                confirmation_video_time=5.0,
                confirmation_wall_time="2026-01-01T00:00:00",
                candidate_start_video_time=1.0,
                candidate_end_video_time=5.0,
                observation_count=3,
                confidence_stats=None,
                event_confidence=None,
                association_score=None,
                contributing_track_ids=frozenset({track_id}),
            ),
            gate_result=ConfirmationGateResult(
                entity_key=entity_key,
                candidate_class=class_name,
                confirmed=confirmed,
                reason=reason,
                unmet_requirements=unmet or [],
            ),
            recorded_at_video_time=5.0,
        )
        return ev

    def test_events_empty(self):
        client = build_app(_mk_mock_pipeline()).test_client()
        data = client.get("/api/events").get_json()
        self.assertEqual(data, {"events": []})

    def test_events_with_data(self):
        ev = self._mk_event("test-ev-001", ModelSource.POSE, 1,
                            "provide-object", True, "all gates met")
        pipeline = _mk_mock_pipeline(shadow_events=[ev])
        client = build_app(pipeline).test_client()
        data = client.get("/api/events").get_json()
        self.assertEqual(len(data["events"]), 1)
        self.assertEqual(data["events"][0]["event_id"], "test-ev-001")
        self.assertEqual(data["events"][0]["class_name"], "provide-object")
        self.assertTrue(data["events"][0]["gate_confirmed"])

    def test_event_detail_404(self):
        client = build_app(_mk_mock_pipeline()).test_client()
        res = client.get("/api/events/nonexistent")
        self.assertEqual(res.status_code, 404)

    def test_event_detail_found(self):
        ev = self._mk_event("ev-xyz", ModelSource.OBJECT, 9,
                            "smartphone", False, "insufficient observations",
                            unmet=["min_observations"])
        pipeline = _mk_mock_pipeline(shadow_events=[ev])
        client = build_app(pipeline).test_client()
        data = client.get("/api/events/ev-xyz").get_json()
        self.assertEqual(data["event"]["event_id"], "ev-xyz")
        self.assertFalse(data["event"]["gate_confirmed"])


# ---------------------------------------------------------------------------
# /api/legacy-tracks
# ---------------------------------------------------------------------------


class TestApiLegacyTracks(unittest.TestCase):

    def test_shape(self):
        from inference.web.app import build_app
        pipeline = _mk_mock_pipeline(pose_tracks=3, object_tracks=2)
        client = build_app(pipeline).test_client()
        data = client.get("/api/legacy-tracks").get_json()
        self.assertIn("pose", data)
        self.assertIn("object", data)
        self.assertEqual(len(data["pose"]), 3)
        self.assertEqual(len(data["object"]), 2)


# ---------------------------------------------------------------------------
# /api/config
# ---------------------------------------------------------------------------


class TestApiConfig(unittest.TestCase):

    def test_no_shadow_config(self):
        client = build_app(_mk_mock_pipeline()).test_client()
        data = client.get("/api/config").get_json()
        self.assertIn("legacy", data)
        self.assertIn("shadow", data)
        self.assertIsNone(data["shadow"])
        self.assertIn("CONF_THRESH_POSE", data["legacy"])
        self.assertIn("NMS_IOU_THRESHOLD", data["legacy"])

    def test_with_shadow_config(self):
        from inference.config.shadow_config import ShadowConfig
        cfg = ShadowConfig()
        client = build_app(_mk_mock_pipeline(shadow_config=cfg)).test_client()
        data = client.get("/api/config").get_json()
        self.assertIsNotNone(data["shadow"])
        self.assertIn("confirmation", data["shadow"])
        self.assertIn("cooldown", data["shadow"])


# ---------------------------------------------------------------------------
# /api/calibration — no fabricated values
# ---------------------------------------------------------------------------


class TestApiCalibration(unittest.TestCase):

    def test_not_ready_empty_gt(self):
        from inference.web.app import build_app
        client = build_app(_mk_mock_pipeline()).test_client()
        data = client.get("/api/calibration").get_json()
        self.assertIn("readiness", data)
        self.assertIn("r10_blockers", data)
        self.assertFalse(data["readiness"]["usable_for_r10"])
        self.assertGreater(len(data["r10_blockers"]), 0)

    def test_no_operating_point_key(self):
        """No operating point, recall, or precision may appear in the response."""
        from inference.web.app import build_app
        client = build_app(_mk_mock_pipeline()).test_client()
        data = client.get("/api/calibration").get_json()
        for banned in ("operating_point", "recall", "precision", "f1"):
            self.assertNotIn(banned, data["readiness"],
                             f"'{banned}' must not appear when R10 is blocked")


# ---------------------------------------------------------------------------
# / — dashboard.html
# ---------------------------------------------------------------------------


class TestIndex(unittest.TestCase):

    def test_serves_dashboard_html(self):
        from inference.web.app import build_app
        client = build_app(_mk_mock_pipeline()).test_client()
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        body = res.data.decode("utf-8")
        self.assertIn("dashboard", body.lower())
    def test_video_feed_mimetype_unchanged(self):
        from inference.web.app import build_app
        client = build_app(_mk_mock_pipeline()).test_client()
        res = client.get("/video_feed")
        self.assertIn("multipart/x-mixed-replace", res.content_type)


# ---------------------------------------------------------------------------
# /api/environment
# ---------------------------------------------------------------------------


class TestApiEnvironment(unittest.TestCase):

    def test_environment_returns_platform(self):
        """GET /api/environment must include a 'platform' key (from
        inference.utils.telemetry.get_environment_info)."""
        from inference.web.app import build_app
        client = build_app(_mk_mock_pipeline()).test_client()
        res = client.get("/api/environment")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("platform", data)


# ---------------------------------------------------------------------------
# /api/events/<id> — full evidence bundle serialization
# ---------------------------------------------------------------------------


def _mk_full_evidence_bundle_event(event_id: str) -> "ShadowEventRecord":
    """Build a ShadowEventRecord whose evidence_bundle is fully populated with
    all Phase 2B.6 / 2E-R6 additive fields — used to exercise the nested
    serialization path that ShadowEventRecord.to_dict() calls via
    evidence_bundle.to_dict().
    """
    from inference.core.events import (
        EntityKey, EventRecord, ModelSource, ConfidenceStats,
        EvidenceBundle, EventConfidenceEvidence, TrackContinuity,
    )
    from inference.core.confirmation import ConfirmationGateResult
    from inference.core.shadow_lifecycle import ShadowEventRecord

    entity_key = EntityKey(source=ModelSource.POSE, track_id=7)
    stats = ConfidenceStats(count=4, mean=0.72, min=0.61, max=0.89)
    continuity = TrackContinuity(
        track_ids=(7,), observation_count=4,
        possible_id_switch_count=0, primary_track_id=7,
    )
    event_conf_ev = EventConfidenceEvidence(
        observation_count=4,
        confidence_stats=stats,
        video_time_span=5.0,
        continuity=continuity,
        association_status=None,
        association_score=None,
        confirmation_decision=None,
        event_confidence=None,
        scoring_status="not_calibrated",
    )
    ev = EventRecord(
        event_id=event_id,
        entity_key=entity_key,
        class_name="smartphone",
        confirmation_video_time=5.0,
        confirmation_wall_time="2026-01-01T00:00:00",
        candidate_start_video_time=0.0,
        candidate_end_video_time=5.0,
        observation_count=4,
        confidence_stats=stats,
        event_confidence=None,
        association_score=None,
        contributing_track_ids=frozenset({7}),
    )
    bundle = EvidenceBundle(
        event_record=ev,
        detection_confidence_stats=stats,
        detection_observation_count=4,
        confirmation_video_time=5.0,
        confirmation_wall_time="2026-01-01T00:00:00",
        candidate_start_video_time=0.0,
        candidate_end_video_time=5.0,
        track_ids_in_window=frozenset({7}),
        decision=None,
        association=None,
        human=None,
        identity=None,
        event_confidence_evidence=event_conf_ev,
    )
    gate = ConfirmationGateResult(
        entity_key=entity_key,
        candidate_class="smartphone",
        confirmed=True,
        reason="all gates met",
        unmet_requirements=[],
    )
    return ShadowEventRecord(
        event=ev,
        gate_result=gate,
        recorded_at_video_time=5.0,
        evidence_bundle=bundle,
    )


class TestApiEventEvidenceBundle(unittest.TestCase):

    def test_full_evidence_bundle_serialization(self):
        """GET /api/events/<id> must serialize nested evidence_bundle
        (detection_confidence_stats, event_confidence_evidence,
        continuity) without error when all fields are populated."""
        from inference.web.app import build_app
        ev = _mk_full_evidence_bundle_event("ev-full")
        pipeline = _mk_mock_pipeline(shadow_events=[ev])
        client = build_app(pipeline).test_client()
        res = client.get("/api/events/ev-full")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        bundle = data["event"]["evidence_bundle"]
        self.assertIsNotNone(bundle, "evidence_bundle must be present")
        # nested confidence stats
        self.assertIsNotNone(bundle["detection_confidence_stats"])
        self.assertEqual(bundle["detection_confidence_stats"]["mean"], 0.72)
        # event_confidence_evidence with continuity
        evc = bundle["event_confidence_evidence"]
        self.assertIsNotNone(evc)
        self.assertEqual(evc["scoring_status"], "not_calibrated")
        self.assertIsNone(evc["event_confidence"])
        self.assertEqual(evc["continuity"]["track_ids"], [7])
        # None sub-fields serialize as null
        self.assertIsNone(bundle["association"])
        self.assertIsNone(bundle["human"])

    def test_confidence_stats_none_serializes_as_null(self):
        """EventRecord with confidence_stats=None must serialize to
        "confidence_stats": null in /api/events/<id> without raising."""
        from inference.web.app import build_app
        from inference.core.events import EntityKey, EventRecord, ModelSource
        from inference.core.confirmation import ConfirmationGateResult
        from inference.core.shadow_lifecycle import ShadowEventRecord

        entity_key = EntityKey(source=ModelSource.OBJECT, track_id=3)
        ev = EventRecord(
            event_id="ev-none-stats",
            entity_key=entity_key,
            class_name="no-cheating",
            confirmation_video_time=1.0,
            confirmation_wall_time="2026-01-01T00:00:00",
            candidate_start_video_time=0.0,
            candidate_end_video_time=1.0,
            observation_count=1,
            confidence_stats=None,
            event_confidence=None,
            association_score=None,
            contributing_track_ids=frozenset({3}),
        )
        gate = ConfirmationGateResult(
            entity_key=entity_key,
            candidate_class="no-cheating",
            confirmed=False,
            reason="insufficient",
            unmet_requirements=["min_observations"],
        )
        shadow_ev = ShadowEventRecord(
            event=ev,
            gate_result=gate,
            recorded_at_video_time=1.0,
            evidence_bundle=None,
        )
        pipeline = _mk_mock_pipeline(shadow_events=[shadow_ev])
        client = build_app(pipeline).test_client()
        res = client.get("/api/events/ev-none-stats")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsNone(data["event"]["confidence_stats"])


if __name__ == "__main__":
    unittest.main()
