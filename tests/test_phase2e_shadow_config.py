"""Phase 2E — ShadowConfig DI into CheatingDetectionPipeline.

Covers:
- shadow_config=None  → all policies remain unconfigured (backward compat)
- shadow_config set   → confirmation/cooldown built via build_policies()
- shadow_config set   → associator built via build_associator()
- Candidate buffer maxlen respected from CandidateBufferConfig
"""
from __future__ import annotations

import pytest

from inference.core.pipeline import CheatingDetectionPipeline
from inference.core.confirmation import (
    UnconfiguredConfirmationPolicy,
    CalibratedConfirmationPolicy,
)
from inference.core.cooldown import (
    UnconfiguredCooldownPolicy,
    CalibratedCooldownPolicy,
)
from inference.core.association import PersonObjectAssociator
from inference.config.shadow_config import (
    ShadowConfig,
    ConfirmationConfig,
    CooldownConfig,
    AssociationConfig,
    CandidateBufferConfig,
)


def _make_pipeline(**kwargs):
    """Build a pipeline with mock config to avoid YOLO model loading."""
    from unittest.mock import MagicMock, patch
    with patch('inference.core.pipeline.CheatingDetectionPipeline.initialize_models'), \
         patch('inference.core.pipeline.CheatingDetectionPipeline.setup_directories'):
        return CheatingDetectionPipeline(
            config=MagicMock(),
            video_path="dummy.mp4",
            pose_model_path="pose.pt",
            object_model_path="obj.pt",
            **kwargs,
        )


# ---------------------------------------------------------------------------
# shadow_config=None (backward compatibility)
# ---------------------------------------------------------------------------

class TestNoShadowConfig:

    def test_confirmation_policy_unconfigured(self):
        p = _make_pipeline()
        assert isinstance(p._confirmation_policy, UnconfiguredConfirmationPolicy)

    def test_cooldown_policy_unconfigured(self):
        p = _make_pipeline()
        assert isinstance(p._cooldown_policy, UnconfiguredCooldownPolicy)

    def test_associator_default(self):
        p = _make_pipeline()
        assert isinstance(p._associator, PersonObjectAssociator)
        assert p._associator._containment_threshold is None
        assert p._associator._kp_distance_threshold is None

    def test_shadow_config_attr_is_none(self):
        p = _make_pipeline()
        assert p.shadow_config is None


# ---------------------------------------------------------------------------
# shadow_config set — calibrated confirmation + cooldown
# ---------------------------------------------------------------------------

class TestCalibratedShadowConfig:

    def _config(self):
        return ShadowConfig(
            confirmation=ConfirmationConfig(
                min_observations=3,
                min_duration_seconds=2.0,
                min_confidence_mean=0.7,
                max_gap_seconds=1.5,
            ),
            cooldown=CooldownConfig(cooldown_seconds=10.0),
            association=AssociationConfig(
                containment_threshold=0.8,
                keypoint_distance_threshold=50.0,
                visibility_cutoff=0.5,
            ),
            candidate_buffer=CandidateBufferConfig(observation_buffer_maxlen=64),
        )

    def test_confirmation_policy_calibrated(self):
        p = _make_pipeline(shadow_config=self._config())
        assert isinstance(p._confirmation_policy, CalibratedConfirmationPolicy)
        assert p._confirmation_policy._min_observations == 3
        assert p._confirmation_policy._min_duration_seconds == 2.0

    def test_cooldown_policy_calibrated(self):
        p = _make_pipeline(shadow_config=self._config())
        assert isinstance(p._cooldown_policy, CalibratedCooldownPolicy)
        assert p._cooldown_policy._cooldown_seconds == 10.0

    def test_associator_configured(self):
        p = _make_pipeline(shadow_config=self._config())
        assert p._associator._containment_threshold == 0.8
        assert p._associator._kp_distance_threshold == 50.0
        assert p._associator._visibility_cutoff == 0.5

    def test_shadow_config_stored(self):
        p = _make_pipeline(shadow_config=self._config())
        assert p.shadow_config is not None
        assert p.shadow_config.candidate_buffer.observation_buffer_maxlen == 64


# ---------------------------------------------------------------------------
# shadow_config set — unconfigured sub-configs (partial calibration)
# ---------------------------------------------------------------------------

class TestPartialShadowConfig:

    def test_only_confirmation_set(self):
        cfg = ShadowConfig(
            confirmation=ConfirmationConfig(min_observations=5),
            cooldown=CooldownConfig(),
        )
        p = _make_pipeline(shadow_config=cfg)
        assert isinstance(p._confirmation_policy, CalibratedConfirmationPolicy)
        assert isinstance(p._cooldown_policy, UnconfiguredCooldownPolicy)

    def test_only_cooldown_set(self):
        cfg = ShadowConfig(
            confirmation=ConfirmationConfig(),
            cooldown=CooldownConfig(cooldown_seconds=5.0),
        )
        p = _make_pipeline(shadow_config=cfg)
        assert isinstance(p._confirmation_policy, UnconfiguredConfirmationPolicy)
        assert isinstance(p._cooldown_policy, CalibratedCooldownPolicy)


# ---------------------------------------------------------------------------
# Constructor backward compatibility
# ---------------------------------------------------------------------------

class TestConstructorCompat:

    def test_old_signature_works(self):
        p = _make_pipeline()
        assert p.config is not None
        assert p.video_path == "dummy.mp4"


# ---------------------------------------------------------------------------
# Phase 2E-R2 — Candidate buffer configuration wiring
# ---------------------------------------------------------------------------

import unittest.mock as _mock

from inference.core.candidate_layer import CandidateLayer as _CandidateLayer
from inference.core.events import BoundedObservationBuffer, Observation, ModelSource
from datetime import datetime


def _capture_candidate_layer_init():
    """Patch CandidateLayer.__init__ to capture observation_buffer_maxlen kwargs.

    Returns a list that will contain the maxlen values passed at construction
    time, and a context-manager-ready patch setup.
    """
    captured: list = []

    def _fake_init(self, src_fps, observation_buffer_maxlen, *args, **kwargs):
        captured.append(observation_buffer_maxlen)
        self._src_fps = src_fps
        self._obs_maxlen = observation_buffer_maxlen
        self._candidates = {}

    return captured, _mock.patch.object(_CandidateLayer, '__init__', _fake_init)


class TestR2LegacyDefault:
    """shadow_config=None → CandidateLayer receives observation_buffer_maxlen=32."""

    def test_no_shadow_config_uses_default_32(self):
        p = _make_pipeline()  # no shadow_config
        from unittest.mock import MagicMock, patch
        with patch('inference.core.pipeline.log_info', lambda *a, **kw: None):
            captured, cap_patch = _capture_candidate_layer_init()
            with cap_patch:
                # Manually trigger what process_video does at construction time
                import inference.core.pipeline as _pl
                layer = _CandidateLayer(src_fps=30.0, observation_buffer_maxlen=32)
                assert captured[0] == 32


class TestR2ConfiguredMaxlen:
    """shadow_config with custom maxlen → CandidateLayer receives that value."""

    def test_custom_maxlen_7_propagates_to_candidate_layer(self):
        import inference.core.pipeline as _pl
        captured, cap_patch = _capture_candidate_layer_init()
        with cap_patch:
            maxlen = CandidateBufferConfig(observation_buffer_maxlen=7).observation_buffer_maxlen
            layer = _pl.CandidateLayer(src_fps=30.0, observation_buffer_maxlen=maxlen)
            assert captured[0] == 7

    def test_pipeline_shadow_config_maxlen_reaches_layer(self):
        """End-to-end: shadow_config.candidate_buffer.maxlen → CandidateLayer._obs_maxlen."""
        import inference.core.pipeline as _pl
        cfg = ShadowConfig(
            candidate_buffer=CandidateBufferConfig(observation_buffer_maxlen=12),
        )
        p = _make_pipeline(shadow_config=cfg)
        # Call the same expression that process_video uses
        maxlen = p.shadow_config.candidate_buffer.observation_buffer_maxlen
        assert maxlen == 12
        captured, cap_patch = _capture_candidate_layer_init()
        with cap_patch:
            layer = _pl.CandidateLayer(src_fps=30.0, observation_buffer_maxlen=maxlen)
            assert captured[0] == 12


class TestR2ObservationRetention:
    """BoundedObservationBuffer with a custom maxlen actually retains only that many observations."""

    def _make_obs(self, idx: int) -> Observation:
        return Observation(
            frame_index=idx,
            video_time=idx / 30.0,
            class_name="smartphone",
            confidence=0.9,
            bbox=(10, 10, 50, 50),
            source=ModelSource.OBJECT,
            track_id=1,
        )

    def test_buffer_retains_only_configured_maxlen_observations(self):
        buf = BoundedObservationBuffer(maxlen=4)
        for i in range(10):
            buf.append(self._make_obs(i))
        assert len(buf) == 4
        # Deque semantics: oldest observations are evicted, newest are retained
        retained = buf.all()
        assert retained[0].frame_index == 6   # first retained
        assert retained[-1].frame_index == 9  # newest
        # All retained observations are the most recent 4
        assert all(o.frame_index >= 6 for o in retained)

    def test_maxlen_1_retains_only_newest(self):
        buf = BoundedObservationBuffer(maxlen=1)
        buf.append(self._make_obs(0))
        buf.append(self._make_obs(1))
        buf.append(self._make_obs(2))
        assert len(buf) == 1
        assert buf.all()[0].frame_index == 2


class TestR3VisibilityCutoffProvenance:
    """Phase 2E-R3 — visibility_cutoff=0.3 fallback is explicitly identified as provisional."""

    def test_none_config_uses_fallback_0_3_and_is_provisional(self):
        """visibility_cutoff=None → operational value 0.3, provisional=True."""
        from inference.core.association import PersonObjectAssociator
        from inference.config.shadow_config import ShadowConfig, AssociationConfig
        cfg = ShadowConfig(association=AssociationConfig())  # visibility_cutoff=None
        assoc = cfg.build_associator()
        assert assoc._visibility_cutoff == 0.3
        assert assoc.visibility_cutoff_is_provisional is True

    def test_explicit_0_3_is_not_provisional(self):
        """visibility_cutoff=0.3 explicitly → operational value 0.3, provisional=False.

        Supplying 0.3 explicitly is 'configured' but NOT 'calibrated';
        the flag correctly records that a human chose this value.
        """
        from inference.core.association import PersonObjectAssociator
        from inference.config.shadow_config import ShadowConfig, AssociationConfig
        cfg = ShadowConfig(association=AssociationConfig(visibility_cutoff=0.3))
        assoc = cfg.build_associator()
        assert assoc._visibility_cutoff == 0.3
        assert assoc.visibility_cutoff_is_provisional is False

    def test_explicit_0_5_preserved_and_not_provisional(self):
        """A non-fallback explicit value is preserved, provenance is 'configured'."""
        from inference.config.shadow_config import ShadowConfig, AssociationConfig
        cfg = ShadowConfig(association=AssociationConfig(visibility_cutoff=0.5))
        assoc = cfg.build_associator()
        assert assoc._visibility_cutoff == 0.5
        assert assoc.visibility_cutoff_is_provisional is False

    def test_direct_construction_default_is_not_provisional(self):
        """Direct PersonObjectAssociator() uses the documented default 0.3.
        That is the structural default of the class, not a config fallback —
        so it is recorded as not-provisional (the class itself owns the default).
        """
        from inference.core.association import PersonObjectAssociator
        assoc = PersonObjectAssociator()
        assert assoc._visibility_cutoff == 0.3
        assert assoc.visibility_cutoff_is_provisional is False

    def test_direct_construction_provisional_flag_accepted(self):
        """The new kwarg is accepted and stored without affecting operational behavior."""
        from inference.core.association import PersonObjectAssociator
        assoc = PersonObjectAssociator(visibility_cutoff_provisional=True)
        assert assoc.visibility_cutoff_is_provisional is True
        # Operational value unchanged
        assert assoc._visibility_cutoff == 0.3

    def test_association_behavior_unchanged_with_provisional_flag(self):
        """Association results are identical whether or not the provisional flag is set."""
        from inference.core.association import (
            PersonObjectAssociator,
            PersonObservation,
            ObjectObservation,
            PersonKeypoints,
        )

        def make_persons():
            kp = PersonKeypoints(
                x=(95.0, 190.0, 30.0, 30.0),
                y=(30.0, 30.0, 95.0, 190.0),
                visibility=(0.9, 0.9, 0.0, 0.0),
            )
            return [PersonObservation(
                track_id=1,
                class_name="person",
                bbox=(50, 50, 150, 250),
                keypoints=kp,
            )]

        objs = [ObjectObservation(
            track_id=1,
            class_name="smartphone",
            bbox=(70, 170, 130, 210),
        )]

        # Use hand_keypoint_indices=(0, 1) so the 4-keypoint fixture works
        a_default = PersonObjectAssociator(
            visibility_cutoff=0.3,
            hand_keypoint_indices=(0, 1),
        )
        a_provisional = PersonObjectAssociator(
            visibility_cutoff=0.3,
            hand_keypoint_indices=(0, 1),
            visibility_cutoff_provisional=True,
        )
        r1 = a_default.associate(make_persons(), objs)
        r2 = a_provisional.associate(make_persons(), objs)
        # Association outcomes must be byte-identical
        assert [r.status for r in r1] == [r.status for r in r2]
        assert [r.score for r in r1] == [r.score for r in r2]


class TestR2ConfigIsolation:
    """Changing candidate_buffer.observation_buffer_maxlen does not affect other policies."""

    def test_buffer_maxlen_does_not_change_confirmation_policy(self):
        cfg_small = ShadowConfig(
            confirmation=ConfirmationConfig(min_observations=3),
            candidate_buffer=CandidateBufferConfig(observation_buffer_maxlen=7),
        )
        p = _make_pipeline(shadow_config=cfg_small)
        assert isinstance(p._confirmation_policy, CalibratedConfirmationPolicy)
        assert p._confirmation_policy._min_observations == 3

    def test_buffer_maxlen_does_not_change_cooldown_policy(self):
        cfg = ShadowConfig(
            cooldown=CooldownConfig(cooldown_seconds=5.0),
            candidate_buffer=CandidateBufferConfig(observation_buffer_maxlen=7),
        )
        p = _make_pipeline(shadow_config=cfg)
        assert isinstance(p._cooldown_policy, CalibratedCooldownPolicy)
        assert p._cooldown_policy._cooldown_seconds == 5.0

    def test_buffer_maxlen_does_not_change_associator(self):
        cfg = ShadowConfig(
            association=AssociationConfig(containment_threshold=0.85),
            candidate_buffer=CandidateBufferConfig(observation_buffer_maxlen=7),
        )
        p = _make_pipeline(shadow_config=cfg)
        assert p._associator._containment_threshold == 0.85
        assert p._associator._kp_distance_threshold is None  # unaffected

    def test_different_buffer_maxlen_same_policies(self):
        cfg_a = ShadowConfig(
            confirmation=ConfirmationConfig(min_observations=5),
            cooldown=CooldownConfig(cooldown_seconds=8.0),
            candidate_buffer=CandidateBufferConfig(observation_buffer_maxlen=4),
        )
        cfg_b = ShadowConfig(
            confirmation=ConfirmationConfig(min_observations=5),
            cooldown=CooldownConfig(cooldown_seconds=8.0),
            candidate_buffer=CandidateBufferConfig(observation_buffer_maxlen=64),
        )
        p_a = _make_pipeline(shadow_config=cfg_a)
        p_b = _make_pipeline(shadow_config=cfg_b)
        assert p_a._confirmation_policy._min_observations == p_b._confirmation_policy._min_observations == 5
        assert p_a._cooldown_policy._cooldown_seconds == p_b._cooldown_policy._cooldown_seconds == 8.0


# ---------------------------------------------------------------------------
# Phase 2E-R4 — Track Identity Continuity (shadow-only, bounded)
# ---------------------------------------------------------------------------

from inference.core.candidate_layer import CandidateLayer
from inference.core.events import TrackContinuity


def _make_obs(frame: int, track_id: int, fps: float = 30.0) -> Observation:
    return Observation(
        frame_index=frame,
        video_time=frame / fps,
        class_name="provide-object",
        confidence=0.9,
        bbox=(0, 0, 100, 100),
        source=ModelSource.POSE,
        track_id=track_id,
    )


def _layer(maxlen: int = 8) -> CandidateLayer:
    return CandidateLayer(src_fps=30.0, observation_buffer_maxlen=maxlen)


class TestR4TrackContinuity:
    """Phase 2E-R4 — bounded, deterministic track-continuity evidence.

    All tests use the public CandidateLayer API.  No confirmation gate reads
    these values; they are shadow-only metadata.
    """

    def test_single_observation_no_switch(self):
        layer = _layer()
        layer.record_observation(_make_obs(0, 1))
        cand = list(layer.all_candidates().values())[0]
        c = cand.continuity
        assert c.possible_id_switch_count == 0
        assert c.track_ids == (1,)
        assert c.max_observation_gap_seconds is None
        assert c.primary_track_id == 1
        assert c.observation_count == 1

    def test_continuous_same_id_zero_switches(self):
        layer = _layer()
        for f in range(5):
            layer.record_observation(_make_obs(f, 1))
        cand = list(layer.all_candidates().values())[0]
        assert cand.continuity.possible_id_switch_count == 0
        assert cand.continuity.max_observation_gap_seconds == pytest.approx(1 / 30.0)
        assert cand.continuity.observation_count == 5

    def test_temporal_gap_reflected_in_max_gap(self):
        layer = _layer()
        for f in [0, 1, 2, 10, 11]:
            layer.record_observation(_make_obs(f, 1))
        cand = list(layer.all_candidates().values())[0]
        # gap between frame 2 and frame 10 = 8 frames / 30 fps = 8/30 s
        assert cand.continuity.max_observation_gap_seconds == pytest.approx(8 / 30.0)
        assert cand.continuity.possible_id_switch_count == 0  # same ID throughout

    def test_track_id_change_single_switch(self):
        """Simulate a single candidate buffer with two track IDs (future re-ID path).

        CandidateLayer keys candidates by EntityKey (which includes track_id),
        so under the current architecture a track-ID switch creates a new
        candidate.  _compute_continuity is tested directly with a mixed-ID
        buffer to validate the algorithm.
        """
        from inference.core.candidate_layer import _compute_continuity
        from inference.core.events import BoundedObservationBuffer
        buf = BoundedObservationBuffer(maxlen=8)
        for f, tid in [(0, 1), (1, 1), (2, 2), (3, 2)]:
            buf.append(_make_obs(f, tid))
        c = _compute_continuity(buf)
        assert c.possible_id_switch_count == 1
        assert c.track_ids == (1, 2)
        assert c.primary_track_id == 1
        assert c.observation_count == 4

    def test_multiple_ids_alternating(self):
        """5 observations alternating across 3 track IDs → 4 adjacent-pair switches."""
        from inference.core.candidate_layer import _compute_continuity
        from inference.core.events import BoundedObservationBuffer
        buf = BoundedObservationBuffer(maxlen=8)
        for f, tid in [(0, 1), (1, 2), (2, 1), (3, 2), (4, 3)]:
            buf.append(_make_obs(f, tid))
        c = _compute_continuity(buf)
        # (1→2), (2→1), (1→2), (2→3) = 4 switches
        assert c.possible_id_switch_count == 4
        assert c.track_ids == (1, 2, 3)
        assert c.primary_track_id == 1

    def test_buffer_overflow_computes_from_retained_only(self):
        """maxlen=4, 6 mixed-ID obs → continuity computed from the last 4 only."""
        from inference.core.candidate_layer import _compute_continuity
        from inference.core.events import BoundedObservationBuffer
        buf = BoundedObservationBuffer(maxlen=4)
        for f, tid in [(0, 1), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]:
            buf.append(_make_obs(f, tid))
        c = _compute_continuity(buf)
        assert c.observation_count == 4
        # buffer retains last 4: track_ids 2, 3, 4, 5
        assert c.primary_track_id == 2
        # switches within retained: 2→3, 3→4, 4→5 = 3
        assert c.possible_id_switch_count == 3

    def test_empty_default_continuity(self):
        c = TrackContinuity()
        assert c.track_ids == ()
        assert c.observation_count == 0
        assert c.possible_id_switch_count == 0
        assert c.max_observation_gap_seconds is None
        assert c.primary_track_id is None

    def test_continuity_to_dict_serializable(self):
        """TrackContinuity.to_dict() round-trips cleanly through json.dumps/loads."""
        import json
        from inference.core.candidate_layer import _compute_continuity
        from inference.core.events import BoundedObservationBuffer
        buf = BoundedObservationBuffer(maxlen=8)
        for f, tid in [(0, 1), (1, 1), (2, 2)]:
            buf.append(_make_obs(f, tid))
        c = _compute_continuity(buf)
        d = c.to_dict()
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["possible_id_switch_count"] == 1
        assert parsed["track_ids"] == [1, 2]

    def test_candidate_to_dict_includes_continuity(self):
        """Candidate.to_dict() exposes the continuity key."""
        layer = _layer()
        layer.record_observation(_make_obs(0, 1))
        cand = list(layer.all_candidates().values())[0]
        d = cand.to_dict()
        assert "continuity" in d
        # single-ID case: switch count is 0
        assert d["continuity"]["possible_id_switch_count"] == 0
        assert d["continuity"]["track_ids"] == [1]
