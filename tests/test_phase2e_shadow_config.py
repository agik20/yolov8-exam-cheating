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
