"""Phase 2B.6 — Config centralization test suite.

Run:  python -m pytest tests/test_phase2b6_config.py -v

Pins the ShadowConfig module and its relationship to legacy parameters:
- All shadow parameters default to None (unconfigured / not calibrated)
- Legacy constants are NOT duplicated inside ShadowConfig
- build_policies() returns safe unconfigured policies when nothing is set
- observation_buffer_maxlen is independent from min_observations
"""
import os
import sys
import json
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


class TestShadowConfigDefaults(unittest.TestCase):
    """Unconfigured semantics: no shadow parameter may hide a production default."""

    def test_module_imports(self):
        from inference.config.shadow_config import (
            ShadowConfig,
            ConfirmationConfig,
            CooldownConfig,
            AssociationConfig,
            CandidateBufferConfig,
        )
        self.assertTrue(True)

    def test_confirmation_all_none_by_default(self):
        from inference.config.shadow_config import ShadowConfig
        cfg = ShadowConfig()
        self.assertIsNone(cfg.confirmation.min_observations)
        self.assertIsNone(cfg.confirmation.min_duration_seconds)
        self.assertIsNone(cfg.confirmation.min_confidence_mean)
        self.assertIsNone(cfg.confirmation.max_gap_seconds)

    def test_cooldown_none_by_default(self):
        from inference.config.shadow_config import ShadowConfig
        cfg = ShadowConfig()
        self.assertIsNone(cfg.cooldown.cooldown_seconds)

    def test_association_all_none_by_default(self):
        from inference.config.shadow_config import ShadowConfig
        cfg = ShadowConfig()
        self.assertIsNone(cfg.association.containment_threshold)
        self.assertIsNone(cfg.association.keypoint_distance_threshold)
        self.assertIsNone(cfg.association.reference_scale)

    def test_buffer_maxlen_is_structural_default(self):
        """observation_buffer_maxlen is a structural capacity, not a policy value."""
        from inference.config.shadow_config import ShadowConfig
        cfg = ShadowConfig()
        self.assertEqual(cfg.candidate_buffer.observation_buffer_maxlen, 32)

    def test_buffer_independent_from_min_observations(self):
        """Changing buffer size must not change min_observations."""
        from inference.config.shadow_config import ShadowConfig
        cfg = ShadowConfig()
        self.assertIsNone(cfg.confirmation.min_observations)
        cfg2 = ShadowConfig()
        cfg2.candidate_buffer = type(cfg.candidate_buffer)(observation_buffer_maxlen=1024)
        self.assertIsNone(cfg2.confirmation.min_observations)

    def test_to_dict_json_serializable(self):
        from inference.config.shadow_config import ShadowConfig
        cfg = ShadowConfig()
        dumped = json.dumps(cfg.to_dict())
        self.assertIsInstance(dumped, str)


class TestShadowConfigPopulated(unittest.TestCase):

    def test_confirmation_populated(self):
        from inference.config.shadow_config import ShadowConfig
        cfg = ShadowConfig()
        cfg.confirmation.min_observations = 5
        cfg.confirmation.min_duration_seconds = 2.0
        cfg.confirmation.min_confidence_mean = 0.7
        cfg.confirmation.max_gap_seconds = 0.5
        self.assertEqual(cfg.confirmation.min_observations, 5)
        self.assertEqual(cfg.confirmation.min_duration_seconds, 2.0)

    def test_cooldown_populated(self):
        from inference.config.shadow_config import ShadowConfig
        cfg = ShadowConfig()
        cfg.cooldown.cooldown_seconds = 90.0
        self.assertEqual(cfg.cooldown.cooldown_seconds, 90.0)


class TestBuildPolicies(unittest.TestCase):
    """build_policies() returns safe unconfigured policies when nothing is set."""

    def test_build_policies_unconfigured(self):
        from inference.config.shadow_config import ShadowConfig
        from inference.core.confirmation import UnconfiguredConfirmationPolicy
        from inference.core.cooldown import UnconfiguredCooldownPolicy
        cfg = ShadowConfig()
        confirmation, cooldown = cfg.build_policies()
        self.assertIsInstance(confirmation, UnconfiguredConfirmationPolicy)
        self.assertIsInstance(cooldown, UnconfiguredCooldownPolicy)

    def test_build_policies_calibrated_confirmation(self):
        from inference.config.shadow_config import ShadowConfig
        from inference.core.confirmation import CalibratedConfirmationPolicy
        cfg = ShadowConfig()
        cfg.confirmation.min_observations = 3
        cfg.confirmation.min_duration_seconds = 1.0
        confirmation, _ = cfg.build_policies()
        self.assertIsInstance(confirmation, CalibratedConfirmationPolicy)
        self.assertTrue(confirmation.is_configured)

    def test_build_policies_calibrated_cooldown(self):
        from inference.config.shadow_config import ShadowConfig
        from inference.core.cooldown import CalibratedCooldownPolicy
        cfg = ShadowConfig()
        cfg.cooldown.cooldown_seconds = 90.0
        _, cooldown = cfg.build_policies()
        self.assertIsInstance(cooldown, CalibratedCooldownPolicy)

    def test_build_policies_partial_gives_calibrated_with_unmet_gates(self):
        """Only some gates configured -> CalibratedConfirmationPolicy with others None."""
        from inference.config.shadow_config import ShadowConfig
        from inference.core.confirmation import CalibratedConfirmationPolicy
        cfg = ShadowConfig()
        cfg.confirmation.min_observations = 5
        confirmation, _ = cfg.build_policies()
        self.assertIsInstance(confirmation, CalibratedConfirmationPolicy)
        self.assertEqual(confirmation._min_observations, 5)
        self.assertIsNone(confirmation._min_duration_seconds)


class TestBuildAssociator(unittest.TestCase):

    def test_build_associator_unconfigured(self):
        from inference.config.shadow_config import ShadowConfig
        from inference.core.association import PersonObjectAssociator
        cfg = ShadowConfig()
        assoc = cfg.build_associator()
        self.assertIsInstance(assoc, PersonObjectAssociator)
        self.assertIsNone(assoc._containment_threshold)

    def test_build_associator_populated(self):
        from inference.config.shadow_config import ShadowConfig
        cfg = ShadowConfig()
        cfg.association.containment_threshold = 0.8
        assoc = cfg.build_associator()
        self.assertEqual(assoc._containment_threshold, 0.8)


class TestLegacyShadowSeparation(unittest.TestCase):
    """Legacy constants (5 s persistence, 90 s cooldown) must NOT live in ShadowConfig."""

    def test_shadow_config_does_not_import_legacy_constants(self):
        """The ShadowConfig module must not reference legacy Config constants."""
        import inspect
        from inference.config import shadow_config
        src = inspect.getsource(shadow_config)
        self.assertNotIn("POSE_COOLDOWN", src)
        self.assertNotIn("REQUIRED_DETECTION_TIME", src)

    def test_shadow_config_legacy_is_separate_module(self):
        """ShadowConfig lives in its own module, distinct from the legacy Config class."""
        from inference.config.shadow_config import ShadowConfig
        from inference.config.constants import Config
        self.assertIsNot(ShadowConfig, Config)
        self.assertFalse(issubclass(ShadowConfig, Config))


class TestValidation(unittest.TestCase):
    """Constructor-time validation: __post_init__ rejects bad values."""

    def test_confirmation_min_observations_negative_raises(self):
        from inference.config.shadow_config import ConfirmationConfig
        with self.assertRaises(ValueError):
            ConfirmationConfig(min_observations=-1)

    def test_confirmation_min_observations_zero_raises(self):
        from inference.config.shadow_config import ConfirmationConfig
        with self.assertRaises(ValueError):
            ConfirmationConfig(min_observations=0)

    def test_confirmation_min_confidence_mean_out_of_range_raises(self):
        from inference.config.shadow_config import ConfirmationConfig
        with self.assertRaises(ValueError):
            ConfirmationConfig(min_confidence_mean=1.5)

    def test_cooldown_zero_raises(self):
        from inference.config.shadow_config import CooldownConfig
        with self.assertRaises(ValueError):
            CooldownConfig(cooldown_seconds=0.0)

    def test_association_containment_out_of_range_raises(self):
        from inference.config.shadow_config import AssociationConfig
        with self.assertRaises(ValueError):
            AssociationConfig(containment_threshold=1.5)


if __name__ == "__main__":
    unittest.main()
