"""Phase 2B.6 — Shadow-path configuration.

All numeric parameters that belong to the shadow decision path (candidate
buffer, confirmation gates, cooldown, association) are declared here in one
typed, inspectable structure.  Every policy parameter defaults to None so that
an unconfigured value is always representable; a None value means "this gate
is not calibrated yet" and must never be silently replaced by a magic number.

This module is deliberately distinct from the legacy ``Config`` class in
``inference.config.constants`` — the two layers share no numeric source of
truth.  Legacy values (5 s persistence, 90 s cooldown, conf thresholds, NMS
IoU, crop scales) remain frozen in ``constants.py``; shadow values live here
and are uncalibrated by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Structural default only — a memory bound, NOT a confirmation policy value.
OBSERVATION_BUFFER_MAXLEN_DEFAULT = 32


@dataclass
class ConfirmationConfig:
    """Shadow confirmation-gate thresholds.

    All fields default to None (gate not enforced).  A None value is
    semantically "unconfigured", not "failed" — see ConfirmationDecisionEvidence.
    """
    min_observations: Optional[int] = None
    min_duration_seconds: Optional[float] = None
    min_confidence_mean: Optional[float] = None
    max_gap_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.min_observations is not None and self.min_observations <= 0:
            raise ValueError(
                f"min_observations must be positive or None, got {self.min_observations}"
            )
        if self.min_duration_seconds is not None and self.min_duration_seconds <= 0:
            raise ValueError(
                f"min_duration_seconds must be positive or None, got {self.min_duration_seconds}"
            )
        if self.min_confidence_mean is not None and not (0.0 <= self.min_confidence_mean <= 1.0):
            raise ValueError(
                f"min_confidence_mean must be in [0.0, 1.0] or None, got {self.min_confidence_mean}"
            )
        if self.max_gap_seconds is not None and self.max_gap_seconds <= 0:
            raise ValueError(
                f"max_gap_seconds must be positive or None, got {self.max_gap_seconds}"
            )

    @property
    def is_any_configured(self) -> bool:
        """True when at least one gate threshold is explicitly set."""
        return any(v is not None for v in (
            self.min_observations,
            self.min_duration_seconds,
            self.min_confidence_mean,
            self.max_gap_seconds,
        ))


@dataclass
class CooldownConfig:
    """Shadow cooldown window (source-video seconds).

    Defaults to None (unconfigured).  The value must come from §12 calibration;
    this module never supplies a production default.
    """
    cooldown_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.cooldown_seconds is not None and self.cooldown_seconds <= 0:
            raise ValueError(
                f"cooldown_seconds must be positive or None, got {self.cooldown_seconds}"
            )


@dataclass
class AssociationConfig:
    """Person-object association thresholds.

    All fields default to None (unconfigured).  A None threshold means the
    corresponding gate is not enforced at build time.
    """
    containment_threshold: Optional[float] = None
    keypoint_distance_threshold: Optional[float] = None
    visibility_cutoff: Optional[float] = None   # None -> default 0.3 at build time
    reference_scale: Optional[float] = None

    def __post_init__(self) -> None:
        if self.containment_threshold is not None and not (0.0 <= self.containment_threshold <= 1.0):
            raise ValueError(
                f"containment_threshold must be in [0.0, 1.0] or None, got {self.containment_threshold}"
            )
        if self.keypoint_distance_threshold is not None and self.keypoint_distance_threshold < 0:
            raise ValueError(
                f"keypoint_distance_threshold must be non-negative or None, got {self.keypoint_distance_threshold}"
            )
        if self.visibility_cutoff is not None and not (0.0 <= self.visibility_cutoff <= 1.0):
            raise ValueError(
                f"visibility_cutoff must be in [0.0, 1.0] or None, got {self.visibility_cutoff}"
            )


@dataclass
class CandidateBufferConfig:
    """Structural (non-policy) capacity for the candidate observation buffer.

    ``observation_buffer_maxlen`` is a memory bound, NOT a confirmation gate.
    It is independent of ``ConfirmationConfig.min_observations`` — changing the
    buffer size must never change which confirmation gates are enforced.
    """
    observation_buffer_maxlen: int = OBSERVATION_BUFFER_MAXLEN_DEFAULT

    def __post_init__(self) -> None:
        if self.observation_buffer_maxlen <= 0:
            raise ValueError(
                f"observation_buffer_maxlen must be positive, got {self.observation_buffer_maxlen}"
            )


@dataclass
class ShadowConfig:
    """Top-level shadow-path configuration.

    All sub-configs default to their unconfigured state.  Use
    ``build_policies()`` / ``build_associator()`` to obtain ready-to-use
    policy objects without hand-constructing each one.
    """
    confirmation: ConfirmationConfig = field(default_factory=ConfirmationConfig)
    cooldown: CooldownConfig = field(default_factory=CooldownConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    candidate_buffer: CandidateBufferConfig = field(default_factory=CandidateBufferConfig)

    def to_dict(self) -> dict:
        return {
            "confirmation": {
                "min_observations": self.confirmation.min_observations,
                "min_duration_seconds": self.confirmation.min_duration_seconds,
                "min_confidence_mean": self.confirmation.min_confidence_mean,
                "max_gap_seconds": self.confirmation.max_gap_seconds,
            },
            "cooldown": {
                "cooldown_seconds": self.cooldown.cooldown_seconds,
            },
            "association": {
                "containment_threshold": self.association.containment_threshold,
                "keypoint_distance_threshold": self.association.keypoint_distance_threshold,
                "visibility_cutoff": self.association.visibility_cutoff,
                "reference_scale": self.association.reference_scale,
            },
            "candidate_buffer": {
                "observation_buffer_maxlen": self.candidate_buffer.observation_buffer_maxlen,
            },
        }

    # ------------------------------------------------------------------
    # Policy builders
    # ------------------------------------------------------------------

    def build_policies(self):
        """Return ``(ConfirmationPolicy, CooldownPolicy)`` for the shadow path.

        Unconfigured (safe) policy objects when no thresholds are set;
        calibrated objects when at least one threshold is populated.
        """
        from inference.core.confirmation import (
            CalibratedConfirmationPolicy,
            UnconfiguredConfirmationPolicy,
        )
        from inference.core.cooldown import (
            CalibratedCooldownPolicy,
            UnconfiguredCooldownPolicy,
        )

        confirmation = (
            CalibratedConfirmationPolicy(
                min_observations=self.confirmation.min_observations,
                min_duration_seconds=self.confirmation.min_duration_seconds,
                min_confidence_mean=self.confirmation.min_confidence_mean,
                max_gap_seconds=self.confirmation.max_gap_seconds,
            )
            if self.confirmation.is_any_configured
            else UnconfiguredConfirmationPolicy()
        )

        cooldown = (
            CalibratedCooldownPolicy(self.cooldown.cooldown_seconds)
            if self.cooldown.cooldown_seconds is not None
            else UnconfiguredCooldownPolicy()
        )

        return confirmation, cooldown

    def build_associator(self):
        """Return a ``PersonObjectAssociator`` from the association sub-config.

        Unpopulated thresholds stay None (gate not enforced).
        """
        from inference.core.association import PersonObjectAssociator

        visibility_cutoff = (
            self.association.visibility_cutoff
            if self.association.visibility_cutoff is not None
            else 0.3
        )

        return PersonObjectAssociator(
            containment_threshold=self.association.containment_threshold,
            keypoint_distance_threshold=self.association.keypoint_distance_threshold,
            visibility_cutoff=visibility_cutoff,
            reference_scale=self.association.reference_scale,
        )
