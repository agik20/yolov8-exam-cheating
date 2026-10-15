"""Phase 2B.3 — Shadow confirmation decision layer.

This module implements the confirmation gate described in
PHASE2A_EVENT_ARCHITECTURE.md §3, §5, §7, and the Phase 2B.3 brief.

What this module does:
- Provide ConfirmationGateResult: a structured, JSON-serializable shadow
  result carrying the measured evidence dimensions and the gate decision.
- Provide UnconfiguredConfirmationPolicy: the safe default.  Always
  returns confirmed=False with reason="confirmation_policy_unconfigured".
  Measured evidence is still included so the gate is diagnostic even
  before calibration.
- Provide CalibratedConfirmationPolicy: evaluates a candidate against
  explicitly supplied calibration parameters.  Any parameter set to
  None is treated as unenforced (the gate is skipped for that dimension).
  When all four parameters are None the result is equivalent to
  UnconfiguredConfirmationPolicy.

What this module intentionally does NOT do (Phase 2B.3 scope):
- No production event creation, no file I/O, no JPEG/CSV writes.
- No person-object association (Phase 2B.5); association_score is None.
- No calibrated event-confidence formula; event_confidence is None.
- No changes to the legacy event path (process_detection, id_info_*,
  cooldown, REQUIRED_DETECTION_TIME, evidence output).
- No arbitrary production defaults: all calibration parameters must be
  explicitly supplied by the caller.

The legacy system in inference/core/pipeline.py remains the production
source of truth.  This module is observational only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple

from .events import (
    ModelSource,
    EntityKey,
    Candidate,
    CandidateState,
    ConfidenceStats,
)


# ---------------------------------------------------------------------------
# ConfirmationGateResult
# ---------------------------------------------------------------------------

@dataclass
class ConfirmationGateResult:
    """Structured shadow confirmation result for one candidate.

    All evidence dimensions are populated regardless of the gate outcome,
    so a rejected result still carries the measured values needed for
    later calibration and FP/FN comparison.
    """
    entity_key: EntityKey
    candidate_class: str
    confirmed: bool
    reason: str
    already_confirmed: bool = False
    observation_count: int = 0
    video_time_span: float = 0.0
    confidence_stats: Optional[ConfidenceStats] = None
    gap_count: Optional[int] = None
    max_gap_seconds: Optional[float] = None
    track_ids: FrozenSet[int] = field(default_factory=frozenset)
    association_score: Optional[float] = None
    event_confidence: Optional[float] = None
    unmet_requirements: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "entity_key": self.entity_key.as_string(),
            "candidate_class": self.candidate_class,
            "confirmed": self.confirmed,
            "reason": self.reason,
            "already_confirmed": self.already_confirmed,
            "observation_count": self.observation_count,
            "video_time_span": self.video_time_span,
            "confidence_stats": self.confidence_stats.to_dict()
                                if self.confidence_stats else None,
            "gap_count": self.gap_count,
            "max_gap_seconds": self.max_gap_seconds,
            "track_ids": sorted(self.track_ids),
            "association_score": self.association_score,
            "event_confidence": self.event_confidence,
            "unmet_requirements": list(self.unmet_requirements),
        }


# ---------------------------------------------------------------------------
# ConfirmationPolicy base
# ---------------------------------------------------------------------------

class ConfirmationPolicy:
    """Abstract confirmation policy.

    Subclasses must implement evaluate().  The pipeline holds one
    ConfirmationPolicy instance; changing policy is a one-line swap —
    no pipeline logic changes required.
    """

    def evaluate(self, candidate: Candidate) -> ConfirmationGateResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# UnconfiguredConfirmationPolicy
# ---------------------------------------------------------------------------

class UnconfiguredConfirmationPolicy(ConfirmationPolicy):
    """Safe default: no calibrated gates active.

    Always returns confirmed=False with reason=
    "confirmation_policy_unconfigured".  Measured evidence is included
    so the result is still useful for diagnostics and calibration.
    """

    def evaluate(self, candidate: Candidate) -> ConfirmationGateResult:
        already = candidate.state == CandidateState.CONFIRMED
        return ConfirmationGateResult(
            entity_key=candidate.entity_key,
            candidate_class=candidate.candidate_class,
            confirmed=False,
            reason="confirmation_policy_unconfigured",
            already_confirmed=already,
            observation_count=candidate.observation_count(),
            video_time_span=candidate.video_time_span(),
            confidence_stats=candidate.confidence_stats(),
            gap_count=candidate.gap_count,
            max_gap_seconds=candidate.max_gap_seconds,
            track_ids=candidate.observed_track_ids,
            association_score=None,
            event_confidence=None,
            unmet_requirements=(),
        )


# ---------------------------------------------------------------------------
# CalibratedConfirmationPolicy
# ---------------------------------------------------------------------------

class CalibratedConfirmationPolicy(ConfirmationPolicy):
    """Calibrated confirmation gate with explicit, caller-supplied thresholds.

    Each parameter controls one evidence dimension.  A parameter set to
    None means "this gate is not enforced."  When *all four* parameters
    are None the policy behaves identically to
    UnconfiguredConfirmationPolicy.

    Parameters
    ----------
    min_observations : Optional[int]
    min_duration_seconds : Optional[float]   # video-time, NOT wall-clock
    min_confidence_mean : Optional[float]    # detection confidence, not event confidence
    max_gap_seconds : Optional[float]         # max acceptable gap; candidate with no
                                               # recorded gaps always passes this gate
    """

    def __init__(self,
                 min_observations: Optional[int] = None,
                 min_duration_seconds: Optional[float] = None,
                 min_confidence_mean: Optional[float] = None,
                 max_gap_seconds: Optional[float] = None) -> None:
        if min_observations is not None and min_observations <= 0:
            raise ValueError(f"min_observations must be positive or None, got {min_observations}")
        if min_duration_seconds is not None and min_duration_seconds <= 0:
            raise ValueError(f"min_duration_seconds must be positive or None, got {min_duration_seconds}")
        if min_confidence_mean is not None and not (0.0 <= min_confidence_mean <= 1.0):
            raise ValueError(f"min_confidence_mean must be in [0.0, 1.0] or None, got {min_confidence_mean}")
        if max_gap_seconds is not None and max_gap_seconds <= 0:
            raise ValueError(f"max_gap_seconds must be positive or None, got {max_gap_seconds}")
        self._min_observations = min_observations
        self._min_duration_seconds = min_duration_seconds
        self._min_confidence_mean = min_confidence_mean
        self._max_gap_seconds = max_gap_seconds

    @property
    def is_configured(self) -> bool:
        """True when at least one gate parameter is explicitly set."""
        return any((self._min_observations is not None,
                    self._min_duration_seconds is not None,
                    self._min_confidence_mean is not None,
                    self._max_gap_seconds is not None))

    def evaluate(self, candidate: Candidate) -> ConfirmationGateResult:
        already = candidate.state == CandidateState.CONFIRMED
        obs_count = candidate.observation_count()
        span = candidate.video_time_span()
        conf_stats = candidate.confidence_stats()

        # If no gate is enforced at all, behave as unconfigured
        if not self.is_configured:
            return ConfirmationGateResult(
                entity_key=candidate.entity_key,
                candidate_class=candidate.candidate_class,
                confirmed=False,
                reason="confirmation_policy_unconfigured",
                already_confirmed=already,
                observation_count=obs_count,
                video_time_span=span,
                confidence_stats=conf_stats,
                gap_count=candidate.gap_count,
                max_gap_seconds=candidate.max_gap_seconds,
                track_ids=candidate.observed_track_ids,
                association_score=None,
                event_confidence=None,
                unmet_requirements=(),
            )

        unmet: List[str] = []

        # Gate A — observation count
        if self._min_observations is not None and obs_count < self._min_observations:
            unmet.append("min_observations")

        # Gate B — temporal span (video-time, NOT wall-clock)
        if self._min_duration_seconds is not None and span < self._min_duration_seconds:
            unmet.append("min_duration")

        # Gate C — mean detection confidence
        if self._min_confidence_mean is not None and conf_stats.mean < self._min_confidence_mean:
            unmet.append("min_confidence")

        # Gate D — maximum observed gap
        # A candidate with no recorded gaps (max_gap_seconds is None) passes.
        if self._max_gap_seconds is not None:
            if candidate.max_gap_seconds is not None and candidate.max_gap_seconds > self._max_gap_seconds:
                unmet.append("max_gap")

        confirmed = len(unmet) == 0
        reason = "all_gates_satisfied" if confirmed else f"gate_failure:{','.join(unmet)}"

        return ConfirmationGateResult(
            entity_key=candidate.entity_key,
            candidate_class=candidate.candidate_class,
            confirmed=confirmed,
            reason=reason,
            already_confirmed=already,
            observation_count=obs_count,
            video_time_span=span,
            confidence_stats=conf_stats,
            gap_count=candidate.gap_count,
            max_gap_seconds=candidate.max_gap_seconds,
            track_ids=candidate.observed_track_ids,
            association_score=None,   # Phase 2B.5
            event_confidence=None,    # no calibrated formula yet
            unmet_requirements=tuple(unmet),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ConfirmationGateResult",
    "ConfirmationPolicy",
    "UnconfiguredConfirmationPolicy",
    "CalibratedConfirmationPolicy",
]
