"""Phase 2B.4 — Post-confirmation cooldown policy.

This module implements the cooldown state described in
PHASE2A_EVENT_ARCHITECTURE.md §7 (state machine) and the Phase 2B.4 brief.

What this module does:
- Provide CooldownGateResult: a structured, JSON-serializable shadow result
  for a cooldown-state check.
- Provide UnconfiguredCooldownPolicy: the safe default.  No suppression
  window is applied; the candidate is immediately eligible after
  confirmation.  This is correct when no cooldown value has been
  calibrated yet.
- Provide CalibratedCooldownPolicy: explicitly caller-supplied cooldown
  duration in source-video seconds.  The duration is in video-time,
  NOT wall-clock, NOT processing-seconds.  This ensures the suppression
  window is tied to the source content regardless of how fast or slow
  the processor runs.

What this module intentionally does NOT do (Phase 2B.4 scope):
- No production event creation, no file I/O, no JPEG/CSV writes.
- No person-object association (Phase 2B.5).
- No changes to the legacy event path (process_detection, POSE_COOLDOWN,
  OBJECT_COOLDOWN, REQUIRED_DETECTION_TIME, evidence output).
- No hardcoded production cooldown value: the duration must be explicitly
  supplied by the caller.  UnconfiguredCooldownPolicy is the safe default
  when no calibrated value is available.

The legacy system in inference/core/pipeline.py remains the production
source of truth.  This module is observational only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .events import EntityKey, Candidate, CandidateState


# ---------------------------------------------------------------------------
# CooldownGateResult
# ---------------------------------------------------------------------------

@dataclass
class CooldownGateResult:
    """Structured shadow cooldown-check result for one candidate.

    All fields are populated so the result is usable for diagnostics even
    when in_cooldown is False.
    """
    entity_key: EntityKey
    candidate_class: str
    in_cooldown: bool
    cooldown_remaining_seconds: float
    expiry_video_time: Optional[float]
    reason: str

    def to_dict(self) -> dict:
        return {
            "entity_key": self.entity_key.as_string(),
            "candidate_class": self.candidate_class,
            "in_cooldown": self.in_cooldown,
            "cooldown_remaining_seconds": self.cooldown_remaining_seconds,
            "expiry_video_time": self.expiry_video_time,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# CooldownPolicy base
# ---------------------------------------------------------------------------

class CooldownPolicy:
    """Abstract cooldown policy.

    Subclasses implement enter_cooldown() and check_eligibility().
    The ShadowLifecycleManager holds one CooldownPolicy instance;
    changing policy is a one-line swap — no pipeline logic changes.

    enter_cooldown   — called at the moment of shadow confirmation.
                       Mutates the candidate: sets state to COOLDOWN and
                       (for calibrated policies) sets cooldown_expiry_video_time.

    check_eligibility— pure query: "is this candidate eligible to accumulate
                       new evidence?"  Does NOT mutate the candidate.  The
                       caller (CandidateLayer.advance_cooldown) is responsible
                       for the reset side-effect when the answer is "eligible".
    """

    def enter_cooldown(self, candidate: Candidate,
                    confirmation_video_time: float) -> CooldownGateResult:
        raise NotImplementedError

    def check_eligibility(self, candidate: Candidate,
                       current_video_time: float) -> CooldownGateResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# UnconfiguredCooldownPolicy
# ---------------------------------------------------------------------------

class UnconfiguredCooldownPolicy(CooldownPolicy):
    """Safe default: no calibrated cooldown window.

    enter_cooldown() returns the candidate to CANDIDATE state immediately
    (no suppression).  check_eligibility() always reports in_cooldown=False.

    Use this when no production-calibrated cooldown value is available.
    Shadow confirmation still creates an in-memory event; only the
    suppression window is absent.
    """

    def enter_cooldown(self, candidate: Candidate,
                    confirmation_video_time: float) -> CooldownGateResult:
        candidate.state = CandidateState.CANDIDATE
        return CooldownGateResult(
            entity_key=candidate.entity_key,
            candidate_class=candidate.candidate_class,
            in_cooldown=False,
            cooldown_remaining_seconds=0.0,
            expiry_video_time=None,
            reason="cooldown_policy_unconfigured",
        )

    def check_eligibility(self, candidate: Candidate,
                       current_video_time: float) -> CooldownGateResult:
        return CooldownGateResult(
            entity_key=candidate.entity_key,
            candidate_class=candidate.candidate_class,
            in_cooldown=False,
            cooldown_remaining_seconds=0.0,
            expiry_video_time=None,
            reason="cooldown_policy_unconfigured",
        )


# ---------------------------------------------------------------------------
# CalibratedCooldownPolicy
# ---------------------------------------------------------------------------

class CalibratedCooldownPolicy(CooldownPolicy):
    """Calibrated cooldown window in source-video seconds.

    The cooldown duration is in video-time, NOT wall-clock, NOT
    processing-seconds.  This is the critical constraint from the
    Phase 2B.4 brief: the suppression window must be tied to the source
    video content, not to how fast or slow the processor runs.

    Parameters
    ----------
    cooldown_seconds : float
        Suppression window duration in source-video seconds.
        Must be positive; zero or negative raises ValueError.
        This value must come from §12 calibration — do not hardcode
        a production value.
    """

    def __init__(self, cooldown_seconds: float) -> None:
        if cooldown_seconds is None:
            raise ValueError("cooldown_seconds cannot be None")
        if not isinstance(cooldown_seconds, (int, float)):
            raise ValueError(f"cooldown_seconds must be a number, got {type(cooldown_seconds).__name__}")
        if cooldown_seconds <= 0:
            raise ValueError(f"cooldown_seconds must be positive, got {cooldown_seconds}")
        self._cooldown_seconds = float(cooldown_seconds)

    @property
    def cooldown_seconds(self) -> float:
        return self._cooldown_seconds

    def enter_cooldown(self, candidate: Candidate,
                    confirmation_video_time: float) -> CooldownGateResult:
        candidate.cooldown_expiry_video_time = confirmation_video_time + self._cooldown_seconds
        candidate.state = CandidateState.COOLDOWN
        return CooldownGateResult(
            entity_key=candidate.entity_key,
            candidate_class=candidate.candidate_class,
            in_cooldown=True,
            cooldown_remaining_seconds=self._cooldown_seconds,
            expiry_video_time=candidate.cooldown_expiry_video_time,
            reason="cooldown_entered",
        )

    def check_eligibility(self, candidate: Candidate,
                       current_video_time: float) -> CooldownGateResult:
        # Pure query — does NOT mutate the candidate.
        if candidate.state != CandidateState.COOLDOWN:
            return CooldownGateResult(
                entity_key=candidate.entity_key,
                candidate_class=candidate.candidate_class,
                in_cooldown=False,
                cooldown_remaining_seconds=0.0,
                expiry_video_time=None,
                reason="not_in_cooldown",
            )
        if candidate.cooldown_expiry_video_time is None:
            # Defensive: should not happen with this policy, but
            # treat as immediately eligible rather than locking forever.
            return CooldownGateResult(
                entity_key=candidate.entity_key,
                candidate_class=candidate.candidate_class,
                in_cooldown=False,
                cooldown_remaining_seconds=0.0,
                expiry_video_time=None,
                reason="no_expiry_recorded",
            )
        remaining = candidate.cooldown_expiry_video_time - current_video_time
        if remaining <= 0.0:
            return CooldownGateResult(
                entity_key=candidate.entity_key,
                candidate_class=candidate.candidate_class,
                in_cooldown=False,
                cooldown_remaining_seconds=0.0,
                expiry_video_time=None,
                reason="cooldown_elapsed",
            )
        return CooldownGateResult(
            entity_key=candidate.entity_key,
            candidate_class=candidate.candidate_class,
            in_cooldown=True,
            cooldown_remaining_seconds=remaining,
            expiry_video_time=candidate.cooldown_expiry_video_time,
            reason="cooldown_active",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CooldownGateResult",
    "CooldownPolicy",
    "UnconfiguredCooldownPolicy",
    "CalibratedCooldownPolicy",
]
