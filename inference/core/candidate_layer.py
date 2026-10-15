"""Phase 2B.2 — Shadow candidate observation layer.

This module implements the temporal observation layer described in
PHASE2A_EVENT_ARCHITECTURE.md §3 (2B.2) and §4.

What this module does:
- Provide VideoTimeSource: converts source frame index to video-time using
  the pipeline's existing src_fps value.  Fails explicitly (raises
  InvalidSourceFps) when src_fps is zero, negative, or None — it never
  silently falls back to wall-clock time.
- Provide CandidateKey: explicit compound identity for a candidate,
  (EntityKey, candidate_class).  Namespace isolation is guaranteed by
  the EntityKey component.
- Provide CandidateLayer: a bounded shadow store of Candidate objects.
  Observation accumulation, gap bookkeeping (measured, no policy), LOST
  state representation, and stale-candidate eviction are all implemented
  here.  No confirmation decision is made; no event is created; no
  production output is written.

What this module intentionally does NOT do (Phase 2B.2 scope):
- No confirmation gate, no event confidence, no event creation.
- No person-object association.
- No production event-policy numbers (candidate duration, minimum
  observation count, cooldown, association threshold, keypoint threshold,
  ID-switch threshold).  All structural parameters are explicit
  constructor arguments — no hidden production defaults.
- No changes to the legacy event path (process_detection, id_info_*,
  cooldown, REQUIRED_DETECTION_TIME, evidence output).

The legacy system in inference/core/pipeline.py remains the production
source of truth.  This layer is observational only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from .events import (
    ModelSource,
    EntityKey,
    Observation,
    BoundedObservationBuffer,
    CandidateState,
    Candidate,
    ConfidenceStats,
)
from .cooldown import CooldownPolicy


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InvalidSourceFps(Exception):
    """Raised when src_fps is zero, negative, None, or otherwise unusable
    for video-time computation.

    The candidate layer must never silently fall back to wall-clock time;
    an invalid FPS is a data-availability problem that the operator must
    resolve (fix the video file or the capture pipeline) rather than a
    value the code can paper over.
    """


# ---------------------------------------------------------------------------
# VideoTimeSource
# ---------------------------------------------------------------------------

class VideoTimeSource:
    """Converts a source frame index to video-time (seconds).

    Uses the pipeline's existing src_fps (obtained from
    cv2.CAP_PROP_FPS in process_video).  video_time = frame_index / src_fps.

    Construction validates that src_fps is a positive finite number;
    InvalidSourceFps is raised otherwise.
    """

    def __init__(self, src_fps: float) -> None:
        if src_fps is None:
            raise InvalidSourceFps(
                "src_fps is None — cannot derive video-time. "
                "Check that the video file has a valid frame-rate property."
            )
        if not isinstance(src_fps, (int, float)):
            raise InvalidSourceFps(
                f"src_fps must be a number, got {type(src_fps).__name__}"
            )
        if src_fps <= 0:
            raise InvalidSourceFps(
                f"src_fps must be positive, got {src_fps}. "
                "cv2.CAP_PROP_FPS returned an unusable value."
            )
        self._src_fps = float(src_fps)

    @property
    def src_fps(self) -> float:
        return self._src_fps

    @property
    def is_available(self) -> bool:
        """True once constructed successfully (i.e., src_fps is valid)."""
        return True

    def video_time_for(self, frame_index: int) -> float:
        """Return the video-time (seconds) for the given source frame index.

        frame_index 0 is the start of the source video (video_time 0.0).
        The pipeline increments self.frame_idx *before* processing a frame,
        so frame_index here is 1-based in the pipeline's own convention.
        Both conventions are valid as long as they are used consistently;
        the layer does not impose a specific offset.
        """
        if frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {frame_index}")
        return frame_index / self._src_fps


# ---------------------------------------------------------------------------
# CandidateKey
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateKey:
    """Explicit compound identity for a candidate in the observation layer.

    A candidate is uniquely identified by (EntityKey, candidate_class).
    This guarantees that pose:3 + smartphone and object:3 + smartphone
    are never conflated, and that pose:3 + smartphone and
    pose:3 + provide-object are distinct candidates.

    Use the classmethod from_observation to derive a CandidateKey from
    an Observation instance.
    """
    entity_key: EntityKey
    candidate_class: str

    @classmethod
    def from_observation(cls, obs: Observation) -> CandidateKey:
        entity_key = EntityKey(source=obs.source, track_id=obs.track_id)
        return cls(entity_key=entity_key, candidate_class=obs.class_name)

    def as_string(self) -> str:
        return f"{self.entity_key.as_string()}:{self.candidate_class}"

    def __str__(self) -> str:
        return self.as_string()


# ---------------------------------------------------------------------------
# CandidateLayer
# ---------------------------------------------------------------------------

class CandidateLayer:
    """Bounded shadow store of Candidate objects, keyed by CandidateKey.

    Observation-only: no confirmation, no event creation, no production
    output.  The layer is safe to run in parallel with the legacy event
    path; it does not influence when or whether a legacy event fires.

    Parameters
    ----------
    src_fps : float
        Source video frame rate (from cv2.CAP_PROP_FPS).  Must be
        positive; InvalidSourceFps is raised otherwise.
    observation_buffer_maxlen : int
        Maximum number of observations retained per candidate.
        Structural parameter (memory bound), not an event-policy value.
        Must be explicitly provided — no hidden default.
    max_active_candidates : Optional[int]
        Maximum number of candidates the layer holds at once.
        When the cap is exceeded the least-recently-touched candidate
        is evicted.  Structural parameter (memory bound), not policy.
        Defaults to None (no cap); callers who need a cap must pass
        an explicit positive integer.
    """

    def __init__(self,
                 src_fps: float,
                 observation_buffer_maxlen: int,
                 max_active_candidates: Optional[int] = None) -> None:
        self._vts = VideoTimeSource(src_fps)
        if observation_buffer_maxlen <= 0:
            raise ValueError(
                f"observation_buffer_maxlen must be positive, got {observation_buffer_maxlen}"
            )
        if max_active_candidates is not None and max_active_candidates <= 0:
            raise ValueError(
                f"max_active_candidates must be positive or None, got {max_active_candidates}"
            )
        self._obs_maxlen = observation_buffer_maxlen
        self._max_active = max_active_candidates
        # LRU ordering: candidates most-recently touched go to the end.
        self._candidates: Dict[CandidateKey, Candidate] = {}
        self._lru_order: List[CandidateKey] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def video_time_source(self) -> VideoTimeSource:
        return self._vts

    @property
    def observation_buffer_maxlen(self) -> int:
        return self._obs_maxlen

    @property
    def max_active_candidates(self) -> Optional[int]:
        return self._max_active

    # ------------------------------------------------------------------
    # Observation recording
    # ------------------------------------------------------------------

    def record_observation(self, obs: Observation) -> Candidate:
        """Record a detector observation into the candidate layer.

        Creates a new Candidate (in CANDIDATE state) if one does not yet
        exist for this (EntityKey, class_name), or appends to an
        existing one.  If the candidate was in LOST state, it is
        transitioned back to CANDIDATE (recovery — representation only,
        no policy decision about whether the gap was "acceptable").

        Returns the Candidate so the caller can inspect accumulated
        statistics without a second lookup.
        """
        key = CandidateKey.from_observation(obs)
        cand = self._candidates.get(key)

        if cand is None:
            entity_key = EntityKey(source=obs.source, track_id=obs.track_id)
            cand = Candidate(
                entity_key=entity_key,
                candidate_class=obs.class_name,
                state=CandidateState.CANDIDATE,
                observations=BoundedObservationBuffer(maxlen=self._obs_maxlen),
                gap_count=0,
                max_gap_seconds=None,
            )
            self._candidates[key] = cand
            self._lru_order.append(key)
            self._evict_if_over_cap()

        # Recovery from LOST: a new observation means the track is back.
        if cand.state == CandidateState.LOST:
            cand.state = CandidateState.CANDIDATE

        cand.add_observation(obs)
        self._touch_lru(key)
        return cand

    # ------------------------------------------------------------------
    # Gap bookkeeping
    # ------------------------------------------------------------------

    def note_gap(self, entity_key: EntityKey, candidate_class: str,
                 gap_seconds: Optional[float] = None) -> Optional[Candidate]:
        """Record that the entity has entered an observation gap.

        Increments gap_count and updates max_gap_seconds on the candidate.
        Transitions the candidate to LOST state (representation only —
        no policy decision about whether the gap is acceptable; that is
        Phase 2B.3+ work).

        If the candidate does not exist, returns None.
        If gap_seconds is None, the layer uses the candidate's observed
        video-time span as a conservative lower bound for the gap
        duration.  The caller should prefer to pass gap_seconds
        explicitly when the gap length is known.
        """
        key = CandidateKey(entity_key=entity_key, candidate_class=candidate_class)
        cand = self._candidates.get(key)
        if cand is None:
            return None

        cand.gap_count = (cand.gap_count or 0) + 1

        if gap_seconds is not None:
            if cand.max_gap_seconds is None or gap_seconds > cand.max_gap_seconds:
                cand.max_gap_seconds = gap_seconds
        elif cand.latest_video_time is not None and cand.start_video_time is not None:
            observed_span = cand.latest_video_time - cand.start_video_time
            if cand.max_gap_seconds is None or observed_span > cand.max_gap_seconds:
                cand.max_gap_seconds = observed_span

        cand.state = CandidateState.LOST
        self._touch_lru(key)
        return cand

    # ------------------------------------------------------------------
    # Candidate access
    # ------------------------------------------------------------------

    def has_candidate(self, entity_key: EntityKey, candidate_class: str) -> bool:
        return CandidateKey(entity_key=entity_key, candidate_class=candidate_class) in self._candidates

    def get_candidate(self, entity_key: EntityKey, candidate_class: str) -> Optional[Candidate]:
        return self._candidates.get(
            CandidateKey(entity_key=entity_key, candidate_class=candidate_class)
        )

    def active_candidates(self) -> List[CandidateKey]:
        """Return the list of active candidate keys in LRU order."""
        return list(self._lru_order)

    def all_candidates(self) -> Dict[CandidateKey, Candidate]:
        """Return a snapshot of all current candidates (for diagnostics)."""
        return dict(self._candidates)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict_candidate(self, entity_key: EntityKey, candidate_class: str) -> bool:
        """Explicitly remove a candidate from the layer.

        Returns True if the candidate was present and removed, False if
        it was not found.  The caller is responsible for deciding which
        candidates are stale — the layer does not apply a hidden timer.
        """
        key = CandidateKey(entity_key=entity_key, candidate_class=candidate_class)
        if key in self._candidates:
            del self._candidates[key]
            self._lru_order.remove(key)
            return True
        return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> List[Dict]:
        """Return a JSON-serializable diagnostic snapshot of all candidates.

        Intended for logging, test inspection, and future Phase 2B.6
        evidence-generation.  Does NOT write to production output.
        """
        return [cand.to_dict() for cand in self._candidates.values()]

    # ------------------------------------------------------------------
    # Phase 2B.4 — Cooldown lifecycle
    # ------------------------------------------------------------------

    def advance_cooldown(self, entity_key: EntityKey, candidate_class: str,
                         current_video_time: float,
                         policy: "CooldownPolicy") -> Optional[Candidate]:
        """Check whether a candidate's cooldown has expired and reset it.

        If the candidate for (entity_key, candidate_class) is in COOLDOWN
        state and the policy reports it is eligible (cooldown_elapsed),
        the candidate is structurally reset for a new incident cycle:
        observation buffer cleared, gap stats cleared, state → CANDIDATE.

        This reset is structural, not a policy decision about whether the
        incident was "resolved."  It simply frees the candidate to
        accumulate fresh evidence.

        Returns the candidate, or None if not found.
        """
        key = CandidateKey(entity_key=entity_key, candidate_class=candidate_class)
        cand = self._candidates.get(key)
        if cand is None:
            return None
        result = policy.check_eligibility(cand, current_video_time)
        if not result.in_cooldown and cand.state == CandidateState.COOLDOWN:
            self._reset_candidate(cand)
        return cand

    def _reset_candidate(self, cand: Candidate) -> None:
        """Structural reset for a new incident cycle.

        Clears all accumulated evidence so the candidate starts fresh.
        No policy value is applied here — this is a memory reset, not a
        decision about what the new incident will require.
        """
        cand.observations = BoundedObservationBuffer(maxlen=self._obs_maxlen)
        cand.gap_count = 0
        cand.max_gap_seconds = None
        cand.start_video_time = None
        cand.latest_video_time = None
        cand.observed_track_ids = frozenset()
        cand.state = CandidateState.CANDIDATE
        cand.cooldown_expiry_video_time = None

    # ------------------------------------------------------------------
    # Internal LRU helpers
    # ------------------------------------------------------------------

    def _touch_lru(self, key: CandidateKey) -> None:
        if key in self._lru_order:
            self._lru_order.remove(key)
        self._lru_order.append(key)

    def _evict_if_over_cap(self) -> None:
        if self._max_active is None:
            return
        while len(self._lru_order) > self._max_active:
            oldest = self._lru_order.pop(0)
            self._candidates.pop(oldest, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "InvalidSourceFps",
    "VideoTimeSource",
    "CandidateKey",
    "CandidateLayer",
]
