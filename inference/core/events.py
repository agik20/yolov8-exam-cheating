"""Phase 2B.1 — Event data structures and entity normalization.

This module defines the data model for the future candidate/confirmation event
engine (see PHASE2A_EVENT_ARCHITECTURE.md, §3, §5, §7, §8, §9).

What this module does:
- Introduce an EntityKey that disambiguates track IDs across the pose and
  object model namespaces.
- Define Observation, Candidate, EventRecord, and EvidenceBundle dataclasses.
- Define the CandidateState enum (IDLE / CANDIDATE / CONFIRMED / COOLDOWN,
  plus LOST as a future transitional state).
- Provide a bounded observation buffer (BoundedObservationBuffer) designed so
  that future Phase 2B.2 accumulation can be memory-safe.

What this module intentionally does NOT do (Phase 2B.1 scope):
- No confirmation logic, no confirmation score.
- No temporal persistence changes.
- No person-object association.
- No evidence generation (JPEG, CSV, etc.).
- No numeric event-policy defaults (no candidate duration, no minimum
  observation count, no cooldown value, no association threshold, no
  keypoint threshold, no ID-switch threshold).
- No changes to the legacy event path (id_info_pose / id_info_obj /
  process_detection / cooldown / REQUIRED_DETECTION_TIME).

The legacy system in inference/core/pipeline.py remains the production source
of truth until a future phase explicitly cuts over.
"""
from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, FrozenSet, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Model namespace
# ---------------------------------------------------------------------------

class ModelSource(Enum):
    """Which model produced a detection or observation.

    The two models each run their own independent BoT-SORT tracker with
    independent numeric track-ID spaces.  ModelSource is the namespace
    discriminator used by EntityKey to prevent cross-model ID collision.
    """
    POSE = "pose"
    OBJECT = "object"


# ---------------------------------------------------------------------------
# Entity key
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntityKey:
    """Pipeline-internal entity key, unique across model namespaces.

    A raw numeric track_id is only meaningful within one model namespace:
    pose track_id 3 and object track_id 3 are unrelated entities.  EntityKey
    binds a track ID to its model source so the two are never conflated.

    Use the classmethods from_pose / from_object; do not construct with
    raw strings — they encode the model source prefix.

    Examples:
        EntityKey.from_pose(3)       -> EntityKey(source=ModelSource.POSE, track_id=3)
        EntityKey.from_object(3)     -> EntityKey(source=ModelSource.OBJECT, track_id=3)
        EntityKey.from_pose(3) == EntityKey.from_object(3)  # False
    """
    source: ModelSource
    track_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, ModelSource):
            raise TypeError(f"source must be a ModelSource, got {type(self.source).__name__}")
        if not isinstance(self.track_id, int):
            raise TypeError(f"track_id must be int, got {type(self.track_id).__name__}")

    @classmethod
    def from_pose(cls, track_id: int) -> EntityKey:
        return cls(ModelSource.POSE, track_id)

    @classmethod
    def from_object(cls, track_id: int) -> EntityKey:
        return cls(ModelSource.OBJECT, track_id)

    def as_string(self) -> str:
        """Human-readable key: 'pose:3', 'object:17', etc."""
        return f"{self.source.value}:{self.track_id}"

    def __str__(self) -> str:
        return self.as_string()


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """One detector observation.

    Small and serializable.  Does not store image data — only the geometric
    and confidence information needed for future confirmation scoring.

    The keypoints field is optional (Phase 2B.5).  It is None for object-model
    observations and for pose observations where the pipeline did not capture
    keypoint data.  When present, it holds the raw per-keypoint (x, y, visibility)
    tuples in source-frame pixel coordinates.
    """
    frame_index: int
    video_time: float          # seconds into source video (frame_index / src_fps)
    class_name: str
    confidence: float          # model detection confidence (0-1), NOT event confidence
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in source-frame pixels
    source: ModelSource
    track_id: int
    # Phase 2B.5 — optional per-keypoint data.
    # Each entry is (x, y, visibility) in source-frame pixels.
    # None means "no keypoint data for this observation."
    keypoints: Optional[Tuple[Tuple[float, float, float], ...]] = None

    def to_dict(self) -> Dict:
        """Flat dict suitable for JSON serialization or CSV row construction."""
        result: Dict = {
            "frame_index": self.frame_index,
            "video_time": self.video_time,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "source": self.source.value,
            "track_id": self.track_id,
        }
        if self.keypoints is not None:
            result["keypoints"] = [list(kp) for kp in self.keypoints]
        return result


# ---------------------------------------------------------------------------
# Bounded observation buffer
# ---------------------------------------------------------------------------

class BoundedObservationBuffer:
    """Fixed-capacity ring of Observations (oldest evicted first).

    Capacity is a constructor parameter — no production default value is
    chosen here.  The future Phase 2B.2 calibration (PHASE2A §12) will
    determine the appropriate capacity; for now callers must pass an
    explicit ``maxlen``.

    Thread-safety: not designed for concurrent access; the pipeline
    processes frames sequentially on one thread.
    """

    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")
        self._buffer: Deque[Observation] = deque(maxlen=maxlen)

    @property
    def maxlen(self) -> int:
        return self._buffer.maxlen  # type: ignore[return-value]

    def append(self, obs: Observation) -> None:
        self._buffer.append(obs)

    def all(self) -> List[Observation]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)

    def is_full(self) -> bool:
        return len(self._buffer) >= self.maxlen


# ---------------------------------------------------------------------------
# Candidate state
# ---------------------------------------------------------------------------

class CandidateState(Enum):
    """Explicit state values for the future candidate event engine.

    Phase 2B.1: these values can be *represented* on a Candidate instance.
    Phase 2B.1 does NOT implement state transitions — that is future work.

    IDLE        — no active suspicious candidate for this entity-class.
    CANDIDATE   — suspicious detections are accumulating evidence.
                  A candidate must be able to dissolve (IDLE) with no
                  event produced.
    CONFIRMED   — evidence is sufficient to produce a reportable event.
                  This is the only state that should trigger event creation.
    COOLDOWN    — post-confirmation suppression window for the same incident.
                  Does not block observation or buffer updates.
    LOST        — (future transitional) the entity's track has temporarily
                  vanished; a grace period may keep accumulated evidence.
                  Whether this state is retained depends on Phase 2A
                  open question #2 (track-ID reliability under occlusion).
    """
    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    COOLDOWN = "COOLDOWN"
    LOST = "LOST"


# ---------------------------------------------------------------------------
# Confidence statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfidenceStats:
    """Aggregate confidence statistics over a candidate window.

    Replaces the legacy single ``last_conf`` value (which is set once at
    track-creation and never updated — see PHASE2A §2.C).  Computed from
    the Observation buffer at confirmation time; stored here for the
    evidence bundle.
    """
    count: int
    mean: float
    min: float
    max: float

    @classmethod
    def from_observations(cls, obs_list: List[Observation]) -> ConfidenceStats:
        if not obs_list:
            return cls(count=0, mean=0.0, min=0.0, max=0.0)
        confs = [o.confidence for o in obs_list]
        return cls(
            count=len(confs),
            mean=sum(confs) / len(confs),
            min=min(confs),
            max=max(confs),
        )

    def to_dict(self) -> Dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
        }


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """An accumulating suspicious behavior candidate for one entity-class.

    A Candidate represents the future "CANDIDATE" state from PHASE2A §3.2.
    It holds an evidence buffer and metadata needed for a future confirmation
    decision.  Phase 2B.1 does not compute a confirmation score or make a
    confirmation decision — those are Phase 2B.3+.

    Fields intentionally left as Optional or with no policy defaults:
    - start_video_time / latest_video_time: set by the pipeline at
      open/update time; no pre-filled default that implies a threshold.
    - confidence_stats: computed on demand from the buffer.
    - observed_track_ids: tracks that contributed to this candidate (an
      ID switch inside a candidate window is a confirmation red flag,
      PHASE2A §5 item 5).
    - gap_count / max_gap_seconds: populated by future gap-handling logic
      (PHASE2A §5 item 3); None until then.
    - class_consistency: fraction of observations matching the candidate's
      class (PHASE2A §5 item 4); None until computed.
    - association_score: person-object association placeholder
      (PHASE2A §6); None until Phase 2B.5.
    - event_confidence: independent quality score for a confirmed event
      (PHASE2A §7); None until Phase 2B.3+ computes it.
    """
    entity_key: EntityKey
    candidate_class: str
    state: CandidateState = CandidateState.CANDIDATE
    start_video_time: Optional[float] = None
    latest_video_time: Optional[float] = None
    observations: BoundedObservationBuffer = field(
        default_factory=lambda: BoundedObservationBuffer(maxlen=128)
    )
    observed_track_ids: FrozenSet[int] = field(default_factory=frozenset)
    gap_count: Optional[int] = None
    max_gap_seconds: Optional[float] = None
    class_consistency: Optional[float] = None
    association_score: Optional[float] = None
    event_confidence: Optional[float] = None
    # Phase 2B.4 — video-time at which the current cooldown window expires.
    # Set by CooldownPolicy.enter_cooldown(); None means no active cooldown.
    cooldown_expiry_video_time: Optional[float] = None

    # -- convenience accessors (no policy, no confirmation) --

    def observation_count(self) -> int:
        return len(self.observations)

    def video_time_span(self) -> float:
        """Elapsed video-time (seconds) from first to latest observation.

        Returns 0.0 when no observations have been recorded yet.
        """
        if self.start_video_time is None or self.latest_video_time is None:
            return 0.0
        return self.latest_video_time - self.start_video_time

    def confidence_stats(self) -> ConfidenceStats:
        return ConfidenceStats.from_observations(self.observations.all())

    def add_observation(self, obs: Observation) -> None:
        """Append an observation to the buffer and update latest_video_time.

        Does NOT transition state.  State transition is Phase 2B.2+ work.
        """
        self.observations.append(obs)
        self.latest_video_time = obs.video_time
        if self.start_video_time is None:
            self.start_video_time = obs.video_time
        self.observed_track_ids = self.observed_track_ids | frozenset([obs.track_id])

    def to_dict(self) -> Dict:
        return {
            "entity_key": self.entity_key.as_string(),
            "candidate_class": self.candidate_class,
            "state": self.state.value,
            "start_video_time": self.start_video_time,
            "latest_video_time": self.latest_video_time,
            "observation_count": self.observation_count(),
            "confidence_stats": self.confidence_stats().to_dict(),
            "observed_track_ids": sorted(self.observed_track_ids),
            "gap_count": self.gap_count,
            "max_gap_seconds": self.max_gap_seconds,
            "class_consistency": self.class_consistency,
            "association_score": self.association_score,
            "event_confidence": self.event_confidence,
            "cooldown_expiry_video_time": self.cooldown_expiry_video_time,
        }


# ---------------------------------------------------------------------------
# EventRecord
# ---------------------------------------------------------------------------

@dataclass
class EventRecord:
    """A future confirmed event.

    Phase 2B.1: the structure exists so the event model can be represented
    and tested.  Event creation is not yet wired into the pipeline — that
    happens at Phase 2B.4 (confirmation decision + cutover).

    event_id is pipeline-unique (uuid4); it is NOT a track ID.  Using a
    track ID as the event ID would collide across models and would not
    survive a future ID switch (PHASE2A §2.D, G).
    """
    event_id: str
    entity_key: EntityKey
    class_name: str
    confirmation_video_time: float
    confirmation_wall_time: str          # ISO-8601 wall-clock timestamp at confirmation
    candidate_start_video_time: float
    candidate_end_video_time: float
    observation_count: int
    confidence_stats: ConfidenceStats
    event_confidence: Optional[float]    # populated at confirmation (Phase 2B.3+)
    association_score: Optional[float]   # populated at association (Phase 2B.5)
    contributing_track_ids: FrozenSet[int]

    @classmethod
    def from_candidate(cls, candidate: Candidate, confirmation_video_time: float,
                       confirmation_wall_time: str) -> EventRecord:
        """Construct an EventRecord from a Candidate that has met confirmation.

        This is a convenience factory; it does not perform the confirmation
        decision itself.  Callers are responsible for verifying the
        confirmation gate before calling this.
        """
        return cls(
            event_id=str(uuid.uuid4()),
            entity_key=candidate.entity_key,
            class_name=candidate.candidate_class,
            confirmation_video_time=confirmation_video_time,
            confirmation_wall_time=confirmation_wall_time,
            candidate_start_video_time=candidate.start_video_time or 0.0,
            candidate_end_video_time=candidate.latest_video_time or 0.0,
            observation_count=candidate.observation_count(),
            confidence_stats=candidate.confidence_stats(),
            event_confidence=candidate.event_confidence,
            association_score=candidate.association_score,
            contributing_track_ids=candidate.observed_track_ids,
        )

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "entity_key": self.entity_key.as_string(),
            "class_name": self.class_name,
            "confirmation_video_time": self.confirmation_video_time,
            "confirmation_wall_time": self.confirmation_wall_time,
            "candidate_start_video_time": self.candidate_start_video_time,
            "candidate_end_video_time": self.candidate_end_video_time,
            "observation_count": self.observation_count,
            "confidence_stats": self.confidence_stats.to_dict(),
            "event_confidence": self.event_confidence,
            "association_score": self.association_score,
            "contributing_track_ids": sorted(self.contributing_track_ids),
        }


# ---------------------------------------------------------------------------
# Phase 2B.6 — Gate outcome + ConfirmationDecisionEvidence
# ---------------------------------------------------------------------------

class GateOutcome(Enum):
    """Per-gate auditability outcome.

    A gate that is not configured is NOT shown as failed.  The four values
    keep "unconfigured" (never enforced) semantically distinct from "failed"
    (enforced, threshold not met).
    """
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    UNAVAILABLE = "UNAVAILABLE"   # gate configured but required data missing

    def to_dict(self) -> str:
        return self.value


@dataclass
class ConfirmationDecisionEvidence:
    """Why a confirmation shadow decision produced the result it did.

    Records, per gate, whether it was configured, the required threshold, the
    actually-measured value, and the outcome (PASS / FAIL / NOT_CONFIGURED /
    UNAVAILABLE).  A not-configured gate is represented explicitly so a reader
    can never mistake "skipped" for "failed".

    policy_type: "calibrated" (CalibratedConfirmationPolicy) or "unconfigured"
        (UnconfiguredConfirmationPolicy).

    Phase 2B.6 audit note: "unconfigured" means the gate was never enforced
    (policy has no threshold), NOT that a calibrated gate failed.
    """
    policy_type: str
    confirmed: bool
    # -- min_observations gate --
    min_observations_configured: bool = False
    min_observations_required: Optional[int] = None
    actual_observations: int = 0
    # -- min_duration gate --
    min_duration_seconds_configured: bool = False
    min_duration_seconds_required: Optional[float] = None
    actual_duration: Optional[float] = None
    # -- min_confidence gate --
    min_confidence_mean_configured: bool = False
    min_confidence_mean_required: Optional[float] = None
    actual_confidence_mean: Optional[float] = None
    # -- max_gap gate --
    max_gap_seconds_configured: bool = False
    max_gap_seconds_required: Optional[float] = None
    actual_max_gap: Optional[float] = None

    @classmethod
    def from_gate_result(cls,
                         result: "ConfirmationGateResult",
                         policy: "ConfirmationPolicy") -> "ConfirmationDecisionEvidence":
        """Build decision evidence from a ConfirmationGateResult + its policy.

        Which gates are *configured* is read from the policy's public
        threshold properties (CalibratedConfirmationPolicy exposes
        ``min_observations`` / ``min_duration_seconds`` /
        ``min_confidence_mean`` / ``max_gap_seconds`` as properties;
        UnconfiguredConfirmationPolicy exposes none).  ``getattr`` with a None
        default therefore reports "not configured" for the unconfigured policy
        and the real thresholds for a calibrated one, without importing
        confirmation.py (events.py must stay free of that circular import).
        The measured values come from ``result``.
        """
        stats = result.confidence_stats
        obs_cfg = cls._read_threshold(policy, "min_observations")
        dur_cfg = cls._read_threshold(policy, "min_duration_seconds")
        conf_cfg = cls._read_threshold(policy, "min_confidence_mean")
        gap_cfg = cls._read_threshold(policy, "max_gap_seconds")
        calibrated = any(v is not None for v in (obs_cfg, dur_cfg, conf_cfg, gap_cfg))

        return cls(
            policy_type="calibrated" if calibrated else "unconfigured",
            confirmed=result.confirmed,
            min_observations_configured=obs_cfg is not None,
            min_observations_required=obs_cfg,
            actual_observations=result.observation_count,
            min_duration_seconds_configured=dur_cfg is not None,
            min_duration_seconds_required=dur_cfg,
            actual_duration=result.video_time_span,
            min_confidence_mean_configured=conf_cfg is not None,
            min_confidence_mean_required=conf_cfg,
            actual_confidence_mean=(stats.mean if stats else None),
            max_gap_seconds_configured=gap_cfg is not None,
            max_gap_seconds_required=gap_cfg,
            actual_max_gap=result.max_gap_seconds,
        )

    @staticmethod
    def _read_threshold(policy: "ConfirmationPolicy", name: str):
        """Read a policy's threshold for ``name``, public or private attribute.

        CalibratedConfirmationPolicy stores its thresholds as private
        attributes (``_min_observations`` etc.) with no public properties;
        UnconfiguredConfirmationPolicy has none.  Probing both the public and
        private names keeps this circular-import-free (no confirmation.py
        import) while still reporting the real thresholds.
        """
        val = getattr(policy, name, None)
        if isinstance(val, (int, float)):
            return val
        return getattr(policy, "_" + name, None)

    # Comparison direction each gate uses for PASS/FAIL:
    #   min_*   : actual >= required
    #   max_gap : actual <= required (a candidate with no recorded gap passes)
    _GATE_DIRECTION = {
        "min_observations": "min",
        "min_duration": "min",
        "min_confidence": "min",
        "max_gap": "max",
    }

    def _gate_outcome(self, label: str, configured: bool, required, actual) -> "GateOutcome":
        if not configured or required is None:
            return GateOutcome.NOT_CONFIGURED
        if actual is None:
            return GateOutcome.UNAVAILABLE
        if self._GATE_DIRECTION[label] == "min":
            return GateOutcome.PASS if actual >= required else GateOutcome.FAIL
        return GateOutcome.PASS if actual <= required else GateOutcome.FAIL

    def _gate_dict(self, label: str, configured: bool, required, actual) -> Dict:
        if not configured or required is None:
            return {"configured": False,
                    "outcome": GateOutcome.NOT_CONFIGURED.to_dict(),
                    "required": None,
                    "actual": actual}
        return {"configured": True,
                "outcome": self._gate_outcome(label, configured, required, actual).to_dict(),
                "required": required,
                "actual": actual}

    def to_dict(self) -> Dict:
        """Deterministic dict; enum outcomes serialized as strings."""
        return {
            "policy_type": self.policy_type,
            "confirmed": self.confirmed,
            "gates": {
                "min_observations": self._gate_dict(
                    "min_observations", self.min_observations_configured,
                    self.min_observations_required, self.actual_observations),
                "min_duration": self._gate_dict(
                    "min_duration", self.min_duration_seconds_configured,
                    self.min_duration_seconds_required, self.actual_duration),
                "min_confidence": self._gate_dict(
                    "min_confidence", self.min_confidence_mean_configured,
                    self.min_confidence_mean_required, self.actual_confidence_mean),
                "max_gap": self._gate_dict(
                    "max_gap", self.max_gap_seconds_configured,
                    self.max_gap_seconds_required, self.actual_max_gap),
            },
        }


# ---------------------------------------------------------------------------
# Phase 2B.6 — TrackIdentityEvidence
# ---------------------------------------------------------------------------

@dataclass
class TrackIdentityEvidence:
    """Identity information needed for audit.

    ``track_id_switches`` is None unless the system actually measured a
    confirmed switch.  A candidate whose observed track-ID set contains more
    than one distinct value is NOT, by itself, evidence of a confirmed switch
    — the two are deliberately kept separate so a reader never infers a
    switch from a plain ID difference.
    """
    entity_key: EntityKey
    track_ids: FrozenSet[int] = field(default_factory=frozenset)
    track_id_switches: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "entity_key": self.entity_key.as_string(),
            "track_ids": sorted(self.track_ids),
            "track_id_switches": self.track_id_switches,
        }


# ---------------------------------------------------------------------------
# Phase 2B.6 — AssociationEvidenceSummary
# ---------------------------------------------------------------------------

@dataclass
class AssociationEvidenceSummary:
    """Phase 2B.5 association result carried into the evidence bundle.

    ``association_score`` mirrors AssociationResult.score and stays None
    unless a caller explicitly injects a calibrated value — no synthetic
    score is fabricated here.  ``association_observations`` is a bounded
    tuple of per-timepoint diagnostics so an ambiguous candidate can still
    be audited without unbounded growth.
    """
    association_result: "AssociationResult"
    person_entity: Optional[str] = None
    object_entity: Optional[str] = None
    association_observations: Tuple[Dict, ...] = ()

    @property
    def association_score(self) -> Optional[float]:
        return self.association_result.score

    @property
    def status(self) -> str:
        return self.association_result.status.value

    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "association_score": self.association_score,
            "person_entity": self.person_entity,
            "object_entity": self.object_entity,
            "association_observations": [dict(o) for o in self.association_observations],
            "result": self.association_result.to_dict(),
        }


# ---------------------------------------------------------------------------
# Phase 2B.6 — HumanEvidenceMetadata
# ---------------------------------------------------------------------------

@dataclass
class HumanEvidenceMetadata:
    """Metadata for human verification of an event.

    All reference fields are Optional[str] file references.  Phase 2B.6 does
    NOT perform image saving; a None reference means "not yet generated."  No
    image bytes are stored here — only path references.

    The metric fields (evidence_quality / brightness / blur / resolution) are
    populated only when the corresponding value is already available upstream;
    None means "not measured", never a fabricated 0.
    """
    full_frame_reference: Optional[str] = None
    expanded_crop_reference: Optional[str] = None
    person_object_crop_reference: Optional[str] = None
    evidence_quality: Optional[float] = None
    brightness: Optional[float] = None
    blur: Optional[float] = None
    resolution: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "full_frame_reference": self.full_frame_reference,
            "expanded_crop_reference": self.expanded_crop_reference,
            "person_object_crop_reference": self.person_object_crop_reference,
            "evidence_quality": self.evidence_quality,
            "brightness": self.brightness,
            "blur": self.blur,
            "resolution": self.resolution,
        }


# ---------------------------------------------------------------------------
# EvidenceBundle
# ---------------------------------------------------------------------------

@dataclass
class EvidenceBundle:
    """Structured evidence container for a confirmed EventRecord.

    Three layers, kept separate so a human reviewer can reconstruct the
    decision without trusting any single model number (PHASE2A §8):

    1. detection_evidence  — what the model saw (boxes, confs over the
       candidate window; ConfidenceStats summary).
    2. decision_evidence   — why the event was confirmed (observation count,
       video-time span, which confirmation terms fired, track IDs, any
       switch events).
    3. human_evidence      — what a reviewer inspects (annotated frame
       reference, crop file path, person-object pair crop, temporal summary,
       one-line human-readable verdict).

    Phase 2B.1: the structure exists.  Evidence *generation* (writing files,
       producing the annotated frame) is not implemented here — that is
       Phase 2B.6.  This class does NOT duplicate the existing JPEG/CSV
       output; it describes where future evidence will live.

    Phase 2B.6 (additive): four Optional sections are appended —
    decision (ConfirmationDecisionEvidence), association
    (AssociationEvidenceSummary), human (HumanEvidenceMetadata) and
    identity (TrackIdentityEvidence).  All default to None; the bundle stays
    fully backward-compatible with Phase 2B.1 construction sites.

    File paths are Optional; they are populated by the evidence-generation
       layer at Phase 2B.6.  A None path means "not yet generated."
    """
    event_record: EventRecord

    # -- detection evidence --
    detection_confidence_stats: ConfidenceStats
    detection_observation_count: int

    # -- decision evidence --
    confirmation_video_time: float
    confirmation_wall_time: str
    candidate_start_video_time: float
    candidate_end_video_time: float
    track_ids_in_window: FrozenSet[int]
    # Populated by Phase 2B.3+ confirmation logic:
    confirmation_terms_met: Tuple[str, ...] = ()
    id_switch_count: Optional[int] = None

    # -- human evidence --
    annotated_frame_path: Optional[str] = None
    crop_path: Optional[str] = None
    person_object_pair_crop_path: Optional[str] = None
    temporal_summary: Optional[str] = None
    human_verdict_line: Optional[str] = None

    # -- Phase 2B.6 additive sections (all default None) --
    decision: Optional[ConfirmationDecisionEvidence] = None
    association: Optional[AssociationEvidenceSummary] = None
    human: Optional[HumanEvidenceMetadata] = None
    identity: Optional[TrackIdentityEvidence] = None

    def to_dict(self) -> Dict:
        """Flat dict for serialization; None values indicate not-yet-generated.

        Phase 2B.6 sections are included only when populated so the
        serialized shape stays stable for pre-2B.6 callers.
        """
        result: Dict = {
            "event_id": self.event_record.event_id,
            "entity_key": self.event_record.entity_key.as_string(),
            "class_name": self.event_record.class_name,
            "detection_confidence_stats": self.detection_confidence_stats.to_dict(),
            "detection_observation_count": self.detection_observation_count,
            "confirmation_video_time": self.confirmation_video_time,
            "confirmation_wall_time": self.confirmation_wall_time,
            "candidate_start_video_time": self.candidate_start_video_time,
            "candidate_end_video_time": self.candidate_end_video_time,
            "track_ids_in_window": sorted(self.track_ids_in_window),
            "confirmation_terms_met": list(self.confirmation_terms_met),
            "id_switch_count": self.id_switch_count,
            "annotated_frame_path": self.annotated_frame_path,
            "crop_path": self.crop_path,
            "person_object_pair_crop_path": self.person_object_pair_crop_path,
            "temporal_summary": self.temporal_summary,
            "human_verdict_line": self.human_verdict_line,
        }
        result["decision"] = self.decision.to_dict() if self.decision is not None else None
        result["association"] = self.association.to_dict() if self.association is not None else None
        result["human"] = self.human.to_dict() if self.human is not None else None
        result["identity"] = self.identity.to_dict() if self.identity is not None else None
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ModelSource",
    "EntityKey",
    "Observation",
    "BoundedObservationBuffer",
    "CandidateState",
    "ConfidenceStats",
    "Candidate",
    "EventRecord",
    "GateOutcome",
    "ConfirmationDecisionEvidence",
    "TrackIdentityEvidence",
    "AssociationEvidenceSummary",
    "HumanEvidenceMetadata",
    "EvidenceBundle",
]
