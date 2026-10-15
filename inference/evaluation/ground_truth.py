"""Phase 2C.1 — Ground-truth dataset & annotation schema.

Model-independent ground-truth structures for exam-cheating evaluation.
This module defines the annotation schema, dataset manifest, and validation
utilities required by Phase 2C.2/2C.3 (matching + calibration).

What this module does NOT do:
- No inference pipeline changes
- No model threshold calibration
- No prediction-vs-ground-truth matching algorithm
- No numeric ground-truth values (the dataset must be annotated by humans)
"""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple

# -- Known model class names (from inference/config/constants.py) --
# These are the ONLY valid behavior_class values in ground-truth incidents.
# They are copied here for validation, NOT as a source of truth for inference.
VALID_POSE_CLASSES: Tuple[str, ...] = (
    "no-cheating",
    "provide-object",
    "see-friends-work",
)
VALID_OBJECT_CLASSES: Tuple[str, ...] = (
    "smartphone",
    "calculator",
)
VALID_BEHAVIOR_CLASSES: Tuple[str, ...] = (
    *VALID_POSE_CLASSES,
    *VALID_OBJECT_CLASSES,
)

__all__ = [
    "GTLabel",
    "EntityReferenceStatus",
    "SceneTag",
    "ReviewStatus",
    "AnnotationProvenance",
    "GroundTruthIncident",
    "NormalPeriod",
    "AmbiguousPeriod",
    "SourceVideoMetadata",
    "DatasetManifest",
    "GroundTruthDataset",
    "validate_incident",
    "validate_dataset",
    "find_overlapping_incidents",
    "VALID_POSE_CLASSES",
    "VALID_OBJECT_CLASSES",
    "VALID_BEHAVIOR_CLASSES",
]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GTLabel(Enum):
    """Ground-truth label for an incident interval.

    NORMAL     — no ground-truth suspicious incident present.
    SUSPICIOUS — a human annotator considers the behavior a suspicious incident.
    AMBIGUOUS  — available video evidence is insufficient for a reliable
                 human decision.  AMBIGUOUS must NOT be silently coerced to
                 NORMAL or SUSPICIOUS by any consumer of this schema.
    """
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    AMBIGUOUS = "AMBIGUOUS"


class EntityReferenceStatus(Enum):
    """Status of a person or object identity reference in a ground-truth annotation.

    KNOWN            — the annotator can reliably identify the entity.
    UNKNOWN          — the entity exists but its identity is not reliably
                       determinable from the video.
    NOT_APPLICABLE   — no entity of this type is relevant to this annotation.
    AMBIGUOUS        — the entity's identity is genuinely uncertain and the
                       annotator cannot resolve it from the available footage.
    """
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    AMBIGUOUS = "AMBIGUOUS"


class SceneTag(Enum):
    """Optional scene/context difficulty tags for a source video.

    These are dataset metadata, NOT model predictions.  A tag describes
    conditions present in the footage; it does not claim that the model
    must fail under those conditions.
    """
    NORMAL_EXAM       = "normal_exam"
    MULTI_PERSON      = "multi_person"
    OCCLUSION         = "occlusion"
    LIGHTING_CHANGE   = "lighting_change"
    BLUR              = "blur"
    CROWDED           = "crowded"
    OBJECT_OVERLAP    = "object_overlap"
    PARTIAL_PERSON    = "partial_person"
    CAMERA_MOTION     = "camera_motion"


class ReviewStatus(Enum):
    """Annotation review status for provenance tracking.

    DRAFT        — initial annotation, not yet reviewed.
    PENDING      — submitted for review.
    APPROVED     — review complete, annotation accepted.
    REJECTED     — review complete, annotation requires correction.
    ADJUDICATED  — multi-annotator disagreement resolved by adjudication.
    """
    DRAFT       = "DRAFT"
    PENDING     = "PENDING"
    APPROVED    = "APPROVED"
    REJECTED    = "REJECTED"
    ADJUDICATED = "ADJUDICATED"


# ---------------------------------------------------------------------------
# Validation helpers (internal)
# ---------------------------------------------------------------------------


def _is_finite_number(value: float) -> bool:
    """Return True if value is a finite real number (no NaN / inf)."""
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _is_valid_video_id(video_id: str) -> bool:
    """A video ID must be a non-empty string with no filesystem path chars."""
    return (
        isinstance(video_id, str)
        and len(video_id) > 0
        and "/" not in video_id
        and "\\" not in video_id
    )


def _is_valid_incident_id(incident_id: str) -> bool:
    """An incident ID must be a non-empty string."""
    return isinstance(incident_id, str) and len(incident_id) > 0


def _validate_interval(
    start: float,
    end: float,
    incident_id: str = "<unknown>",
) -> List[str]:
    """Return a list of human-readable validation errors for an interval.

    Zero-duration intervals (start == end) are accepted and documented as
    a valid edge case (a point-in-time annotation).  Negative or NaN/inf
    values are rejected.
    """
    errors: List[str] = []
    if not _is_finite_number(start):
        errors.append(f"{incident_id}: start_video_time must be finite, got {start!r}")
    if not _is_finite_number(end):
        errors.append(f"{incident_id}: end_video_time must be finite, got {end!r}")
    if _is_finite_number(start) and start < 0:
        errors.append(f"{incident_id}: start_video_time must be >= 0, got {start}")
    if _is_finite_number(end) and end < 0:
        errors.append(f"{incident_id}: end_video_time must be >= 0, got {end}")
    if _is_finite_number(start) and _is_finite_number(end) and start > end:
        errors.append(
            f"{incident_id}: start_video_time ({start}) must be <= "
            f"end_video_time ({end})"
        )
    return errors


def _validate_frame_time_consistency(
    start_frame: Optional[int],
    end_frame: Optional[int],
    start_time: float,
    end_time: float,
    fps: Optional[float],
    incident_id: str = "<unknown>",
) -> List[str]:
    """Check that frame indices and video times are internally consistent.

    If both frame and time are provided and fps is known, the time must equal
    frame / fps within a small tolerance (one frame).  If fps is unknown,
    only check that frames are non-negative and start <= end.
    """
    errors: List[str] = []
    if start_frame is not None and start_frame < 0:
        errors.append(f"{incident_id}: start_frame must be >= 0, got {start_frame}")
    if end_frame is not None and end_frame < 0:
        errors.append(f"{incident_id}: end_frame must be >= 0, got {end_frame}")
    if start_frame is not None and end_frame is not None and start_frame > end_frame:
        errors.append(
            f"{incident_id}: start_frame ({start_frame}) must be <= "
            f"end_frame ({end_frame})"
        )
    if (
        fps is not None
        and _is_finite_number(fps)
        and fps > 0
    ):
        if start_frame is not None and _is_finite_number(start_time):
            expected = start_frame / fps
            if abs(start_time - expected) > 1.0 / fps + 1e-9:
                errors.append(
                    f"{incident_id}: start_frame {start_frame} at fps {fps} "
                    f"implies video_time {expected:.6f}, got {start_time:.6f}"
                )
        if end_frame is not None and _is_finite_number(end_time):
            expected = end_frame / fps
            if abs(end_time - expected) > 1.0 / fps + 1e-9:
                errors.append(
                    f"{incident_id}: end_frame {end_frame} at fps {fps} "
                    f"implies video_time {expected:.6f}, got {end_time:.6f}"
                )
    return errors


# ---------------------------------------------------------------------------
# Annotation provenance
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class AnnotationProvenance:
    """Provenance metadata for a ground-truth annotation.

    Supports double-annotation and review workflows without requiring a
    database.  ``annotation_id`` is a stable, human-readable identifier
    (e.g. "ann_sk2_0001") that lets multiple annotators reference the
    same interval.  ``schema_version`` ties the record to a known schema
    generation so future schema evolution does not silently break older
    annotations.

    All fields are Optional; a minimal annotation may carry only
    ``schema_version``.  No personally-identifying information is stored.
    """
    schema_version: str = SCHEMA_VERSION
    annotator_id: Optional[str] = None       # anonymous or pseudonymous ID
    annotation_id: Optional[str] = None       # stable interval reference
    created_at: Optional[str] = None          # ISO-8601 UTC timestamp
    review_status: Optional[ReviewStatus] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        result: Dict = {
            "schema_version": self.schema_version,
            "annotator_id": self.annotator_id,
            "annotation_id": self.annotation_id,
            "created_at": self.created_at,
            "notes": self.notes,
        }
        result["review_status"] = (
            self.review_status.value if self.review_status is not None else None
        )
        return result


# ---------------------------------------------------------------------------
# Ground-truth incident (the primary annotation unit)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruthIncident:
    """A human-annotated behavioral incident interval.

    This is the fundamental annotation unit: one incident is a single
    suspicious (or explicitly ambiguous) behavior episode observed in a
    source video.  It is NOT a model observation, NOT a track ID, and NOT
    derived from any inference output.

    Fields:
        video_id           — stable portable identifier for the source video.
        incident_id        — unique identifier for this annotation record.
        label              — GTLabel.NORMAL, SUSPICIOUS, or AMBIGUOUS.
        start_video_time   — source video seconds (frame_index / src_fps).
        end_video_time     — source video seconds; must be >= start.
        behavior_class     — the annotated behavior class name.  Must be a
                              value from VALID_BEHAVIOR_CLASSES.  For NORMAL
                              incidents this may be "no-cheating".  For
                              SUSPICIOUS/AMBIGUOUS incidents it should name
                              the observed behavior (e.g. "provide-object",
                              "see-friends-work", "smartphone").
        person_ref_status  — EntityReferenceStatus for the person involved.
        object_ref_status  — EntityReferenceStatus for the object involved.
        person_reference   — optional free-text identifier (e.g. "student_3",
                              "person_at_seat_B2").  NOT a tracker track ID.
        object_reference   — optional free-text identifier (e.g. "phone_blue").
                              NOT a tracker track ID.
        start_frame        — optional source frame index at incident start.
        end_frame          — optional source frame index at incident end.
        provenance         — AnnotationProvenance record.
        notes              — free-text annotation notes (difficult-negative
                              description, ambiguity reason, etc.).

    Validation: call validate_incident() before adding to a dataset.
    Zero-duration incidents (start == end) are a valid edge case and are
    documented as such; they represent a point-in-time annotation.
    """
    video_id: str
    incident_id: str
    label: GTLabel
    start_video_time: float
    end_video_time: float
    behavior_class: str
    person_ref_status: EntityReferenceStatus = EntityReferenceStatus.UNKNOWN
    object_ref_status: EntityReferenceStatus = EntityReferenceStatus.NOT_APPLICABLE
    person_reference: Optional[str] = None
    object_reference: Optional[str] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    provenance: AnnotationProvenance = field(default_factory=AnnotationProvenance)
    notes: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        return self.end_video_time - self.start_video_time

    def to_dict(self) -> Dict:
        return {
            "video_id": self.video_id,
            "incident_id": self.incident_id,
            "label": self.label.value,
            "start_video_time": self.start_video_time,
            "end_video_time": self.end_video_time,
            "duration_seconds": self.duration_seconds,
            "behavior_class": self.behavior_class,
            "person_ref_status": self.person_ref_status.value,
            "object_ref_status": self.object_ref_status.value,
            "person_reference": self.person_reference,
            "object_reference": self.object_reference,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "provenance": self.provenance.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "GroundTruthIncident":
        return cls(
            video_id=d["video_id"],
            incident_id=d["incident_id"],
            label=GTLabel(d["label"]),
            start_video_time=float(d["start_video_time"]),
            end_video_time=float(d["end_video_time"]),
            behavior_class=d["behavior_class"],
            person_ref_status=EntityReferenceStatus(d.get("person_ref_status", "UNKNOWN")),
            object_ref_status=EntityReferenceStatus(d.get("object_ref_status", "NOT_APPLICABLE")),
            person_reference=d.get("person_reference"),
            object_reference=d.get("object_reference"),
            start_frame=d.get("start_frame"),
            end_frame=d.get("end_frame"),
            provenance=AnnotationProvenance(
                schema_version=d.get("provenance", {}).get("schema_version", SCHEMA_VERSION),
                annotator_id=d.get("provenance", {}).get("annotator_id"),
                annotation_id=d.get("provenance", {}).get("annotation_id"),
                created_at=d.get("provenance", {}).get("created_at"),
                review_status=(
                    ReviewStatus(d["provenance"]["review_status"])
                    if d.get("provenance", {}).get("review_status")
                    else None
                ),
                notes=d.get("provenance", {}).get("notes"),
            ),
            notes=d.get("notes"),
        )


# ---------------------------------------------------------------------------
# Normal / Ambiguous period annotations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalPeriod:
    """An explicitly annotated normal (non-suspicious) period in a video.

    Representing normal periods explicitly is required so that the
    evaluation dataset can distinguish "annotator confirmed this period is
    normal" from "annotator did not annotate this period."  A NORMAL label
    means the annotator reviewed the footage and found no suspicious
    behavior in the interval.
    """
    video_id: str
    period_id: str
    start_video_time: float
    end_video_time: float
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    provenance: AnnotationProvenance = field(default_factory=AnnotationProvenance)
    notes: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        return self.end_video_time - self.start_video_time

    def to_dict(self) -> Dict:
        return {
            "video_id": self.video_id,
            "period_id": self.period_id,
            "start_video_time": self.start_video_time,
            "end_video_time": self.end_video_time,
            "duration_seconds": self.duration_seconds,
            "provenance": self.provenance.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "NormalPeriod":
        return cls(
            video_id=d["video_id"],
            period_id=d["period_id"],
            start_video_time=float(d["start_video_time"]),
            end_video_time=float(d["end_video_time"]),
            start_frame=d.get("start_frame"),
            end_frame=d.get("end_frame"),
            notes=d.get("notes"),
            provenance=AnnotationProvenance(
                schema_version=d.get("provenance", {}).get("schema_version", SCHEMA_VERSION),
            ),
        )


@dataclass(frozen=True)
class AmbiguousPeriod:
    """An explicitly annotated ambiguous period in a video.

    AMBIGUOUS means the available video evidence is insufficient for a
    reliable human decision.  An ambiguous period is NOT a normal period
    and NOT a suspicious incident — it is a distinct annotation class that
    must remain identifiable in the dataset.  Consumers must not silently
    convert AMBIGUOUS to NORMAL or SUSPICIOUS.
    """
    video_id: str
    period_id: str
    start_video_time: float
    end_video_time: float
    reason: str                       # why the annotator could not decide
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    provenance: AnnotationProvenance = field(default_factory=AnnotationProvenance)
    notes: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        return self.end_video_time - self.start_video_time

    def to_dict(self) -> Dict:
        return {
            "video_id": self.video_id,
            "period_id": self.period_id,
            "start_video_time": self.start_video_time,
            "end_video_time": self.end_video_time,
            "duration_seconds": self.duration_seconds,
            "reason": self.reason,
            "provenance": self.provenance.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "AmbiguousPeriod":
        return cls(
            video_id=d["video_id"],
            period_id=d["period_id"],
            start_video_time=float(d["start_video_time"]),
            end_video_time=float(d["end_video_time"]),
            reason=d["reason"],
            start_frame=d.get("start_frame"),
            end_frame=d.get("end_frame"),
            notes=d.get("notes"),
            provenance=AnnotationProvenance(
                schema_version=d.get("provenance", {}).get("schema_version", SCHEMA_VERSION),
            ),
        )


# ---------------------------------------------------------------------------
# Source video metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceVideoMetadata:
    """Metadata for a source video in the ground-truth dataset.

    ``reference`` is a stable, portable identifier (e.g. "sk2" or
    "exam_room_A_2026-01-01") — NOT an absolute filesystem path.  This
    allows the dataset to be moved between machines without breaking
    annotation references.

    ``duration_seconds`` and ``source_fps`` are required for validating
    incident intervals.  ``frame_count`` is optional and used only when
    available.  ``scene_tags`` are optional difficulty/context labels.
    """
    video_id: str
    reference: str
    duration_seconds: Optional[float] = None
    source_fps: Optional[float] = None
    resolution: Optional[Tuple[int, int]] = None   # (width, height)
    frame_count: Optional[int] = None
    scene_tags: Tuple[SceneTag, ...] = ()
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not _is_valid_video_id(self.video_id):
            raise ValueError(
                f"video_id must be a non-empty string with no path separators, got {self.video_id!r}"
            )
        if not self.reference:
            raise ValueError(
                f"reference must be a non-empty string, got {self.reference!r}"
            )
        if self.source_fps is not None:
            if not _is_finite_number(self.source_fps) or self.source_fps <= 0:
                raise ValueError(
                    f"source_fps must be positive and finite, got {self.source_fps!r}"
                )
        if self.duration_seconds is not None:
            if not _is_finite_number(self.duration_seconds) or self.duration_seconds < 0:
                raise ValueError(
                    f"duration_seconds must be non-negative and finite, got {self.duration_seconds!r}"
                )
        if self.frame_count is not None and self.frame_count < 0:
            raise ValueError(
                f"frame_count must be non-negative, got {self.frame_count!r}"
            )

    def to_dict(self) -> Dict:
        result: Dict = {
            "video_id": self.video_id,
            "reference": self.reference,
            "duration_seconds": self.duration_seconds,
            "source_fps": self.source_fps,
            "frame_count": self.frame_count,
            "scene_tags": [t.value for t in self.scene_tags],
            "notes": self.notes,
        }
        if self.resolution is not None:
            result["resolution"] = list(self.resolution)
        else:
            result["resolution"] = None
        return result

    @classmethod
    def from_dict(cls, d: Dict) -> "SourceVideoMetadata":
        res = d.get("resolution")
        return cls(
            video_id=d["video_id"],
            reference=d["reference"],
            duration_seconds=d.get("duration_seconds"),
            source_fps=d.get("source_fps"),
            resolution=tuple(res) if res else None,
            frame_count=d.get("frame_count"),
            scene_tags=tuple(SceneTag(t) for t in d.get("scene_tags", [])),
            notes=d.get("notes"),
        )


# ---------------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------------


@dataclass
class DatasetManifest:
    """Top-level manifest for a ground-truth evaluation dataset.

    A manifest describes which source videos are included, their
    metadata, and the annotations (incidents + normal + ambiguous periods)
    associated with each video.  It is the entry point for loading and
    validating a complete dataset.

    ``version`` is the manifest schema version string, NOT a dataset
    content version.  ``created_at`` is an ISO-8601 UTC timestamp.
    """
    version: str
    created_at: str
    videos: List[SourceVideoMetadata] = field(default_factory=list)
    incidents: List[GroundTruthIncident] = field(default_factory=list)
    normal_periods: List[NormalPeriod] = field(default_factory=list)
    ambiguous_periods: List[AmbiguousPeriod] = field(default_factory=list)
    notes: Optional[str] = None

    # -- helpers --

    def videos_by_id(self) -> Dict[str, SourceVideoMetadata]:
        return {v.video_id: v for v in self.videos}

    def incidents_for_video(self, video_id: str) -> List[GroundTruthIncident]:
        return [i for i in self.incidents if i.video_id == video_id]

    def add_video(self, video: SourceVideoMetadata) -> None:
        self.videos.append(video)

    def add_incident(self, incident: GroundTruthIncident) -> None:
        self.incidents.append(incident)

    def add_normal_period(self, period: NormalPeriod) -> None:
        self.normal_periods.append(period)

    def add_ambiguous_period(self, period: AmbiguousPeriod) -> None:
        self.ambiguous_periods.append(period)

    # -- serialization --

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "videos": [v.to_dict() for v in self.videos],
            "incidents": [i.to_dict() for i in self.incidents],
            "normal_periods": [p.to_dict() for p in self.normal_periods],
            "ambiguous_periods": [p.to_dict() for p in self.ambiguous_periods],
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict) -> "DatasetManifest":
        return cls(
            version=d["version"],
            created_at=d["created_at"],
            videos=[SourceVideoMetadata.from_dict(v) for v in d.get("videos", [])],
            incidents=[GroundTruthIncident.from_dict(i) for i in d.get("incidents", [])],
            normal_periods=[NormalPeriod.from_dict(p) for p in d.get("normal_periods", [])],
            ambiguous_periods=[AmbiguousPeriod.from_dict(p) for p in d.get("ambiguous_periods", [])],
            notes=d.get("notes"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "DatasetManifest":
        return cls.from_dict(json.loads(raw))

    @classmethod
    def from_file(cls, path: str) -> "DatasetManifest":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_json(fh.read())


# ---------------------------------------------------------------------------
# Aggregate dataset (in-memory collection of one or more manifests)
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthDataset:
    """In-memory collection of ground-truth annotations.

    Wraps one or more DatasetManifest objects for convenient validation
    and lookup.  Use ``validate_dataset()`` before running any evaluation
    to catch malformed annotations early.
    """
    manifests: List[DatasetManifest] = field(default_factory=list)

    def all_incidents(self) -> List[GroundTruthIncident]:
        out: List[GroundTruthIncident] = []
        for m in self.manifests:
            out.extend(m.incidents)
        return out

    def all_videos(self) -> List[SourceVideoMetadata]:
        out: List[SourceVideoMetadata] = []
        for m in self.manifests:
            out.extend(m.videos)
        return out

    def video_ids(self) -> FrozenSet[str]:
        return frozenset(v.video_id for m in self.manifests for v in m.videos)

    def incident_ids(self) -> FrozenSet[str]:
        return frozenset(i.incident_id for m in self.manifests for i in m.incidents)


# ---------------------------------------------------------------------------
# Public validation utilities
# ---------------------------------------------------------------------------


def validate_incident(
    incident: GroundTruthIncident,
    video_meta: Optional[SourceVideoMetadata] = None,
) -> List[str]:
    """Validate a single GroundTruthIncident.

    Returns a list of actionable error strings.  An empty list means the
    incident passed all checks.  No errors are raised; all issues are
    reported so a caller can address them all at once.

    Checks performed:
      - video_id and incident_id are non-empty
      - start_video_time <= end_video_time, both finite and non-negative
      - behavior_class is in VALID_BEHAVIOR_CLASSES
      - label is a valid GTLabel (always true if constructed properly)
      - frame/time consistency when both frames and fps are available
      - if video_meta is supplied: incident interval is within video duration
    """
    errors: List[str] = []
    iid = incident.incident_id or "<empty incident_id>"

    if not _is_valid_incident_id(incident.incident_id):
        errors.append(f"{iid}: incident_id must be a non-empty string")
    if not _is_valid_video_id(incident.video_id):
        errors.append(f"{iid}: video_id must be a non-empty string with no path separators")

    errors.extend(_validate_interval(
        incident.start_video_time,
        incident.end_video_time,
        incident_id=iid,
    ))

    if incident.behavior_class not in VALID_BEHAVIOR_CLASSES:
        errors.append(
            f"{iid}: behavior_class {incident.behavior_class!r} is not in "
            f"VALID_BEHAVIOR_CLASSES {VALID_BEHAVIOR_CLASSES}"
        )

    # Frame/time consistency when fps is known
    fps = video_meta.source_fps if video_meta is not None else None
    errors.extend(_validate_frame_time_consistency(
        incident.start_frame,
        incident.end_frame,
        incident.start_video_time,
        incident.end_video_time,
        fps,
        incident_id=iid,
    ))

    # Duration boundary check when video duration is known
    if video_meta is not None and video_meta.duration_seconds is not None:
        if _is_finite_number(incident.end_video_time):
            if incident.end_video_time > video_meta.duration_seconds:
                errors.append(
                    f"{iid}: end_video_time ({incident.end_video_time}) exceeds "
                    f"video duration ({video_meta.duration_seconds}) for video "
                    f"{incident.video_id!r}"
                )

    return errors


def validate_dataset(dataset: GroundTruthDataset) -> List[str]:
    """Validate a complete GroundTruthDataset.

    Returns a list of actionable diagnostic strings.  An empty list means
    the dataset passed all checks.  No exceptions are raised; all issues
    are collected and returned together.

    Checks performed:
      - duplicate incident IDs across all manifests
      - duplicate video IDs across all manifests
      - every incident's video_id has a matching SourceVideoMetadata entry
      - every incident passes validate_incident()
      - every normal/ambiguous period has valid interval bounds
      - dataset is not empty (at least one video)
    """
    issues: List[str] = []
    all_videos = dataset.all_videos()
    video_ids_seen: set = set()
    incident_ids_seen: set = set()

    # Duplicate video IDs
    for v in all_videos:
        if v.video_id in video_ids_seen:
            issues.append(f"DUPLICATE video_id: {v.video_id!r}")
        video_ids_seen.add(v.video_id)

    # Duplicate incident IDs
    for inc in dataset.all_incidents():
        if inc.incident_id in incident_ids_seen:
            issues.append(f"DUPLICATE incident_id: {inc.incident_id!r}")
        incident_ids_seen.add(inc.incident_id)

    # Build video lookup
    video_map: Dict[str, SourceVideoMetadata] = {v.video_id: v for v in all_videos}

    # Per-incident validation
    for inc in dataset.all_incidents():
        if inc.video_id not in video_map:
            issues.append(
                f"MISSING VIDEO: incident {inc.incident_id!r} references "
                f"video_id {inc.video_id!r} which has no SourceVideoMetadata entry"
            )
        errors = validate_incident(inc, video_map.get(inc.video_id))
        issues.extend(errors)

    # Normal period interval checks
    for m in dataset.manifests:
        for p in m.normal_periods:
            p_errs = _validate_interval(
                p.start_video_time, p.end_video_time,
                incident_id=f"normal_period {p.period_id!r}",
            )
            issues.extend(p_errs)

        # Ambiguous period interval checks
        for p in m.ambiguous_periods:
            p_errs = _validate_interval(
                p.start_video_time, p.end_video_time,
                incident_id=f"ambiguous_period {p.period_id!r}",
            )
            issues.extend(p_errs)

    if not all_videos:
        issues.append("EMPTY DATASET: no source videos registered")

    return issues


def find_overlapping_incidents(
    incidents: List[GroundTruthIncident],
) -> List[Tuple[GroundTruthIncident, GroundTruthIncident]]:
    """Return all pairs of incidents that overlap in video time.

    Two incidents overlap if their [start, end] intervals intersect.
    Overlapping incidents are valid when they involve different persons
    or different behaviors; this function reports overlaps for human
    review, it does NOT merge or reject them automatically.

    Only pairs within the same video_id are compared.
    """
    by_video: Dict[str, List[GroundTruthIncident]] = {}
    for inc in incidents:
        by_video.setdefault(inc.video_id, []).append(inc)

    overlaps: List[Tuple[GroundTruthIncident, GroundTruthIncident]] = []
    for video_id, incs in by_video.items():
        n = len(incs)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = incs[i], incs[j]
                # Intervals [a.start, a.end] and [b.start, b.end] overlap
                # if a.start < b.end and b.start < a.end
                if a.start_video_time < b.end_video_time and b.start_video_time < a.end_video_time:
                    overlaps.append((a, b))
    return overlaps

