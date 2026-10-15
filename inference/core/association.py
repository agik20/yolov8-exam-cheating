"""Phase 2B.5 — Person↔object association data structures and geometry helpers.

This module implements the spatial and temporal association evidence described
in PHASE2A_EVENT_ARCHITECTURE.md §6 and the Phase 2B.5 brief.

What this module does:
- Define AssociationStatus enum: ASSOCIATED, AMBIGUOUS, UNASSIGNED, UNAVAILABLE.
- Define SpatialEvidence: center_inside, containment_ratio, normalized_center_distance,
  iou (diagnostic only — NOT an association gate).
- Define KeypointEvidence: available flag, nearest_hand_distance, confidence.
  Distinguishes "keypoints unavailable" from "keypoints available but not
  supporting association."
- Define TemporalEvidence: overlap flag, duration, observation_count.
- Define AssociationResult: per-pair evidence bundle, JSON-serializable.
  score stays None unless an explicit calibrated formula is supplied.
- Provide pure geometry helpers: center_inside, containment_ratio,
  normalized_center_distance, iou — all take plain (x1,y1,x2,y2) tuples,
  no model objects, no image data.

What this module intentionally does NOT do (Phase 2B.5 scope):
- No person-object assignment heuristic (that is PersonObjectAssociator,
  task 2B.5-2).
- No production thresholds hardcoded: all threshold parameters are
  injectable and default to None (gate unenforced).
- No image I/O, no model inference, no legacy path changes.
- No changes to process_detection or any legacy event path.

The legacy system in inference/core/pipeline.py remains the production
source of truth.  This module is observational only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# AssociationStatus
# ---------------------------------------------------------------------------

class AssociationStatus(Enum):
    """Per-pair association verdict.

    ASSOCIATED   — spatial evidence (and optionally keypoint evidence)
                   supports the object being held by / in the immediate
                   vicinity of *this* person, not another.
    AMBIGUOUS    — evidence is compatible with more than one person;
                   no forced single-pair assignment is made.
    UNASSIGNED   — no person-object pair has sufficient evidence.
    UNAVAILABLE  — keypoint data (or another required input) is not
                   available and the spatial evidence alone is insufficient
                   to make a confident call.
    """
    ASSOCIATED   = "ASSOCIATED"
    AMBIGUOUS    = "AMBIGUOUS"
    UNASSIGNED   = "UNASSIGNED"
    UNAVAILABLE  = "UNAVAILABLE"


# ---------------------------------------------------------------------------
# Geometry helpers — pure, no model dependencies
# ---------------------------------------------------------------------------

def _validate_bbox(bbox: Tuple[float, float, float, float]) -> None:
    x1, y1, x2, y2 = bbox
    if x2 < x1:
        raise ValueError(f"bbox x2 ({x2}) < x1 ({x1}): degenerate box")
    if y2 < y1:
        raise ValueError(f"bbox y2 ({y2}) < y1 ({y1}): degenerate box")


def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_inside(person_bbox: Tuple[float, float, float, float],
                  object_bbox: Tuple[float, float, float, float]) -> bool:
    """Return True when the object's center point falls inside the person bbox.

    This is approach #2 from PHASE2A §6 and is the recommended primary
    spatial signal.  A held object (phone, calculator) has its center well
    inside the person's crop region even when the person box is large.

    Does NOT use IoU — IoU is scale-mismatched for person/phone pairs and
    is explicitly rejected as an association gate in PHASE2A §6.
    """
    _validate_bbox(person_bbox)
    _validate_bbox(object_bbox)
    cx, cy = _center(object_bbox)
    px1, py1, px2, py2 = person_bbox
    return px1 <= cx <= px2 and py1 <= cy <= py2


def containment_ratio(person_bbox: Tuple[float, float, float, float],
                      object_bbox: Tuple[float, float, float, float]) -> float:
    """Return the fraction of the object bbox area that falls inside the person bbox.

    Approach #3 from PHASE2A §6.  1.0 = object fully inside person bbox.
    0.0 = no overlap.  Values > 1.0 are clamped (object larger than person
    bbox — rare but geometrically possible with a very small person box).

    The ratio is a continuous signal, useful for calibration; it is not
    itself a gate.  Callers pass a threshold to decide "associated" vs.
    "ambiguous" vs. "unassigned."
    """
    _validate_bbox(person_bbox)
    _validate_bbox(object_bbox)
    px1, py1, px2, py2 = person_bbox
    ox1, oy1, ox2, oy2 = object_bbox

    overlap_x1 = max(px1, ox1)
    overlap_y1 = max(py1, oy1)
    overlap_x2 = min(px2, ox2)
    overlap_y2 = min(py2, oy2)

    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return 0.0

    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    object_area = (ox2 - ox1) * (oy2 - oy1)
    if object_area <= 0:
        return 0.0

    ratio = overlap_area / object_area
    return min(ratio, 1.0)


def normalized_center_distance(
    person_bbox: Tuple[float, float, float, float],
    object_bbox: Tuple[float, float, float, float],
    reference_scale: float,
) -> float:
    """Euclidean distance between the two bboxes' centers, normalized.

    The distance is divided by reference_scale (a positive float, e.g. the
    object bbox width or a camera-calibration distance value) so the result
    is scale-invariant.  A value near 0.0 means the centers coincide.

    reference_scale must be > 0; a value of 0 or negative raises ValueError
    (division-by-zero guard).
    """
    if reference_scale is None or reference_scale <= 0:
        raise ValueError(f"reference_scale must be positive, got {reference_scale}")
    _validate_bbox(person_bbox)
    _validate_bbox(object_bbox)
    pcx, pcy = _center(person_bbox)
    ocx, ocy = _center(object_bbox)
    return math.sqrt((pcx - ocx) ** 2 + (pcy - ocy) ** 2) / reference_scale


def iou(
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of two axis-aligned bboxes.

    DIAGNOSTIC ONLY — not an association gate.  PHASE2A §6 explicitly
    rejects IoU as a primary signal because the person crop box is much
    larger than a held phone, making IoU near zero even for a correctly
    associated pair.  This helper is provided for logging and review,
    not for making the associated/ambiguous decision.
    """
    _validate_bbox(bbox_a)
    _validate_bbox(bbox_b)
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        inter = 0.0
    else:
        inter = (ix2 - ix1) * (iy2 - iy1)

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# SpatialEvidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpatialEvidence:
    """Measured spatial evidence for one person-object pair.

    All geometry values are computed from the bboxes only.  IoU is
    recorded for diagnostics but is NOT used as an association gate
    (PHASE2A §6 rejects IoU for this use case).

    The containment_ratio is the primary continuous spatial signal;
    center_inside is its binary projection.  normalized_center_distance
    complements them for cases where the object is near but not fully
    inside the person box.
    """
    center_inside: bool
    containment_ratio: float
    normalized_center_distance: Optional[float]  # None when reference_scale not supplied
    iou: float                                    # diagnostic only, not a gate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center_inside": self.center_inside,
            "containment_ratio": round(self.containment_ratio, 6),
            "normalized_center_distance": (
                round(self.normalized_center_distance, 6)
                if self.normalized_center_distance is not None else None
            ),
            "iou": round(self.iou, 6),
        }


# ---------------------------------------------------------------------------
# KeypointEvidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeypointEvidence:
    """Keypoint-based proximity signal for one person-object pair.

    available=True  — hand keypoints were present on the pose result and
                      at least one hand keypoint had confidence above
                      the visibility cutoff.  nearest_hand_distance is
                      the minimum Euclidean pixel distance from any
                      visible hand keypoint to the object bbox center,
                      normalized by reference_scale (same scale as
                      SpatialEvidence.normalized_center_distance).
                      confidence is the minimum visibility score among
                      the hand keypoints considered.

    available=False — keypoints were absent (pose model did not return
                      them for this frame) OR all hand keypoints fell
                      below the visibility cutoff.  nearest_hand_distance
                      and confidence are None.

    This distinction is critical: "keypoints unavailable" is not the
    same as "keypoints available but not supporting association."
    A reviewer must see which case applies.
    """
    available: bool
    nearest_hand_distance: Optional[float]   # normalized; None when unavailable
    confidence: Optional[float]              # min visibility among hand kps; None when unavailable
    hand_keypoint_count: int                  # how many hand kp slots were visible

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "nearest_hand_distance": (
                round(self.nearest_hand_distance, 6)
                if self.nearest_hand_distance is not None else None
            ),
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "hand_keypoint_count": self.hand_keypoint_count,
        }


# ---------------------------------------------------------------------------
# TemporalEvidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalEvidence:
    """Temporal consistency signal for one person-object pair.

    overlap          — whether the person and object tracks were both
                       active during at least one shared frame in the
                       candidate window.
    duration_seconds — total video-time (seconds) the pair co-occurred,
                       or 0.0 if no overlap.
    observation_count — number of frames in which BOTH the person and
                       object tracks were present in the window.

    These values are supporting signals only (PHASE2A §6 approach #6).
    They are recorded for the association result but do not drive the
    status decision on their own.
    """
    overlap: bool
    duration_seconds: float
    observation_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overlap": self.overlap,
            "duration_seconds": round(self.duration_seconds, 3),
            "observation_count": self.observation_count,
        }


# ---------------------------------------------------------------------------
# AssociationEvidence — bundled per-pair
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssociationEvidence:
    """All evidence layers for one person-object pair, ready for
    AssociationResult or serialization.

    person_track_id and object_track_id are raw int values within
    their respective model namespaces.  Use EntityKey when you need
    cross-namespace disambiguation (pose:3 ≠ object:3).
    """
    person_track_id: int
    object_track_id: int
    spatial: SpatialEvidence
    keypoint: KeypointEvidence
    temporal: Optional[TemporalEvidence]  # None when no temporal window is available

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_track_id": self.person_track_id,
            "object_track_id": self.object_track_id,
            "spatial": self.spatial.to_dict(),
            "keypoint": self.keypoint.to_dict(),
            "temporal": self.temporal.to_dict() if self.temporal is not None else None,
        }


# ---------------------------------------------------------------------------
# AssociationResult
# ---------------------------------------------------------------------------

@dataclass
class AssociationResult:
    """Verdict for a set of candidate person-object pairs in one frame
    or candidate window.

    status is the per-object verdict:
    - ASSOCIATED  : exactly one person pair has sufficient evidence.
    - AMBIGUOUS   : two or more person pairs have comparable evidence;
                    the object could belong to any of them.  No forced
                    single-pair assignment is made; all candidate pairs
                    are listed in evidence.
    - UNASSIGNED  : no person pair has sufficient evidence.
    - UNAVAILABLE : keypoint data is entirely absent and spatial evidence
                    alone is insufficient (only reached when the
                    association policy requires keypoint confirmation).

    score is intentionally Optional[float] = None.  A non-None score is
    only set when an explicitly supplied calibrated formula produces a
    value — the module never invents one.  Callers who want to assign
    a float score must compute it from the evidence fields and inject it.

    evidence lists every candidate pair evaluated, so an AMBIGUOUS
    result still carries the full picture for a human reviewer.
    """
    status: AssociationStatus
    # Populated only when a calibrated formula is explicitly supplied:
    score: Optional[float] = None
    # The object this result describes (track id in the object namespace):
    object_track_id: int = 0
    # All pairs evaluated for this object:
    evidence: List[AssociationEvidence] = field(default_factory=list)
    # Human-readable one-line summary (not data):
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "score": self.score,
            "object_track_id": self.object_track_id,
            "evidence": [e.to_dict() for e in self.evidence],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# PersonObjectAssociator
# ---------------------------------------------------------------------------

# COCO hand keypoint indices (17-keypoint format).  Override per project
# if a different skeleton is used.
HAND_KEYPOINT_INDICES: Tuple[int, ...] = (9, 10, 11, 12, 13, 14)


@dataclass(frozen=True)
class PersonKeypoints:
    """Per-person keypoint data, decoupled from ultralytics internals.

    Data layout:
    - x: list of x-coordinates in source-frame pixels, one per keypoint.
      Length must equal the number of keypoints in the skeleton
      (typically 17 for COCO).
    - y: list of y-coordinates, same length as x.
    - visibility: list of per-keypoint visibility scores (0.0–1.0).
      0.0 means the keypoint was not detected (occluded, out of frame, etc.).
      A keypoint with visibility below the visibility_cutoff is treated
      as unavailable for proximity measurement.

    All three lists must have the same length.  An empty PersonKeypoints
    (all three lists are empty) is valid and means "no keypoint data
    for this person in this frame."

    Pass x, y, and visibility as lists of floats; they are stored as
    tuples (immutable) on construction.
    """
    x: Tuple[float, ...]
    y: Tuple[float, ...]
    visibility: Tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.x)
        if len(self.y) != n:
            raise ValueError(
                f"PersonKeypoints: x has {n} elements, y has {len(self.y)} — lengths must match"
            )
        if len(self.visibility) != n:
            raise ValueError(
                f"PersonKeypoints: x has {n} elements, visibility has {len(self.visibility)} — lengths must match"
            )

    @property
    def num_keypoints(self) -> int:
        return len(self.x)

    def hand_distances_to_point(
        self,
        point_x: float,
        point_y: float,
        visibility_cutoff: float = 0.3,
        hand_indices: Tuple[int, ...] = HAND_KEYPOINT_INDICES,
    ) -> Tuple[Optional[float], Optional[float], int]:
        """Compute minimum Euclidean pixel distance from visible hand keypoints
        to (point_x, point_y).

        Returns (nearest_distance, min_visibility, count_visible).
        - nearest_distance is None when no hand keypoint is visible
          (visibility >= visibility_cutoff for at least one hand index).
        - min_visibility is the minimum visibility among the *visible*
          hand keypoints considered (0.0 when no hand kp is visible).
        - count_visible is how many hand keypoints were visible.

        Raises ValueError if any hand index is out of range for this
        keypoint set.
        """
        valid_indices = []
        for idx in hand_indices:
            if self.num_keypoints == 0:
                break
            if idx >= self.num_keypoints:
                raise ValueError(
                    f"hand keypoint index {idx} out of range "
                    f"(skeleton has {self.num_keypoints} keypoints)"
                )
            if self.visibility[idx] >= visibility_cutoff:
                valid_indices.append(idx)

        if not valid_indices:
            return None, None, 0

        min_dist = float('inf')
        min_vis = float('inf')
        for idx in valid_indices:
            dx = self.x[idx] - point_x
            dy = self.y[idx] - point_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                min_dist = dist
            if self.visibility[idx] < min_vis:
                min_vis = self.visibility[idx]

        return min_dist, min_vis, len(valid_indices)


# ---------------------------------------------------------------------------
# PersonObjectAssociator
# ---------------------------------------------------------------------------

@dataclass
class PersonObservation:
    """One person detected in the pose model output for this frame.

    bbox is in source-frame pixels (pre-crop-expansion, i.e. the raw
    detector box, NOT the display-expanded crop).  keypoints is optional
    and is None when the pose model returned no keypoint data for this
    person (e.g. the person is partially occluded or the model's pose
    branch produced no visible skeleton).

    track_id is the pose-model tracker ID — it lives in the pose
    namespace; use EntityKey.from_pose(track_id) when you need
    cross-namespace disambiguation.
    """
    track_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    keypoints: Optional[PersonKeypoints] = None

    @property
    def entity_key(self) -> "Any":
        from .events import EntityKey, ModelSource
        return EntityKey.from_pose(self.track_id)


@dataclass
class ObjectObservation:
    """One object detected in the object model output for this frame.

    bbox is in source-frame pixels.  track_id is the object-model
    tracker ID (object namespace).
    """
    track_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]

    @property
    def entity_key(self) -> "Any":
        from .events import EntityKey, ModelSource
        return EntityKey.from_object(self.track_id)


class PersonObjectAssociator:
    """Evaluates spatial and keypoint evidence for every person-object
    pair and returns one AssociationResult per object.

    Construction is pure: no I/O, no model inference, no image data.
    All threshold parameters are injectable and default to None, meaning
    that gate is not enforced.  When no threshold is set the associator
    still measures and reports all evidence — the status will be
    AMBIGUOUS (multiple candidates) or UNASSIGNED (no pair passes
    center_inside with containment_ratio > 0).  The caller is responsible
    for supplying calibrated thresholds to make an ASSOCIATED call.

    Parameters
    ----------
    containment_threshold : Optional[float]
        Minimum containment_ratio for a pair to be considered a
        "sufficient" spatial candidate.  0.0 means "any overlap."
        None (default) disables this gate — all pairs are reported.
    keypoint_distance_threshold : Optional[float]
        Maximum normalized hand-to-object-center distance (in the same
        units as reference_scale) for a pair to be considered
        keypoint-supported.  None disables the keypoint gate.
    visibility_cutoff : float
        Minimum keypoint visibility score for a hand keypoint to be
        counted.  Below this, the keypoint is treated as not visible.
        Default 0.3 — a conservative, project-tunable value; override
        per deployment.
    hand_keypoint_indices : Tuple[int, ...]
        Which skeleton indices are "hand" keypoints.  Default is the
        COCO hand set (9,10,11,12,13,14).
    reference_scale : Optional[float]
        Pixel scale used to normalize distances.  When None,
        normalized_center_distance and keypoint distances are reported
        in raw pixels (reference_scale=1.0 implicitly).  Supply the
        object bbox width, or a camera-calibration value, for
        scale-invariant results.

    All parameters are structural — no production values are hardcoded.
    """

    def __init__(
        self,
        containment_threshold: Optional[float] = None,
        keypoint_distance_threshold: Optional[float] = None,
        visibility_cutoff: float = 0.3,
        hand_keypoint_indices: Tuple[int, ...] = HAND_KEYPOINT_INDICES,
        reference_scale: Optional[float] = None,
    ) -> None:
        if containment_threshold is not None and not (0.0 <= containment_threshold <= 1.0):
            raise ValueError(
                f"containment_threshold must be in [0.0, 1.0] or None, got {containment_threshold}"
            )
        if keypoint_distance_threshold is not None and keypoint_distance_threshold < 0:
            raise ValueError(
                f"keypoint_distance_threshold must be non-negative or None, got {keypoint_distance_threshold}"
            )
        if not (0.0 <= visibility_cutoff <= 1.0):
            raise ValueError(
                f"visibility_cutoff must be in [0.0, 1.0], got {visibility_cutoff}"
            )
        self._containment_threshold = containment_threshold
        self._kp_distance_threshold = keypoint_distance_threshold
        self._visibility_cutoff = visibility_cutoff
        self._hand_indices = hand_keypoint_indices
        self._reference_scale = reference_scale if reference_scale is not None else 1.0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def associate(
        self,
        persons: List[PersonObservation],
        objects: List[ObjectObservation],
        temporal: Optional[Dict[int, TemporalEvidence]] = None,
    ) -> List[AssociationResult]:
        """Evaluate all person-object pairs and return one
        AssociationResult per object.

        Parameters
        ----------
        persons: pose-model person detections for this frame.
        objects: object-model detections for this frame.
        temporal: optional dict mapping object_track_id → TemporalEvidence
                  for the candidate window.  When None, the temporal
                  field in each AssociationEvidence is None.

        Returns a list of AssociationResult, one per object, in the
        same order as the objects list.

        Raises ValueError when persons or objects is not a list.
        """
        if not isinstance(persons, (list, tuple)):
            raise ValueError(f"persons must be a list, got {type(persons).__name__}")
        if not isinstance(objects, (list, tuple)):
            raise ValueError(f"objects must be a list, got {type(objects).__name__}")

        if not objects:
            return []

        results: List[AssociationResult] = []
        for obj in objects:
            results.append(self._associate_one_object(obj, persons, temporal))
        return results

    def _associate_one_object(
        self,
        obj: ObjectObservation,
        persons: List[PersonObservation],
        temporal: Optional[Dict[int, TemporalEvidence]],
    ) -> AssociationResult:
        if not persons:
            return AssociationResult(
                status=AssociationStatus.UNAVAILABLE,
                object_track_id=obj.track_id,
                summary=f"object {obj.track_id}: no person detections in frame",
            )

        temp = temporal.get(obj.track_id) if temporal else None
        evidence_list: List[AssociationEvidence] = []
        sufficient_pairs: List[int] = []  # indices into evidence_list

        for person in persons:
            # -- spatial evidence --
            c_inside = center_inside(person.bbox, obj.bbox)
            c_ratio = containment_ratio(person.bbox, obj.bbox)
            c_dist = normalized_center_distance(
                person.bbox, obj.bbox, self._reference_scale
            )
            c_iou = iou(person.bbox, obj.bbox)
            spatial = SpatialEvidence(
                center_inside=c_inside,
                containment_ratio=c_ratio,
                normalized_center_distance=c_dist,
                iou=c_iou,
            )

            # -- keypoint evidence --
            if person.keypoints is not None and person.keypoints.num_keypoints > 0:
                dist, conf, count = person.keypoints.hand_distances_to_point(
                    point_x=(obj.bbox[0] + obj.bbox[2]) / 2.0,
                    point_y=(obj.bbox[1] + obj.bbox[3]) / 2.0,
                    visibility_cutoff=self._visibility_cutoff,
                    hand_indices=self._hand_indices,
                )
                # Normalize distance by reference_scale to match spatial distance units
                norm_dist = dist / self._reference_scale if dist is not None else None
                keypoint = KeypointEvidence(
                    available=(dist is not None),
                    nearest_hand_distance=norm_dist,
                    confidence=conf,
                    hand_keypoint_count=count,
                )
            else:
                keypoint = KeypointEvidence(
                    available=False,
                    nearest_hand_distance=None,
                    confidence=None,
                    hand_keypoint_count=0,
                )

            ae = AssociationEvidence(
                person_track_id=person.track_id,
                object_track_id=obj.track_id,
                spatial=spatial,
                keypoint=keypoint,
                temporal=temp,
            )
            evidence_list.append(ae)

            # Determine if this pair is "sufficient" for ASSOCIATED
            if self._is_sufficient(ae):
                sufficient_pairs.append(len(evidence_list) - 1)

        # Decide status
        if len(sufficient_pairs) == 1:
            status = AssociationStatus.ASSOCIATED
            summary = (
                f"object {obj.track_id} ASSOCIATED with person "
                f"{evidence_list[sufficient_pairs[0]].person_track_id}"
            )
        elif len(sufficient_pairs) > 1:
            status = AssociationStatus.AMBIGUOUS
            ids = [evidence_list[i].person_track_id for i in sufficient_pairs]
            summary = (
                f"object {obj.track_id} AMBIGUOUS — could belong to any of persons {ids}"
            )
        elif not self._has_any_spatial_support(evidence_list):
            # No spatial support at all and no keypoint support
            status = AssociationStatus.UNAVAILABLE
            summary = (
                f"object {obj.track_id} UNAVAILABLE — no spatial or keypoint support "
                f"across {len(evidence_list)} person(s)"
            )
        else:
            status = AssociationStatus.UNASSIGNED
            summary = (
                f"object {obj.track_id} UNASSIGNED — "
                f"{len(evidence_list)} person(s) evaluated, none meet threshold"
            )

        return AssociationResult(
            status=status,
            object_track_id=obj.track_id,
            evidence=evidence_list,
            summary=summary,
        )

    def _is_sufficient(self, ae: AssociationEvidence) -> bool:
        """Return True when a pair has sufficient evidence to be ASSOCIATED.

        A pair is sufficient when BOTH:
        1. Spatial gate passes:
           - if containment_threshold is set, containment_ratio >= threshold
             AND center_inside is True
           - if containment_threshold is None, center_inside is True
        2. Keypoint gate passes (only when keypoint_distance_threshold is set):
           - keypoint is available AND
             nearest_hand_distance is not None AND
             nearest_hand_distance <= keypoint_distance_threshold

        When keypoint_distance_threshold is None, the keypoint gate is
        unenforced and the spatial gate alone determines sufficiency.
        """
        spatial_ok = False
        if self._containment_threshold is not None:
            spatial_ok = ae.spatial.center_inside and ae.spatial.containment_ratio >= self._containment_threshold
        else:
            spatial_ok = ae.spatial.center_inside

        if not spatial_ok:
            return False

        if self._kp_distance_threshold is not None:
            if not ae.keypoint.available:
                return False
            if ae.keypoint.nearest_hand_distance is None:
                return False
            if ae.keypoint.nearest_hand_distance > self._kp_distance_threshold:
                return False

        return True

    def _has_any_spatial_support(self, evidence_list: List[AssociationEvidence]) -> bool:
        """True when at least one pair has containment_ratio > 0
        (object bbox overlaps person bbox at all), regardless of thresholds.
        Used to distinguish UNASSIGNED (spatial support exists but
        below threshold) from UNAVAILABLE (no spatial support whatsoever).
        """
        return any(ee.spatial.containment_ratio > 0.0 for ee in evidence_list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AssociationStatus",
    "SpatialEvidence",
    "KeypointEvidence",
    "TemporalEvidence",
    "AssociationEvidence",
    "AssociationResult",
    "PersonKeypoints",
    "PersonObservation",
    "ObjectObservation",
    "PersonObjectAssociator",
    "HAND_KEYPOINT_INDICES",
    # geometry helpers
    "center_inside",
    "containment_ratio",
    "normalized_center_distance",
    "iou",
]
