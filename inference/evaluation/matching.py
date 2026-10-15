"""Phase 2C.2 — Deterministic prediction-vs-ground-truth matching.

This module implements the offline evaluation matching layer described in
PHASE2C2_MATCHING.md.  It is strictly an evaluation tool: it never modifies
production inference behavior, never invents a calibrated threshold, and
never writes to production output paths.

What this module does:
- Define the evaluation-only data structures: NormalizedPrediction,
  TemporalRelationship, MatchResult, EvaluationSummary, MatchingPolicy.
- Provide PredictionNormalizer: an explicit adapter from EventRecord to
  NormalizedPrediction, marking unavailable fields as None rather than
  fabricating values.
- Implement compute_temporal_relationship: deterministic interval
  intersection metrics (overlap duration, ratios relative to prediction and
  to ground truth).
- Implement compute_class_match_status: CLASS_MATCH / CLASS_MISMATCH /
  CLASS_UNAVAILABLE / CLASS_AMBIGUOUS.
- Implement match_predictions_to_ground_truth: deterministic one-to-one
  greedy assignment with a documented lexicographic priority, duplicate
  detection, FN/FP/AMBIGUOUS outcome classification.
- Implement compute_evaluation_summary: precision, recall, F1, duplicate
  rate with documented denominators and None (not 0.0) for undefined
  values.
- Implement compute_confirmation_latency: video-time latency, None when
  the confirmation timestamp is unavailable.

What this module intentionally does NOT do:
- No threshold calibration (that is Phase 2C.3+).
- No changes to EventRecord, Candidate, EvidenceBundle, or any production
  data structure.
- No use of wall-clock time.
- No filesystem I/O.
- No model imports beyond inference.core.events (EventRecord, EntityKey,
  ModelSource, ConfidenceStats) and inference.evaluation.ground_truth
  (GroundTruthIncident, NormalPeriod, AmbiguousPeriod, GTLabel,
  EntityReferenceStatus).

Matching policy is an evaluation methodology and is not a production
detection threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PredictionBoundary(Enum):
    """How the prediction's temporal extent is represented.

    EXACT_INTERVAL — both start and end video times are known.
    POINT_EVENT    — only the confirmation (or single) video time is known;
                     start_video_time is None and end_video_time holds the
                     point time.
    UNAVAILABLE    — no usable temporal information; both times are None.
    """
    EXACT_INTERVAL = "exact_interval"
    POINT_EVENT    = "point_event"
    UNAVAILABLE    = "unavailable"


class EvaluationOutcome(Enum):
    """Final evaluation category for one prediction–GT relationship.

    TP      — prediction matched a GT incident (primary match).
    FP      — prediction matched no eligible GT incident; it is a
              false positive.
    FN      — GT incident had no eligible prediction; it is a
              false negative.
    DUPLICATE — a second (or later) prediction matched the same GT
              incident that was already consumed by a primary TP.
    AMBIGUOUS_GROUND_TRUTH — GT incident is labeled AMBIGUOUS and the
              policy excludes it from TP/FN accounting.
    UNMATCHED — prediction/GT pair that did not meet the temporal
              threshold under the current policy.
    """
    TP                       = "TP"
    FP                       = "FP"
    FN                       = "FN"
    DUPLICATE                = "DUPLICATE"
    AMBIGUOUS_GROUND_TRUTH   = "AMBIGUOUS_GROUND_TRUTH"
    UNMATCHED                = "UNMATCHED"


class ClassMatchStatus(Enum):
    """Outcome of comparing prediction class to ground-truth behavior class."""
    CLASS_MATCH       = "CLASS_MATCH"
    CLASS_MISMATCH    = "CLASS_MISMATCH"
    CLASS_UNAVAILABLE = "CLASS_UNAVAILABLE"   # prediction class is None
    CLASS_AMBIGUOUS   = "CLASS_AMBIGUOUS"     # GT label is AMBIGUOUS


class IdentityMatchStatus(Enum):
    """Outcome of comparing prediction identity to ground-truth person reference.

    IDENTITY_MATCH      — both sides have a non-None identity/reference and
                          they are equal.
    IDENTITY_MISMATCH   — both sides have a non-None value and they differ.
    IDENTITY_UNAVAILABLE— either side has no usable identity value.
    """
    IDENTITY_MATCH       = "IDENTITY_MATCH"
    IDENTITY_MISMATCH    = "IDENTITY_MISMATCH"
    IDENTITY_UNAVAILABLE = "IDENTITY_UNAVAILABLE"


class TemporalMatchStatus(Enum):
    """Whether the pair passed the temporal threshold.

    MATCHED     — overlap ratio meets or exceeds the policy threshold.
    UNMATCHED   — overlap ratio is below the threshold.
    UNAVAILABLE — the ratio could not be computed (zero-duration interval).
    """
    MATCHED     = "MATCHED"
    UNMATCHED   = "UNMATCHED"
    UNAVAILABLE = "UNAVAILABLE"


# ---------------------------------------------------------------------------
# NormalizedPrediction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedPrediction:
    """Evaluation representation of one prediction event.

    Contains only the information needed for matching; all unavailable
    fields are explicitly None (never a fabricated value).

    Fields
    ------
    prediction_id   — unique evaluation ID for this prediction.
    video_id        — source video this prediction was generated from.
    start_video_time— interval start in video seconds; None when the
                      prediction is a POINT_EVENT or UNAVAILABLE.
    end_video_time  — interval end in video seconds; for POINT_EVENT this
                      holds the single point time; None for UNAVAILABLE.
    class_name      — model behavior class; None when the prediction did
                      not carry a class label.
    identity        — identity string (e.g. "pose:1"); None when
                      identity is unavailable.
    confirmation_video_time — video time at which the event was confirmed;
                      None when not recorded.
    event_confidence  — model event confidence (0-1); None when unavailable.
    boundary_type   — PredictionBoundary enum.
    source_metadata — free-form dict for additional provenance; optional.
    """
    prediction_id: str
    video_id: str
    start_video_time: Optional[float]
    end_video_time: Optional[float]
    class_name: Optional[str]
    identity: Optional[str] = None
    confirmation_video_time: Optional[float] = None
    event_confidence: Optional[float] = None
    boundary_type: PredictionBoundary = PredictionBoundary.EXACT_INTERVAL
    source_metadata: Optional[Dict[str, Any]] = None

    def _interval(self) -> Tuple[float, float]:
        """Return (start, end) for temporal computation.

        For POINT_EVENT, start is set equal to end (zero-duration point).
        For UNAVAILABLE, raises ValueError.
        """
        if self.boundary_type == PredictionBoundary.UNAVAILABLE:
            raise ValueError(
                f"prediction {self.prediction_id!r} has UNAVAILABLE boundary — "
                "cannot compute temporal relationship"
            )
        if self.boundary_type == PredictionBoundary.POINT_EVENT:
            pt = self.end_video_time if self.end_video_time is not None else 0.0
            return (pt, pt)
        s = self.start_video_time if self.start_video_time is not None else 0.0
        e = self.end_video_time   if self.end_video_time   is not None else 0.0
        return (s, e)


# ---------------------------------------------------------------------------
# TemporalRelationship
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalRelationship:
    """Deterministic temporal metrics for one prediction–GT pair.

    overlap_duration          — seconds of shared time (always >= 0.0).
    overlap_ratio_prediction  — overlap_duration / prediction_duration;
                                None when prediction duration is 0.
    overlap_ratio_ground_truth— overlap_duration / gt_duration;
                                None when GT duration is 0.
    """
    overlap_duration: float
    overlap_ratio_prediction: Optional[float]
    overlap_ratio_ground_truth: Optional[float]

    def __repr__(self) -> str:
        return (
            f"TemporalRelationship(overlap={self.overlap_duration:.3f}s, "
            f"pred_ratio={self.overlap_ratio_prediction}, "
            f"gt_ratio={self.overlap_ratio_ground_truth})"
        )


# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Typed result for one prediction–GT evaluation relationship.

    ground_truth_id is None for FP results (no GT was matched).
    The reason field explains WHY the outcome was reached, making every
    result auditable.
    """
    prediction_id: str
    ground_truth_id: Optional[str]
    outcome: EvaluationOutcome
    class_match: ClassMatchStatus
    identity_match: IdentityMatchStatus
    temporal_status: TemporalMatchStatus
    overlap_duration: float
    overlap_ratio_prediction: Optional[float]
    overlap_ratio_ground_truth: Optional[float]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "ground_truth_id": self.ground_truth_id,
            "outcome": self.outcome.value,
            "class_match": self.class_match.value,
            "identity_match": self.identity_match.value,
            "temporal_status": self.temporal_status.value,
            "overlap_duration": self.overlap_duration,
            "overlap_ratio_prediction": self.overlap_ratio_prediction,
            "overlap_ratio_ground_truth": self.overlap_ratio_ground_truth,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# MatchingPolicy
# ---------------------------------------------------------------------------

@dataclass
class MatchingPolicy:
    """Evaluation-only matching parameters.

    All fields are evaluation methodology parameters — they are NOT
    production detection thresholds.  They control how the matching
    algorithm decides whether a prediction and a GT incident are
    "close enough" to be considered a match.

    Fields
    ------
    min_overlap_ratio   — minimum overlap ratio required for a temporal
                          match.  Provisional/unvalidated value; must be
                          calibrated in Phase 2C.3 before production use.
                          Default 0.5 is a clearly-labelled provisional
                          evaluation default, NOT a calibrated threshold.
    overlap_metric      — which ratio to use: "gt_ratio" (default,
                          overlap relative to GT duration), "pred_ratio"
                          (overlap relative to prediction duration), or
                          "absolute" (raw overlap seconds; min_overlap_ratio
                          is then interpreted as minimum overlap seconds).
    require_class_match — when True, a CLASS_MISMATCH prevents a TP match
                          (the pair is treated as FP/FN).  When False
                          (default), class mismatch is recorded but does
                          not block the temporal match.
    identity_mode       — "ignore" (default): identity is recorded but
                          never blocks a match.  "strict": an
                          IDENTITY_MISMATCH prevents a TP match.
                          IDENTITY_UNAVAILABLE never prevents a match in
                          either mode.
    ambiguous_handling  — "exclude" (default): AMBIGUOUS GT incidents are
                          reported as AMBIGUOUS_GROUND_TRUTH and excluded
                          from TP/FN accounting.  "report": AMBIGUOUS GT
                          incidents are treated as eligible for matching
                          (i.e., can produce a TP).
    """
    min_overlap_ratio: float = 0.5
    overlap_metric: str = "gt_ratio"
    require_class_match: bool = False
    identity_mode: str = "ignore"
    ambiguous_handling: str = "exclude"

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_overlap_ratio <= 1.0):
            raise ValueError(
                f"min_overlap_ratio must be in [0.0, 1.0], got {self.min_overlap_ratio}"
            )
        if self.overlap_metric not in ("gt_ratio", "pred_ratio", "absolute"):
            raise ValueError(
                f"overlap_metric must be 'gt_ratio', 'pred_ratio', or 'absolute', "
                f"got {self.overlap_metric!r}"
            )
        if self.identity_mode not in ("ignore", "strict"):
            raise ValueError(
                f"identity_mode must be 'ignore' or 'strict', "
                f"got {self.identity_mode!r}"
            )
        if self.ambiguous_handling not in ("exclude", "report"):
            raise ValueError(
                f"ambiguous_handling must be 'exclude' or 'report', "
                f"got {self.ambiguous_handling!r}"
            )

    @property
    def is_provisional(self) -> bool:
        """True when the policy is using uncalibrated default values.

        A policy is considered "configured" only when the caller explicitly
        sets min_overlap_ratio to a non-default value.  This property exists
        to make it impossible to silently present the provisional default as
        a calibrated threshold.
        """
        return self.min_overlap_ratio == 0.5 and self.overlap_metric == "gt_ratio"


# ---------------------------------------------------------------------------
# EvaluationSummary
# ---------------------------------------------------------------------------

@dataclass
class EvaluationSummary:
    """Aggregate evaluation metrics.

    Denominator rules (documented, must not be changed silently):
    - precision = TP / (TP + FP)         — None when denominator is 0
    - recall    = TP / (TP + FN)         — None when denominator is 0
    - F1        = 2*p*r / (p+r)          — None when p or r is None
    - duplicate_rate = DUPLICATE / total_predictions — None when total is 0

    AMBIGUOUS_GROUND_TRUTH results are excluded from all denominators.
    """
    tp_count: int
    fp_count: int
    fn_count: int
    duplicate_count: int
    ambiguous_count: int
    total_predictions: int
    total_gt_incidents: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    duplicate_rate: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp_count": self.tp_count,
            "fp_count": self.fp_count,
            "fn_count": self.fn_count,
            "duplicate_count": self.duplicate_count,
            "ambiguous_count": self.ambiguous_count,
            "total_predictions": self.total_predictions,
            "total_gt_incidents": self.total_gt_incidents,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "duplicate_rate": self.duplicate_rate,
        }


# ---------------------------------------------------------------------------
# PredictionNormalizer
# ---------------------------------------------------------------------------

class PredictionNormalizer:
    """Explicit adapter from EventRecord to NormalizedPrediction.

    All fields are mapped explicitly.  Unavailable EventRecord fields
    (event_confidence=None, association_score=None) are passed through
    as None — never replaced with a fabricated value.

    The adapter is a static method so it can be used without
    constructing a PredictionNormalizer instance.
    """

    @staticmethod
    def event_record_to_normalized(
        event: Any,
        video_id: str,
        prediction_id: str,
    ) -> NormalizedPrediction:
        """Convert an EventRecord to a NormalizedPrediction.

        Parameters
        ----------
        event       — an EventRecord (inference.core.events.EventRecord).
        video_id    — the source video ID this event was generated from.
                      Must be supplied by the caller; it is not embedded
                      in EventRecord.
        prediction_id — evaluation-specific ID for this prediction.
                      Typically event.event_id, but can be any stable string.

        Returns
        -------
        A NormalizedPrediction with all available fields populated.
        Fields that are None on the EventRecord remain None.
        """
        from inference.core.events import EventRecord  # noqa: F811

        if not isinstance(event, EventRecord):
            raise TypeError(
                f"expected EventRecord, got {type(event).__name__}"
            )

        # Determine boundary type:
        #  - if candidate_start and candidate_end are both finite and
        #    start < end → EXACT_INTERVAL
        #  - if start == end → POINT_EVENT (zero-duration point)
        #  - otherwise → EXACT_INTERVAL with 0.0 fallback
        start = event.candidate_start_video_time
        end   = event.candidate_end_video_time
        if start is not None and end is not None and start < end:
            boundary  = PredictionBoundary.EXACT_INTERVAL
            start_val = start
            end_val   = end
        elif start is not None and end is not None and start == end:
            boundary = PredictionBoundary.POINT_EVENT
            # For POINT_EVENT, start is None in the NormalizedPrediction
            # and end holds the point time.
            start_val = None
            end_val   = end
        else:
            boundary = PredictionBoundary.EXACT_INTERVAL
            start_val = start if start is not None else 0.0
            end_val   = end   if end   is not None else 0.0

        identity: Optional[str] = event.entity_key.as_string() if event.entity_key else None

        return NormalizedPrediction(
            prediction_id=prediction_id,
            video_id=video_id,
            start_video_time=start_val,
            end_video_time=end_val,
            class_name=event.class_name,
            identity=identity,
            confirmation_video_time=event.confirmation_video_time,
            event_confidence=event.event_confidence,
            boundary_type=boundary,
            source_metadata={
                "source_event_id": event.event_id,
                "entity_key": event.entity_key.as_string() if event.entity_key else None,
                "observation_count": event.observation_count,
            },
        )


# ---------------------------------------------------------------------------
# Temporal relationship
# ---------------------------------------------------------------------------

def compute_temporal_relationship(
    pred: NormalizedPrediction,
    gt: Any,
) -> TemporalRelationship:
    """Compute deterministic interval-overlap metrics for one pair.

    Parameters
    ----------
    pred — a NormalizedPrediction.
    gt   — a GroundTruthIncident (inference.evaluation.ground_truth).

    Returns
    -------
    TemporalRelationship with:
      - overlap_duration: seconds of shared time (clamped to >= 0.0).
      - overlap_ratio_prediction: overlap / pred_duration; None when
        pred_duration is 0 (POINT_EVENT or zero-duration interval).
      - overlap_ratio_ground_truth: overlap / gt_duration; None when
        gt_duration is 0.

    The intervals are half-open [start, end).  Touching endpoints
    (pred.end == gt.start or pred.start == gt.end) produce overlap 0.
    """
    from inference.evaluation.ground_truth import GroundTruthIncident

    try:
        ps, pe = pred._interval()
    except ValueError:
        # UNAVAILABLE boundary — no usable temporal information.
        return TemporalRelationship(0.0, None, None)

    gs = gt.start_video_time
    ge = gt.end_video_time

    overlap = max(0.0, min(pe, ge) - max(ps, gs))

    pred_dur = pe - ps
    gt_dur   = ge - gs

    pred_ratio: Optional[float]
    gt_ratio: Optional[float]
    pred_ratio = (overlap / pred_dur) if pred_dur > 0.0 else None
    gt_ratio   = (overlap / gt_dur)   if gt_dur   > 0.0 else None

    return TemporalRelationship(
        overlap_duration=overlap,
        overlap_ratio_prediction=pred_ratio,
        overlap_ratio_ground_truth=gt_ratio,
    )


# ---------------------------------------------------------------------------
# Class match status
# ---------------------------------------------------------------------------

def compute_class_match_status(
    pred: NormalizedPrediction,
    gt: Any,
) -> ClassMatchStatus:
    """Determine the class-match outcome for one prediction–GT pair.

    Priority order (first match wins):
    1. GT label is AMBIGUOUS          → CLASS_AMBIGUOUS
    2. pred.class_name is None       → CLASS_UNAVAILABLE
    3. pred.class_name == gt.behavior_class → CLASS_MATCH
    4. otherwise                      → CLASS_MISMATCH
    """
    from inference.evaluation.ground_truth import GTLabel

    if gt.label == GTLabel.AMBIGUOUS:
        return ClassMatchStatus.CLASS_AMBIGUOUS
    if pred.class_name is None:
        return ClassMatchStatus.CLASS_UNAVAILABLE
    if pred.class_name == gt.behavior_class:
        return ClassMatchStatus.CLASS_MATCH
    return ClassMatchStatus.CLASS_MISMATCH


# ---------------------------------------------------------------------------
# Identity match status (helper for match_predictions_to_ground_truth)
# ---------------------------------------------------------------------------

def _compute_identity_match_status(
    pred: NormalizedPrediction,
    gt: Any,
) -> IdentityMatchStatus:
    """Determine the identity-match outcome for one pair.

    IDENTITY_MATCH      — both sides have a non-None identity/reference
                          and they are equal.
    IDENTITY_MISMATCH   — both sides have a non-None value and they differ.
    IDENTITY_UNAVAILABLE— either side has no usable identity value
                          (pred.identity is None, or
                          gt.person_ref_status != KNOWN, or
                          gt.person_reference is None).
    """
    from inference.evaluation.ground_truth import EntityReferenceStatus

    gt_identity = None
    if (
        gt.person_ref_status == EntityReferenceStatus.KNOWN
        and gt.person_reference is not None
    ):
        gt_identity = gt.person_reference

    if pred.identity is None or gt_identity is None:
        return IdentityMatchStatus.IDENTITY_UNAVAILABLE
    if pred.identity == gt_identity:
        return IdentityMatchStatus.IDENTITY_MATCH
    return IdentityMatchStatus.IDENTITY_MISMATCH


# ---------------------------------------------------------------------------
# Temporal match decision (helper)
# ---------------------------------------------------------------------------

def _temporal_status_for_pair(
    pred: NormalizedPrediction,
    gt: Any,
    rel: TemporalRelationship,
    policy: MatchingPolicy,
) -> TemporalMatchStatus:
    """Return TemporalMatchStatus for one pair under the given policy.

    The threshold is selected based on policy.overlap_metric:
      - "gt_ratio":   compare rel.overlap_ratio_ground_truth
      - "pred_ratio": compare rel.overlap_ratio_prediction
      - "absolute":   compare rel.overlap_duration directly

    A ratio of None (zero-duration denominator) yields UNAVAILABLE.
    """
    if policy.overlap_metric == "absolute":
        val = rel.overlap_duration
        return TemporalMatchStatus.MATCHED if val >= policy.min_overlap_ratio \
            else TemporalMatchStatus.UNMATCHED
    elif policy.overlap_metric == "pred_ratio":
        ratio = rel.overlap_ratio_prediction
    else:  # "gt_ratio" (default)
        ratio = rel.overlap_ratio_ground_truth

    if ratio is None:
        return TemporalMatchStatus.UNAVAILABLE
    return TemporalMatchStatus.MATCHED if ratio >= policy.min_overlap_ratio \
        else TemporalMatchStatus.UNMATCHED


# ---------------------------------------------------------------------------
# match_predictions_to_ground_truth
# ---------------------------------------------------------------------------

def match_predictions_to_ground_truth(
    predictions: List[NormalizedPrediction],
    ground_truth_incidents: List[Any],
    policy: MatchingPolicy,
    normal_periods: Optional[List[Any]] = None,
    ambiguous_periods: Optional[List[Any]] = None,
) -> List[MatchResult]:
    """Deterministic one-to-one greedy matching with documented priority.

    The algorithm:

    1. Sort predictions by (start_video_time, prediction_id) for a
       deterministic processing order.  (Input order is NOT relied upon.)

    2. Sort GT incidents by (start_video_time, incident_id) for the same
       reason.  In report mode, AMBIGUOUS GTs are included in the eligible
       set; in exclude mode they are processed but produce
       AMBIGUOUS_GROUND_TRUTH results rather than TP/FN.

    3. For each prediction (in sorted order), find the best eligible GT:
       - Compute the temporal relationship.
       - Temporal status must be MATCHED (or UNAVAILABLE → no match).
       - If require_class_match is True, CLASS_MISMATCH blocks the match.
       - If identity_mode is "strict" and identity status is
         IDENTITY_MISMATCH, the match is blocked.
       - Among all eligible GTs, pick the one with the highest
         overlap_ratio (per the policy's metric); ties broken by
         lexicographic incident_id.

    4. Once a GT is consumed (assigned a primary TP), any subsequent
       prediction that would have matched the same GT produces a
       DUPLICATE result.

    5. After all predictions are processed, any GT that was never
       consumed and is not ambiguous (under the active ambiguous_handling
       mode) produces an FN result.

    6. Predictions that matched no GT produce an FP result.  If
       normal_periods is supplied and a prediction's interval falls
       inside a normal period (same video_id), the reason field notes
       this explicitly.

    Parameters
    ----------
    predictions          — list of NormalizedPrediction.
    ground_truth_incidents — list of GroundTruthIncident (SUSPICIOUS or
                            AMBIGUOUS labels; NORMAL incidents should not
                            appear here — they belong in normal_periods).
    policy               — MatchingPolicy.
    normal_periods       — optional list of NormalPeriod (Phase 2C.1).
    ambiguous_periods    — optional list of AmbiguousPeriod (Phase 2C.1).

    Returns
    -------
    A list of MatchResult, one per prediction plus one per unmatched GT.
    The list is deterministic for a given input set (order-invariant
    with respect to the GT list).
    """
    from inference.evaluation.ground_truth import GTLabel, NormalPeriod

    if normal_periods is None:
        normal_periods = []

    # Build lookup: which normal periods apply to each video_id.
    normal_periods_by_video: Dict[str, List[NormalPeriod]] = {}
    for np_ in normal_periods:
        normal_periods_by_video.setdefault(np_.video_id, []).append(np_)

    # Sort for determinism (independent of input order).
    sorted_preds = sorted(
        predictions,
        key=lambda p: (
            p.start_video_time if p.start_video_time is not None else
            (p.end_video_time or 0.0),
            p.prediction_id,
        ),
    )
    sorted_gts = sorted(
        ground_truth_incidents,
        key=lambda g: (g.start_video_time, g.incident_id),
    )

    # Determine which GTs are eligible under the active ambiguous mode.
    if policy.ambiguous_handling == "report":
        eligible_gts: List[Any] = sorted_gts
    else:  # "exclude"
        eligible_gts = [
            g for g in sorted_gts if g.label != GTLabel.AMBIGUOUS
        ]
        ambiguous_gts = [g for g in sorted_gts if g.label == GTLabel.AMBIGUOUS]

    # Track which GT incidents have been consumed (assigned a primary TP).
    consumed_gt_ids: FrozenSet[str] = frozenset()
    # For each GT, record the prediction that consumed it (for DUPLICATE).
    consumed_by: Dict[str, str] = {}

    results: List[MatchResult] = []
    matched_pred_ids: set = set()

    for pred in sorted_preds:
        # Compute relationships to all eligible GTs.
        best_gt: Optional[Any] = None
        best_ratio: Optional[float] = None
        best_rel: Optional[TemporalRelationship] = None
        best_temporal_status: TemporalMatchStatus = TemporalMatchStatus.UNMATCHED
        # Track the best GT that was blocked (class/identity gate) so the
        # FP result can carry the actual identity status when applicable.
        blocked_gt: Optional[Any] = None
        blocked_ratio: float = -1.0
        blocked_rel: Optional[TemporalRelationship] = None

        for gt in eligible_gts:
            # Skip already-consumed GTs for the PRIMARY match.
            if gt.incident_id in consumed_gt_ids:
                continue

            rel = compute_temporal_relationship(pred, gt)
            t_status = _temporal_status_for_pair(pred, gt, rel, policy)

            if t_status != TemporalMatchStatus.MATCHED:
                continue

            # Compute the metric value up-front so we can rank blocked candidates.
            if policy.overlap_metric == "absolute":
                metric_val = rel.overlap_duration
            elif policy.overlap_metric == "pred_ratio":
                metric_val = rel.overlap_ratio_prediction
            else:
                metric_val = rel.overlap_ratio_ground_truth

            # Class gate.
            class_status = compute_class_match_status(pred, gt)
            if policy.require_class_match and class_status == ClassMatchStatus.CLASS_MISMATCH:
                # Record as blocked (for FP identity reporting) and continue.
                if metric_val is not None and metric_val > blocked_ratio:
                    blocked_ratio = metric_val
                    blocked_gt = gt
                    blocked_rel = rel
                continue

            # Identity gate.
            id_status = _compute_identity_match_status(pred, gt)
            if policy.identity_mode == "strict" and id_status == IdentityMatchStatus.IDENTITY_MISMATCH:
                # Record as blocked (for FP identity reporting) and continue.
                if metric_val is not None and metric_val > blocked_ratio:
                    blocked_ratio = metric_val
                    blocked_gt = gt
                    blocked_rel = rel
                continue

            if metric_val is None:
                continue

            if best_ratio is None or metric_val > best_ratio or \
               (metric_val == best_ratio and best_gt is not None and
                gt.incident_id < best_gt.incident_id):
                best_ratio = metric_val
                best_gt = gt
                best_rel = rel
                best_temporal_status = t_status

        if best_gt is not None:
            # Primary TP.
            rel = best_rel  # type: ignore[assignment]
            class_status = compute_class_match_status(pred, best_gt)
            id_status = _compute_identity_match_status(pred, best_gt)
            consumed_gt_ids = consumed_gt_ids | {best_gt.incident_id}
            consumed_by[best_gt.incident_id] = pred.prediction_id
            matched_pred_ids.add(pred.prediction_id)
            results.append(MatchResult(
                prediction_id=pred.prediction_id,
                ground_truth_id=best_gt.incident_id,
                outcome=EvaluationOutcome.TP,
                class_match=class_status,
                identity_match=id_status,
                temporal_status=TemporalMatchStatus.MATCHED,
                overlap_duration=rel.overlap_duration,
                overlap_ratio_prediction=rel.overlap_ratio_prediction,
                overlap_ratio_ground_truth=rel.overlap_ratio_ground_truth,
                reason=(
                    f"primary TP: pred {pred.prediction_id!r} matched GT "
                    f"{best_gt.incident_id!r} "
                    f"(overlap={rel.overlap_duration:.3f}s, "
                    f"ratio={rel.overlap_ratio_ground_truth})"
                ),
            ))
        else:
            # Check for duplicate: did this prediction's temporal window
            # overlap a GT that was already consumed by another prediction?
            duplicate_gt: Optional[Any] = None
            for gt in eligible_gts:
                if gt.incident_id not in consumed_gt_ids:
                    continue
                rel = compute_temporal_relationship(pred, gt)
                t_status = _temporal_status_for_pair(pred, gt, rel, policy)
                if t_status == TemporalMatchStatus.MATCHED:
                    duplicate_gt = gt
                    break

            if duplicate_gt is not None:
                rel = compute_temporal_relationship(pred, duplicate_gt)
                results.append(MatchResult(
                    prediction_id=pred.prediction_id,
                    ground_truth_id=duplicate_gt.incident_id,
                    outcome=EvaluationOutcome.DUPLICATE,
                    class_match=compute_class_match_status(pred, duplicate_gt),
                    identity_match=_compute_identity_match_status(pred, duplicate_gt),
                    temporal_status=TemporalMatchStatus.MATCHED,
                    overlap_duration=rel.overlap_duration,
                    overlap_ratio_prediction=rel.overlap_ratio_prediction,
                    overlap_ratio_ground_truth=rel.overlap_ratio_ground_truth,
                    reason=(
                        f"DUPLICATE: pred {pred.prediction_id!r} overlaps "
                        f"GT {duplicate_gt.incident_id!r} which was already "
                        f"consumed by {consumed_by.get(duplicate_gt.incident_id)!r}"
                    ),
                ))
                matched_pred_ids.add(pred.prediction_id)
            else:
                # FP — no eligible GT matched.
                # Use the blocked GT (if any) to carry the actual identity
                # status rather than defaulting to UNAVAILABLE.
                fp_identity = IdentityMatchStatus.IDENTITY_UNAVAILABLE
                fp_class = ClassMatchStatus.CLASS_UNAVAILABLE
                fp_overlap_dur = 0.0
                fp_overlap_pred_ratio: Optional[float] = None
                fp_overlap_gt_ratio: Optional[float] = None
                fp_reason_extra = ""
                if blocked_gt is not None and blocked_rel is not None:
                    fp_identity = _compute_identity_match_status(pred, blocked_gt)
                    fp_class = compute_class_match_status(pred, blocked_gt)
                    fp_overlap_dur = blocked_rel.overlap_duration
                    fp_overlap_pred_ratio = blocked_rel.overlap_ratio_prediction
                    fp_overlap_gt_ratio = blocked_rel.overlap_ratio_ground_truth
                    fp_reason_extra = (
                        f" (blocked by identity gate: GT "
                        f"{blocked_gt.incident_id!r} had identity_mismatch)"
                    )

                in_normal_period: Optional[str] = None
                for np_ in normal_periods_by_video.get(pred.video_id, []):
                    if pred.start_video_time is not None and \
                       np_.start_video_time <= pred.start_video_time and \
                       np_.end_video_time >= (
                           pred.end_video_time if pred.end_video_time is not None
                           else pred.start_video_time
                       ):
                        in_normal_period = np_.period_id
                        break

                reason = (
                    f"FP: pred {pred.prediction_id!r} did not match any "
                    f"eligible GT incident"
                )
                if fp_reason_extra:
                    reason += fp_reason_extra
                if in_normal_period:
                    reason += f"; falls inside normal period {in_normal_period!r}"

                results.append(MatchResult(
                    prediction_id=pred.prediction_id,
                    ground_truth_id=blocked_gt.incident_id if blocked_gt is not None else None,
                    outcome=EvaluationOutcome.FP,
                    class_match=fp_class,
                    identity_match=fp_identity,
                    temporal_status=TemporalMatchStatus.UNMATCHED,
                    overlap_duration=fp_overlap_dur,
                    overlap_ratio_prediction=fp_overlap_pred_ratio,
                    overlap_ratio_ground_truth=fp_overlap_gt_ratio,
                    reason=reason,
                ))
                matched_pred_ids.add(pred.prediction_id)

    # FN results: GTs that were never consumed and are not ambiguous-excluded.
    if policy.ambiguous_handling == "exclude":
        # AMBIGUOUS GTs that were never matched → AMBIGUOUS_GROUND_TRUTH
        for gt in ambiguous_gts:
            if gt.incident_id in consumed_gt_ids:
                continue
            results.append(MatchResult(
                prediction_id=None,  # type: ignore[arg-type]
                ground_truth_id=gt.incident_id,
                outcome=EvaluationOutcome.AMBIGUOUS_GROUND_TRUTH,
                class_match=ClassMatchStatus.CLASS_AMBIGUOUS,
                identity_match=IdentityMatchStatus.IDENTITY_UNAVAILABLE,
                temporal_status=TemporalMatchStatus.UNMATCHED,
                overlap_duration=0.0,
                overlap_ratio_prediction=None,
                overlap_ratio_ground_truth=None,
                reason=(
                    f"AMBIGUOUS_GROUND_TRUTH: GT {gt.incident_id!r} is "
                    f"labeled AMBIGUOUS and is excluded from TP/FN "
                    f"accounting (ambiguous_handling='exclude')"
                ),
            ))

    for gt in eligible_gts:
        if gt.incident_id in consumed_gt_ids:
            continue
        # GT was never matched by any prediction → FN.
        results.append(MatchResult(
            prediction_id=None,  # type: ignore[arg-type]
            ground_truth_id=gt.incident_id,
            outcome=EvaluationOutcome.FN,
            class_match=ClassMatchStatus.CLASS_UNAVAILABLE,
            identity_match=IdentityMatchStatus.IDENTITY_UNAVAILABLE,
            temporal_status=TemporalMatchStatus.UNMATCHED,
            overlap_duration=0.0,
            overlap_ratio_prediction=None,
            overlap_ratio_ground_truth=None,
            reason=(
                f"FN: GT {gt.incident_id!r} had no eligible prediction "
                f"within the temporal threshold"
            ),
        ))

    return results


# ---------------------------------------------------------------------------
# compute_evaluation_summary
# ---------------------------------------------------------------------------

def compute_evaluation_summary(results: List[MatchResult]) -> EvaluationSummary:
    """Aggregate a list of MatchResults into an EvaluationSummary.

    Denominator rules:
    - precision      = TP / (TP + FP)     → None when (TP+FP) == 0
    - recall         = TP / (TP + FN)     → None when (TP+FN) == 0
    - f1             = 2*p*r / (p+r)      → None when p or r is None
    - duplicate_rate = DUPLICATE / total_predictions → None when total==0

    total_predictions = count of results with a non-None prediction_id.
    total_gt_incidents = count of results with a non-None ground_truth_id.
    AMBIGUOUS_GROUND_TRUTH results are excluded from TP/FN denominators.
    """
    tp_count = sum(1 for r in results if r.outcome == EvaluationOutcome.TP)
    fp_count = sum(1 for r in results if r.outcome == EvaluationOutcome.FP)
    fn_count = sum(1 for r in results if r.outcome == EvaluationOutcome.FN)
    dup_count = sum(1 for r in results if r.outcome == EvaluationOutcome.DUPLICATE)
    amb_count = sum(1 for r in results if r.outcome == EvaluationOutcome.AMBIGUOUS_GROUND_TRUTH)

    # total_predictions: predictions with a definite TP/FP classification.
    # DUPLICATE and AMBIGUOUS results are excluded from this count.
    total_predictions = tp_count + fp_count
    # total_gt_incidents: GTs with a definite TP/FN classification.
    # DUPLICATE and AMBIGUOUS results are excluded from this count.
    total_gt_incidents = tp_count + fn_count

    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    duplicate_rate: Optional[float]

    if tp_count + fp_count == 0:
        precision = None
    else:
        precision = tp_count / (tp_count + fp_count)

    if tp_count + fn_count == 0:
        recall = None
    else:
        recall = tp_count / (tp_count + fn_count)

    if precision is not None and recall is not None and \
       (precision + recall) > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = None

    # duplicate_rate denominator: all prediction-side outcomes
    # (TP + FP + DUPLICATE), NOT just TP + FP.
    pred_denom = tp_count + fp_count + dup_count
    if pred_denom == 0:
        duplicate_rate = None
    else:
        duplicate_rate = dup_count / pred_denom

    return EvaluationSummary(
        tp_count=tp_count,
        fp_count=fp_count,
        fn_count=fn_count,
        duplicate_count=dup_count,
        ambiguous_count=amb_count,
        total_predictions=total_predictions,
        total_gt_incidents=total_gt_incidents,
        precision=precision,
        recall=recall,
        f1=f1,
        duplicate_rate=duplicate_rate,
    )


# ---------------------------------------------------------------------------
# compute_confirmation_latency
# ---------------------------------------------------------------------------

def compute_confirmation_latency(
    pred: NormalizedPrediction,
    gt: Any,
) -> Optional[float]:
    """Video-time latency from GT incident start to prediction confirmation.

    Returns
    -------
    confirmation_video_time - gt.start_video_time  (in video seconds).
    A negative value means the prediction was confirmed before the GT
    incident started (possible for pre-emptive detection).

    Returns None when confirmation_video_time is unavailable.
    """
    if pred.confirmation_video_time is None:
        return None
    return pred.confirmation_video_time - gt.start_video_time
