"""Phase 2C.3 — Offline evaluation runner.

This module implements the offline evaluation runner described in
PHASE2C3_EVALUATION.md.  It consumes a Phase 2C.1 GroundTruthDataset and
a Phase 2C.2-compatible prediction artifact, runs per-video matching,
and produces a deterministic, machine-readable JSON evaluation report.

What this module does:
- Define PredictionRecord: a JSON-serializable prediction artifact record
  that preserves all information needed for later threshold sweeps.
- Define PredictionArtifact: a versioned container of PredictionRecords
  with provenance metadata.
- Define VideoEvaluationResult, LatencyDistribution,
  AggregateEvaluationResult, and EvaluationReport.
- Implement run_evaluation: the main entry point that groups predictions
  and GT incidents by video_id, calls the Phase 2C.2 matching engine,
  and produces a complete evaluation report.

What this module intentionally does NOT do:
- No threshold calibration (that is Phase 2C.4).
- No changes to production inference behavior.
- No use of wall-clock time in metric values.
- No OpenCV, YOLO, or GPU imports.

Phase 2C.3 evaluates existing predictions.  It does not calibrate
production thresholds.  The provisional min_overlap_ratio=0.5 is
preserved in the matching configuration snapshot and is explicitly
labelled as provisional via MatchingPolicy.is_provisional.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from inference.evaluation.matching import (
    NormalizedPrediction,
    PredictionBoundary,
    EvaluationOutcome,
    ClassMatchStatus,
    IdentityMatchStatus,
    TemporalMatchStatus,
    TemporalRelationship,
    MatchResult,
    MatchingPolicy,
    EvaluationSummary,
    compute_temporal_relationship,
    compute_class_match_status,
    match_predictions_to_ground_truth,
    compute_evaluation_summary,
    compute_confirmation_latency,
)
from inference.evaluation.ground_truth import (
    GroundTruthDataset,
    DatasetManifest,
    GroundTruthIncident,
    NormalPeriod,
    AmbiguousPeriod,
    SourceVideoMetadata,
    GTLabel,
    validate_dataset,
)


EVALUATION_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# PredictionRecord
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """A single prediction record in a prediction artifact.

    This is the canonical evaluation-side representation of one model
    prediction.  It is JSON-serializable and preserves all raw
    information needed for Phase 2C.4 threshold sweeps without
    re-running inference.

    Fields that are unavailable (not recorded by the inference pipeline)
    are explicitly None — never a fabricated value.
    """
    prediction_id: str
    video_id: str
    start_video_time: Optional[float] = None
    end_video_time: Optional[float] = None
    class_name: Optional[str] = None
    identity: Optional[str] = None
    confirmation_video_time: Optional[float] = None
    event_confidence: Optional[float] = None
    boundary_type: str = PredictionBoundary.EXACT_INTERVAL.value
    source_metadata: Optional[Dict[str, Any]] = None

    # -- helpers --

    def to_normalized(self) -> NormalizedPrediction:
        """Convert to a Phase 2C.2 NormalizedPrediction for matching."""
        return NormalizedPrediction(
            prediction_id=self.prediction_id,
            video_id=self.video_id,
            start_video_time=self.start_video_time,
            end_video_time=self.end_video_time,
            class_name=self.class_name,
            identity=self.identity,
            confirmation_video_time=self.confirmation_video_time,
            event_confidence=self.event_confidence,
            boundary_type=PredictionBoundary(self.boundary_type),
            source_metadata=self.source_metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "video_id": self.video_id,
            "start_video_time": self.start_video_time,
            "end_video_time": self.end_video_time,
            "class_name": self.class_name,
            "identity": self.identity,
            "confirmation_video_time": self.confirmation_video_time,
            "event_confidence": self.event_confidence,
            "boundary_type": self.boundary_type,
            "source_metadata": self.source_metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionRecord":
        return cls(
            prediction_id=d["prediction_id"],
            video_id=d["video_id"],
            start_video_time=d.get("start_video_time"),
            end_video_time=d.get("end_video_time"),
            class_name=d.get("class_name"),
            identity=d.get("identity"),
            confirmation_video_time=d.get("confirmation_video_time"),
            event_confidence=d.get("event_confidence"),
            boundary_type=d.get("boundary_type",
                                PredictionBoundary.EXACT_INTERVAL.value),
            source_metadata=d.get("source_metadata"),
        )

    @staticmethod
    def from_event_record_dict(d: Dict[str, Any], video_id: str) -> "PredictionRecord":
        """Adapt a serialized EventRecord dict (from EventRecord.to_dict())
        into a PredictionRecord.

        The caller must supply video_id (it is not embedded in EventRecord).
        prediction_id defaults to the EventRecord's event_id.
        """
        start = d.get("candidate_start_video_time")
        end   = d.get("candidate_end_video_time")
        if start is not None and end is not None and start < end:
            boundary = PredictionBoundary.EXACT_INTERVAL.value
            start_val, end_val = start, end
        elif start is not None and end is not None and start == end:
            boundary = PredictionBoundary.POINT_EVENT.value
            start_val, end_val = None, end
        else:
            boundary = PredictionBoundary.EXACT_INTERVAL.value
            start_val = start if start is not None else 0.0
            end_val   = end   if end   is not None else 0.0

        entity_key = d.get("entity_key")
        return cls(
            prediction_id=d.get("event_id", ""),
            video_id=video_id,
            start_video_time=start_val,
            end_video_time=end_val,
            class_name=d.get("class_name"),
            identity=entity_key,
            confirmation_video_time=d.get("confirmation_video_time"),
            event_confidence=d.get("event_confidence"),
            boundary_type=boundary,
            source_metadata={
                "entity_key": entity_key,
                "observation_count": d.get("observation_count"),
                "confidence_stats": d.get("confidence_stats"),
            },
        )


# ---------------------------------------------------------------------------
# PredictionArtifact
# ---------------------------------------------------------------------------

@dataclass
class PredictionArtifact:
    """A versioned, JSON-serializable collection of prediction records.

    This is the offline prediction input for the evaluation runner.  It
    preserves all raw information needed for Phase 2C.4 threshold sweeps
    without re-running inference.

    artifact_version identifies the serialization format (not the model
    version).  source_identifier is a human-readable label for where this
    artifact came from (e.g. a shadow run ID or manual export name).
    """
    artifact_version: str = "1.0"
    source_identifier: str = "unknown"
    records: List[PredictionRecord] = field(default_factory=list)
    notes: Optional[str] = None

    # -- helpers --

    def records_by_video(self) -> Dict[str, List[PredictionRecord]]:
        """Group records by video_id (deterministic order within each group)."""
        out: Dict[str, List[PredictionRecord]] = {}
        for r in sorted(self.records, key=lambda r: (r.video_id, r.prediction_id)):
            out.setdefault(r.video_id, []).append(r)
        return out

    def video_ids(self) -> List[str]:
        """Sorted unique video IDs present in this artifact."""
        return sorted({r.video_id for r in self.records})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "source_identifier": self.source_identifier,
            "records": [r.to_dict() for r in self.records],
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionArtifact":
        return cls(
            artifact_version=d.get("artifact_version", "1.0"),
            source_identifier=d.get("source_identifier", "unknown"),
            records=[PredictionRecord.from_dict(r) for r in d.get("records", [])],
            notes=d.get("notes"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "PredictionArtifact":
        return cls.from_dict(json.loads(raw))

    @classmethod
    def from_file(cls, path: str) -> "PredictionArtifact":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_json(fh.read())

    def validate(self) -> List[str]:
        """Return actionable validation errors for this artifact.

        An empty list means the artifact is structurally valid.
        """
        errors: List[str] = []
        seen_ids: set = set()
        for r in self.records:
            if not r.prediction_id:
                errors.append(
                    f"MISSING prediction_id in record for video {r.video_id!r}"
                )
            elif r.prediction_id in seen_ids:
                errors.append(f"DUPLICATE prediction_id: {r.prediction_id!r}")
            seen_ids.add(r.prediction_id)
            if not r.video_id:
                errors.append(f"MISSING video_id in record {r.prediction_id!r}")
            if r.start_video_time is not None and r.end_video_time is not None:
                if r.start_video_time > r.end_video_time:
                    errors.append(
                        f"INCONSISTENT INTERVAL: prediction {r.prediction_id!r} "
                        f"start ({r.start_video_time}) > end ({r.end_video_time})"
                    )
                if r.start_video_time < 0 or r.end_video_time < 0:
                    errors.append(
                        f"NEGATIVE TIME: prediction {r.prediction_id!r} "
                        f"has negative video time"
                    )
        if not self.records:
            errors.append("EMPTY ARTIFACT: no prediction records")
        return errors


# ---------------------------------------------------------------------------
# LatencyDistribution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatencyDistribution:
    """Statistical summary of confirmation latencies (video seconds).

    All fields are None when no valid latency values exist.
    Negative values are valid (pre-emptive detection).
    """
    count: int
    mean: Optional[float]
    median: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]

    @classmethod
    def compute(cls, latencies: List[Optional[float]]) -> "LatencyDistribution":
        """Compute distribution stats from a list of optional latency values.

        None values are excluded.  Returns all-None when the list is
        empty or contains only None values.
        """
        valid = [v for v in latencies if v is not None]
        if not valid:
            return cls(count=0, mean=None, median=None, minimum=None, maximum=None)
        return cls(
            count=len(valid),
            mean=sum(valid) / len(valid),
            median=statistics.median(valid),
            minimum=min(valid),
            maximum=max(valid),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


# ---------------------------------------------------------------------------
# VideoEvaluationResult
# ---------------------------------------------------------------------------

@dataclass
class VideoEvaluationResult:
    """Evaluation result for one video.

    Covers all MatchResults and the EvaluationSummary computed for that
    video, plus latency statistics.  Videos with GT but zero predictions
    (or vice versa) are still represented here.
    """
    video_id: str
    prediction_count: int
    gt_incident_count: int
    summary: EvaluationSummary
    latency: LatencyDistribution
    match_results: List[MatchResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "prediction_count": self.prediction_count,
            "gt_incident_count": self.gt_incident_count,
            "summary": self.summary.to_dict(),
            "latency": self.latency.to_dict(),
            "match_results": [r.to_dict() for r in self.match_results],
        }


# ---------------------------------------------------------------------------
# AggregateEvaluationResult
# ---------------------------------------------------------------------------

@dataclass
class AggregateEvaluationResult:
    """Dataset-level aggregate evaluation metrics.

    Primary aggregate metrics are based on total counts across all videos:
    - precision = TP_total / (TP_total + FP_total)
    - recall    = TP_total / (TP_total + FN_total)
    - F1        = 2*p*r / (p+r)

    None when the denominator is zero.
    """
    video_count: int
    total_predictions: int
    total_gt_incidents: int
    tp_count: int
    fp_count: int
    fn_count: int
    duplicate_count: int
    ambiguous_count: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    duplicate_rate: Optional[float]
    latency: LatencyDistribution

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_count": self.video_count,
            "total_predictions": self.total_predictions,
            "total_gt_incidents": self.total_gt_incidents,
            "tp_count": self.tp_count,
            "fp_count": self.fp_count,
            "fn_count": self.fn_count,
            "duplicate_count": self.duplicate_count,
            "ambiguous_count": self.ambiguous_count,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "duplicate_rate": self.duplicate_rate,
            "latency": self.latency.to_dict(),
        }


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """The complete, JSON-serializable evaluation report.

    This is the primary machine-readable output of Phase 2C.3.  It is
    deterministic given the same inputs (excluding provenance.timestamp).
    """
    schema_version: str
    provenance: Dict[str, Any]
    matching_config: Dict[str, Any]
    videos: List[VideoEvaluationResult]
    aggregate: AggregateEvaluationResult
    # Flat list of all MatchResults across all videos (preserves raw
    # per-event distribution data needed by Phase 2C.4).
    records: List[MatchResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provenance": self.provenance,
            "matching_config": self.matching_config,
            "videos": [v.to_dict() for v in self.videos],
            "aggregate": self.aggregate.to_dict(),
            "records": [r.to_dict() for r in self.records],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluationReport":
        """Reconstruct an EvaluationReport from its dict representation.

        This supports JSON round-trip for the report.
        """
        from inference.evaluation.matching import (
            MatchResult as _MR,
            EvaluationOutcome as _EO,
            ClassMatchStatus as _CMS,
            IdentityMatchStatus as _IMS,
            TemporalMatchStatus as _TMS,
        )
        videos = []
        for vd in d.get("videos", []):
            vs = EvaluationSummary(
                tp_count=vd["summary"]["tp_count"],
                fp_count=vd["summary"]["fp_count"],
                fn_count=vd["summary"]["fn_count"],
                duplicate_count=vd["summary"]["duplicate_count"],
                ambiguous_count=vd["summary"]["ambiguous_count"],
                total_predictions=vd["summary"]["total_predictions"],
                total_gt_incidents=vd["summary"]["total_gt_incidents"],
                precision=vd["summary"]["precision"],
                recall=vd["summary"]["recall"],
                f1=vd["summary"]["f1"],
                duplicate_rate=vd["summary"]["duplicate_rate"],
            )
            lat_d = vd.get("latency", {})
            lat = LatencyDistribution(
                count=lat_d.get("count", 0),
                mean=lat_d.get("mean"),
                median=lat_d.get("median"),
                minimum=lat_d.get("minimum"),
                maximum=lat_d.get("maximum"),
            )
            results = [
                _MR(
                    prediction_id=r.get("prediction_id"),
                    ground_truth_id=r.get("ground_truth_id"),
                    outcome=_EO(r["outcome"]),
                    class_match=_CMS(r["class_match"]),
                    identity_match=_IMS(r["identity_match"]),
                    temporal_status=_TMS(r["temporal_status"]),
                    overlap_duration=r["overlap_duration"],
                    overlap_ratio_prediction=r.get("overlap_ratio_prediction"),
                    overlap_ratio_ground_truth=r.get("overlap_ratio_ground_truth"),
                    reason=r.get("reason", ""),
                )
                for r in vd.get("match_results", [])
            ]
            videos.append(VideoEvaluationResult(
                video_id=vd["video_id"],
                prediction_count=vd.get("prediction_count", 0),
                gt_incident_count=vd.get("gt_incident_count", 0),
                summary=vs,
                latency=lat,
                match_results=results,
            ))

        ag = d.get("aggregate", {})
        ag_lat = ag.get("latency", {})
        aggregate = AggregateEvaluationResult(
            video_count=ag.get("video_count", 0),
            total_predictions=ag.get("total_predictions", 0),
            total_gt_incidents=ag.get("total_gt_incidents", 0),
            tp_count=ag.get("tp_count", 0),
            fp_count=ag.get("fp_count", 0),
            fn_count=ag.get("fn_count", 0),
            duplicate_count=ag.get("duplicate_count", 0),
            ambiguous_count=ag.get("ambiguous_count", 0),
            precision=ag.get("precision"),
            recall=ag.get("recall"),
            f1=ag.get("f1"),
            duplicate_rate=ag.get("duplicate_rate"),
            latency=LatencyDistribution(
                count=ag_lat.get("count", 0),
                mean=ag_lat.get("mean"),
                median=ag_lat.get("median"),
                minimum=ag_lat.get("minimum"),
                maximum=ag_lat.get("maximum"),
            ),
        )

        records = [
            _MR(
                prediction_id=r.get("prediction_id"),
                ground_truth_id=r.get("ground_truth_id"),
                outcome=_EO(r["outcome"]),
                class_match=_CMS(r["class_match"]),
                identity_match=_IMS(r["identity_match"]),
                temporal_status=_TMS(r["temporal_status"]),
                overlap_duration=r["overlap_duration"],
                overlap_ratio_prediction=r.get("overlap_ratio_prediction"),
                overlap_ratio_ground_truth=r.get("overlap_ratio_ground_truth"),
                reason=r.get("reason", ""),
            )
            for r in d.get("records", [])
        ]

        return cls(
            schema_version=d.get("schema_version", EVALUATION_SCHEMA_VERSION),
            provenance=d.get("provenance", {}),
            matching_config=d.get("matching_config", {}),
            videos=videos,
            aggregate=aggregate,
            records=records,
        )


# ---------------------------------------------------------------------------
# run_evaluation
# ---------------------------------------------------------------------------

def _make_provenance(
    gt_dataset: GroundTruthDataset,
    artifact: PredictionArtifact,
    policy: MatchingPolicy,
) -> Dict[str, Any]:
    """Build the provenance block for the evaluation report.

    The timestamp is isolated here (not in semantic fields) so
    deterministic equality tests can exclude it.
    """
    gt_versions = [m.version for m in gt_dataset.manifests]
    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "gt_dataset_versions": gt_versions,
        "prediction_artifact_version": artifact.artifact_version,
        "prediction_source_identifier": artifact.source_identifier,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _make_matching_config_dict(policy: MatchingPolicy) -> Dict[str, Any]:
    """Serialize the MatchingPolicy for the report's matching_config block."""
    return {
        "min_overlap_ratio": policy.min_overlap_ratio,
        "overlap_metric": policy.overlap_metric,
        "require_class_match": policy.require_class_match,
        "identity_mode": policy.identity_mode,
        "ambiguous_handling": policy.ambiguous_handling,
        "is_provisional": policy.is_provisional,
    }


def run_evaluation(
    gt_dataset: GroundTruthDataset,
    artifact: PredictionArtifact,
    policy: Optional[MatchingPolicy] = None,
    *,
    validation_errors: Optional[List[str]] = None,
) -> EvaluationReport:
    """Run the full offline evaluation and return an EvaluationReport.

    Parameters
    ----------
    gt_dataset  — a Phase 2C.1 GroundTruthDataset.
    artifact    — a Phase 2C.3 PredictionArtifact (offline predictions).
    policy      — a Phase 2C.2 MatchingPolicy.  Defaults to
                  MatchingPolicy() (provisional, uncalibrated).
    validation_errors — optional pre-collected list of GT dataset
                  validation errors (from validate_dataset).  When
                  supplied and non-empty, they are included in the
                  provenance block for auditability but do NOT block
                  the evaluation (the runner still runs on the data it
                  has, and reports the errors).

    Returns
    -------
    A deterministic EvaluationReport.  The report is independent of
    input ordering (both predictions and GT incidents are sorted
    internally by the Phase 2C.2 matching engine).

    Raises
    ------
    ValueError — if the GT dataset has fatal validation errors that
                 prevent evaluation (e.g. no videos at all).
    """
    if policy is None:
        policy = MatchingPolicy()

    # Validate GT dataset (non-blocking — report errors in provenance).
    gt_errors = validate_dataset(gt_dataset)
    if validation_errors is None:
        validation_errors = []
    all_gt_errors = gt_errors + validation_errors
    # Fatal error: no videos registered at all → cannot evaluate.
    if not gt_dataset.all_videos():
        if all_gt_errors:
            raise ValueError(
                f"GT dataset has fatal errors, cannot evaluate: {all_gt_errors}"
            )
        # Empty GT dataset is valid (video with zero GT case); proceed.

    artifact_errors = artifact.validate()
    # Fatal artifact errors (missing IDs) → raise; structural issues are
    # logged but do not block the run.
    fatal_artifact_errors = [
        e for e in artifact_errors
        if e.startswith("MISSING") or e.startswith("DUPLICATE")
    ]
    if fatal_artifact_errors:
        raise ValueError(
            f"Prediction artifact has fatal errors: {fatal_artifact_errors}"
        )

    # Group GT incidents and normal/ambiguous periods by video_id.
    gt_by_video: Dict[str, List[GroundTruthIncident]] = {}
    for inc in gt_dataset.all_incidents():
        # Only SUSPICIOUS and AMBIGUOUS incidents participate in matching.
        # NORMAL incidents are represented via NormalPeriods instead.
        if inc.label in (GTLabel.SUSPICIOUS, GTLabel.AMBIGUOUS):
            gt_by_video.setdefault(inc.video_id, []).append(inc)

    normal_periods_by_video: Dict[str, List[NormalPeriod]] = {}
    ambiguous_periods_by_video: Dict[str, List[AmbiguousPeriod]] = {}
    for manifest in gt_dataset.manifests:
        for np_ in manifest.normal_periods:
            normal_periods_by_video.setdefault(np_.video_id, []).append(np_)
        for ap in manifest.ambiguous_periods:
            ambiguous_periods_by_video.setdefault(ap.video_id, []).append(ap)

    # Group predictions by video_id.
    preds_by_video = artifact.records_by_video()

    # Union of all video IDs (deterministic sort).
    all_video_ids = sorted(
        set(gt_by_video.keys()) | set(preds_by_video.keys())
    )

    video_results: List[VideoEvaluationResult] = []
    all_match_results: List[MatchResult] = []
    all_latencies: List[Optional[float]] = []

    for video_id in all_video_ids:
        preds = preds_by_video.get(video_id, [])
        gts   = gt_by_video.get(video_id, [])
        nps   = normal_periods_by_video.get(video_id, [])
        aps   = ambiguous_periods_by_video.get(video_id, [])

        normalized_preds = [p.to_normalized() for p in preds]

        results = match_predictions_to_ground_truth(
            predictions=normalized_preds,
            ground_truth_incidents=gts,
            policy=policy,
            normal_periods=nps,
            ambiguous_periods=aps,
        )
        all_match_results.extend(results)

        # Collect latency values for this video.
        # Latency is only meaningful for TP results (confirmed prediction
        # matched a GT incident).
        video_latencies: List[Optional[float]] = []
        for r in results:
            if r.outcome == EvaluationOutcome.TP:
                # Find the prediction record to get confirmation_video_time.
                pred = next(
                    (p for p in preds if p.prediction_id == r.prediction_id),
                    None,
                )
                gt = next(
                    (g for g in gts if g.incident_id == r.ground_truth_id),
                    None,
                )
                if pred is not None and gt is not None:
                    normalized_pred = pred.to_normalized()
                    video_latencies.append(
                        compute_confirmation_latency(normalized_pred, gt)
                    )
        all_latencies.extend(video_latencies)
        video_latency = LatencyDistribution.compute(video_latencies)

        summary = compute_evaluation_summary(results)
        video_results.append(VideoEvaluationResult(
            video_id=video_id,
            prediction_count=len(preds),
            gt_incident_count=len(gts),
            summary=summary,
            latency=video_latency,
            match_results=results,
        ))

    # Build aggregate from total counts.
    total_tp = sum(v.summary.tp_count for v in video_results)
    total_fp = sum(v.summary.fp_count for v in video_results)
    total_fn = sum(v.summary.fn_count for v in video_results)
    total_dup = sum(v.summary.duplicate_count for v in video_results)
    total_amb = sum(v.summary.ambiguous_count for v in video_results)
    total_preds = sum(v.prediction_count for v in video_results)
    total_gts = sum(v.gt_incident_count for v in video_results)

    agg_latency = LatencyDistribution.compute(all_latencies)

    agg_precision: Optional[float]
    agg_recall: Optional[float]
    agg_f1: Optional[float]
    agg_dup_rate: Optional[float]

    if total_tp + total_fp == 0:
        agg_precision = None
    else:
        agg_precision = total_tp / (total_tp + total_fp)

    if total_tp + total_fn == 0:
        agg_recall = None
    else:
        agg_recall = total_tp / (total_tp + total_fn)

    if agg_precision is not None and agg_recall is not None and \
       (agg_precision + agg_recall) > 0.0:
        agg_f1 = 2.0 * agg_precision * agg_recall / (agg_precision + agg_recall)
    else:
        agg_f1 = None

    pred_denom = total_tp + total_fp + total_dup
    if pred_denom == 0:
        agg_dup_rate = None
    else:
        agg_dup_rate = total_dup / pred_denom

    aggregate = AggregateEvaluationResult(
        video_count=len(video_results),
        total_predictions=total_preds,
        total_gt_incidents=total_gts,
        tp_count=total_tp,
        fp_count=total_fp,
        fn_count=total_fn,
        duplicate_count=total_dup,
        ambiguous_count=total_amb,
        precision=agg_precision,
        recall=agg_recall,
        f1=agg_f1,
        duplicate_rate=agg_dup_rate,
        latency=agg_latency,
    )

    # Build the provenance and matching config blocks.
    provenance = _make_provenance(gt_dataset, artifact, policy)
    # Include GT validation errors in provenance for auditability.
    if all_gt_errors:
        provenance["gt_validation_errors"] = all_gt_errors
    if artifact_errors:
        provenance["artifact_validation_errors"] = artifact_errors

    matching_config = _make_matching_config_dict(policy)

    return EvaluationReport(
        schema_version=EVALUATION_SCHEMA_VERSION,
        provenance=provenance,
        matching_config=matching_config,
        videos=video_results,
        aggregate=aggregate,
        records=all_match_results,
    )


# ---------------------------------------------------------------------------
# human_readable_summary
# ---------------------------------------------------------------------------

def human_readable_summary(report: EvaluationReport) -> str:
    """Produce a concise text summary suitable for terminal display.

    The JSON report remains the authoritative machine-readable artifact.
    This is for human review only.
    """
    lines: List[str] = []
    a = report.aggregate
    lines.append("=" * 60)
    lines.append("PHASE 2C.3 EVALUATION SUMMARY")
    lines.append(f"  Schema version : {report.schema_version}")
    lines.append(f"  Provisional    : {report.matching_config.get('is_provisional')}")
    lines.append("-" * 60)
    lines.append(f"  Videos         : {a.video_count}")
    lines.append(f"  Predictions    : {a.total_predictions}")
    lines.append(f"  GT incidents   : {a.total_gt_incidents}")
    lines.append(f"  TP / FP / FN   : {a.tp_count} / {a.fp_count} / {a.fn_count}")
    lines.append(f"  Duplicates     : {a.duplicate_count}")
    lines.append(f"  Ambiguous      : {a.ambiguous_count}")
    p = f"{a.precision:.4f}" if a.precision is not None else "None (undefined)"
    r = f"{a.recall:.4f}" if a.recall is not None else "None (undefined)"
    f = f"{a.f1:.4f}" if a.f1 is not None else "None (undefined)"
    d = f"{a.duplicate_rate:.4f}" if a.duplicate_rate is not None else "None (undefined)"
    lines.append(f"  Precision      : {p}")
    lines.append(f"  Recall         : {r}")
    lines.append(f"  F1             : {f}")
    lines.append(f"  Duplicate rate : {d}")
    if a.latency.count > 0:
        lines.append(
            f"  Latency (s)    : n={a.latency.count} "
            f"mean={a.latency.mean:.3f} median={a.latency.median:.3f} "
            f"min={a.latency.minimum:.3f} max={a.latency.maximum:.3f}"
        )
    else:
        lines.append("  Latency (s)    : no valid values")
    lines.append("=" * 60)
    return "\n".join(lines)
