"""Phase 2E-R9 — Calibration execution readiness.

Orchestrates a calibration run from a GroundTruthDataset and a
PredictionArtifact, producing a CalibrationExecutionResult that
honestly reports what was computed and what is still blocked.

This module is read-only: no file I/O, no mutation of inputs, no
fabricated results.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from inference.evaluation.ground_truth import (
    AmbiguousPeriod,
    DatasetManifest,
    GTLabel,
    GroundTruthDataset,
    GroundTruthIncident,
    NormalPeriod,
    SourceVideoMetadata,
    SceneTag,
    validate_dataset,
)
from inference.evaluation.matching import (
    MatchingPolicy,
    NormalizedPrediction,
    match_predictions_to_ground_truth,
)
from inference.evaluation.runner import (
    PredictionArtifact,
    PredictionRecord,
)
from inference.evaluation.calibration import (
    CalibrationParameterGrid,
    CalibrationReport,
    CalibrationSplit,
    CandidateEvaluationConfig,
    OperatingPointPolicy,
    OperatingPointResult,
    build_calibration_report,
    compute_subgroup_metrics,
    select_operating_point,
    sweep_matching_parameters,
    _default_total_video_hours,
)
from inference.evaluation.shadow_bridge import CalibrationReadinessReport


@dataclass
class DatasetValidationReport:
    video_count: int
    incident_count: int
    normal_period_count: int
    ambiguous_period_count: int
    total_duration_seconds: Optional[float]
    sufficient_for_calibration: bool
    issues: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    dataset_validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_count": self.video_count,
            "incident_count": self.incident_count,
            "normal_period_count": self.normal_period_count,
            "ambiguous_period_count": self.ambiguous_period_count,
            "total_duration_seconds": self.total_duration_seconds,
            "sufficient_for_calibration": self.sufficient_for_calibration,
            "issues": list(self.issues),
            "limitations": list(self.limitations),
            "dataset_validation_errors": list(self.dataset_validation_errors),
        }


def validate_gt_dataset_for_calibration(
    gt: GroundTruthDataset,
) -> DatasetValidationReport:
    videos = gt.all_videos()
    incidents = gt.all_incidents()
    normal_periods: List[NormalPeriod] = []
    ambiguous_periods: List[AmbiguousPeriod] = []
    for m in gt.manifests:
        normal_periods.extend(m.normal_periods)
        ambiguous_periods.extend(m.ambiguous_periods)

    durations = [v.duration_seconds for v in videos if v.duration_seconds is not None]
    total_duration = sum(durations) if durations else None

    issues: List[str] = []
    if not incidents:
        issues.append("no GT incidents")
    if not videos:
        issues.append("no source video metadata")

    sufficient = bool(videos) and bool(incidents)

    limitations: List[str] = []
    if len(videos) < 3:
        limitations.append("insufficient_videos_for_split")
    if not normal_periods:
        limitations.append("no_normal_periods")
    if not ambiguous_periods:
        limitations.append("no_ambiguous_periods")
    if total_duration is None:
        limitations.append("no_duration_metadata")

    validation_errors = validate_dataset(gt)

    return DatasetValidationReport(
        video_count=len(videos),
        incident_count=len(incidents),
        normal_period_count=len(normal_periods),
        ambiguous_period_count=len(ambiguous_periods),
        total_duration_seconds=total_duration,
        sufficient_for_calibration=sufficient,
        issues=issues,
        limitations=limitations,
        dataset_validation_errors=validation_errors,
    )


@dataclass
class CalibrationExecutionResult:
    executed: bool
    missing_prerequisites: List[str]
    dataset_summary: Dict[str, Any]
    readiness: CalibrationReadinessReport
    report: Optional[CalibrationReport]
    operating_point: Optional[OperatingPointResult]
    subgroup_metrics: Optional[Dict[str, Any]]
    confirmation_results: Optional[List[Dict[str, Any]]]
    cutover_status: str = "NOT_READY"
    production_ready: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executed": self.executed,
            "missing_prerequisites": list(self.missing_prerequisites),
            "dataset_summary": self.dataset_summary,
            "readiness": self.readiness.to_dict(),
            "report": self.report.to_dict() if self.report is not None else None,
            "operating_point": self.operating_point.to_dict() if self.operating_point is not None else None,
            "subgroup_metrics": self.subgroup_metrics,
            "confirmation_results": self.confirmation_results,
            "cutover_status": self.cutover_status,
            "production_ready": self.production_ready,
        }


def _build_dataset_summary(
    gt: GroundTruthDataset,
    artifact: PredictionArtifact,
    split_seed: int,
    train_frac: float,
    val_frac: float,
) -> Dict[str, Any]:
    dv = validate_gt_dataset_for_calibration(gt)
    video_ids = sorted(gt.video_ids())
    split = CalibrationSplit.create(video_ids, train_frac=train_frac, val_frac=val_frac, seed=split_seed)
    train_split = split["train"]
    validation_split = split["validation"]
    holdout_split = split["holdout"]

    return {
        "video_count": dv.video_count,
        "incident_count": dv.incident_count,
        "normal_period_count": dv.normal_period_count,
        "ambiguous_period_count": dv.ambiguous_period_count,
        "total_duration_seconds": dv.total_duration_seconds,
        "split_seed": split_seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "train_videos": list(train_split.videos),
        "validation_videos": list(validation_split.videos),
        "holdout_videos": list(holdout_split.videos),
        "total_predictions": len(artifact.records),
    }


def _strip_confidence_params(
    grid: CalibrationParameterGrid,
) -> CalibrationParameterGrid:
    stripped = {k: list(v) for k, v in grid.parameters.items() if k != "min_confidence_mean"}
    if not stripped:
        return grid
    return CalibrationParameterGrid(stripped)


def execute_calibration(
    gt_dataset: GroundTruthDataset,
    artifact: PredictionArtifact,
    matching_grid: Optional[CalibrationParameterGrid],
    operating_point_policy: OperatingPointPolicy,
    split_seed: int = 42,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    total_video_hours: Optional[float] = None,
) -> CalibrationExecutionResult:
    readiness = CalibrationReadinessReport.from_prediction_records(artifact.records)
    dv = validate_gt_dataset_for_calibration(gt_dataset)
    dataset_summary = _build_dataset_summary(gt_dataset, artifact, split_seed, train_frac, val_frac)

    if not dv.sufficient_for_calibration:
        prereqs: List[str] = []
        if dv.incident_count == 0:
            prereqs.append("no GT incidents - a GroundTruthDataset with annotated incidents is required")
        if dv.video_count == 0:
            prereqs.append("no source video metadata - at least one SourceVideoMetadata entry is required")
        for item in dv.issues:
            if item not in prereqs:
                prereqs.append(item)
        for item in dv.limitations:
            prereqs.append("dataset limitation: " + item)
        for item in dv.dataset_validation_errors:
            prereqs.append("GT validation error: " + item)
        return CalibrationExecutionResult(
            executed=False,
            missing_prerequisites=prereqs,
            dataset_summary=dataset_summary,
            readiness=readiness,
            report=None,
            operating_point=None,
            subgroup_metrics=None,
            confirmation_results=None,
        )

    if matching_grid is None:
        return CalibrationExecutionResult(
            executed=False,
            missing_prerequisites=["no matching_grid supplied - no sweep can run"],
            dataset_summary=dataset_summary,
            readiness=readiness,
            report=None,
            operating_point=None,
            subgroup_metrics=None,
            confirmation_results=None,
        )

    all_none_conf = all(r.event_confidence is None for r in artifact.records) if artifact.records else True
    effective_grid = _strip_confidence_params(matching_grid) if all_none_conf else matching_grid

    video_ids = sorted(gt_dataset.video_ids())
    split = CalibrationSplit.create(video_ids, train_frac=train_frac, val_frac=val_frac, seed=split_seed)
    train_split = split["train"]

    matching_sweep: List[Dict[str, Any]] = []
    if effective_grid.parameters:
        matching_sweep = sweep_matching_parameters(
            gt_dataset, artifact, effective_grid, train_split,
            total_video_hours=total_video_hours,
        )

    op_result: Optional[OperatingPointResult] = None
    matching_selected: Optional[Dict[str, Any]] = None
    if matching_sweep:
        op_result = select_operating_point(matching_sweep, operating_point_policy)
        matching_selected = op_result.selected_combination

    subgroup_metrics: Optional[Dict[str, Any]] = None
    has_tags = any(v.scene_tags for v in gt_dataset.all_videos())
    if has_tags and matching_sweep:
        gt_by_video: Dict[str, List[GroundTruthIncident]] = {}
        for inc in gt_dataset.all_incidents():
            if inc.label in (GTLabel.SUSPICIOUS, GTLabel.AMBIGUOUS):
                gt_by_video.setdefault(inc.video_id, []).append(inc)
        preds_by_video = artifact.records_by_video()
        default_policy = MatchingPolicy()
        all_results = []
        for vid in train_split.videos:
            preds = [p.to_normalized() for p in preds_by_video.get(vid, [])]
            gts = gt_by_video.get(vid, [])
            norms: List[NormalPeriod] = []
            ambs: List[AmbiguousPeriod] = []
            for m in gt_dataset.manifests:
                norms.extend(p for p in m.normal_periods if p.video_id == vid)
                ambs.extend(p for p in m.ambiguous_periods if p.video_id == vid)
            results = match_predictions_to_ground_truth(
                predictions=preds,
                ground_truth_incidents=gts,
                policy=default_policy,
                normal_periods=norms or None,
                ambiguous_periods=ambs or None,
            )
            all_results.extend(results)
        subgroup_metrics = compute_subgroup_metrics(gt_dataset, artifact, all_results)

    report = build_calibration_report(
        gt_dataset, artifact,
        matching_sweep=matching_sweep or None,
        matching_selected=matching_selected,
        confirmation_sweep=None,
        confirmation_selected=None,
    )

    dataset_summary["split_seed"] = split_seed
    dataset_summary["train_videos"] = list(train_split.videos)

    return CalibrationExecutionResult(
        executed=True,
        missing_prerequisites=[],
        dataset_summary=dataset_summary,
        readiness=readiness,
        report=report,
        operating_point=op_result,
        subgroup_metrics=subgroup_metrics,
        confirmation_results=None,
    )


__all__ = [
    "DatasetValidationReport",
    "CalibrationExecutionResult",
    "validate_gt_dataset_for_calibration",
    "execute_calibration",
]
