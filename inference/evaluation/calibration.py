"""Phase 2C.4 — Deterministic calibration for evaluation parameters.

This module provides deterministic calibration utilities for the offline
evaluation system described in PHASE2C.  It sweeps matching and
confirmation parameters over a deterministic grid, selects an operating
point (precision-first), computes subgroup metrics, and assembles a
JSON-serializable CalibrationReport.

What this module does:
- Define deterministic parameter grids (CalibrationParameterGrid).
- Compute per-combination metrics (CalibrationMetrics) from MatchResults.
- Provide a deterministic video-level split (CalibrationSplit / split_videos).
- Sweep matching and confirmation parameters over the grid.
- Select an operating point (precision-first, optional Pareto set).
- Compute per-scene-tag subgroup metrics.
- Build a full CalibrationReport (JSON-serializable, deterministic).

What this module intentionally does NOT do:
- No production threshold writes or inference changes.
- No OpenCV, YOLO, or GPU imports.
- No wall-clock time in metric values (timestamp is isolated in the report).
- No randomness: all splits and grid orderings are deterministic.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

from inference.evaluation.matching import (
    EvaluationOutcome,
    MatchResult,
    MatchingPolicy,
    NormalizedPrediction,
    compute_confirmation_latency,
    compute_evaluation_summary,
    match_predictions_to_ground_truth,
)
from inference.evaluation.runner import (
    PredictionArtifact,
    PredictionRecord,
)
from inference.evaluation.ground_truth import (
    AmbiguousPeriod,
    GroundTruthDataset,
    GroundTruthIncident,
    GTLabel,
    NormalPeriod,
    SceneTag,
    SourceVideoMetadata,
)


CALIBRATION_VERSION = "1.0"


def _default_total_video_hours(gt_dataset: GroundTruthDataset) -> float:
    """Derive total video hours from GT dataset metadata.

    Sums duration_seconds / 3600 over all SourceVideoMetadata entries
    that have a non-None duration_seconds.  Returns 0.0 when no
    duration is available.
    """
    total = 0.0
    for video in gt_dataset.all_videos():
        if video.duration_seconds is not None:
            total += video.duration_seconds
    return total / 3600.0


# ---------------------------------------------------------------------------
# CalibrationParameterGrid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationParameterGrid:
    """A deterministic grid of parameter combinations to sweep.

    parameters: Dict[str, List[Any]] — parameter name to candidate values.
    Combinations are the lexicographic product, deterministically sorted
    (no randomness).
    """
    parameters: Dict[str, List[Any]]

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("CalibrationParameterGrid.parameters must be non-empty")
        for name, values in self.parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"parameter name must be a non-empty string, got {name!r}"
                )
            if not isinstance(values, list):
                raise ValueError(
                    f"parameter {name!r} values must be a list, "
                    f"got {type(values).__name__}"
                )
            if not values:
                raise ValueError(f"parameter {name!r} values must be a non-empty list")

    def combinations(self) -> List[Dict[str, Any]]:
        """Return all parameter combinations in deterministic lexicographic order.

        The Cartesian product is taken over sorted parameter names, and the
        resulting combinations are sorted by the tuple of their values so the
        output is stable regardless of insertion order.
        """
        names = sorted(self.parameters.keys())
        value_lists = [self.parameters[n] for n in names]
        combos = [dict(zip(names, combo)) for combo in product(*value_lists)]
        combos.sort(key=lambda c: tuple(c[n] for n in names))
        return combos


# ---------------------------------------------------------------------------
# CalibrationMetrics
# ---------------------------------------------------------------------------

@dataclass
class CalibrationMetrics:
    """Per-combination metrics from one calibration run.

    All rate fields are None when their denominator is 0.
    latency_p95 is None when the valid sample count is fewer than 10.
    """
    tp: int
    fp: int
    fn: int
    duplicates: int
    ambiguous: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    duplicate_rate: Optional[float]
    fp_per_hour: Optional[float]   # None if total_video_hours is 0
    fn_per_hour: Optional[float]   # None if total_video_hours is 0
    latency_count: int
    latency_mean: Optional[float]
    latency_median: Optional[float]
    latency_p95: Optional[float]   # None if count < 10
    total_videos: int
    total_gt_incidents: int
    total_predictions: int
    per_video_metrics: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "duplicates": self.duplicates,
            "ambiguous": self.ambiguous,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "duplicate_rate": self.duplicate_rate,
            "fp_per_hour": self.fp_per_hour,
            "fn_per_hour": self.fn_per_hour,
            "latency_count": self.latency_count,
            "latency_mean": self.latency_mean,
            "latency_median": self.latency_median,
            "latency_p95": self.latency_p95,
            "total_videos": self.total_videos,
            "total_gt_incidents": self.total_gt_incidents,
            "total_predictions": self.total_predictions,
            "per_video_metrics": self.per_video_metrics,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationMetrics":
        return cls(
            tp=d.get("tp", 0),
            fp=d.get("fp", 0),
            fn=d.get("fn", 0),
            duplicates=d.get("duplicates", 0),
            ambiguous=d.get("ambiguous", 0),
            precision=d.get("precision"),
            recall=d.get("recall"),
            f1=d.get("f1"),
            duplicate_rate=d.get("duplicate_rate"),
            fp_per_hour=d.get("fp_per_hour"),
            fn_per_hour=d.get("fn_per_hour"),
            latency_count=d.get("latency_count", 0),
            latency_mean=d.get("latency_mean"),
            latency_median=d.get("latency_median"),
            latency_p95=d.get("latency_p95"),
            total_videos=d.get("total_videos", 0),
            total_gt_incidents=d.get("total_gt_incidents", 0),
            total_predictions=d.get("total_predictions", 0),
            per_video_metrics=d.get("per_video_metrics", []),
        )


# ---------------------------------------------------------------------------
# CalibrationSplit
# ---------------------------------------------------------------------------

@dataclass
class CalibrationSplit:
    """Deterministic video-level split.

    split_name: 'train' | 'validation' | 'holdout' | 'all'
    videos: sorted video_id list.
    All incidents/predictions from a video stay in the same split.
    """
    split_name: str
    videos: List[str]

    @classmethod
    def create(
        cls,
        video_ids: List[str],
        *,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        seed: int = 42,
    ) -> Dict[str, "CalibrationSplit"]:
        """Deterministic video-level split.

        Delegates to split_videos() and wraps each part in a CalibrationSplit.
        Returns a dict with keys 'train', 'validation', 'holdout' plus 'all'
        (containing every video_id, sorted).
        """
        parts = split_videos(
            video_ids, train_frac=train_frac, val_frac=val_frac, seed=seed
        )
        result: Dict[str, "CalibrationSplit"] = {
            key: CalibrationSplit(split_name=key, videos=vals)
            for key, vals in parts.items()
        }
        result["all"] = CalibrationSplit(split_name="all", videos=sorted(video_ids))
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {"split_name": self.split_name, "videos": list(self.videos)}


# ---------------------------------------------------------------------------
# OperatingPointPolicy / OperatingPointResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatingPointPolicy:
    """Precision-first operating-point selection policy.

    fp_per_hour_target: when set, candidates must satisfy fp_per_hour <= target.
    min_recall:         when set, candidates must satisfy recall >= min_recall.
    If both are None, a Pareto set is returned instead of a single selection.
    """
    fp_per_hour_target: Optional[float] = None
    min_recall: Optional[float] = None


@dataclass
class OperatingPointResult:
    """Result of operating-point selection from a sweep."""
    selected_combination: Dict[str, Any]
    metrics: CalibrationMetrics
    reason: str
    is_pareto_candidate: bool = False
    pareto_set: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_combination": self.selected_combination,
            "metrics": self.metrics.to_dict(),
            "reason": self.reason,
            "is_pareto_candidate": self.is_pareto_candidate,
            "pareto_set": self.pareto_set,
        }


# ---------------------------------------------------------------------------
# CandidateEvaluationConfig
# ---------------------------------------------------------------------------

@dataclass
class CandidateEvaluationConfig:
    """Calibrated candidate configuration for matching + confirmation.

    Each sub-dict carries a 'status' field: 'calibrated', 'provisional',
    or 'not_calibrated'.  This is a review artifact, NOT a runtime config.
    """
    matching: Dict[str, Any]
    confirmation: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"matching": self.matching, "confirmation": self.confirmation}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidateEvaluationConfig":
        return cls(
            matching=d.get("matching", {}),
            confirmation=d.get("confirmation", {}),
        )


# ---------------------------------------------------------------------------
# CalibrationReport
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReport:
    """Full calibration report — JSON-serializable, deterministic
    (excluding the timestamp_utc field)."""
    calibration_version: str
    timestamp_utc: str
    dataset_version: str
    artifact_version: str
    artifact_identifier: str
    split_definition: Dict[str, Any]
    matching_grid: Dict[str, Any]
    matching_results: List[Dict[str, Any]]
    matching_selected: Optional[Dict[str, Any]]
    confirmation_grid: Optional[Dict[str, Any]]
    confirmation_results: Optional[List[Dict[str, Any]]]
    confirmation_selected: Optional[Dict[str, Any]]
    candidate_configuration: Dict[str, Any]
    parameter_status: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_version": self.calibration_version,
            "timestamp_utc": self.timestamp_utc,
            "dataset_version": self.dataset_version,
            "artifact_version": self.artifact_version,
            "artifact_identifier": self.artifact_identifier,
            "split_definition": self.split_definition,
            "matching_grid": self.matching_grid,
            "matching_results": self.matching_results,
            "matching_selected": self.matching_selected,
            "confirmation_grid": self.confirmation_grid,
            "confirmation_results": self.confirmation_results,
            "confirmation_selected": self.confirmation_selected,
            "candidate_configuration": self.candidate_configuration,
            "parameter_status": self.parameter_status,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationReport":
        return cls(
            calibration_version=d.get("calibration_version", CALIBRATION_VERSION),
            timestamp_utc=d.get("timestamp_utc", ""),
            dataset_version=d.get("dataset_version", ""),
            artifact_version=d.get("artifact_version", ""),
            artifact_identifier=d.get("artifact_identifier", ""),
            split_definition=d.get("split_definition", {}),
            matching_grid=d.get("matching_grid", {}),
            matching_results=d.get("matching_results", []),
            matching_selected=d.get("matching_selected"),
            confirmation_grid=d.get("confirmation_grid"),
            confirmation_results=d.get("confirmation_results"),
            confirmation_selected=d.get("confirmation_selected"),
            candidate_configuration=d.get("candidate_configuration", {}),
            parameter_status=d.get("parameter_status", {}),
        )


# ---------------------------------------------------------------------------
# split_videos
# ---------------------------------------------------------------------------

def split_videos(
    video_ids: List[str],
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Deterministic video-level split.

    Returns {'train': [...], 'validation': [...], 'holdout': [...]}.
    All incidents/predictions from a video stay in the same split.
    """
    vids = sorted(video_ids)
    if len(vids) < 3:
        return {"train": vids, "validation": [], "holdout": []}

    def _bucket(vid: str) -> int:
        h = hashlib.sha256(f"{seed}:{vid}".encode()).digest()
        return int.from_bytes(h[:4], "big") % 100

    buckets: Dict[str, List[str]] = {"train": [], "validation": [], "holdout": []}
    threshold_val = int(train_frac * 100)
    threshold_holdout = int((train_frac + val_frac) * 100)

    for vid in vids:
        b = _bucket(vid)
        if b < threshold_val:
            buckets["train"].append(vid)
        elif b < threshold_holdout:
            buckets["validation"].append(vid)
        else:
            buckets["holdout"].append(vid)

    if not buckets["train"] and vids:
        buckets["train"].append(vids[0])

    for k in buckets:
        buckets[k].sort()
    return buckets


# ---------------------------------------------------------------------------
# compute_calibration_metrics
# ---------------------------------------------------------------------------

def _p95(values: List[float]) -> Optional[float]:
    """Return the 95th percentile, or None if fewer than 10 values."""
    if len(values) < 10:
        return None
    s = sorted(values)
    idx = min(int(0.95 * len(s)), len(s) - 1)
    return s[idx]


def compute_calibration_metrics(
    match_results: List[MatchResult],
    total_video_hours: float = 0.0,
    latencies: Optional[List[Optional[float]]] = None,
    total_videos: int = 0,
    total_gt_incidents: int = 0,
    total_predictions: int = 0,
    per_video_metrics: Optional[List[Dict[str, Any]]] = None,
) -> CalibrationMetrics:
    """Compute CalibrationMetrics from Phase 2C.2 MatchResults.

    - precision = tp/(tp+fp), None if denominator is 0
    - recall    = tp/(tp+fn), None if denominator is 0
    - f1        = 2*p*r/(p+r), None if p or r is None
    - fp_per_hour = fp / total_video_hours, None if hours is 0
    - fn_per_hour = fn / total_video_hours, None if hours is 0
    - latency_p95 = None if fewer than 10 valid latency samples
    """
    tp  = sum(1 for r in match_results if r.outcome == EvaluationOutcome.TP)
    fp  = sum(1 for r in match_results if r.outcome == EvaluationOutcome.FP)
    fn  = sum(1 for r in match_results if r.outcome == EvaluationOutcome.FN)
    dup = sum(1 for r in match_results if r.outcome == EvaluationOutcome.DUPLICATE)
    amb = sum(1 for r in match_results if r.outcome == EvaluationOutcome.AMBIGUOUS_GROUND_TRUTH)

    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None

    if tp + fp > 0:
        precision = tp / (tp + fp)
    if tp + fn > 0:
        recall = tp / (tp + fn)
    if precision is not None and recall is not None and (precision + recall) > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)

    pred_denom = tp + fp + dup
    dup_rate: Optional[float] = (dup / pred_denom) if pred_denom > 0 else None

    fp_per_hour: Optional[float] = (fp / total_video_hours) if total_video_hours > 0 else None
    fn_per_hour: Optional[float] = (fn / total_video_hours) if total_video_hours > 0 else None

    valid_lats = [v for v in (latencies or []) if v is not None]
    lat_count = len(valid_lats)
    lat_mean = (sum(valid_lats) / lat_count) if lat_count > 0 else None
    lat_median = statistics.median(valid_lats) if lat_count > 0 else None
    lat_p95 = _p95(valid_lats)

    return CalibrationMetrics(
        tp=tp, fp=fp, fn=fn, duplicates=dup, ambiguous=amb,
        precision=precision, recall=recall, f1=f1, duplicate_rate=dup_rate,
        fp_per_hour=fp_per_hour, fn_per_hour=fn_per_hour,
        latency_count=lat_count,
        latency_mean=lat_mean, latency_median=lat_median, latency_p95=lat_p95,
        total_videos=total_videos,
        total_gt_incidents=total_gt_incidents,
        total_predictions=total_predictions,
        per_video_metrics=per_video_metrics or [],
    )


# ---------------------------------------------------------------------------
# _run_match_for_video  (internal helper)
# ---------------------------------------------------------------------------

def _run_match_for_video(
    video_id: str,
    predictions: List[PredictionRecord],
    gt_incidents: List[GroundTruthIncident],
    normal_periods: List[NormalPeriod],
    ambiguous_periods: List[AmbiguousPeriod],
    policy: MatchingPolicy,
) -> List[MatchResult]:
    """Run matching for a single video and return the result list."""
    normalized = [p.to_normalized() for p in predictions]
    return match_predictions_to_ground_truth(
        predictions=normalized,
        ground_truth_incidents=gt_incidents,
        policy=policy,
        normal_periods=normal_periods or None,
        ambiguous_periods=ambiguous_periods or None,
    )


def _collect_latencies(
    match_results: List[MatchResult],
    predictions: List[PredictionRecord],
    gt_incidents: List[GroundTruthIncident],
) -> List[Optional[float]]:
    """Collect latency values for TP results only."""
    gt_by_id = {g.incident_id: g for g in gt_incidents}
    pred_by_id = {p.prediction_id: p for p in predictions}
    latencies: List[Optional[float]] = []
    for r in match_results:
        if r.outcome != EvaluationOutcome.TP:
            continue
        pred = pred_by_id.get(r.prediction_id)
        gt = gt_by_id.get(r.ground_truth_id) if r.ground_truth_id else None
        if pred is None or gt is None:
            latencies.append(None)
        else:
            latencies.append(compute_confirmation_latency(
                pred.to_normalized(), gt
            ))
    return latencies


def _filter_to_split(
    gt_dataset: GroundTruthDataset,
    artifact: PredictionArtifact,
    split: CalibrationSplit,
) -> Dict[str, Any]:
    """Filter GT incidents and predictions to only those in split.videos."""
    split_vids = set(split.videos)
    all_incidents = gt_dataset.all_incidents()
    all_norm = [np for m in gt_dataset.manifests for np in m.normal_periods]
    all_amb = [ap for m in gt_dataset.manifests for ap in m.ambiguous_periods]

    gt_by_video: Dict[str, List[GroundTruthIncident]] = {}
    for inc in all_incidents:
        if inc.video_id in split_vids:
            gt_by_video.setdefault(inc.video_id, []).append(inc)

    norm_by_video: Dict[str, List[NormalPeriod]] = {}
    for np_ in all_norm:
        if np_.video_id in split_vids:
            norm_by_video.setdefault(np_.video_id, []).append(np_)

    amb_by_video: Dict[str, List[AmbiguousPeriod]] = {}
    for ap in all_amb:
        if ap.video_id in split_vids:
            amb_by_video.setdefault(ap.video_id, []).append(ap)

    preds_by_video = artifact.records_by_video()
    filtered_preds: Dict[str, List[PredictionRecord]] = {
        vid: recs for vid, recs in preds_by_video.items() if vid in split_vids
    }

    gt_vids = {i.video_id for m in gt_dataset.manifests for i in m.incidents} & split_vids
    pred_vids = set(filtered_preds.keys())
    all_vids = sorted(gt_vids | pred_vids)

    return {
        "gt_by_video": gt_by_video,
        "norm_by_video": norm_by_video,
        "amb_by_video": amb_by_video,
        "preds_by_video": filtered_preds,
        "video_ids": all_vids,
    }


# ---------------------------------------------------------------------------
# sweep_matching_parameters
# ---------------------------------------------------------------------------

def sweep_matching_parameters(
    gt_dataset: GroundTruthDataset,
    artifact: PredictionArtifact,
    grid: CalibrationParameterGrid,
    split: CalibrationSplit,
    *,
    total_video_hours: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run matching calibration sweep over a parameter grid.

    For each combination in grid.combinations():
      1. Filter predictions and GT incidents to only those in split.videos.
      2. Build a MatchingPolicy from the combination.
      3. Run per-video matching.
      4. Compute CalibrationMetrics.
      5. Append row: {**combination, "metrics": ..., "split": split.split_name}

    Returns a deterministic list of result rows.
    """
    if total_video_hours is None:
        total_video_hours = _default_total_video_hours(gt_dataset)

    filt = _filter_to_split(gt_dataset, artifact, split)
    all_rows: List[Dict[str, Any]] = []

    for combo in grid.combinations():
        min_overlap = combo.get("min_overlap_ratio", 0.5)
        overlap_metric = combo.get("overlap_metric", "gt_ratio")
        require_class = combo.get("require_class_match", False)
        identity_mode = combo.get("identity_mode", "ignore")
        amb_handling = combo.get("ambiguous_handling", "exclude")

        try:
            policy = MatchingPolicy(
                min_overlap_ratio=min_overlap,
                overlap_metric=overlap_metric,
                require_class_match=require_class,
                identity_mode=identity_mode,
                ambiguous_handling=amb_handling,
            )
        except ValueError:
            # Invalid combination (e.g. min_overlap_ratio out of range)
            all_rows.append({**combo, "error": f"invalid combination: {combo}", "metrics": None, "split": split.split_name})
            continue

        all_match_results: List[MatchResult] = []
        per_video_metrics: List[Dict[str, Any]] = []
        all_latencies: List[Optional[float]] = []

        for vid in filt["video_ids"]:
            preds = filt["preds_by_video"].get(vid, [])
            gts = filt["gt_by_video"].get(vid, [])
            norms = filt["norm_by_video"].get(vid, [])
            ambs = filt["amb_by_video"].get(vid, [])

            results = _run_match_for_video(vid, preds, gts, norms, ambs, policy)
            all_match_results.extend(results)
            all_latencies.extend(_collect_latencies(results, preds, gts))

            summary = compute_evaluation_summary(results)
            per_video_metrics.append({
                "video_id": vid,
                "tp": summary.tp_count,
                "fp": summary.fp_count,
                "fn": summary.fn_count,
                "duplicate": summary.duplicate_count,
                "ambiguous": summary.ambiguous_count,
                "precision": summary.precision,
                "recall": summary.recall,
                "f1": summary.f1,
            })

        metrics = compute_calibration_metrics(
            match_results=all_match_results,
            total_video_hours=total_video_hours,
            latencies=all_latencies,
            total_videos=len(filt["video_ids"]),
            total_gt_incidents=sum(len(gts) for gts in filt["gt_by_video"].values()),
            total_predictions=sum(len(ps) for ps in filt["preds_by_video"].values()),
            per_video_metrics=per_video_metrics,
        )

        all_rows.append({
            **combo,
            "metrics": metrics,
            "split": split.split_name,
        })

    return all_rows


# ---------------------------------------------------------------------------
# select_operating_point
# ---------------------------------------------------------------------------

def _is_dominated(
    candidate: Dict[str, Any],
    pareto_set: List[Dict[str, Any]],
) -> bool:
    """Return True if candidate is dominated by any row in pareto_set.

    A row dominates candidate when:
      - fp_per_hour <= candidate.fp_per_hour (lower is better) AND
      - recall >= candidate.recall (higher is better)
      - precision >= candidate.precision (higher is better)
      - AND at least one strict inequality holds.
    None fp_per_hour is treated as worst (inf); None recall as worst (0.0).
    """
    def _fph(row: Dict[str, Any]) -> float:
        m = row.get("metrics")
        if m is None:
            return float("inf")
        return m.fp_per_hour if m.fp_per_hour is not None else float("inf")

    def _rec(row: Dict[str, Any]) -> float:
        m = row.get("metrics")
        if m is None:
            return 0.0
        return m.recall if m.recall is not None else 0.0

    def _prec(row: Dict[str, Any]) -> float:
        m = row.get("metrics")
        if m is None:
            return 0.0
        return m.precision if m.precision is not None else 0.0

    c_fp, c_rec, c_prec = _fph(candidate), _rec(candidate), _prec(candidate)
    for row in pareto_set:
        r_fp, r_rec, r_prec = _fph(row), _rec(row), _prec(row)
        if r_fp <= c_fp and r_rec >= c_rec and r_prec >= c_prec:
            if r_fp < c_fp or r_rec > c_rec or r_prec > c_prec:
                return True
    return False


def select_operating_point(
    sweep_results: List[Dict[str, Any]],
    policy: OperatingPointPolicy,
) -> OperatingPointResult:
    """Select the best operating point from a list of sweep result rows.

    - If policy.fp_per_hour_target is set:
      filter to rows where fp_per_hour <= target (or fp_per_hour is None,
      meaning no FPs occurred), then pick the row with the best recall
      (ties broken by higher precision).
    - If policy.min_recall is set: additionally filter to rows where
      recall >= min_recall.
    - If no constraint (fp_per_hour_target is None and min_recall is None):
      compute the Pareto set (rows not dominated in precision + recall +
      low-FP space), return with is_pareto_candidate=True and pareto_set
      populated.
    - Raises ValueError if no rows satisfy the active constraint.
    """
    valid_rows = [
        r for r in sweep_results
        if r.get("metrics") is not None and "error" not in r
    ]
    if not valid_rows:
        raise ValueError("select_operating_point: no valid sweep rows to select from")

    has_fp_target = policy.fp_per_hour_target is not None
    has_recall_floor = policy.min_recall is not None

    if has_fp_target or has_recall_floor:
        # Apply FP-rate filter: fp_per_hour <= target, or fp_per_hour is None (no FPs)
        if has_fp_target:
            target = policy.fp_per_hour_target
            valid_rows = [
                r for r in valid_rows
                if (r["metrics"].fp_per_hour is None)
                or (r["metrics"].fp_per_hour <= target)
            ]
        # Apply recall floor
        if has_recall_floor:
            floor = policy.min_recall
            valid_rows = [
                r for r in valid_rows
                if (r["metrics"].recall is not None)
                and (r["metrics"].recall >= floor)
            ]

        if not valid_rows:
            raise ValueError(
                "select_operating_point: no sweep row satisfies the active constraint "
                f"(fp_per_hour_target={policy.fp_per_hour_target}, "
                f"min_recall={policy.min_recall})"
            )

        # Best recall; tie-break by higher precision; final tie-break by lower fp_per_hour.
        def _rank(r: Dict[str, Any]) -> tuple:
            m: CalibrationMetrics = r["metrics"]
            recall_v = m.recall if m.recall is not None else -1.0
            prec_v   = m.precision if m.precision is not None else -1.0
            fph_v    = m.fp_per_hour if m.fp_per_hour is not None else 0.0
            return (-recall_v, -prec_v, fph_v)

        valid_rows.sort(key=_rank)
        best = valid_rows[0]
        reason_parts = []
        if has_fp_target:
            reason_parts.append(f"fp_per_hour <= {policy.fp_per_hour_target}")
        if has_recall_floor:
            reason_parts.append(f"recall >= {policy.min_recall}")
        reason = "best recall among " + " AND ".join(reason_parts) if reason_parts else "best recall"
        reason += f" (from {len(valid_rows)} qualifying rows)"
        return OperatingPointResult(
            selected_combination={k: v for k, v in best.items() if k not in ("metrics", "split", "error")},
            metrics=best["metrics"],
            reason=reason,
            is_pareto_candidate=False,
            pareto_set=[],
        )

    # No constraint: compute Pareto frontier.
    pareto: List[Dict[str, Any]] = []
    for candidate in valid_rows:
        if not _is_dominated(candidate, pareto):
            pareto.append(candidate)

    if not pareto:
        raise ValueError("select_operating_point: Pareto set is empty (all rows dominated)")

    # Deterministic ordering: sort by recall desc, then precision desc.
    def _pareto_key(r: Dict[str, Any]) -> tuple:
        m: CalibrationMetrics = r["metrics"]
        return (
            -(m.recall if m.recall is not None else -1.0),
            -(m.precision if m.precision is not None else -1.0),
        )
    pareto.sort(key=_pareto_key)

    best = pareto[0]
    pareto_combos = [
        {k: v for k, v in r.items() if k not in ("metrics", "split", "error")}
        for r in pareto
    ]
    reason = f"Pareto frontier with {len(pareto)} non-dominated row(s); selected best-recall member"
    return OperatingPointResult(
        selected_combination={k: v for k, v in best.items() if k not in ("metrics", "split", "error")},
        metrics=best["metrics"],
        reason=reason,
        is_pareto_candidate=True,
        pareto_set=pareto_combos,
    )


# ---------------------------------------------------------------------------
# compute_subgroup_metrics
# ---------------------------------------------------------------------------

def compute_subgroup_metrics(
    gt_dataset: GroundTruthDataset,
    artifact: PredictionArtifact,
    match_results: List[MatchResult],
) -> Dict[str, Dict[str, Any]]:
    """Compute per-scene-tag subgroup metrics.

    Uses SourceVideoMetadata.scene_tags from the GT dataset.
    Returns {'scene_tag_name': {tag, tp, fp, fn, precision, recall, video_count}}
    for each tag actually present in the dataset.
    Tags not in the dataset are omitted (no fabricated empty groups).
    """
    # Build tag → set of video_ids that carry that tag.
    tag_to_videos: Dict[SceneTag, set] = {}
    video_to_tags: Dict[str, List[SceneTag]] = {}
    for video in gt_dataset.all_videos():
        for tag in video.scene_tags:
            tag_to_videos.setdefault(tag, set()).add(video.video_id)
            video_to_tags.setdefault(video.video_id, []).append(tag)

    # Build lookups: prediction_id → video_id, and GT incident_id → video_id.
    pred_vid: Dict[str, str] = {p.prediction_id: p.video_id for p in artifact.records}
    gt_vid_lookup: Dict[str, str] = {
        inc.incident_id: inc.video_id for inc in gt_dataset.all_incidents()
    }

    # Per-tag counters.
    tag_counts: Dict[SceneTag, Dict[str, int]] = {
        tag: {"tp": 0, "fp": 0, "fn": 0} for tag in tag_to_videos
    }
    tag_videos_seen: Dict[SceneTag, set] = {tag: set() for tag in tag_to_videos}

    for r in match_results:
        # Determine the video_id for this result.
        vid: Optional[str] = None
        if r.prediction_id is not None and r.prediction_id in pred_vid:
            vid = pred_vid[r.prediction_id]
        elif r.ground_truth_id is not None and r.ground_truth_id in gt_vid_lookup:
            vid = gt_vid_lookup[r.ground_truth_id]
        if vid is None:
            continue

        tags = video_to_tags.get(vid, [])
        for tag in tags:
            counts = tag_counts[tag]
            if r.outcome == EvaluationOutcome.TP:
                counts["tp"] += 1
            elif r.outcome == EvaluationOutcome.FP:
                counts["fp"] += 1
            elif r.outcome == EvaluationOutcome.FN:
                counts["fn"] += 1
            tag_videos_seen[tag].add(vid)

    result: Dict[str, Dict[str, Any]] = {}
    for tag in sorted(tag_to_videos, key=lambda t: t.value):
        counts = tag_counts[tag]
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
        result[tag.value] = {
            "tag": tag.value,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "video_count": len(tag_videos_seen.get(tag, set())),
        }

    return result


# ---------------------------------------------------------------------------
# build_calibration_report
# ---------------------------------------------------------------------------

def build_calibration_report(
    gt_dataset: GroundTruthDataset,
    artifact: PredictionArtifact,
    matching_sweep: Optional[List[Dict[str, Any]]] = None,
    matching_selected: Optional[Dict[str, Any]] = None,
    confirmation_sweep: Optional[List[Dict[str, Any]]] = None,
    confirmation_selected: Optional[Dict[str, Any]] = None,
    candidate_config: Optional[CandidateEvaluationConfig] = None,
) -> CalibrationReport:
    """Build a CalibrationReport from sweep results.

    Deterministic except for timestamp_utc (isolated for reproducibility
    testing).  parameter_status is derived from which parameters were
    actually swept:
      - swept → "calibrated"
      - not swept → "not_calibrated"
    """
    from datetime import datetime, timezone

    # Dataset version: first manifest's version, or "unknown".
    dataset_version: str = "unknown"
    if gt_dataset.manifests:
        versions = [m.version for m in gt_dataset.manifests]
        dataset_version = versions[0] if len(versions) == 1 else ";".join(versions)

    # Determine which parameters were swept.
    matching_params: List[str] = []
    if matching_sweep:
        row0 = matching_sweep[0]
        matching_params = sorted(
            k for k in row0 if k not in ("metrics", "split", "error")
        )

    confirmation_params: List[str] = []
    if confirmation_sweep:
        row0 = confirmation_sweep[0]
        confirmation_params = sorted(
            k for k in row0 if k not in ("metrics", "split", "error")
        )

    # parameter_status: swept → "calibrated", not swept → "not_calibrated"
    parameter_status: Dict[str, str] = {}
    for p in matching_params:
        parameter_status[f"matching.{p}"] = "calibrated"
    for p in confirmation_params:
        parameter_status[f"confirmation.{p}"] = "calibrated"
    # Parameters that were not swept default to "not_calibrated"
    if not matching_sweep:
        for p in ("min_overlap_ratio", "overlap_metric"):
            parameter_status.setdefault(f"matching.{p}", "not_calibrated")
    if not confirmation_sweep:
        for p in ("min_confidence_mean",):
            parameter_status.setdefault(f"confirmation.{p}", "not_calibrated")

    # Build candidate_config if not supplied.
    if candidate_config is None:
        matching_cfg: Dict[str, Any] = {}
        for p in matching_params:
            matching_cfg[p] = (
                matching_selected.get(p) if matching_selected else "not_calibrated"
            )
        matching_cfg["status"] = "calibrated" if matching_selected else "not_calibrated"

        confirmation_cfg: Dict[str, Any] = {}
        for p in confirmation_params:
            confirmation_cfg[p] = (
                confirmation_selected.get(p) if confirmation_selected else "not_calibrated"
            )
        confirmation_cfg["status"] = "calibrated" if confirmation_selected else "not_calibrated"
        candidate_config = CandidateEvaluationConfig(
            matching=matching_cfg,
            confirmation=confirmation_cfg,
        )

    # Build split_definition from the first sweep row if available.
    split_definition: Dict[str, Any] = {}
    if matching_sweep:
        split_name = matching_sweep[0].get("split", "")
        split_definition = {"split_name": split_name}
    elif confirmation_sweep:
        split_name = confirmation_sweep[0].get("split", "")
        split_definition = {"split_name": split_name}

    # Serialize sweep rows (CalibrationMetrics → dict).
    def _serialize_rows(
        rows: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if rows is None:
            return None
        out: List[Dict[str, Any]] = []
        for row in rows:
            r: Dict[str, Any] = dict(row)
            m = r.get("metrics")
            if isinstance(m, CalibrationMetrics):
                r["metrics"] = m.to_dict()
            out.append(r)
        return out

    serialized_matching = _serialize_rows(matching_sweep)
    serialized_confirmation = _serialize_rows(confirmation_sweep)

    # Serialize selected combinations (strip metrics/split/error keys).
    def _serialize_selected(
        sel: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if sel is None:
            return None
        return {k: v for k, v in sel.items() if k not in ("metrics", "split", "error")}

    return CalibrationReport(
        calibration_version=CALIBRATION_VERSION,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        dataset_version=dataset_version,
        artifact_version=artifact.artifact_version,
        artifact_identifier=artifact.source_identifier,
        split_definition=split_definition,
        matching_grid={"parameters": {p: "swept" for p in matching_params}},
        matching_results=serialized_matching if serialized_matching else [],
        matching_selected=_serialize_selected(matching_selected),
        confirmation_grid=(
            {"parameters": {p: "swept" for p in confirmation_params}}
            if confirmation_params else None
        ),
        confirmation_results=serialized_confirmation,
        confirmation_selected=_serialize_selected(confirmation_selected),
        candidate_configuration=candidate_config.to_dict(),
        parameter_status=parameter_status,
    )


# ---------------------------------------------------------------------------
# sweep_confirmation_parameters
# ---------------------------------------------------------------------------

def sweep_confirmation_parameters(
    gt_dataset: GroundTruthDataset,
    artifact: PredictionArtifact,
    grid: CalibrationParameterGrid,
    split: CalibrationSplit,
    *,
    total_video_hours: Optional[float] = None,
    matching_policy: Optional[MatchingPolicy] = None,
) -> List[Dict[str, Any]]:
    """Sweep confirmation parameters over a grid.

    For each combination:
      1. Filter predictions to split.videos.
      2. Apply confirmation gate: a prediction passes if
         event_confidence is not None AND event_confidence >= min_confidence_mean
         (when min_confidence_mean is present in the combination).
      3. Run matching on the filtered predictions using matching_policy
         (default: MatchingPolicy()).
      4. Compute metrics.

    Returns a deterministic list of result rows.
    """
    if total_video_hours is None:
        total_video_hours = _default_total_video_hours(gt_dataset)
    if matching_policy is None:
        matching_policy = MatchingPolicy()

    filt = _filter_to_split(gt_dataset, artifact, split)
    all_rows: List[Dict[str, Any]] = []

    for combo in grid.combinations():
        min_conf = combo.get("min_confidence_mean")
        filtered_preds: Dict[str, List[PredictionRecord]] = {}
        for vid, recs in filt["preds_by_video"].items():
            if min_conf is not None:
                filtered_preds[vid] = [
                    p for p in recs
                    if p.event_confidence is not None and p.event_confidence >= min_conf
                ]
            else:
                filtered_preds[vid] = recs

        all_match_results: List[MatchResult] = []
        all_latencies: List[Optional[float]] = []
        per_video_metrics: List[Dict[str, Any]] = []

        for vid in filt["video_ids"]:
            preds = filtered_preds.get(vid, [])
            gts = filt["gt_by_video"].get(vid, [])
            norms = filt["norm_by_video"].get(vid, [])
            ambs = filt["amb_by_video"].get(vid, [])

            results = _run_match_for_video(vid, preds, gts, norms, ambs, matching_policy)
            all_match_results.extend(results)
            all_latencies.extend(_collect_latencies(results, preds, gts))

            summary = compute_evaluation_summary(results)
            per_video_metrics.append({
                "video_id": vid,
                "tp": summary.tp_count,
                "fp": summary.fp_count,
                "fn": summary.fn_count,
                "duplicate": summary.duplicate_count,
                "ambiguous": summary.ambiguous_count,
                "precision": summary.precision,
                "recall": summary.recall,
                "f1": summary.f1,
            })

        metrics = compute_calibration_metrics(
            match_results=all_match_results,
            total_video_hours=total_video_hours,
            latencies=all_latencies,
            total_videos=len(filt["video_ids"]),
            total_gt_incidents=sum(len(g) for g in filt["gt_by_video"].values()),
            total_predictions=sum(len(p) for p in filtered_preds.values()),
            per_video_metrics=per_video_metrics,
        )

        all_rows.append({
            **combo,
            "metrics": metrics,
            "split": split.split_name,
        })

    return all_rows
