# Phase 2C.4 — Deterministic Calibration Runner

**Status:** IN PROGRESS — 43 tests in `tests/test_phase2c4_calibration.py`
**Created:** 2026-10-15
**Owner:** A

---

## 1. Purpose

Phase 2C.4 implements the offline calibration runner that:
- Sweeps matching and confirmation parameters over a deterministic grid.
- Computes per-combination `CalibrationMetrics` from Phase 2C.2 `MatchResult` lists.
- Selects an operating point (recall-first with optional FP-rate and recall constraints, or a Pareto set when no constraints are set).
- Computes per-scene-tag subgroup metrics.
- Builds a JSON-serializable `CalibrationReport` with provenance.

What this module intentionally does NOT do:
- No production threshold writes or inference behaviour changes.
- No OpenCV, YOLO, or GPU imports.
- No randomness: all splits and grid orderings are deterministic (SHA-256 bucketing).
- No wall-clock time in metric values (timestamp is isolated in `CalibrationReport.timestamp_utc`).

> **Important:** `min_overlap_ratio=0.5` remains provisional and uncalibrated
> until a calibration sweep is run and an operating point is selected.

---

## 2. Files Touched

| File | Change |
|------|--------|
| `inference/evaluation/calibration.py` | **NEW** — all dataclasses and functions |
| `tests/test_phase2c4_calibration.py` | **NEW** — 43 unit tests |
| `inference/evaluation/matching.py` | Untouched (Phase 2C.2 regression passes) |
| `inference/evaluation/ground_truth.py` | Untouched |
| `inference/evaluation/runner.py` | Untouched (Phase 2C.3 regression passes) |
| `inference/core/events.py` | Untouched |
| `inference/config/shadow_config.py` | Untouched |
| `inference/core/pipeline.py` | Untouched |

---

## 3. Data Structures

### 3.1 `CalibrationParameterGrid`

A frozen dataclass holding `parameters: Dict[str, List[Any]]`.
`combinations()` returns the lexicographic Cartesian product over sorted
parameter names — deterministic, no randomness.

Validation:
- `parameters` must be a non-empty dict.
- Each parameter name must be a non-empty string.
- Each value list must be a non-empty Python list.

### 3.2 `CalibrationMetrics`

Per-combination metrics computed from a `List[MatchResult]`:

| Field | Type | Notes |
|-------|------|-------|
| `tp`, `fp`, `fn`, `duplicates`, `ambiguous` | `int` | Counts |
| `precision` | `Optional[float]` | `None` when `tp+fp == 0` |
| `recall` | `Optional[float]` | `None` when `tp+fn == 0` |
| `f1` | `Optional[float]` | `None` when `precision` or `recall` is `None` |
| `duplicate_rate` | `Optional[float]` | `None` when `tp+fp+duplicates == 0` |
| `fp_per_hour` | `Optional[float]` | `None` when `total_video_hours == 0` |
| `fn_per_hour` | `Optional[float]` | `None` when `total_video_hours == 0` |
| `latency_count` | `int` | Count of non-`None` latency values |
| `latency_mean`, `latency_median` | `Optional[float]` | `None` when count is 0 |
| `latency_p95` | `Optional[float]` | `None` when count < 10 |
| `per_video_metrics` | `List[Dict]` | Per-video `EvaluationSummary` dicts |

### 3.3 `CalibrationSplit`

A video-level split with `split_name` and `videos` (sorted list of video_ids).
All incidents and predictions from a video stay in the same split (no leakage).

Created via `CalibrationSplit.create(video_ids, train_frac, val_frac, seed)`.

### 3.4 `OperatingPointPolicy` (frozen)

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `fp_per_hour_target` | `Optional[float]` | `None` | Max acceptable FP/hour rate |
| `min_recall` | `Optional[float]` | `None` | Minimum acceptable recall |

When both are `None`, a Pareto set is returned instead of a single selection.

### 3.5 `OperatingPointResult`

| Field | Type |
|-------|------|
| `selected_combination` | `Dict[str, Any]` |
| `metrics` | `CalibrationMetrics` |
| `reason` | `str` |
| `is_pareto_candidate` | `bool` |
| `pareto_set` | `List[Dict[str, Any]]` |

### 3.6 `CandidateEvaluationConfig`

A dict-of-dicts container: `matching` and `confirmation` sub-dicts, each
carrying a `status` field (`"calibrated"`, `"provisional"`, or `"not_calibrated"`).

### 3.7 `CalibrationReport`

The full JSON-serializable report. See `to_dict()` / `to_json()` / `from_dict()`.

---

## 4. Functions

### 4.1 `split_videos(video_ids, *, train_frac=0.6, val_frac=0.2, seed=42)`

Deterministic video-level split using SHA-256 bucketing:

```python
bucket = int.from_bytes(sha256(f"{seed}:{vid}").digest()[:4], "big") % 100
```

- `bucket < train_frac*100` → `train`
- `train_frac*100 <= bucket < (train_frac+val_frac)*100` → `validation`
- else → `holdout`

Fewer than 3 videos: all go to `train`; `validation` and `holdout` are empty.
Guarantee: no video appears in more than one split.

### 4.2 `compute_calibration_metrics(match_results, ...)`

Computes `CalibrationMetrics` from a `List[MatchResult]`.

Edge cases:
- `precision` is `None` when `tp+fp == 0`
- `recall` is `None` when `tp+fn == 0`
- `f1` is `None` when either `precision` or `recall` is `None`
- `fp_per_hour` / `fn_per_hour` are `None` when `total_video_hours == 0`
- `latency_p95` is `None` when fewer than 10 valid latency samples
- `None` latency values are excluded from statistics

### 4.3 `sweep_matching_parameters(gt_dataset, artifact, grid, split, *, total_video_hours=None)`

For each combination in `grid.combinations()`:
1. Filters predictions and GT incidents to `split.videos`.
2. Builds a `MatchingPolicy` from the combination's keys.
3. Runs per-video matching.
4. Computes `CalibrationMetrics`.
5. Appends a row: `{**combination, "metrics": CalibrationMetrics, "split": split_name}`.

Invalid combinations (e.g. out-of-range `min_overlap_ratio`) produce rows with
`"error"` and `"metrics": None`.

### 4.4 `sweep_confirmation_parameters(gt_dataset, artifact, grid, split, *, total_video_hours=None, matching_policy=None)`

Same structure as `sweep_matching_parameters` but applies a confidence gate:
- A prediction passes when `event_confidence is not None AND event_confidence >= min_confidence_mean`.
- Predictions with `event_confidence=None` are always excluded when a gate is active.

### 4.5 `select_operating_point(sweep_results, policy)`

Selection logic:

- **No constraints** (`fp_per_hour_target=None, min_recall=None`):
  Compute Pareto set (non-dominated on `fp_per_hour ↓`, `recall ↑`, `precision ↑`).
  Returns `is_pareto_candidate=True` with `pareto_set` populated.

- **Constraints active**:
  Filter to qualifying rows:
  - `fp_per_hour is None OR fp_per_hour <= fp_per_hour_target`
  - `recall is not None AND recall >= min_recall`

  Among qualifiers, select the row with:
  1. Highest recall
  2. Tie: highest precision
  3. Tie: lowest `fp_per_hour` (None treated as 0)

  Returns `is_pareto_candidate=False`.

- **Raises `ValueError`** when no valid rows exist, or no row satisfies the active constraints.

### 4.6 `compute_subgroup_metrics(gt_dataset, artifact, match_results)`

Computes per-`SceneTag` TP/FP/FN counts by mapping each `MatchResult` back to
its video_id (via `PredictionRecord.video_id` or `GroundTruthIncident.video_id`)
and then to that video's `SceneTag` list.

Returns `{tag_value_str: {"tag", "video_count", "tp", "fp", "fn", "precision", "recall"}}`.
Only tags actually present in the dataset appear.

### 4.7 `build_calibration_report(gt_dataset, artifact, matching_sweep, matching_selected, confirmation_sweep, confirmation_selected, candidate_config)`

Assembles a `CalibrationReport`.

Provenance:
- `calibration_version`: `"1.0"`
- `dataset_version`: first manifest version, or semicolon-joined when multiple
- `artifact_version`: from `PredictionArtifact.artifact_version`
- `artifact_identifier`: from `PredictionArtifact.source_identifier`
- `timestamp_utc`: ISO-8601 UTC (the only non-deterministic field)

`parameter_status` is derived:
- Parameters in `matching_sweep[0]` → `"calibrated"`
- Parameters in `confirmation_sweep[0]` → `"calibrated"`
- Unswept parameters default to `"not_calibrated"`

---

## 5. Split Leakage Guarantee

`split_videos` uses SHA-256 bucketing on `(seed, video_id)` pairs. This ensures:
- The same video always lands in the same bucket for a given seed.
- All incidents and predictions for a video stay in the same split.
- Different video_ids are statistically independent.

No cross-video leakage is possible by construction.

---

## 6. Calibration Readiness

To calibrate `min_overlap_ratio` for your dataset:

```python
from inference.evaluation.calibration import (
    CalibrationParameterGrid,
    CalibrationSplit,
    OperatingPointPolicy,
    build_calibration_report,
    select_operating_point,
    split_videos,
    sweep_matching_parameters,
)

split_dict = split_videos(dataset.video_ids(), seed=42)
split = CalibrationSplit(split_name="train", videos=split_dict["train"])

grid = CalibrationParameterGrid(
    parameters={
        "min_overlap_ratio": [0.3, 0.4, 0.5, 0.6, 0.7],
        "overlap_metric": ["gt_ratio"],
    }
)

sweep = sweep_matching_parameters(dataset, artifact, grid, split)
selected = select_operating_point(sweep, OperatingPointPolicy(
    fp_per_hour_target=5.0,
    min_recall=0.7,
))

report = build_calibration_report(
    gt_dataset=dataset,
    artifact=artifact,
    matching_sweep=sweep,
    matching_selected=selected,
)
print(report.to_json())
```

---

## 7. Known Limitations

1. `compute_subgroup_metrics` maps `MatchResult` to a video via prediction_id
   or GT incident_id; a result with `prediction_id=None` and `ground_truth_id=None`
   is skipped.
2. `sweep_matching_parameters` silently skips invalid `MatchingPolicy` combinations
   (rows with `"error"` key); they do not participate in operating-point selection.
3. `fp_per_hour=None` (no FPs observed) is treated as satisfying any FP/hour target
   constraint in `select_operating_point`, but as worst-case (`inf`) in the Pareto
   frontier calculation.
4. The timestamp field makes `CalibrationReport` non-deterministic at the byte level;
   semantic equality tests should exclude `timestamp_utc`.

---

## 8. Change Log

| Date | Entry |
|------|-------|
| 2026-10-15 | Created `inference/evaluation/calibration.py` and `tests/test_phase2c4_calibration.py`; 43 tests green |
