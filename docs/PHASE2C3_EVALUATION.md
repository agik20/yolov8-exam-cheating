# Phase 2C.3 — Offline Evaluation Runner

**Status:** IN PROGRESS — 31 tests in `tests/test_phase2c3_evaluation.py`
**Created:** 2026-10-15
**Owner:** A

---

## 1. Purpose

Phase 2C.3 implements the offline evaluation runner that consumes:
- a Phase 2C.1 `GroundTruthDataset`, and
- a Phase 2C.3 `PredictionArtifact` (offline prediction records)

and produces a deterministic, machine-readable `EvaluationReport`.

What this module does:
- Defines `PredictionRecord`: a JSON-serializable prediction artifact record
  that preserves all information needed for later threshold sweeps.
- Defines `PredictionArtifact`: a versioned container of `PredictionRecord`
  objects with provenance metadata and structural validation.
- Defines `LatencyDistribution`, `VideoEvaluationResult`,
  `AggregateEvaluationResult`, and `EvaluationReport`.
- Implements `run_evaluation`: the main entry point that groups predictions
  and GT incidents by `video_id` (hard per-video partition boundary), calls
  the Phase 2C.2 matching engine per video, and produces a complete report.
- Implements `human_readable_summary`: a concise terminal-display string.

What this module intentionally does NOT do:
- No threshold calibration (that is Phase 2C.4).
- No changes to production inference behaviour.
- No use of wall-clock time in metric values.
- No OpenCV, YOLO, or GPU imports.

> **Important:** `min_overlap_ratio=0.5` remains provisional and uncalibrated.
> No calibration has occurred. The provisional status is explicitly visible
> via `MatchingPolicy.is_provisional` and is snapshotted into
> `EvaluationReport.matching_config`.

---

## 2. Files Touched

| File | Change |
|------|--------|
| `inference/evaluation/runner.py` | **NEW** — all dataclasses, `run_evaluation`, `human_readable_summary` |
| `tests/test_phase2c3_evaluation.py` | **NEW** — 31 unit tests |
| `inference/evaluation/matching.py` | Untouched (Phase 2C.2 regression passes) |
| `inference/evaluation/ground_truth.py` | Untouched |
| `inference/core/events.py` | Untouched |
| `inference/config/shadow_config.py` | Untouched |
| `inference/core/pipeline.py` | Untouched |

---

## 3. Input Artifacts

### 3.1 `PredictionRecord`

A single prediction in a `PredictionArtifact`. JSON-serializable.

| Field | Type | Notes |
|-------|------|-------|
| `prediction_id` | `str` | Required, non-empty. |
| `video_id` | `str` | Required, non-empty. |
| `start_video_time` | `Optional[float]` | `None` when unavailable. |
| `end_video_time` | `Optional[float]` | `None` when unavailable. |
| `class_name` | `Optional[str]` | Model class name; `None` when not recorded. |
| `identity` | `Optional[str]` | Entity key string; `None` when not recorded. |
| `confirmation_video_time` | `Optional[float]` | Used for latency; `None` when unavailable. |
| `event_confidence` | `Optional[float]` | Raw model confidence; `None` when not recorded. |
| `boundary_type` | `str` | One of `PredictionBoundary` enum values. |
| `source_metadata` | `Optional[Dict]` | Free-form provenance from inference. |

### 3.2 `PredictionArtifact`

A versioned collection of `PredictionRecord` objects.

| Field | Default | Meaning |
|-------|---------|---------|
| `artifact_version` | `"1.0"` | Serialization format version (not model version). |
| `source_identifier` | `"unknown"` | Human-readable label (shadow run ID, export name). |
| `records` | `[]` | List of `PredictionRecord`. |
| `notes` | `None` | Free-text. |

Helper methods: `records_by_video()`, `video_ids()`, `validate()`,
`to_dict()` / `from_dict()`, `to_json()` / `from_json()`, `from_file()`.

### 3.3 `GroundTruthDataset`

A Phase 2C.1 `GroundTruthDataset` wrapping one or more `DatasetManifest`
objects. The runner uses `all_incidents()`, `all_videos()`, and
`manifests` to access GT data.

---

## 4. `run_evaluation`

```python
run_evaluation(
    gt_dataset: GroundTruthDataset,
    artifact:   PredictionArtifact,
    policy:     Optional[MatchingPolicy] = None,   # defaults to MatchingPolicy()
    *,
    validation_errors: Optional[List[str]] = None, # pre-collected GT errors
) -> EvaluationReport
```

### 4.1 Per-video partitioning (spec §19)

Predictions and GT incidents are grouped by `video_id` before matching.
Cross-video matching is **impossible** by construction — a prediction for
video A can never match a GT incident in video B. The union of all video
IDs (from both GT and predictions) determines which `VideoEvaluationResult`
objects appear in the report.

### 4.2 Fatal errors

| Condition | Behaviour |
|-----------|-----------|
| GT dataset has no videos AND validation errors exist | `ValueError` raised |
| Prediction artifact has `MISSING` or `DUPLICATE` prediction_id errors | `ValueError` raised |
| GT dataset has no videos but no validation errors | Empty report (valid edge case) |
| Artifact is empty | Zero-count report (valid) |

### 4.3 Aggregate metrics (spec §10)

Dataset-level precision/recall/F1 are computed from **summed counts**,
not averaged per-video ratios:

```
precision = TP_total / (TP_total + FP_total)   # None if denominator is 0
recall    = TP_total / (TP_total + FN_total)   # None if denominator is 0
F1        = 2*p*r / (p+r)                      # None if p or r is None
dup_rate  = DUPLICATE_total / (TP_total + FP_total + DUPLICATE_total)
```

### 4.4 Latency (spec §13)

Latency is computed **per video** for TP results only:
`confirmation_video_time - gt.start_video_time` (video seconds).
`None` values (missing confirmation time) are excluded from statistics.
Negative latencies are valid (pre-emptive detection).
Aggregate latency is computed from all TP latencies across all videos.

---

## 5. `EvaluationReport` Structure

```
EvaluationReport
├── schema_version: str          # "1.0"
├── provenance: Dict            # generated_at_utc, GT/artifact versions
├── matching_config: Dict       # MatchingPolicy snapshot
├── videos: List[VideoEvaluationResult]
│       ├── video_id
│       ├── prediction_count
│       ├── gt_incident_count
│       ├── summary: EvaluationSummary
│       ├── latency: LatencyDistribution
│       └── match_results: List[MatchResult]
├── aggregate: AggregateEvaluationResult
│       ├── video_count, total_predictions, total_gt_incidents
│       ├── tp_count, fp_count, fn_count, duplicate_count, ambiguous_count
│       ├── precision, recall, f1, duplicate_rate   # Optional[float]
│       └── latency: LatencyDistribution
└── records: List[MatchResult]  # flat list across all videos
```

### 5.1 Provenance keys

| Key | Description |
|-----|-------------|
| `evaluation_schema_version` | `"1.0"` |
| `gt_dataset_versions` | List of `DatasetManifest.version` strings |
| `prediction_artifact_version` | `artifact.artifact_version` |
| `prediction_source_identifier` | `artifact.source_identifier` |
| `generated_at_utc` | ISO-8601 UTC timestamp (non-deterministic) |
| `gt_validation_errors` | Present only when non-empty |
| `artifact_validation_errors` | Present only when non-empty |

### 5.2 Matching config keys

| Key | Source |
|-----|--------|
| `min_overlap_ratio` | `policy.min_overlap_ratio` |
| `overlap_metric` | `policy.overlap_metric` |
| `require_class_match` | `policy.require_class_match` |
| `identity_mode` | `policy.identity_mode` |
| `ambiguous_handling` | `policy.ambiguous_handling` |
| `is_provisional` | `policy.is_provisional` |

---

## 6. Ambiguous and Duplicate Handling

### 6.1 AMBIGUOUS GT (default `ambiguous_handling="exclude"`)

AMBIGUOUS `GroundTruthIncident` objects produce
`AMBIGUOUS_GROUND_TRUTH` results. They are excluded from all TP/FN
denominators. `ambiguous_count` is reported separately.

### 6.2 Duplicates

When multiple predictions overlap the same GT incident, the first
(by sort order) is the primary TP; subsequent ones produce
`DUPLICATE` results. `duplicate_rate = DUPLICATE / (TP + FP + DUPLICATE)`.

### 6.3 Normal periods

`NormalPeriod` objects are passed to the matching engine per video.
FP results that fall inside a normal period carry an explicit reason
string mentioning the normal period.

### 6.4 Ambiguous periods

`AmbiguousPeriod` objects are passed to the matching engine per video.
They do not affect the outcome classification but are available for
downstream analysis.

---

## 7. Calibration Readiness

The same `PredictionArtifact` can be re-evaluated with different
`MatchingPolicy` configurations without re-running inference:

```python
# Re-run with a different threshold — no inference re-run needed
report_alt = run_evaluation(dataset, artifact, policy=MatchingPolicy(min_overlap_ratio=0.7))
```

`PredictionRecord.event_confidence` is preserved in the report's flat
`records` list, enabling Phase 2C.4 to sweep confidence thresholds
offline.

---

## 8. `human_readable_summary`

Returns a multi-line string suitable for terminal display. Contains:
- Evaluation heading, schema version, provisional flag
- Video / prediction / GT incident counts
- TP / FP / FN / duplicate / ambiguous counts
- Precision, recall, F1, duplicate rate (or "None (undefined)")
- Latency statistics (count, mean, median, min, max) or "no valid values"

The JSON report remains the authoritative machine-readable artifact.
This summary is for human review only.

---

## 9. Known Limitations

1. `min_overlap_ratio=0.5` is provisional and uncalibrated.
2. No automatic threshold calibration in this phase.
3. `EvaluationReport.from_dict()` does not fully round-trip
   `source_metadata` inside `PredictionRecord` (it is preserved in the
   JSON but not restored to typed sub-fields).
4. `generated_at_utc` in provenance makes the report non-deterministic
   at the byte level; semantic equality tests should exclude this key.
5. `PredictionRecord.from_event_record_dict` (adapter from serialized
   `EventRecord`) is a convenience helper; `video_id` must be supplied
   by the caller (it is not embedded in `EventRecord`).

---

## 10. Change Log

| Date | Entry |
|------|-------|
| 2026-10-15 | Created `inference/evaluation/runner.py` and `tests/test_phase2c3_evaluation.py`; 31 tests green |
