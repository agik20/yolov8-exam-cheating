# Phase 2C.2 — Deterministic Prediction-vs-Ground-Truth Matching

**Status:** COMPLETE — 60 tests in `tests/test_phase2c2_matching.py`, full suite 355 passing
**Created:** 2026-10-13
**Owner:** A

---

## 1. Purpose

Phase 2C.2 implements the offline evaluation matching layer that compares
model predictions (from Phase 2B `EventRecord` objects) against the
Phase 2C.1 ground-truth annotation dataset.  It is strictly an evaluation
tool: it never modifies production inference behaviour, never invents a
calibrated threshold, and never writes to production output paths.

What this module does:
- Convert `EventRecord` to `NormalizedPrediction` (evaluation adapter).
- Compute deterministic interval-overlap metrics (overlap duration, ratios).
- Classify temporal, class, and identity match status per pair.
- Run one-to-one greedy assignment with documented lexicographic priority.
- Produce a complete `EvaluationSummary` (precision, recall, F1, duplicate
  rate) with documented denominator rules and `None` (not 0.0) for
  undefined values.
- Compute confirmation latency in video-time.

What this module intentionally does NOT do:
- No threshold calibration (that is Phase 2C.3+).
- No changes to `EventRecord`, `Candidate`, `EvidenceBundle`, or any
  production data structure.
- No use of wall-clock time.
- No filesystem I/O.
- No model imports beyond `inference.core.events` and
  `inference.evaluation.ground_truth`.

---

## 2. Files Touched

| File | Change |
|------|--------|
| `inference/evaluation/matching.py` | **NEW** — all dataclasses, enums, and functions |
| `tests/test_phase2c2_matching.py` | **NEW** — 60 unit tests |
| `inference/evaluation/ground_truth.py` | Untouched (Phase 2C.1 regression passes) |
| `inference/core/events.py` | Untouched |
| `inference/config/shadow_config.py` | Untouched |

---

## 3. Data Model

### 3.1 `PredictionBoundary` (enum)

| Value | Meaning |
|-------|---------|
| `EXACT_INTERVAL` | Both start and end video times are known. |
| `POINT_EVENT` | Only the confirmation/single video time is known; `start_video_time=None`, `end_video_time` holds the point time. |
| `UNAVAILABLE` | No usable temporal information; both times are `None`. |

### 3.2 `EvaluationOutcome` (enum)

| Value | Meaning |
|-------|---------|
| `TP` | Prediction matched a GT incident (primary match). |
| `FP` | Prediction matched no eligible GT; it is a false positive. |
| `FN` | GT incident had no eligible prediction; it is a false negative. |
| `DUPLICATE` | A second (or later) prediction matched the same GT already consumed by a primary TP. |
| `AMBIGUOUS_GROUND_TRUTH` | GT incident is AMBIGUOUS and the policy excludes it from TP/FN accounting. |
| `UNMATCHED` | Pair did not meet the temporal threshold under the current policy. |

### 3.3 `ClassMatchStatus` (enum)

| Value | Meaning |
|-------|---------|
| `CLASS_MATCH` | `pred.class_name == gt.behavior_class`. |
| `CLASS_MISMATCH` | Both sides have a class; they differ. |
| `CLASS_UNAVAILABLE` | `pred.class_name is None`. |
| `CLASS_AMBIGUOUS` | GT label is `AMBIGUOUS`. |

### 3.4 `IdentityMatchStatus` (enum)

| Value | Meaning |
|-------|---------|
| `IDENTITY_MATCH` | Both sides have a non-None identity/reference and they are equal. |
| `IDENTITY_MISMATCH` | Both sides have a non-None value and they differ. |
| `IDENTITY_UNAVAILABLE` | Either side has no usable identity value. |

### 3.5 `TemporalMatchStatus` (enum)

| Value | Meaning |
|-------|---------|
| `MATCHED` | Overlap ratio meets or exceeds the policy threshold. |
| `UNMATCHED` | Overlap ratio is below the threshold. |
| `UNAVAILABLE` | Ratio could not be computed (zero-duration denominator). |

### 3.6 `NormalizedPrediction` (frozen dataclass)

Fields: `prediction_id`, `video_id`, `start_video_time`, `end_video_time`,
`class_name`, `identity`, `confirmation_video_time`, `event_confidence`,
`boundary_type`, `source_metadata`.

All unavailable fields are explicitly `None` — never a fabricated value.

### 3.7 `TemporalRelationship` (frozen dataclass)

| Field | Description |
|-------|-------------|
| `overlap_duration` | Seconds of shared time, clamped to `>= 0.0`. |
| `overlap_ratio_prediction` | `overlap / pred_duration`; `None` when `pred_duration == 0`. |
| `overlap_ratio_ground_truth` | `overlap / gt_duration`; `None` when `gt_duration == 0`. |

Intervals are half-open `[start, end)`.  Touching endpoints produce
`overlap = 0`.

### 3.8 `MatchResult` (dataclass)

One per prediction–GT relationship.  `ground_truth_id` is `None` for FP
results.  The `reason` field explains WHY the outcome was reached, making
every result auditable.

### 3.9 `MatchingPolicy` (dataclass)

| Field | Default | Meaning |
|-------|---------|---------|
| `min_overlap_ratio` | `0.5` | Minimum overlap ratio required for a temporal match. **Provisional / uncalibrated** — must be calibrated in Phase 2C.3 before production use. |
| `overlap_metric` | `"gt_ratio"` | Which ratio to use: `"gt_ratio"`, `"pred_ratio"`, or `"absolute"` (raw seconds). |
| `require_class_match` | `False` | When `True`, `CLASS_MISMATCH` blocks a TP match. |
| `identity_mode` | `"ignore"` | `"ignore"`: identity is recorded but never blocks.  `"strict"`: `IDENTITY_MISMATCH` blocks a TP. |
| `ambiguous_handling` | `"exclude"` | `"exclude"`: AMBIGUOUS GTs produce `AMBIGUOUS_GROUND_TRUTH` and are excluded from TP/FN.  `"report"`: AMBIGUOUS GTs are eligible for matching. |

The `is_provisional` property returns `True` when the policy is using the
default uncalibrated values, making it impossible to silently present the
provisional default as a calibrated threshold.

### 3.10 `EvaluationSummary` (dataclass)

| Field | Denominator |
|-------|-------------|
| `tp_count` / `fp_count` / `fn_count` / `duplicate_count` / `ambiguous_count` | Direct counts. |
| `total_predictions` | `TP + FP` (DUPLICATE and AMBIGUOUS excluded). |
| `total_gt_incidents` | `TP + FN` (DUPLICATE and AMBIGUOUS excluded). |
| `precision` | `TP / (TP + FP)`; `None` when denominator is 0. |
| `recall` | `TP / (TP + FN)`; `None` when denominator is 0. |
| `f1` | `2*p*r / (p+r)`; `None` when `p` or `r` is `None`. |
| `duplicate_rate` | `DUPLICATE / (TP + FP + DUPLICATE)`; `None` when denominator is 0. |

---

## 4. Key Functions

### 4.1 `PredictionNormalizer.event_record_to_normalized(event, video_id, prediction_id)`

Static adapter: `EventRecord` to `NormalizedPrediction`.
- `candidate_start/end_video_time` to `start/end_video_time` (or `POINT_EVENT`
  when they are equal).
- `entity_key.as_string()` to `identity`.
- `event_confidence` passed through as-is (`None` stays `None`).
- `video_id` must be supplied by the caller (it is not embedded in
  `EventRecord`).

### 4.2 `compute_temporal_relationship(pred, gt)`

Returns `TemporalRelationship`.  Half-open interval intersection:
`overlap = max(0, min(pe, ge) - max(ps, gs))`.  Ratios are `None` when the
denominator duration is 0.

### 4.3 `compute_class_match_status(pred, gt)`

Priority: `CLASS_AMBIGUOUS` (GT label AMBIGUOUS) to `CLASS_UNAVAILABLE`
(pred class is None) to `CLASS_MATCH` / `CLASS_MISMATCH`.

### 4.4 `match_predictions_to_ground_truth(predictions, ground_truth_incidents, policy, normal_periods=None, ambiguous_periods=None)`

Deterministic one-to-one greedy matching:

1. Sort predictions by `(start_video_time, prediction_id)`; sort GTs by
   `(start_video_time, incident_id)`.  Input order is not relied upon.
2. For each prediction (in sorted order), find the best eligible GT:
   - Temporal status must be `MATCHED`.
   - `require_class_match=True`: `CLASS_MISMATCH` blocks.
   - `identity_mode="strict"`: `IDENTITY_MISMATCH` blocks.
   - Among eligible GTs, pick highest overlap metric value; ties broken by
     lexicographic `incident_id`.
3. Once a GT is consumed (assigned a primary TP), subsequent predictions
   that overlap the same GT produce `DUPLICATE`.
4. Unmatched GTs (non-ambiguous, under `ambiguous_handling="exclude"`)
   produce `FN`.
5. Unmatched predictions produce `FP`.  If `normal_periods` is supplied
   and the prediction falls inside a normal period, the reason field notes
   it explicitly.
6. In `ambiguous_handling="report"` mode, AMBIGUOUS GTs are included in
   the eligible set and can produce TPs.

FP results that were blocked by the identity gate carry the actual
`IDENTITY_MISMATCH` status (not `UNAVAILABLE`) when a blocked GT was
identified, preserving auditability.

### 4.5 `compute_evaluation_summary(results)`

Aggregates `List[MatchResult]` into `EvaluationSummary`.  Denominator rules
are documented in 3.10.  `None` (not 0.0) for undefined values.

### 4.6 `compute_confirmation_latency(pred, gt)`

Returns `confirmation_video_time - gt.start_video_time` (in video seconds).
`None` when `confirmation_video_time` is unavailable.  Negative values are
valid (pre-emptive detection).

---

## 5. Design Decisions

### 5.1 Provisional thresholds are explicitly labelled

`MatchingPolicy.min_overlap_ratio` defaults to 0.5, which is a
**provisional evaluation default**, not a calibrated production threshold.
The `is_provisional` property makes this visible to callers.  Phase 2C.3
will replace this with a calibrated value derived from the dataset.

### 5.2 Deterministic order-invariance

The matching algorithm sorts both the prediction and GT lists by
`(start_video_time, id)` before processing.  The result is invariant to
the order in which GTs are supplied.  This is pinned by
`test_order_invariance_of_gt_list`.

### 5.3 `None` for undefined metrics

When a denominator is zero (e.g., no TPs and no FPs, so precision is
undefined), the corresponding metric field is set to `None` rather than
0.0.  This prevents callers from silently treating "no data" as "perfect
score."

### 5.4 `AMBIGUOUS_GROUND_TRUTH` is a distinct outcome

AMBIGUOUS GT incidents (under `ambiguous_handling="exclude"`, the
default) produce `AMBIGUOUS_GROUND_TRUTH` results, not FN results.  They
are excluded from all TP/FN denominators.  This is consistent with Phase
2C.1's requirement that AMBIGUOUS never be silently coerced to NORMAL or
SUSPICIOUS.

---

## 6. Test Coverage (60 tests)

| Class | Count | Covers |
|-------|-------|--------|
| `TestNormalizedPrediction` | 5 | boundary types, unavailable fields, required `prediction_id` |
| `TestPredictionNormalizer` | 3 | EventRecord adapter: basic, entity key, event confidence |
| `TestTemporalRelationship` | 11 | exact/partial/no overlap, contained, zero-duration, boundary-touching |
| `TestClassMatching` | 4 | CLASS_MATCH / MISMATCH / UNAVAILABLE / AMBIGUOUS |
| `TestOneToOneMatching` | 4 | 1:1, multi:1 (duplicate), 1:multi, multi:multi |
| `TestDuplicates` | 2 | duplicate distinct from FP; duplicate rate in summary |
| `TestFalseNegatives` | 2 | unmatched GT is FN; ambiguous GT not counted as FN |
| `TestFalsePositives` | 2 | unmatched prediction is FP; prediction in normal period is FP |
| `TestIdentityMatching` | 4 | known match/mismatch, unknown GT, both unavailable |
| `TestDeterminism` | 2 | repeated inputs same output; GT order invariance |
| `TestEvaluationSummary` | 7 | precision/recall/F1, zero-denominator, duplicate rate, total counts |
| `TestConfirmationLatency` | 3 | valid, missing, negative latency |
| `TestMatchResultStructure` | 2 | reason populated; all fields present |
| `TestEmptySets` | 3 | empty predictions, empty GT, both empty |
| `TestAmbiguousHandling` | 2 | exclude mode, report mode |
| `TestAdjacentAndNested` | 3 | adjacent no overlap, nested, identical timestamps |
| `TestConfigurableThreshold` | 2 | higher threshold reduces matches; metric configurable |

**Full suite: 355 passed, 0 failed (5.00 s)**

---

## 7. Downstream Phase Contracts

### For Phase 2C.3 (Threshold Calibration)
- Use `match_predictions_to_ground_truth` with a `MatchingPolicy` whose
  `min_overlap_ratio` and `overlap_metric` are the variables to calibrate.
- Use `compute_evaluation_summary` to track precision/recall at each
  candidate threshold.
- Use `compute_confirmation_latency` to measure video-time latency from
  GT incident start to prediction confirmation.
- AMBIGUOUS GT incidents are already excluded from denominators under the
  default policy.

### For Phase 2C.4+ (Evaluation Reporting)
- `MatchResult.to_dict()` and `EvaluationSummary.to_dict()` provide
  serializable output for report generation.

---

## 8. Change Log

| Date | Entry |
|------|-------|
| 2026-10-13 | Created `inference/evaluation/matching.py` and `tests/test_phase2c2_matching.py`; 60 tests green, full suite 355 green |
