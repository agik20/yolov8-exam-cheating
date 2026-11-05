# Phase 2E-R11 — Ground Truth Dataset Readiness Tooling

**Status: SHADOW-ONLY / NOT READY — tooling is in place; a real annotated
GroundTruthDataset has NOT been committed to the repository**

## What this phase provides

`inference/evaluation/gt_readiness.py` (new module) adds three functions
on top of the existing `ground_truth.py` schema:

| Function | Purpose |
|----------|---------|
| `load_gt_dataset(source, source_name)` | Load a `GroundTruthDataset` from a JSON string, file path, or dict. Returns `(dataset_or_None, issues)`. |
| `assess_gt_dataset_readiness(dataset)` | Pure-function readiness report. No file I/O, no fabricated values. |
| `validate_for_r10(dataset)` | Strict gate: returns `{"ok", "blockers", "report"}`. Blocks R10 when schema errors exist even if counts look fine. |

### What a valid dataset must contain before R10 can execute

A `GroundTruthDataset` is **usable for R10** when ALL of the following
are true:

1. **At least one `SourceVideoMetadata` entry** — `video_id`, `reference`,
   `duration_seconds` (finite, >= 0). Without `duration_seconds` the
   calibration executor cannot bound incident intervals.
2. **At least one `GroundTruthIncident`** — human-annotated with a valid
   `behavior_class` (from `VALID_BEHAVIOR_CLASSES`), a valid `GTLabel`,
   and interval `start_video_time <= end_video_time` within video duration.
3. **No schema errors** — as detected by `validate_dataset()`:
   - no duplicate `incident_id` or `video_id` across all manifests
   - every incident's `video_id` has a matching `SourceVideoMetadata`
   - no negative or NaN/inf interval bounds
   - frame/time consistency holds when both frames and fps are present
4. **Normal and ambiguous periods** — recommended but not required for the
   gate to pass. `no_normal_periods` and `no_ambiguous_periods` appear in
   the R10 blocked-output `missing_prerequisites` list but do not block
   `validate_for_r10()` by themselves.

## The readiness report

`GTDatasetReadinessReport` is a `dataclass` with:

| Field | Type | Notes |
|-------|------|-------|
| `usable_for_r10` | `bool` | `True` only when no blockers |
| `video_count` | `int` | |
| `incident_count` | `int` | |
| `normal_period_count` | `int` | |
| `ambiguous_period_count` | `int` | |
| `schema_error_count` | `int` | from `validate_dataset()` |
| `schema_errors` | `List[str]` | actionable diagnostic strings |
| `r10_blockers` | `List[str]` | high-level blocker labels |
| `unblock_requirements` | `List[str]` | human-readable next steps |

The report is deterministic: calling `assess_gt_dataset_readiness()` twice
on the same dataset produces identical `to_dict()` output. No operating-point
values, recall, precision, or F1 are included — those belong to R10 itself
once a real dataset is available.

## How to use `validate_for_r10`

```python
from inference.evaluation.gt_readiness import load_gt_dataset, validate_for_r10

gt, issues = load_gt_dataset("path/to/gt_dataset.json", source_name="annotated_exam_session")
if gt is None:
    print("Failed to load dataset:", issues)
    raise SystemExit(1)

gate = validate_for_r10(gt)
if not gate["ok"]:
    print("R10 blocked:")
    for b in gate["blockers"]:
        print(f"  - {b}")
    print("Schema errors:")
    for e in gate["report"]["schema_errors"]:
        print(f"  {e}")
    raise SystemExit(1)

print("Dataset is ready for R10 calibration.")
# Proceed to execute_calibration(gt, ...)
```

## Files changed in R11

| File | Change |
|------|--------|
| `inference/evaluation/gt_readiness.py` | **New** — readiness module |
| `tests/test_phase2e_r11_gt_dataset_readiness.py` | **New** — 23 tests |
| `docs/PHASE2E_R11_GT_DATASET_READINESS.md` | This document |

## Test results

- R11 tests: **23/23 passed**
- Full suite: **617/617 passed** (all prior tests preserved)

## What remains before R10 can execute

R11 has not committed a real dataset. The repository still contains only
inference-output CSVs and a benchmark JSON (see
`docs/PHASE2E_R10_CALIBRATION_RUN_BLOCKED.md`). Until a human-annotated
`DatasetManifest` JSON file is committed and a calibrated
`event_confidence` policy is in place (Phase 2C.4+), `execute_calibration()`
continues to return `cutover_status="NOT_READY"` and
`production_ready=False`.

The pipeline remains **SHADOW-ONLY / NOT READY** for cutover.
