# Phase 2E-R10 — Calibration Run and Operating-Point Audit

**Status: BLOCKED — no human-annotated GroundTruthDataset in repository**

## What R10 requires

A real `GroundTruthDataset` (one or more `DatasetManifest` JSON files) containing:
- `SourceVideoMetadata` entries with `duration_seconds`
- `GroundTruthIncident` entries annotated by a human reviewer
- `NormalPeriod` and `AmbiguousPeriod` time windows

Without this, `execute_calibration()` returns `executed=False`, `report=None`,
`operating_point=None`, and `cutover_status="NOT_READY"`. No metric — FP/hour,
recall, precision, F1, duplicate rate, latency, or subgroup breakdown — can be
produced.

## Why R10 is blocked in this repository

All three data-adjacent files in the repo are **inference output**, not GT:

| File | What it actually is | Why it fails as GT |
|------|-------------------|-------------------|
| `2026-10-01_detection_log.csv` | Per-crop detection log (class + confidence + timestamp) | Not JSON → `DatasetManifest.from_file` raises `JSONDecodeError` |
| `inference/2026-10-02_detection_log.csv` | Per-crop detection log (same schema) | Not JSON → `JSONDecodeError` |
| `inference/output/benchmark_results.json` | Model throughput benchmark (torch/frame counts) | JSON but no `version` key → `KeyError: 'version'` |

The `GroundTruthDataset` constructor is only called in test fixtures
(`tests/`) or as an empty default. No production loader writes a GT file to
disk, and no GT file has been committed to the repo.

## Live blocked-state output (2026-10-19)

Running `execute_calibration(GroundTruthDataset(), ...)` with
`CalibrationParameterGrid({"min_overlap_ratio": [0.3, 0.5, 0.7]})`:

```json
{
  "executed": false,
  "missing_prerequisites": [
    "no GT incidents - a GroundTruthDataset with annotated incidents is required",
    "no source video metadata - at least one SourceVideoMetadata entry is required",
    "no GT incidents",
    "no source video metadata",
    "dataset limitation: insufficient_videos_for_split",
    "dataset limitation: no_normal_periods",
    "dataset limitation: no_ambiguous_periods",
    "dataset limitation: no_duration_metadata",
    "GT validation error: EMPTY DATASET: no source videos registered"
  ],
  "dataset_summary": {
    "video_count": 0,
    "incident_count": 0,
    "normal_period_count": 0,
    "ambiguous_period_count": 0,
    "total_duration_seconds": null,
    "split_seed": 42,
    "train_frac": 0.6,
    "val_frac": 0.2,
    "train_videos": [],
    "validation_videos": [],
    "holdout_videos": [],
    "total_predictions": 1
  },
  "readiness": {
    "total_predictions": 1,
    "matching_parameters_sweepable": true,
    "sweepable_parameters": [
      "ambiguous_handling", "identity_mode", "min_overlap_ratio",
      "overlap_metric", "require_class_match"
    ],
    "blocked_parameters": ["min_confidence_mean"],
    "blocked_reasons": {
      "min_confidence_mean": "event_confidence is None on all shadow records; a calibrated event-confidence scoring policy is required"
    },
    "association_evidence_status": "unavailable"
  },
  "report": null,
  "operating_point": null,
  "subgroup_metrics": null,
  "confirmation_results": null,
  "cutover_status": "NOT_READY",
  "production_ready": false
}
```

## What is still missing to unblock R10

| # | Missing prerequisite | Where it will come from |
|---|---------------------|------------------------|
| 1 | Human-annotated `GroundTruthDataset` (JSON) | Annotation pipeline; one `DatasetManifest` per exam session |
| 2 | Calibrated `event_confidence` scoring policy (Phase 2C.4+) | Replaces shadow `None` with a real score |
| 3 | An associator to populate `association_score` | `ShadowLifecycleManager` with a configured `association_policy` |

Until all three are in place the pipeline remains **SHADOW-ONLY / NOT READY**
for cutover.

## No-fabrication guarantee

`tests/test_phase2e_r10_calibration_run_blocked.py` (8 tests) pins:
- `TestR10RepoFilesAreNotGT` — all three repo files fail `DatasetManifest.from_file`
- `TestR10NoFabricationWhenNoGT` — empty GT → `executed=False`, `report=None`,
  `operating_point=None`, `cutover_status="NOT_READY"`, prerequisites are specific
- `TestR10ShadowEventConfidenceBlocked` — `min_confidence_mean` is stripped from
  any grid when all records have `event_confidence=None`
- `TestR10BlockedRunIsDeterministic` — two identical blocked runs produce
  identical output

## Files changed

| File | Change |
|------|--------|
| `tests/test_phase2e_r10_calibration_run_blocked.py` | **New** — 8 tests |
| `docs/PHASE2E_R10_CALIBRATION_RUN_BLOCKED.md` | This document |

## Test results

- R10 tests: **8/8 passed**
- Full suite: **594/594 passed**

## Expected next step

R10 becomes unblocked when a real `GroundTruthDataset` is committed to the repo
and `event_confidence` is calibrated by a Phase 2C scoring policy. At that
point `execute_calibration()` will produce a `CalibrationReport` and
`OperatingPointResult`, and the operating-point audit (FP/hour, recall,
precision, F1, duplicate-rate, subgroup metrics) can be written as a second
R10 document.
