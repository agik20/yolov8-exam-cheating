# Phase 2E-R8 — Calibration Integration Readiness

Status: **SHADOW-ONLY / NOT READY FOR OPERATING-POINT SELECTION**

## What is calibration-ready now (matching parameters)

All matching parameters are mechanically sweepable against shadow data
today via the `shadow_bridge` adapter:

| Parameter | Sweepable now? | Reason |
|-----------|---------------|--------|
| `min_overlap_ratio` | Yes | Depends only on start/end video time (always present) |
| `overlap_metric` | Yes | Depends only on interval geometry |
| `require_class_match` | Yes | Class name is on `EventRecord` |
| `identity_mode` | Yes | Entity key (identity) is on `EventRecord` |
| `ambiguous_handling` | Yes | GT period classification only; no model fields needed |

## What is blocked and why

### Confirmation sweep (`min_confidence_mean`)

**Blocked.** `sweep_confirmation_parameters` filters records with:

```python
p.event_confidence is not None and p.event_confidence >= min_conf
```

In the current shadow pipeline `event_confidence` is always `None`
(`scoring_status = "not_calibrated"`). Every grid cell with a confidence
threshold therefore excludes all shadow records. This is not a data-mapping
problem — it is a genuinely missing calibrated scoring policy.

### Association evidence

`association_score` is `None` unless an `association_policy` was supplied
to `ShadowLifecycleManager`. The default shadow path does not supply one,
so `association_evidence_status = "unavailable"`.

### Ground-truth labels

Calibration requires a `GroundTruthDataset` with human-annotated incidents.
These cannot be invented. Without a real labelled dataset, no operating
point can be selected.

## Parameter provenance table

| Parameter | Status today | Provenance |
|-----------|-------------|------------|
| `min_overlap_ratio` | Provisional (0.5) | Structural default in `MatchingPolicy()` |
| `overlap_metric` | Provisional ("gt_ratio") | Structural default in `MatchingPolicy()` |
| `require_class_match` | Provisional (False) | Structural default |
| `identity_mode` | Provisional ("ignore") | Structural default |
| `ambiguous_handling` | Provisional ("exclude") | Structural default |
| `min_confidence_mean` | Not calibrated | No scoring policy exists |
| `event_confidence` | Not calibrated | R6 evidence-only; no formula |
| Association | Not available | No associator in default path |

## Files changed

| File | Change |
|------|--------|
| `inference/evaluation/shadow_bridge.py` | **New** — `shadow_events_to_prediction_records`, `shadow_events_to_prediction_artifact`, `CalibrationReadinessReport` |
| `inference/evaluation/runner.py` | Fixed bug: `from_event_record_dict` was `@staticmethod` but called `cls(...)`; now calls `PredictionRecord(...)` |
| `tests/test_phase2e_r8_calibration_readiness.py` | **New** — 23 tests covering conversion, no-fabrication, readiness report, sweep-blocking behaviour |
| `docs/PHASE2E_R8_CALIBRATION_INTEGRATION_READINESS.md` | This document |

## Test results

- R8 tests: **23/23 passed**
- Full suite: **567/567 passed**

## Expected next step

A calibrated operating point requires:
1. A `GroundTruthDataset` with real human labels
2. A calibrated event-confidence scoring policy (Phase 2C.4 or later)
3. An associator to populate `association_score`

Until all three are in place, the shadow pipeline remains
**SHADOW-ONLY / NOT READY** for cutover.
