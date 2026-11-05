# Phase 2C.1 — Ground-Truth Annotation Schema & Dataset Validation

**Status:** COMPLETE — 295/295 tests passing (66 new tests in `tests/test_phase2c1_ground_truth.py`)
**Created:** 2026-09-24
**Owner:** A

---

## 1. Purpose

Phase 2C.1 defines the **model-independent ground-truth annotation schema**
required for Phase 2C.2 (prediction-vs-ground-truth matching) and
Phase 2C.3 (threshold calibration).

What it does:
- Defines the annotation data types: `GroundTruthIncident`, `NormalPeriod`,
  `AmbiguousPeriod`, `SourceVideoMetadata`, `DatasetManifest`,
  `GroundTruthDataset`
- Enforces validation invariants (interval bounds, behavior-class
  membership, ID uniqueness, video cross-referencing)
- Provides `validate_dataset()` as a one-call pre-flight check

What it does NOT do:
- No matching algorithm (Phase 2C.2)
- No threshold calibration (Phase 2C.3)
- No changes to the inference pipeline
- No automatic annotation; humans must produce the data

---

## 2. Files Touched

| File | Change |
|------|--------|
| `inference/evaluation/ground_truth.py` | **NEW** — all schema types, validation, overlap detection |
| `tests/test_phase2c1_ground_truth.py` | **NEW** — 66 unit tests |
| `inference/core/events.py` | Untouched (Phase 2B regression check passes) |
| `inference/config/shadow_config.py` | Untouched |
| `inference/core/pipeline.py` | Untouched |

---

## 3. Data Model Overview

```
GroundTruthDataset
  └─ List[DatasetManifest]
       ├─ List[SourceVideoMetadata]     # one entry per source video
       ├─ List[GroundTruthIncident]     # suspicious/ambiguous incidents
       ├─ List[NormalPeriod]            # explicitly-annotated normal spans
       └─ List[AmbiguousPeriod]         # explicitly-annotated ambiguous spans
```

### 3.1 `GTLabel` (enum)

| Value | Meaning |
|-------|---------|
| `NORMAL` | Annotator confirmed no suspicious behavior in interval |
| `SUSPICIOUS` | Annotator confirmed suspicious behavior |
| `AMBIGUOUS` | Evidence insufficient for reliable human decision; must NOT be silently coerced to either of the above |

### 3.2 `EntityReferenceStatus` (enum)

| Value | Meaning |
|-------|---------|
| `KNOWN` | Entity identity reliably determinable |
| `UNKNOWN` | Entity exists but identity not determinable from footage |
| `NOT_APPLICABLE` | No entity of this type relevant to annotation |
| `AMBIGUOUS` | Identity genuinely uncertain, cannot be resolved |

### 3.3 `SceneTag` (enum)

Optional difficulty/context tags for a source video.
Values: `NORMAL_EXAM`, `MULTI_PERSON`, `OCCLUSION`, `LIGHTING_CHANGE`,
`BLUR`, `CROWDED`, `OBJECT_OVERLAP`, `PARTIAL_PERSON`, `CAMERA_MOTION`.

### 3.4 `ReviewStatus` (enum)

Annotation review workflow: `DRAFT` → `PENDING` → `APPROVED` / `REJECTED` / `ADJUDICATED`.

### 3.5 `VALID_BEHAVIOR_CLASSES`

```python
("no-cheating", "provide-object", "see-friends-work",
 "smartphone", "calculator")
```

Copied from model class names. Any `behavior_class` outside this tuple is a
validation error.

---

## 4. Key Design Decisions

### 4.1 No tracker IDs
`GroundTruthIncident` has **no** `track_id` field. The annotation unit is a
time interval in the source video, not a detected track. This keeps the
ground-truth schema fully decoupled from any specific tracker implementation.

### 4.2 AMBIGUOUS is a first-class label
AMBIGUOUS incidents and AMBIGUOUS periods remain distinct from NORMAL and
SUSPICIOUS in all serialization and validation paths. No consumer of this
schema is permitted to silently map AMBIGUOUS → NORMAL or AMBIGUOUS →
SUSPICIOUS. The tests `TestAmbiguityExplicitness` pin this behavior.

### 4.3 Zero-duration intervals are valid
`start_video_time == end_video_time` is a legitimate point-in-time
annotation and passes validation.

### 4.4 Portable video references
`SourceVideoMetadata.video_id` and `.reference` must be plain strings with
no filesystem path separators (`/`, `\`). This allows datasets to be moved
between machines without breaking annotation references.

### 4.5 Validation is non-raising
`validate_incident()` and `validate_dataset()` return lists of actionable
error strings; they never raise exceptions. Callers decide whether to
fail hard or report.

---

## 5. Validation Rules

### 5.1 Per-incident (`validate_incident`)

| Rule | Error prefix |
|------|-------------|
| `incident_id` non-empty | `…: incident_id must be a non-empty string` |
| `video_id` non-empty, no `/` or `\` | `…: video_id must be…` |
| `start_video_time` finite, ≥ 0 | `…: start_video_time must be…` |
| `end_video_time` finite, ≥ 0 | `…: end_video_time must be…` |
| `start_video_time ≤ end_video_time` | `…: start_video_time (…) must be <= end_video_time (…) ` |
| `behavior_class` in `VALID_BEHAVIOR_CLASSES` | `…: behavior_class (…) is not in VALID_BEHAVIOR_CLASSES` |
| frame/time consistency (when fps known) | `…: start_frame (…) at fps (…) implies video_time (…) got (…) ` |
| `end_video_time ≤ video duration` (when known) | `…: end_video_time (…) exceeds video duration (…) ` |

### 5.2 Dataset-level (`validate_dataset`)

| Rule | Error string |
|------|-------------|
| Duplicate `video_id` across manifests | `DUPLICATE video_id: '…'` |
| Duplicate `incident_id` across manifests | `DUPLICATE incident_id: '…'` |
| Incident references missing `SourceVideoMetadata` | `MISSING VIDEO: incident '…' references video_id '…'` |
| Any per-incident validation error | (inherited from `validate_incident`) |
| Normal/ambiguous period interval error | `normal_period '…': …` / `ambiguous_period '…': …` |
| No source videos registered | `EMPTY DATASET: no source videos registered` |

### 5.3 Overlap detection (`find_overlapping_incidents`)

Two incidents overlap iff `a.start < b.end and b.start < a.end`
(touching intervals do NOT count as overlapping). Only pairs within the
same `video_id` are compared. Overlapping incidents are reported for human
review; they are **not** automatically merged or rejected.

---

## 6. Test Coverage (66 tests)

| Class | Count | Covers |
|-------|-------|--------|
| `TestEnums` | 5 | enum value completeness, valid behavior-class match |
| `TestIncidentSchema` | 11 | frozen dataclass, all field defaults, with_frames, zero-duration |
| `TestTemporalValidation` | 12 | negative/NaN/inf intervals, start>end, frame/time consistency |
| `TestBehaviorClassValidation` | 3 | empty, invalid, valid class names |
| `TestSerialization` | 7 | incident/manifest/period round-trips, JSON dump safety |
| `TestIndependence` | 2 | no tracker-ID coupling, no model-class imports |
| `TestOverlappingIncidents` | 7 | disjoint/overlapping/contained/touching/different-video |
| `TestDatasetValidation` | 7 | valid dataset, dupes, missing video, empty, bad interval, bad class |
| `TestPeriodValidation` | 3 | normal valid, normal start>end, ambiguous valid |
| `TestAmbiguityExplicitness` | 2 | AMBIGUOUS label preserved through round-trip; ambiguous ≠ normal |
| `TestPhase2BRegression` | 3 | `events`, `shadow_config`, `pipeline` modules still importable |

**Full suite: 295 passed, 0 failed (6.56 s)**

---

## 7. Downstream Phase Contracts

### For Phase 2C.2 (Matching)
- Load `GroundTruthDataset` via `DatasetManifest.from_file()`
- Validate with `validate_dataset()` before matching
- Ground-truth events = `dataset.all_incidents()` where `label == SUSPICIOUS`
- Predictions = `EvidenceBundle` from inference pipeline (Phase 2B)
- Match on (video_id, time_interval) using `find_overlapping_incidents` as
  a sanity check, not as the matching algorithm itself

### For Phase 2C.3 (Calibration)
- Same loading path; additionally use `NormalPeriod` records as
  false-positive ground-truth
- `AmbiguousPeriod` records must be excluded from precision/recall
  denominator (they are not normal nor suspicious)

---

## 8. Change Log

| Date | Entry |
|------|-------|
| 2026-09-24 | Fix #1 — Created `inference/evaluation/ground_truth.py` and `tests/test_phase2c1_ground_truth.py`; 66 tests green, full suite 295 green |
