# Phase 2B.6 — Evidence Bundle & Shadow Config Architecture

> **Status: Not calibrated.**
> This phase introduces the *structure* for audit-grade evidence and a typed
> shadow configuration layer.  All thresholds default to `None`
> (unconfigured).  No values from this phase should be treated as validated or
> production-ready.  Calibration data (§12 of PHASE2A_EVENT_ARCHITECTURE.md)
> has not been collected.

---

## 1. Overview

Phase 2B.6 adds two things to the shadow decision path without touching the
legacy production path:

| Component | What it provides |
|-----------|-----------------|
| **Evidence structures** (`inference/core/events.py` — additive) | `ConfirmationDecisionEvidence`, `TrackIdentityEvidence`, `AssociationEvidenceSummary`, `HumanEvidenceMetadata`; four new optional fields on the existing `EvidenceBundle`. |
| **`ShadowConfig`** (`inference/config/shadow_config.py` — new module) | A typed, inspectable configuration object for all shadow-path numeric parameters.  Separate module from the legacy `Config` class; no numeric source of truth is shared between the two. |

Neither component writes files.  Image/crop/CSV generation remains out of
scope for this phase.

---

## 2. Evidence Schema

### 2.1 New structures in `events.py`

All are dataclasses with `to_dict()` methods; enums serialize to their string
values.  No field holds a numpy, torch, or OpenCV object.

**`GateOutcome`** (enum) — per-gate auditability:

| Value | Meaning |
|-------|---------|
| `PASS` | Gate configured; measured value meets threshold. |
| `FAIL` | Gate configured; measured value does not meet threshold. |
| `NOT_CONFIGURED` | No threshold set — gate not enforced. **Not a failure.** |
| `UNAVAILABLE` | Gate configured but the required measured value is `None`. |

**`ConfirmationDecisionEvidence`**

```
policy_type: str            # "calibrated" | "unconfigured"
confirmed: bool
min_observations_configured: bool
min_observations_required: Optional[int]
actual_observations: int
min_duration_seconds_configured: bool
min_duration_seconds_required: Optional[float]
actual_duration: Optional[float]
min_confidence_mean_configured: bool
min_confidence_mean_required: Optional[float]
actual_confidence_mean: Optional[float]
max_gap_seconds_configured: bool
max_gap_seconds_required: Optional[float]
actual_max_gap: Optional[float]

to_dict() -> {
    policy_type, confirmed,
    gates: {
        min_observations: {configured, outcome, required, actual},
        min_duration:     {configured, outcome, required, actual},
        min_confidence:   {configured, outcome, required, actual},
        max_gap:          {configured, outcome, required, actual},
    }
}
```

`ConfirmationDecisionEvidence.from_gate_result(result, policy)` reads which
gates are configured from the policy (a `CalibratedConfirmationPolicy`
exposes its thresholds; an `UnconfiguredConfirmationPolicy` reports every
gate as `NOT_CONFIGURED`).  It never fabricates a threshold.

**`TrackIdentityEvidence`**

```
entity_key: EntityKey
track_ids: FrozenSet[int]           # all observed track IDs
track_id_switches: Optional[int]    # None = not confirmed
```

*Identity rule:* a candidate whose `observed_track_ids` set contains more
than one value is **not**, by itself, evidence of a confirmed ID switch.
`track_id_switches` must be set explicitly by a future switch-detection
component; `None` means "unknown / not confirmed".

**`AssociationEvidenceSummary`**

```
association_result: AssociationResult        # wraps Phase 2B.5 result
person_entity: Optional[str]
object_entity: Optional[str]
association_observations: Tuple[Dict, ...]   # bounded diagnostics
association_score (property) -> Optional[float]   # None unless calibrated
status (property) -> str            # "ASSOCIATED" | "AMBIGUOUS" |
                                     "UNASSIGNED" | "UNAVAILABLE"
```

*Score rule:* `AssociationResult.score` is `None` unless a caller
explicitly injects a calibrated value.  The summary propagates that `None` —
no synthetic score is fabricated.

**`HumanEvidenceMetadata`**

```
full_frame_reference:       Optional[str]
expanded_crop_reference:    Optional[str]
person_object_crop_reference: Optional[str]
evidence_quality:           Optional[float]
brightness:                 Optional[float]
blur:                       Optional[float]
resolution:                 Optional[str]
```

All reference fields are `Optional[str]` paths.  `None` means "not yet
generated."  No image bytes are stored.  A `None` metric means "not
measured," never 0.

### 2.2 `EvidenceBundle` additive fields

Four new `Optional[...] = None` fields were appended to the existing
`EvidenceBundle` dataclass.  Existing construction sites remain fully
backward-compatible:

```python
bundle.decision:    Optional[ConfirmationDecisionEvidence]
bundle.association: Optional[AssociationEvidenceSummary]
bundle.human:       Optional[HumanEvidenceMetadata]
bundle.identity:    Optional[TrackIdentityEvidence]
```

`to_dict()` includes each section only when it is not `None`, so the
serialized shape for pre-2B.6 callers is unchanged.

### 2.3 Serialization invariants

- `json.dumps(bundle.to_dict())` must succeed for any fully- or
  partially-populated bundle.
- No `numpy.ndarray`, `torch.Tensor`, or `cv2` object may appear in any
  `to_dict()` output.
- `event_id` is a `uuid4` string (generated in `EventRecord.from_candidate`);
  it is **not** a track ID.

### 2.4 Temporal auditability

| Field | Clock source | Used in gates? |
|-------|-------------|----------------|
| `confirmation_video_time` | `frame_index / src_fps` | Yes (all confirmation gates) |
| `confirmation_wall_time` | `datetime.now().isoformat()` | **No** — audit-only |
| `candidate_start/end_video_time` | video-time | Yes |

The mixed-clock situation (video-time for gates, wall-time for the timestamp)
is documented and preserved from Phase 2B.1; changing it is a future
calibration decision.

---

## 3. Shadow Config

### 3.1 Module structure

```
inference/config/
├── __init__.py          # re-exports legacy Config + paths (unchanged)
├── constants.py         # legacy production constants (frozen, unchanged)
├── paths.py             # root-anchored path helpers (unchanged)
└── shadow_config.py     # NEW — ShadowConfig (this phase)
```

`ShadowConfig` is a standalone dataclass.  It does **not** inherit from or
reference the legacy `Config` class.  The two modules share no numeric source
of truth.

### 3.2 Configuration hierarchy

```
ShadowConfig
├── confirmation:  ConfirmationConfig
│   ├── min_observations:      Optional[int]   # default None (gate unenforced)
│   ├── min_duration_seconds:  Optional[float]  # default None
│   ├── min_confidence_mean:   Optional[float]  # default None
│   └── max_gap_seconds:       Optional[float]  # default None
│
├── cooldown:      CooldownConfig
│   └── cooldown_seconds:      Optional[float]  # default None
│
├── association:   AssociationConfig
│   ├── containment_threshold:       Optional[float]  # default None
│   ├── keypoint_distance_threshold: Optional[float]  # default None
│   ├── visibility_cutoff:           Optional[float]  # default None (→ 0.3 at build time)
│   └── reference_scale:             Optional[float]  # default None (→ 1.0 at build time)
│
└── candidate_buffer:  CandidateBufferConfig
    └── observation_buffer_maxlen: int  # default 32 (structural, not a policy gate)
```

### 3.3 Legacy vs shadow parameter separation

The following legacy values remain frozen in `inference/config/constants.py`
and are **never** read or written by `ShadowConfig`:

| Legacy constant | Value | Meaning |
|----------------|-------|---------|
| `REQUIRED_DETECTION_TIME` | 5 s | Wall-clock persistence gate in `process_detection` |
| `POSE_COOLDOWN` / `OBJECT_COOLDOWN` | 90 s | Wall-clock per-track cooldown in `process_detection` |
| `CONF_THRESH_POSE` / `CONF_THRESH_OBJ` | 0.3 | Model detection confidence threshold |
| `NMS_IOU_THRESHOLD` | 0.45 | NMS IoU suppression |
| `MIN_CROP_BRIGHTNESS` | 30 | Legacy crop validity gate |

The shadow-path equivalents in `ShadowConfig` are **separate fields** and
default to `None`.  No value from `constants.py` is silently copied into
`ShadowConfig` at construction time.

### 3.4 Unconfigured-parameter semantics

When a `ShadowConfig` field is `None`, the corresponding gate is **not
enforced**.  The evidence structures make this distinction explicit:

| Scenario | Evidence representation |
|----------|------------------------|
| Gate unconfigured | `NOT_CONFIGURED` outcome; `configured=False`; `required=None` |
| Gate configured, data missing | `UNAVAILABLE` outcome; `configured=True`; `actual=None` |
| Gate configured, passed | `PASS` outcome; `actual >= required` (`<=` for `max_gap`) |
| Gate configured, failed | `FAIL` outcome; `actual < required` (`>` for `max_gap`) |

`ShadowConfig.build_policies()` returns an `UnconfiguredConfirmationPolicy`
when **no** confirmation gate is set, and a `CalibratedConfirmationPolicy`
when at least one is set.  The same logic applies to `build_associator()`.

### 3.5 Buffer size is independent of observation gate

`CandidateBufferConfig.observation_buffer_maxlen` (default 32) is a memory
bound on the observation ring buffer.  It is **not** a confirmation-gate
threshold.  Setting it to any positive value has no effect on
`ConfirmationConfig.min_observations`.  The two are deliberately decoupled.

---

## 4. Calibration-Required Parameters

The following `ShadowConfig` fields require §12 calibration data before they
can be set to a production value.  Until then they must remain `None`:

| Field | Why it requires calibration |
|-------|----------------------------|
| `ConfirmationConfig.min_observations` | Depends on frame rate and target incident duration |
| `ConfirmationConfig.min_duration_seconds` | Must hold FP/hr under target conditions |
| `ConfirmationConfig.min_confidence_mean` | Depends on model version and scene distribution |
| `ConfirmationConfig.max_gap_seconds` | Depends on tracker reliability and occlusion profile |
| `CooldownConfig.cooldown_seconds` | Suppression window for post-incident re-detection |
| `AssociationConfig.containment_threshold` | Geometry threshold; scene-dependent |
| `AssociationConfig.keypoint_distance_threshold` | Depends on camera resolution and pose accuracy |

**Do not set any of these to a "reasonable" number without calibration data.**

---

## 5. Known Limitations

1. **Association scores are not computed.**
   `AssociationResult.score` is always `None` in this phase.  The
   `PersonObjectAssociator` (Phase 2B.5) does not yet have a calibrated
   scoring formula.  `AssociationEvidenceSummary` propagates `None` faithfully.

2. **Track-ID switches are not detected.**
   `TrackIdentityEvidence.track_id_switches` is `None` by default.  The
   system has no switch-detection component yet.  The structure is in place
   for a future implementation but currently carries no confirmed value.

3. **Human evidence paths are always `None`.**
   `HumanEvidenceMetadata` has no image-saving layer wired to it in this
   phase.  All reference fields remain `None` until a future
   evidence-generation phase populates them.

4. **No evidence is persisted to disk.**
   `EvidenceBundle.to_dict()` produces a JSON-portable dict, but nothing in
   this phase writes it to a file, database, or log.

5. **`ShadowConfig` is not yet wired into `pipeline.py`.**
   The pipeline currently constructs `UnconfiguredConfirmationPolicy` and
   `PersonObjectAssociator()` with no thresholds — correct for the
   uncalibrated state.  When calibration values are available,
   `ShadowConfig.build_policies()` / `build_associator()` will supply them.
   No pipeline code changes were made in this phase.

---

## 6. Test Coverage

| File | Tests | What it pins |
|------|-------|--------------|
| `tests/test_phase2b6_evidence.py` | 34 | Evidence schema serialization, gate outcomes, identity, association, human metadata, `to_dict()` JSON-portability, event ID uniqueness, temporal auditability |
| `tests/test_phase2b6_config.py` | 22 | `ShadowConfig` defaults, populated values, `build_policies()`, `build_associator()`, validation, legacy/shadow separation |

Full regression suite: **229 tests passing** (173 pre-existing + 56 new).

Run:
```bash
python -m pytest tests/ -q
```
