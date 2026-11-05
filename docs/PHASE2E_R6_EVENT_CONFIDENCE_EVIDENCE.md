# Phase 2E-R6 — Event Confidence Evidence (Evidence-Only)

Status: **IMPLEMENTED — 523/523 tests passing**

## What it adds

`EventConfidenceEvidence` is a frozen dataclass that preserves the *inputs*
to a future calibrated event-confidence score.  It is attached to
`EvidenceBundle` as an optional field (`event_confidence_evidence`,
default `None`).

This is **evidence-only**.  No scoring formula is implemented, calibrated,
or implied.  `event_confidence` on the structure is always `None` until
an explicitly calibrated scoring/operating-point policy exists.

## Core principle

> **Detection confidence ≠ event confidence.**

A high model detection confidence (e.g. 0.95 for a "phone" box) does not
automatically mean the *event* "person is providing an object to another
person" is trustworthy.  Event confidence is a human-trust / evidence-quality
concept that requires:

- sufficient temporal persistence
- consistent track identity
- a valid person-object association
- confirmation gate outcomes
- calibrated operating-point data

None of these are computed into a single number by the current
implementation.  `EventConfidenceEvidence` records each component
separately so a future calibrated formula can be applied without loss of
auditability.

## Fields

| Field | Type | Meaning |
|-------|------|---------|
| `observation_count` | `int` | Raw count of observations in the candidate window. |
| `confidence_stats` | `ConfidenceStats` | Detection confidence distribution (count / mean / min / max). |
| `video_time_span` | `float` | Temporal duration of the candidate window (seconds, video-time). |
| `continuity` | `TrackContinuity` | R4 continuity evidence (track IDs, switch count, max gap, primary ID). |
| `association_status` | `Optional[str]` | `AssociationStatus` value string (ASSOCIATED / AMBIGUOUS / UNASSIGNED / UNAVAILABLE); `None` when association was not performed. |
| `association_score` | `Optional[float]` | `AssociationResult.score`; stays `None` unless a calibrated formula explicitly injects a value. |
| `confirmation_decision` | `Optional[ConfirmationDecisionEvidence]` | Per-gate PASS / FAIL / NOT_CONFIGURED / UNAVAILABLE outcomes; `None` when the gate result is not supplied. |
| `event_confidence` | `Optional[float]` | **Always `None`** until a calibrated scoring policy exists. |
| `scoring_status` | `str` | `"not_calibrated"` — the only valid value today.  A value of `"calibrated"` would require an explicit, reviewed operating-point. |

## What is NOT implemented

- No weighted formula (`0.4 * confidence + 0.3 * persistence + ...`).
- No arbitrary threshold or cutoff.
- No production event file write.
- No change to confirmation thresholds, cooldown, or association thresholds.
- No track-ID heuristic beyond what R4 already provides.

## Files changed

| File | Change |
|------|--------|
| `inference/core/events.py` | Added `EventConfidenceEvidence` frozen dataclass; added `event_confidence_evidence: Optional[EventConfidenceEvidence] = None` to `EvidenceBundle`; extended `EvidenceBundle.to_dict()` to include `event_confidence_evidence`; added `EventConfidenceEvidence` to `__all__` |
| `tests/test_phase2e_r6_event_confidence_evidence.py` | New test file with 20 tests |

## Usage

```python
from inference.core.events import EventConfidenceEvidence, EvidenceBundle

# Build from a confirmed Candidate (association and decision optional):
ev = EventConfidenceEvidence.from_candidate(
    candidate,
    association_result=assoc_result,   # Optional
    confirmation_decision=cd,          # Optional
)

# Attach to an EvidenceBundle:
bundle = EvidenceBundle(
    ...,
    event_confidence_evidence=ev,
)

# Serialize:
d = bundle.to_dict()
d["event_confidence_evidence"]["event_confidence"]   # always None
d["event_confidence_evidence"]["scoring_status"]     # "not_calibrated"
```

## Remaining limitations

- **No calibrated score.**  A numerical `event_confidence` requires a
  calibrated operating-point policy (Phase 2C.4) with explicit
  thresholds, weights, or a logistic-calibrated score.  Until that
  exists, `event_confidence` is `None` and `scoring_status` is
  `"not_calibrated"`.
- **Cross-entity-key scope.**  Continuity and association evidence are
  computed within a single `(EntityKey, candidate_class)` candidate;
  cross-entity-key identity is out of scope.
- **No temporal persistence beyond `maxlen`.**  `TrackContinuity` and
  `ConfidenceStats` are recomputed from the bounded buffer each call.
- **Not operationally usable.**  The evidence structure is a record,
  not a decision input.  No gate, cooldown, or event-write reads
  `event_confidence` in the current implementation.

**Cutover status: NOT READY.**
