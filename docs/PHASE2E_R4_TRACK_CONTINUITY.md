# Phase 2E-R4 — Track Identity Continuity (Shadow-Only)

Status: **IMPLEMENTED — 503/503 tests passing**

## What it adds

`TrackContinuity` is a frozen dataclass attached to every `Candidate`
instance.  It is recomputed deterministically from the bounded
`BoundedObservationBuffer` on every `CandidateLayer.record_observation`
call.  No confirmation gate, cooldown, or association path reads this
field.

## Fields

| Field | Type | Meaning |
|-------|------|---------|
| `track_ids` | `Tuple[int, ...]` | All distinct track IDs in the buffer, sorted ascending. |
| `observation_count` | `int` | Number of observations currently in the buffer (= `len(buffer)`, bounded by `maxlen`). |
| `possible_id_switch_count` | `int` | Count of consecutive-in-buffer pairs `(O[i], O[i+1])` where `O[i].track_id != O[i+1].track_id`. Bounded by `observation_count - 1`. |
| `max_observation_gap_seconds` | `Optional[float]` | `max(O[i+1].video_time - O[i].video_time)` over all consecutive buffer pairs. `None` when `observation_count < 2`. |
| `primary_track_id` | `Optional[int]` | Track ID of the oldest observation currently in the buffer (= `buffer.all()[0].track_id`). `None` when buffer is empty. |

## Terminology

`possible_id_switch` — a frame-adjacent track-ID discontinuity within the
buffer.  This is **not** evidence of a physical identity change.  It may
be caused by tracker re-identification after occlusion, two distinct
objects occupying the same entity-key slot, or genuinely different
people.  The name is deliberately non-assertive.

## Boundedness

All values are derived from a `BoundedObservationBuffer` of capacity
`observation_buffer_maxlen` (default 32).  Worst-case computation is
O(N) where N = maxlen.  No unbounded state is stored on `Candidate`.

When the buffer evicts its oldest observation, `continuity` is
recomputed from the remaining N observations.  `primary_track_id` may
change if the evicted observation was the oldest.  `possible_id_switch_count`
may decrease if an evicted pair contained a switch.

## Current Architecture Note

Under the current `CandidateLayer` keying scheme, `CandidateKey` includes
the `track_id` (via `EntityKey`), so a track-ID change creates a *new*
candidate rather than appending to the existing one.  Within a single
candidate's buffer all observations therefore have the same `track_id`,
and `possible_id_switch_count` is always 0 in the current pipeline.

The R4 metric is correctly implemented and will fire when the pipeline
eventually routes re-assigned track IDs to the same logical candidate
(future work).  The `_compute_continuity` helper is also exposed for
direct unit testing with mixed-ID buffers.

## Files changed

| File | Change |
|------|--------|
| `inference/core/events.py` | Added `TrackContinuity` frozen dataclass; added `continuity` field to `Candidate` with `default_factory`; extended `Candidate.to_dict()` to include `continuity`; added `TrackContinuity` to `__all__` |
| `inference/core/candidate_layer.py` | Added `TrackContinuity` to imports; added module-level `_compute_continuity(buffer)` helper; called it in `record_observation` after each `cand.add_observation(obs)` |
| `tests/test_phase2e_shadow_config.py` | Added `TestR4TrackContinuity` (9 tests) |

## Production impact

**SHADOW-ONLY.**  No event file, no CSV column, no confirmation gate
reads `Candidate.continuity`.  The legacy `Config`, `process_detection`,
NMS, and evidence-saving paths are unchanged.

## Remaining limitations

- No calibration — `possible_id_switch_count` is an observational count;
  no threshold has been calibrated to distinguish benign tracker re-ID
  from genuine identity changes.
- Single-entity-key scope — continuity is computed per
  `(EntityKey, candidate_class)` pair; cross-entity-key identity is out
  of scope.
- No temporal persistence — continuity is recomputed from the bounded
  buffer each call; observations older than `maxlen` are forgotten, so
  long-horizon ID history is not retained.
- No physical-identity assertion — by design, the metric does not claim
  whether a switch is a real person change or a tracker artifact.
