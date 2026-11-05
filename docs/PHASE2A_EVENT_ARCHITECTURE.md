# Phase 2A — Event Architecture Design & Code Audit

**Project:** yolov8-exam-cheating (`inference/`)
**Phase status:** analysis/design only — **no production code modified.**
**Verdict:** `PHASE 2A — PASS WITH OPEN DESIGN QUESTIONS`
**Files changed:** `NONE`

---

## 1. Current event architecture (what the code actually does)

The pipeline is in `inference/core/pipeline.py`. Two YOLO models run per frame, each with
Ultralytics' built-in tracker (`persist=True`), and event logic is a wall-clock timer
attached to each track ID.

### Exact flow (verified line-by-line against `pipeline.py` + `constants.py`)

```
per frame, in processing_thread:
  1. cap.read() frame                     pipeline.py:103
  2. current_time = datetime.now()        pipeline.py:112   ← WALL CLOCK, video-time discarded
  3. detect_pose → pose_model.track(conf=0.3, iou=0.45, persist=True, imgsz=640)   :221
  4. apply_nms(pose boxes, 0.45)         :258
  5. for each kept pose box:
        if class in SAVE_POSE_CLASSES ("provide-object","see-friends-work"):
            process_detection(... id_info_pose, POSE_COOLDOWN=90, OUTPUT_POSE_BASE, scale=2.5)  :279-290
  6. detect_objects → object_model.track(same params)              :306
  7. apply_nms(object boxes, 0.45)       :343
  8. for each kept object box:
        if class in SAVE_OBJECT_CLASSES ("smartphone"):
            process_detection(... id_info_obj, OBJECT_COOLDOWN=90, OUTPUT_OBJ_BASE, scale=4.5)  :363-374
  9. push annotated frame to frame_queue (maxsize=1, drop-on-full)  :138-147
```

`process_detection` (pipeline.py:377-420) is the **entire** event engine:

```python
if track_id not in id_info:
    id_info[track_id] = {'first_time': now, 'last_saved': None,
                         'last_class': cls, 'last_conf': conf}   # :380-386
info = id_info[track_id]

if info['last_saved'] and (now - info['last_saved']).total_seconds() < cooldown:   # :388
    return                                                            # COOLDOWN GATE (90 s)

if info['last_class'] != cls:                                         # :390  CLASS-FLIP BRANCH
    info['first_time'] = now                                          # timer restarts
    info['last_class'] = cls                                          # cooldown NOT touched
elif (now - info['first_time']).total_seconds() >= REQUIRED_DETECTION_TIME:  # :393 PERSISTENCE GATE
    crop = expand_crop(...)  # :394
    if is_valid_crop(crop, 30):                                        # BRIGHTNESS GATE :396
        save_crop(...)                                                # :404/417
        log_detection_to_csv(track_id, filename, cls,
                             info['last_conf'], now)                 # :405/418  ← uses STALE conf
        info['last_saved'] = now                                       # :420
```

### What an "event" is today
An event = **a crop JPEG + one CSV row**. It is emitted when **all four** hold for a
`track_id` in `id_info_pose` or `id_info_obj`:
1. class ∈ SAVE list,
2. `now - last_saved ≥ 90 s` (or no prior save),
3. `last_class` unchanged and `now - first_time ≥ 5 s`,
4. crop mean brightness ≥ 30.

### Answers to the 19 flow points

| # | Question | Actual code behavior |
|---|---|---|
| 1 | Raw detections | Two `YOLO.track()` calls/frame (`:221`, `:306`), conf=0.3, iou=0.45, imgsz=640. |
| 2 | NMS | `apply_nms` runs twice/frame (`:258`, `:347`), class-agnostic greedy, IoU 0.45. Redundant after the model's internal NMS. |
| 3 | Tracking | Ultralytics built-in tracker, `persist=True`. **Default = BoT-SORT** (`botsort.yaml`), verified in installed lib. No keypoint input — bbox-only. |
| 4 | Track IDs | From `boxes.id` (`:251`, `:336`); consumed as dict keys in `id_info_pose`/`id_info_obj` (`:33-34`, `:380`). |
| 5 | Class filter | Post-detection, via `SAVE_POSE_CLASSES`/`SAVE_OBJECT_CLASSES` (`:279`, `:363`). `no-cheating` & `calculator` are drawn but never save. |
| 6 | `first_time` init | First sight of a track ID (`:382`). |
| 7 | `last_saved` update | Only after a crop actually passes brightness and is written (`:420`). |
| 8 | `last_class` update | On class change (`:392`). |
| 9 | `last_conf` update | **NEVER updated after first sight.** Set once at `:385`; never reassigned anywhere. Lines `:397`, `:405`, `:412` read the stale first-frame value. |
| 10 | REQUIRED_DETECTION_TIME | `:393`, `(now-first_time) ≥ 5 s`, wall-clock. |
| 11 | Cooldown | `:388`, per track_id, 90 s, wall-clock. |
| 12 | Evidence saved | On gate pass: crop JPEG + CSV row (`:404-405`). Not on first sight. |
| 13 | "Event" | Crop + CSV row, gated as above. |
| 14 | Class change | `first_time` reset, `last_class` set; **cooldown survives**; save skipped that frame (`:390-392`). |
| 15 | Track disappears | **Nothing.** Entry lingers forever in `id_info` (no pruning). |
| 16 | Track ID changes | New ID = fresh entry; **old cooldown/persistence state is orphaned** — a new ID restarts both gates. |
| 17 | Pose↔object association | **None.** Separate dicts, no geometric link. |
| 18 | One person → many events | **Yes.** ID switch, pose+object double, and 90 s re-fire all produce separate events for one physical incident. |
| 19 | Many people share state | Each track ID is isolated, so no cross-person contamination of `first_time`/`last_saved`; the risk is the *opposite* — one person producing many IDs. |

### Clock (point E)
**All** persistence/cooldown comparisons use `datetime.now()` (wall clock). `src_fps`
is read (`:98`) only to print it — it is never used for timing. Video time is discarded.

---

## 2. Current problems (event-semantic, concrete)

### A. Track == Event
The system treats a *persistent track ID* as the unit of suspicion. There is no
candidate/confirmation separation: a track that has been *detected for 5 wall-clock
seconds* is deemed a confirmed event. "Detection persisting" is conflated with
"behavior is cheating." A persistent *false* detection therefore becomes a persistent
*false* accusation — the single biggest precision risk.

### B. No explicit candidate state
There is no state in which transient suspicious detections accumulate evidence without
producing output. `process_detection` either resets a timer or saves. Consequence: the
only protection against a 1-frame false positive is the 5 s timer; anything that
survives 5 s of wall-clock is immediately accused, with zero opportunity to say
"accumulate a little more before reporting."

### C. No separate concept of confirmation evidence
Confirmed: there is **no** distinction between detection confidence, temporal
persistence, track continuity, behavioral consistency, and event confidence. The
CSV's "Confidence Score" column is literally `info['last_conf']` — the *first-frame*
confidence, not a confidence in the event. There is no accumulated-evidence concept at all.

### D. Class-flip behavior (documented, not fixed)
- `first_time` **resets** on flip (`:391`) → the 5 s timer restarts.
- `last_saved` **does not reset** → the 90 s cooldown from the previous class's save
  survives the flip.
- Cooldown therefore **blocks** a fresh event of the *new* class for up to 90 s even
  though the new class's timer just restarted.
- `last_conf` stays stale.
- **Verified bug (pipeline.py:385 vs 397/405/412):** `last_conf` is assigned **only** when a
  track ID is first created (line 385) and is **never reassigned** in `process_detection`.
  The confidence written to the CSV and embedded in the evidence filename is therefore the
  *first-frame* confidence — potentially many seconds (or a class flip) older than the
  frame that actually triggered the save. A track first seen at conf 0.31 that later holds
  at 0.88 still reports "0.31". The real log shows this directly: track `2` reports `0.31`
  and track `57` reports `0.31` while others report 0.7–0.8 for the same class. This is
  the concrete symptom of problem C below — the "Confidence Score" a reviewer sees is stale
  detection confidence, not event confidence.
- This is an **accidental** asymmetry, not a designed semantic. It is observable in the
  real log: track `3` fires `smartphone` at 15:04:12, then `provide-object` at 15:04:21
  (9 s later, a different "event" for the same physical object/person).

### E. Wall-clock vs video-time
All timing is wall-clock. Consequences:
- **Processing slower than source FPS** (the normal case): "5 s of detection" means 5 s
  of *real time* while the video may have advanced far more (or far less) — the
  persistence gate is decoupled from video content.
- **Processing faster than source** (burst/offline): a long video processed quickly may
  never satisfy the 5 s wall-clock gate at all → events silently fail to fire.
- **Dropped frames / variable inference latency:** wall-clock gaps inflate the measured
  duration even when no new observations were made — a track that disappears for 5 s of
  wall-clock and reappears still "passes" persistence because `first_time` is never
  invalidated. **The timer never checks that the track is still being observed.**

---

## 3. Track continuity audit (Step 3)

Verified against the installed `ultralytics` library (`site-packages/.../cfg/`).

**KNOWN (from the library, no project override):**
- Default tracker = **BoT-SORT** (`default.yaml:134 → tracker: botsort.yaml`).
- `track_high_thresh = 0.25`, `track_low_thresh = 0.1`, `new_track_thresh = 0.25`,
  `track_buffer = 30`, `match_thresh = 0.8`, `gmc_method = sparseOptFlow`,
  `with_reid = False`.
- No project file sets `tracker=`, `track_buffer=`, `match_thresh=`, `cmatch=`, etc.
  All are library defaults.

**KNOWN (from project code):**
- Pipeline feeds the tracker **bboxes only**; keypoints are unused in the live path.
- No application-level lost/reappear/reid/merge handling. `id_info` entries are never
  deleted.
- `track_buffer = 30` frames: a track may be dropped after ~30 frames (1 s @ 30 fps) of
  non-detection, then a **new** ID is issued on reappearance.

**INFERRED:**
- With `track_high_thresh = 0.25` (very permissive) and bbox-only matching, ID stability
  under occlusion/low-confidence is weak; new IDs are likely on reappear — the real log
  (track IDs jumping 1→47→51→54→57→64→65→84→85 within ~10 min of one run) is consistent
  with this.

**NEEDS-VALIDATION (empirical, cannot be settled from code):**
- Actual ID-switch rate in real exam footage.
- Whether 30-frame buffer is long enough for typical occlusion durations in the room.
- Whether the observed ID jumps are occlusion, re-lighting, or model flicker.

> Distinguishing: the *configuration* is known; the *behavioral* reliability of IDs
> under real occlusion is **not** established and must be measured (see §12).

---

## 4. Temporal confirmation model (Step 4)

| Model | Definition | Strong point | Weak point for this project |
|---|---|---|---|
| **Frame-count** | N consecutive processed observations | Simple, decoupled from latency | Depends on processing FPS; "3 frames" means 0.1 s at 30 fps, 0.5 s at 6 fps — not a real duration. Fragile if frames are dropped. |
| **Video-time** | T seconds of *source* video timestamps | Matches "how long the behavior lasted in the video" | Requires trusting `cap.get(FPS)` (it is read but unverified); if the source FPS is wrong or frames drop, video-time is also wrong. |
| **Observation-time** | T seconds during which the detector actually *saw* the track | Robust to processing speed and dropped frames; measures real evidence accumulation | Needs a per-track "last observed" stamp and an observation counter; currently absent. |
| **Hybrid** | (duration of continuous observation) AND (≥ min observation count) | Directly encodes "consistent, repeated evidence" — the exact precision requirement | Most complex; must calibrate two knobs. |

**Recommendation (design, no numbers):** **Hybrid on observation-time**, i.e.
"track continuously observed for ≥ T *video/observation* seconds **and** ≥ N
observations." Rationale: the core principle is *conservative confirmation*. Frame-count
alone is meaningless without a known processing rate; pure wall-clock (the current bug)
is decoupled from content and is inflated by gaps. An observation-based duration plus a
minimum count is the only form that directly measures "repeated, consistent evidence"
rather than "time elapsed." The confirmation clock should be tied to **the frame
timestamps / observation sequence of the track**, not the wall clock.

---

## 5. Event confidence (Step 5)

Conceptual separation (no formula invented):

1. **Detection confidence** — the model's per-box `conf`. Permissive. Already filtered by 0.3.
2. **Candidate evidence** — accumulated while the suspicious detection persists:
   observations of the track, class stability, duration observed, confidence trend,
   spatial stability.
3. **Event confirmation** — a *decision gate*: "does the accumulated evidence meet the
   confirmation criteria?" This is the **only** step that may create a reportable event.
4. **Event confidence** — a separate quality score for the *confirmed* event,
   combining the evidence above. It is **not** the detection `conf`.

Evidence categories that can feed confirmation (each is currently *absent*):
model confidence, temporal persistence, confidence stability, track continuity,
class consistency, spatial consistency, person-object relationship, keypoint/pose
consistency, repeated observations, evidence-image quality (brightness is the only one
currently checked).

The current system conflates #1 and #4 (CSV "Confidence Score" = stale detection conf).
The design must keep #4 an independent, higher-order quantity.

---

## 6. Person ↔ object association (Step 6)

No association exists today. Candidate approaches:

| Approach | Strength | Weakness / failure mode | Cost | Fit for exam-cheating |
|---|---|---|---|---|
| 1. bbox IoU | Simple, standard | Person box is large; a held phone has tiny/zero IoU with the whole person box → near-useless here | Low | Poor (scale mismatch) |
| 2. object-center inside person box | Robust scale-wise | Works when the person box is tight; fails if the box is over-expanded (it is: 1.5×/2.5× crop factors) | Low | Good as a primary test |
| 3. person containment ratio | Precise | Requires knowing which box is "the person" | Low | Good, complements #2 |
| 4. center-point distance | Cheap | Needs a distance threshold → camera-geometry dependent | Low | Moderate |
| 5. hand/keypoint proximity | **Most accurate** — uses the pose model's hand keypoints | Pose model currently outputs them but the pipeline ignores them; keypoint detection is unreliable when hand is occluded by the object | Higher | Best signal *if* keypoints are trusted |
| 6. temporal association | Handles the object entering the person box over time | Alone, too permissive | Low | Supporting signal only |
| 7. hybrid spatial + temporal | Most robust | Most complex; needs calibration of multiple thresholds | Highest | Recommended for validation |

**Recommendation for further validation:** **Approach 7 (hybrid)**, whose spatial core is
**#2 (object-center inside person box) OR #5 (hand-keypoint proximity)**. Do **not**
assume IoU ≥ 0.3 (or any IoU) — with a 2.5×-expanded person crop, object/person IoU is
nearly meaningless; center-containment is the scale-robust primary test, and hand-keypoint
proximity is the strongest discriminative signal (it distinguishes *this* student
holding the phone from the phone merely being nearby another student's box). This must be
validated on real crops, not fixed numerically now.

---

## 7. State machine design (Step 7)

```
IDLE ──suspicious detection observed──▶ CANDIDATE
  ▲                                        │ evidence accumulates,
  │ (candidate drops out without event)    │ OR confirmation criteria met
  │                                        ▼
  │                                     CONFIRMED ──create event + evidence bundle──▶
  │                                        │
  │                                        ▼
  └──────────── IDLE ◀───── COOLDOWN ◀────┘ (suppression window for same incident)
```

- **IDLE** — no active suspicious candidate for this track/incident.
- **CANDIDATE** — suspicious detections are accumulating evidence. A candidate may
  dissolve (disappear/stop being suspicious) with **no event produced** — this is the
  key precision safeguard and does not exist today.
- **CONFIRMED** — evidence meets confirmation criteria. **The only state that creates a
  reportable event + evidence bundle.**
- **COOLDOWN** — suppresses re-confirmation of the same underlying incident for a
  configured window; then returns to IDLE.
- **LOST / TRACK_GAP** (separate state): **Yes, needed.** Because confirmation must be
  conservative and the tracker is bbox-only with `track_buffer=30`, a track can
  disappear and reappear with a *new* ID. A `LOST` state lets a candidate survive a short
  ID gap (preserving accumulated evidence) rather than reset on an ID switch, and lets
  long gaps dissolve the candidate. Without it, an ID switch both *resets* evidence
  (false negative) *and* creates a fresh candidate (false positive) for the same incident.
  A `LOST` state with a gap tolerance is what makes track-continuity robust.

---

## 8. Class transitions (Step 8)

Design semantics (avoid spam, avoid unnecessary resets, avoid cooldown bypass):

| Situation | Desired behavior |
|---|---|
| Same class continues | Evidence keeps accumulating; no reset. |
| Class changes **briefly** (≤ small jitter window) | Treat as the *same* candidate; do **not** reset the timer (a flicker between `provide-object`/`see-friends-work` should not restart confirmation). |
| Class changes **persistently** | Reset the confirmation timer for the *new* class; keep the candidate alive; carry continuity if the classes are in the same "suspicious" family. |
| Class disappears temporarily | Enter `LOST`; preserve evidence up to a gap limit, then dissolve to IDLE. |
| Track ID changes (same physical person) | Link new ID to the prior candidate (via spatial proximity / temporal) so the incident does not double-count or reset. |
| Multiple suspicious classes at once | Consolidate into **one incident** (one confirmation, one event) rather than one event per class. |

The current code does the *opposite* of this: it resets on any class change (point D)
and does not consolidate, which is a direct source of duplicate events.

---

## 9. Evidence model (Step 9)

A confirmed event should be an **evidence bundle** containing:
event ID · source video + start/end (video time) · track ID(s) · class(es) ·
event confidence (independent of detection conf) · detection-confidence stats
(min/max/mean over the candidate) · candidate start · confirmation time · source
frame index · annotated full frame · relevant crop · person–object association info ·
temporal-evidence summary (observation count, observed duration, gap count).

Separate the three evidence kinds:
- **Detection evidence** — what the model saw (boxes, confs, keypoints over the candidate window).
- **Decision evidence** — *why* it was confirmed (observation count, duration, consistency,
  which confirmation criteria were met).
- **Human evidence** — what a reviewer inspects (annotated frame + crop + a one-line
  human-readable verdict, e.g. "student held smartphone 8 s, hand-keypoint proximity 0.9").

Goal: a reviewer can judge the event **without** trusting a model number.

---

## 10. False-positive analysis (Step 10)

| FP scenario | Mitigating mechanism (and whether current design supports it) |
|---|---|
| Normal hand movement | Pose-class confirmation + duration; **not** addressed today (no behavioral consistency) |
| Ordinary head movement | Class filter (only `provide-object`/`see-friends-work`/`smartphone` save) — partially addressed |
| Adjusting clothing | Duration gate + class consistency — only duration today |
| Reaching for allowed material | Person-object association (is a *phone* actually held?) — **not** addressed |
| Temporary object overlap | Temporal persistence (≥T obs) — not robust today (wall-clock bug) |
| Object held by another person | Hand-keypoint proximity / center-containment association — **not** addressed |
| Occlusion | `LOST` state + track-buffer; ID-switch linking — not addressed today |
| Tracker ID switch | `LOST`/ID-linking + consolidation of incident — **not** addressed (today it causes *duplicates*) |
| Detector class flip | Class-jitter tolerance + incident consolidation — **not** addressed (today causes duplicate) |
| Low-confidence false detection | conf ≥ 0.3 already; but confirmation should require *stable/high* conf over the candidate — not addressed |
| Single-frame false detection | Minimum-observation-count gate — not addressed (only the 5 s timer today) |
| Crowded scenes | Person-object association + spatial consistency — not addressed |
| Motion blur / lighting | Brightness gate (min 30) — partially addressed; blur unaddressed |

> No mechanism above is claimed to *solve* a risk unless the current code implements it.
> The only FP guards present today are: SAVE-class filter, 5 s wall-clock timer, 90 s
> cooldown, and the brightness gate.

---

## 11. Numeric parameters requiring calibration (Step 11)

| Parameter | Meaning | Candidate unit | Must calibrate against |
|---|---|---|---|
| Candidate duration | min suspicious persistence before confirmation | video/observation seconds | false-positive rate |
| Min observations | observation count floor | frames/observations | sparse-processing & drop-frame robustness |
| Confirmation confidence | evidence-quality threshold | normalized 0–1 score | validation dataset (precision) |
| Cooldown duration | repeat-suppression window | seconds | incident frequency in real exams |
| LOST gap tolerance | max observation gap to keep a candidate | seconds/frames | tracker `track_buffer` + real occlusion durations |
| Association threshold | person-object spatial test | containment ratio / keypoint distance | camera geometry + real crops |
| Class-jitter window | how long a class flicker is "same class" | seconds | detector flicker rate |

All of these are **currently hardcoded with no validation basis** (the 5 s, 90 s, 0.3,
0.45, 30, 2.5, 4.5 in `constants.py` are all unexplained magic numbers). **None may be
hardcoded into Phase 2B without validation data.**

---

## 12. Calibration plan (Step 12)

**Metrics to measure:** FP per hour · FN rate · precision · recall · event-confirmation
latency · duplicate-event rate · track-ID failure rate · association failure rate.

**Validation dataset (to be constructed later, not now):**
- *Normal exam behavior* (sitting, reading, writing, stretching) — must yield **no**
  confirmed events (FP measurement).
- *Genuine suspicious behavior* (holding phone, passing notes, looking at neighbor's
  paper) — must yield confirmed events (recall/FN measurement).
- *Borderline* (brief hand-raise, a calculator legitimately used, an object set down).
- *Occlusion / re-lighting / motion-blur* clips — to measure ID-switch and association
  failure.
- *Multi-person* crowded frames — to measure person-object association and incident
  consolidation.
- *Object interactions* — phone picked up / put down, to calibrate association and
  LOST-gap.

**Procedure:** run the candidate pipeline with each parameter swept; record all metrics;
choose values that hold FP/hr under a target while keeping recall acceptable; freeze only
the validated values into config.

---

## 13. Proposed architecture (Step 13)

```
Model Detection ─▶ NMS ─▶ Tracking ─▶ Association ─▶ Candidate Evidence
Accumulation ─▶ Confirmation Decision ─▶ Event Creation ─▶ Evidence Bundle ─▶ Cooldown
```

Layer ownership (and the invariants that must hold):
- **Model Detection / NMS** — produce raw boxes; **never** create events.
- **Tracking** — produce stable-enough track IDs; no event logic.
- **Association** — link pose↔object into an *incident*; no event logic.
- **Candidate Evidence Accumulation** — build the candidate + its evidence; **never saves**.
- **Confirmation Decision** — the **only** layer that decides "reportable."
- **Event Creation + Evidence Bundle** — emit only on confirmation.
- **Cooldown** — suppress repeats of the same incident.

Current code violates two invariants: model-level persistence (the 5 s timer in
`process_detection`) *does* create events (no candidate/confirmation split), and there is
no association layer at all.

---

## 14. Implementation plan — Phase 2B (Step 14, smallest safe sequence)

The audit shows the confirmation split and the temporal/evidence fixes are prerequisites
for association to be useful. Revised order (association moves **after** evidence
accumulation):

1. **2B.1 — Event data structures.** Introduce `TrackState`, `Candidate`, `ConfirmedEvent`,
   `EvidenceBundle` classes + an `Incident` grouping. No behavior change; pure refactor of
   `id_info`. *Testable:* unit tests on state shapes.
2. **2B.2 — Temporal clock fix + observation accumulation.** Replace wall-clock
   persistence with observation-count + observation-time; add `LOST`/`TRACK_GAP`. *Testable:*
   synthetic frame sequences with dropped frames / slow processing.
3. **2B.3 — Candidate accumulation.** Build/extend candidates per incident; allow
   dissolve-without-event. *Testable:* transient detection produces **no** event.
4. **2B.4 — Confirmation decision.** Gate that turns a candidate into an event only when
   criteria met; emit the event + bundle. *Testable:* criteria not met ⇒ no event.
5. **2B.5 — Cooldown state.** Per-incident suppression; incident consolidation so a
   class flip or ID switch does not spawn a duplicate. *Testable:* flip/ID-switch ⇒ 1 event.
6. **2B.6 — Person↔object association.** Center-containment + hand-keypoint proximity
   (hybrid), feeding confirmation evidence. *Testable:* held phone associates to correct
   person; nearby phone does not.
7. **2B.7 — Structured evidence bundle + calibrated config.** Persist the bundle
   (detection/decision/human evidence); move all §11 parameters into config, values
   pending §12 calibration (use clearly-flagged provisional values until validated).

Each stage is independently testable and independently revertible; 2B.1–2B.5 are the
minimum set that delivers the precision goal, and 2B.6–2B.7 add association + evidence.

---

## 15. Verdict & open design questions

`PHASE 2A — PASS WITH OPEN DESIGN QUESTIONS`

**Open questions (must resolve before Phase 2B numeric work):**
1. **Clock choice** — observation-time + observation-count hybrid is recommended, but the
   exact `T` seconds and `N` observations require §12 validation data that does not yet exist.
2. **Tracker reliability** — BoT-SORT defaults (`track_buffer=30`, `track_high_thresh=0.25`,
   bbox-only) need an *measured* ID-switch rate before the `LOST` gap tolerance can be set.
3. **Association test** — hand-keypoint proximity is the strongest signal but depends on
   keypoint reliability in occlusion; needs real-crop validation to choose containment vs.
   keypoint as the primary test.
4. **Incident grouping** — how aggressively to merge a pose event + object event + ID-switch
   into a single incident is a design decision needing a small labeled set.

No production file was modified in this phase. Expected files changed: `NONE`.
