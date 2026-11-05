# YOLOv8 Exam-Cheating Inference — Third-Pass Audit & Engineering Specification

> **Third-pass audit (2026-06-10).** Every claim is grounded in the *actual* current
> source. The authoritative files are:
> - `inference/main.py` (64 lines) — thin orchestrator
> - `inference/core/pipeline.py` (304 lines) — the entire detection/tracking/event logic
> - `inference/core/nms.py` (47 lines) — custom NMS
> - `inference/config/constants.py` (48 lines) — frozen thresholds
> - `inference/config/paths.py` (63 lines) — root-anchored path resolution
> - `inference/utils/image.py` (28 lines) — crop expand/validate/save
> - `inference/utils/csv_log.py` (32 lines) — CSV append
> - `inference/utils/logging.py` (12 lines) — timestamped log
> - `inference/web/app.py` (35 lines) — Flask MJPEG endpoint
> - `inference/original_backup/main.py` (482 lines) — the original monolith

**IMPORTANT CORRECTION:** The prior (`inference-audit.md` v2) was written against a
*larger, multi-camera version* of `main.py` that included `run_worker`, `--num-cameras`,
`Inference` class, etc. **None of that exists in the current source.** The current
`main.py` is 64 lines; all logic lives in `core/pipeline.py`. This third pass re-
derives everything from the actual 304-line `pipeline.py` and corrects the record.
Line numbers below refer to the current source files.

---

## 1. Executive Summary

The system is a **single-video, single-processing-thread, file-input** pipeline:

```
main.py → CheatingDetectionPipeline (thread, daemon)
  → cv2.VideoCapture(file) → per-frame loop
    → pose_model.track(persist=True) → NMS → process_detection(id_info_pose)
    → object_model.track(persist=True) → NMS → process_detection(id_info_obj)
    → frame_queue(maxsize=1) ← Flask generate_frames consumer
```

**What it does correctly (SOURCE-VERIFIED):**
- Two YOLO models (pose task + detect task) each run BoT-SORT via `track(persist=True)`.
- Custom NMS (`apply_nms`) runs over both model outputs before event logic.
- Per-track state dict `id_info_*` tracks `first_time`, `last_saved`, `last_class`, `last_conf`.
- Cooldown gate (90s) and persistence gate (`REQUIRED_DETECTION_TIME = 5s`) prevent save spam.
- Crop validation rejects blank/oversaturated crops by mean brightness.
- CSV + file log record every saved evidence.

**What it does NOT do (architectural gaps, SOURCE-VERIFIED):**
- No **event** concept: no `IDLE/CANDIDATE/CONFIRMED/COOLDOWN` state machine.
  A track *is* the event — there is no candidate tier.
- No **person↔object association**: the two models produce independent tracks.
  There is no spatial or temporal cross-check between `id_info_pose` and `id_info_obj`.
- No **event confidence** distinct from **model confidence**: `last_conf` is the last
  raw model prediction, not an accumulated measure of how certain the system is that
  a cheating event is confirmed.
- No **class-switching protection**: a single `last_class != class_name` check resets
  `first_time`, but does NOT reset `last_saved` — so a class flip can bypass cooldown.
- No **frame-rate awareness**: `REQUIRED_DETECTION_TIME = 5` is compared against
  `datetime.now()` (wall-clock), not frame count. If inference is slower than
  source FPS, "5 seconds of detection" actually spans fewer video-frames than intended.

**What cannot be confirmed from source (RUNTIME VALIDATION REQUIRED):**
- Actual Jetson Orin Nano FPS (inference + tracking + NMS + crop + JPEG encode).
- BoT-SORT track-ID stability under occlusion (Ultralytics version-specific).
- Model confidence calibration (whether 0.3 is appropriately tuned or too low).
- MJPEG end-to-end latency and multi-client behavior.
- Thermal throttling behavior under sustained load on Jetson.

---

## 2. System Objective & Constraints

| Constraint | Value | Source |
|---|---|---|
| Deployment target | NVIDIA Jetson Orin Nano (edge) | task statement |
| Mode | Real-time exam monitoring (not offline forensics) | task statement |
| Error-cost priority | **False positives > false negatives** (precision-first) | task statement |
| Desired flow | detection → candidate → **confirmed event** → evidence → cooldown | task statement |
| Models | pose (`task='pose'`, 3 classes) + object (`task='detect'`, 2 classes) | `pipeline.py:55,64` |
| Tracker | BoT-SORT via `track(persist=True)` | `pipeline.py:158,208` |
| Detection threshold | `conf=0.3` for both models | `constants.py:18-19` |
| NMS IoU | `0.45` | `constants.py:26` |
| Persistence gate | `REQUIRED_DETECTION_TIME = 5` (seconds, wall-clock) | `constants.py:36` |
| Cooldown | `POSE_COOLDOWN = 90`, `OBJECT_COOLDOWN = 90` (seconds) | `constants.py:34-35` |
| Crop brightness floor | `MIN_CROP_BRIGHTNESS = 30` (mean grayscale) | `constants.py:23` |

**Key design principle for next phase:** detection may be permissive; confirmation
must be conservative. A single noisy model prediction must not become a confirmed
event. The current code's only "confirmation" is wall-clock persistence on a per-track
dict — there is no candidate tier and no cross-model evidence.

---

## 3. Scope & Methodology

- All claims are re-derived from the **current** source listed in the header.
- The prior audit (v2) was written against a *larger* `main.py` (with `run_worker`,
  `--num-cameras`, `Inference` class) that is **not present** in the tree. This pass
  treats the actual 64-line `main.py` + 304-line `pipeline.py` as the source of truth.
- No source code was modified. The only output is this file.
- Each finding is classified: **CONFIRMED**, **REFUTED**, **CHANGED**, **NEW**, or
  **UNVERIFIED — REQUIRES RUNTIME VALIDATION**.

---

## 4. Current Architecture (as-built)

### 4.1 Thread model
```
main()                          process_video() [daemon thread]
 └─ CheatingDetectionPipeline ───┬─ cv2.VideoCapture
    └─ frame_queue(maxsize=1)   │  while cap.isOpened():
    └─ stop_event (Event)       │    frame = cap.read()
                                 │    process_frame(frame):
    web/app.py generate_frames ──┤      detect_pose  → track+NMS+process_detection
       (Flask worker threads)     │      detect_objects → track+NMS+process_detection
                                 │    frame_queue.put(annotated.copy(), block=False)
                                 │    display_results() [cv2.imshow + waitKey]
                                 └─ cap.release(); log_file.close()
```
- **Producer:** one `process_video` thread (daemon).
- **Consumer:** one Flask worker per `/video_feed` browser connection; each runs
  `generate_frames()` which loops `frame_queue.get(timeout=1)`.
- **maxsize=1:** at most one annotated frame is ever in flight; `block=False` +
  `except queue.Full: pass` (pipeline.py:109-111) means a slow consumer silently drops.

### 4.2 Event concept — NOT IMPLEMENTED
There is no `IDLE/CANDIDATE/CONFIRMED/COOLDOWN` state machine. The closest thing is:

```
process_detection(track_id, class, box, ...):   pipeline.py:256-279
  id_info[track_id] = {first_time, last_saved, last_class, last_conf}   (created on first-seen)
  if last_saved and now-last_saved < cooldown:  return         (COOLDOWN gate)
  if last_class != class:  first_time = now; last_class = class   (CLASS-FLIP reset)
  elif now-first_time >= REQUIRED_DETECTION_TIME:  SAVE CROP + CSV   (PERSISTENCE gate)
```
A "confirmed event" is simply: *a track seen under a save-qualifying class for ≥5 wall-clock
seconds, that has not re-saved within the last 90s.* There is no separate candidate tier,
no minimum-frame-count, no confidence aggregation, no class-stability check.

### 4.3 Person↔object association — NOT IMPLEMENTED
`id_info_pose` and `id_info_obj` are **separate dicts keyed by each model's own track IDs**.
There is **no code** that checks whether a pose-track's box overlaps/contains a
object-track's box, or that a person and an object are co-temporal. A `provide-object`
pose class (a person hand-off behavior) and a `smartphone` object detection are
independent. The system cannot answer "which student is holding the phone."

### 4.4 Evidence is a **crop**, not a full frame
- `expand_crop(box, scale)` clamps to image bounds (`image.py:8-16`).
- Pose save scale = `2.5×`, object save scale = `4.5×` (`constants.py:21-22`).
- Full frames are saved *separately* to `output/frames/` every `FRAME_SAVE_INTERVAL`
  seconds and one `room_*.jpg` at `frame_idx == 1`. The **evidence** crop is a region
  of the *annotated* frame — a zoomed box, not the full context.

---

## 5. Findings — Reconciliation of Prior Audit (v2) Against Current Source

Each entry: **claim from v2 → verdict + evidence in current source.**

| # | v2 Claim | Verdict | Evidence |
|---|----------|---------|----------|
| F1 | "Multiple cameras supported via `--num-cameras`" | **REFUTED** | No such flag exists. `main.py` (64 L) takes a single `sys.argv[1]` video path. `original_backup/main.py` (482 L) has a `--num-cameras` argparse but it is the *backup*, not the live code. |
| F2 | "`run_worker` spawns per-camera threads" | **REFUTED** | `run_worker` does not appear in any live file. The live `main.py` spawns exactly one `process_video` thread. |
| F3 | "Inference class wraps model + queue + web" | **REFUTED** | No `Inference` class exists in the live tree. `CheatingDetectionPipeline` (`pipeline.py:38`) is the sole class. |
| F4 | "Per-camera `id_info` dict" | **CHANGED** | Single shared `id_info_pose` / `id_info_obj` (`pipeline.py:67-68`). No camera isolation (irrelevant now: one video input). |
| F5 | "Custom NMS (`apply_nms`) replaces YOLO's built-in NMS" | **CONFIRMED** | `nms.py:8-47`; called at `pipeline.py:163,213`. |
| F6 | "Class-flip resets `first_time` but NOT `last_saved`" | **CONFIRMED** | `pipeline.py:270-273`: `if last_class != class_name: first_time = now; last_class = class_name` — `last_saved` is untouched. |
| F7 | "No person↔object association" | **CONFIRMED** | `id_info_pose` and `id_info_obj` are independent dicts. No IoU / containment / co-temporal check exists. |
| F8 | "Evidence is a crop, not a full frame" | **CONFIRMED** | `pipeline.py:285-293` — `expand_crop(box, scale)` then `save_crop`. |
| F9 | "`REQUIRED_DETECTION_TIME = 5` is wall-clock, not frame-count" | **CONFIRMED** | `pipeline.py:274-275`: `now = datetime.now(); if now - first_time >= timedelta(seconds=REQUIRED_DETECTION_TIME)`. |
| F10 | "No event state machine (IDLE/CANDIDATE/CONFIRMED/COOLDOWN)" | **CONFIRMED** | The only gates are the class-flip reset (line 270) and the cooldown check (line 265-267). No explicit state. |
| F11 | "Flask MJPEG serves annotated frames" | **CONFIRMED** | `web/app.py:28-35` — `generate_frames()` pulls from `pipeline.frame_queue`, encodes JPEG, yields MJPEG chunks. |
| F12 | "Crop brightness floor rejects blank/oversat" | **CONFIRMED** | `image.py:18-28` — `is_crop_valid` computes mean grayscale, rejects < `MIN_CROP_BRIGHTNESS` (30). |

---

## 6. New Findings (Not in v2)

### N1 — `frame_queue` silently drops frames under load
`pipeline.py:109-111`:
```python
try:
    self.frame_queue.put(annotated, block=False)
except queue.Full:
    pass
```
No log, no counter. If the Flask consumer (or `cv2.imshow`) lags, frames are lost
silently. On a Jetson with thermal throttling, this could mean the web feed shows
stale frames while detection continues — operators see a frozen view and assume
the system is idle.

### N2 — `display_results` blocks the processing thread
`pipeline.py:114-118`:
```python
cv2.imshow("Exam Cheating Detection", annotated)
cv2.waitKey(1) & 0xFF
```
`waitKey(1)` adds ~1-15 ms per frame of GUI latency *inside* the single processing
thread. On headless Jetson, `cv2.imshow` requires a virtual framebuffer (`xvfb-run`
or similar). If not present, it may crash or silently no-op depending on the OpenCV
build. **Runtime validation required.**

### N3 — Log file is opened once and never rotated
`pipeline.py:84`: `self.log_file = open(log_path, 'w')` — mode `'w'` truncates any
existing log on each process restart. No size cap, no rotation. A long exam session
could grow the log to gigabytes.

### N4 — `original_backup/main.py` is a 482-line dead monolith in the tree
It references `Inference`, `run_worker`, `--num-cameras` — none of which exist in the
live code. Leaving it in the tree risks a developer copying patterns from the backup
into the live pipeline, reintroducing the multi-camera assumptions.

### N5 — `requirements.txt` lists `cv2` as a separate package
`inference/requirements.txt` line 5: `cv2`. There is no PyPI package named `cv2`;
the correct package is `opencv-python` (already listed on line 3). A clean
`pip install -r requirements.txt` will fail with "No matching distribution found
for cv2" on many PyPI mirrors/versions. Line 6 also duplicates `flask`.

### N6 — No test coverage
No `tests/` directory, no test files anywhere under `inference/`. The NMS
implementation (`nms.py`), the crop logic (`image.py`), and the event gates
(`pipeline.py:256-293`) are all untested.

### N7 — Track-ID drift under occlusion resets persistence
`pipeline.py:158, 208` use `track(persist=True)` which returns track IDs managed
internally by BoT-SORT. If a track is occluded and re-identified, BoT-SORT may
assign a new ID for the same physical person, resetting `first_time` and
potentially bypassing the 5-second persistence gate. This is a known limitation
of all tracker-based persistence and is **UNVERIFIED — REQUIRES RUNTIME
VALIDATION** on Jetson with the actual model weights.

---

## 7. Runtime Validation Checklist (Jetson Orin Nano)

These items **cannot be determined from source** and require on-device testing:

- [ ] **FPS measurement:** Run inference on a 30 fps 720p video on Jetson Orin Nano
      with the exact weights in use. Record per-frame time (inference + BoT-SORT +
      NMS + crop + JPEG encode). The system is real-time-capable only if avg frame
      time ≤ 33.3 ms.
- [ ] **Thermal throttling:** Sustained 30-minute run. Log `tegrastats` GPU/CPU temp.
      Confirm no throttling in the 10 W / 15 W power modes.
- [ ] **BoT-SORT ID stability:** Run a video with a person occluded for 10 s.
      Confirm the same track ID resumes. If the ID changes, `first_time` resets
      and the system misses events.
- [ ] **Conf=0.3 calibration:** On a representative exam-room video, count
      false-positive detections per minute at conf=0.3 vs conf=0.5. The task says
      precision first; 0.3 may be too permissive.
- [ ] **`cv2.imshow` headless:** Confirm whether `display_results` works on Jetson
      without a display. If it crashes, `display_results` must be gated behind a
      headless flag or removed.
- [ ] **MJPEG multi-client:** Open 2+ browser tabs on `/video_feed`. Confirm
      `generate_frames` handles concurrent consumers from a `maxsize=1` queue
      (consumers compete for the single queued frame).
- [ ] **Log file growth:** Measure `detection_log.txt` size after a 2-hour run.
      Plan rotation if >10 MB.
- [ ] **Model weight format:** Confirm whether the deployed models are `.pt`
      (PyTorch) or `.engine` (TensorRT). TensorRT on Jetson is strongly preferred
      for FPS.

---

## 8. Engineering Specification — Target Design (Next Phase)

This section is a **specification**, not a plan to implement now. It describes
the architecture the next development phase should target, based on the task
statement: *detection → candidate → confirmed event → evidence → cooldown.*

### 8.1 Event State Machine

```
IDLE ──[detected]──► CANDIDATE ──[persisted ≥ N frames]──► CONFIRMED ──► COOLDOWN
  ▲                       │                                                   │
  │  [class flip / lost > M frames]                                          │
  └────────────────────────┴────────────────��────────────────────────────────┘
                            (cooldown elapsed → back to IDLE)
```

| State | Entry | Exit |
|---|---|---|
| `IDLE` | default | any detection of a save-qualifying class → `CANDIDATE` |
| `CANDIDATE` | first detection of a save-qualifying class for a track | persisted ≥ `CANDIDATE_FRAMES` frames **AND** conf ≥ `CONFIRM_CONF` → `CONFIRMED`; lost ≥ `LOST_TOLERANCE` frames → `IDLE`; class flip to non-qualifying → `IDLE` |
| `CONFIRMED` | candidate satisfies both frame-count and confidence gates | save evidence bundle, enter cooldown |
| `COOLDOWN` | after save | `COOLDOWN_SECONDS` (per class) elapses → `IDLE` |

**`CANDIDATE_FRAMES`:** minimum number of consecutive frames in which the track is
seen under a save-qualifying class before confirmation. Proposed default: **10 frames**
(~0.33 s at 30 fps, ~2 s at 5 fps — must be tuned to actual measured FPS).

**`CONFIRM_CONF`:** minimum mean model confidence over the candidate window.
Proposed default: **0.45** (above the 0.3 detection floor; tunable).

**`LOST_TOLERANCE`:** maximum consecutive frames a track may be undetected before
the candidate reverts to `IDLE`. Proposed default: **5 frames**.

### 8.2 Person↔Object Association

Add a spatial cross-check per frame:

```python
for pose_track in pose_tracks:
    for obj_track in object_tracks:
        if iou(pose_track.box, obj_track.box) >= ASSOC_IOU_THRESHOLD:  # 0.3
            associate(pose_track.id, obj_track.id, frame_idx)
```

- `ASSOC_IOU_THRESHOLD = 0.3` (IoU ≥ 0.3 = meaningful overlap for a phone-in-hand).
- Association is **per-frame, not persistent**: if the object leaves the person's
  box for > 2 frames, the association breaks.
- Store `(person_id, object_id)` pairs in the event log so evidence is
  *who + what*, not just two independent crops.

### 8.3 Evidence Bundle

On `CONFIRMED`, save:
1. **Full frame** (the annotated frame, not a crop) → `evidence/{event_id}/frame.jpg`
2. **Person crop** (expanded 2.5×) → `evidence/{event_id}/person.jpg`
3. **Object crop** (expanded 4.5×, only if an object was associated) → `evidence/{event_id}/object.jpg`
4. **Event record** (JSON) → `evidence/{event_id}/event.json`:
   ```json
   {
     "timestamp": "ISO-8601",
     "person_track_id": 42,
     "object_track_id": 17,
     "pose_class": "provide-object",
     "object_class": "smartphone",
     "candidate_start_frame": 12040,
     "confirm_frame": 12050,
     "mean_conf": 0.62,
     "frames_observed": 15
   }
   ```
5. **CSV row** (as now) with the added `event_id` field.

### 8.4 Frame-Rate-Aware Persistence

Replace the wall-clock `REQUIRED_DETECTION_TIME = 5` with a **frame-count gate**:

```python
if frames_in_candidate_state >= CANDIDATE_FRAMES:
    confirm()
```

This makes the system FPS-agnostic. The 5-second concept is preserved as
`CANDIDATE_FRAMES = round(FPS * 5)` computed at startup from
`cap.get(cv2.CAP_PROP_FPS)`.

### 8.5 Cooldown Scope Fix

Current bug (F6): class-flip resets `first_time` but not `last_saved`,
allowing a rapid A→B class flip to bypass cooldown. Fix:

```python
if last_class != class_name:
    first_time = now
    last_class = class_name
    last_saved = 0   # RESET — new class gets its own cooldown window
```

### 8.6 `frame_queue` Overflow Safety

Replace the silent `pass`:

```python
try:
    self.frame_queue.put(annotated, block=False)
except queue.Full:
    self.dropped_frames += 1
    if self.dropped_frames % 300 == 1:
        log_info(f"Dropped {self.dropped_frames} frames (queue full)")
```

Expose `dropped_frames` in a `/stats` endpoint.

### 8.7 Headless Mode

```python
if args.headless or not os.environ.get("DISPLAY"):
    skip display_results()
```

On Jetson, run without `xvfb-run` unless a monitor is physically attached.

### 8.8 `requirements.txt` Fix

```
ultralytics>=8.0
flask>=3.0
opencv-python>=4.8
numpy>=1.24
```

(Remove the `cv2` entry — no such PyPI package. Remove the duplicate `flask`.)

### 8.9 Test Skeleton

Minimum tests to write in `tests/`:
- `test_nms.py` — verify `apply_nms` correctly suppresses overlapping boxes.
- `test_crop.py` — verify `expand_crop` clamping and `is_crop_valid` brightness gate.
- `test_event_state.py` — unit-test the IDLE→CANDIDATE→CONFIRMED→COOLDOWN transitions.
- `test_association.py` — verify IoU ≥ 0.3 triggers association, < 0.3 does not.
- `test_cooldown_reset.py` — verify class-flip resets both `first_time` AND `last_saved`.

---

## 9. Prioritisation (Recommended Order of Work)

| Priority | Item | Effort | Risk if Skipped |
|---|---|---|---|
| **P0** | Fix `requirements.txt` (N5) | 5 min | `pip install` fails on clean env |
| **P0** | Cooldown-scope fix (F6, §8.5) | 30 min | Class-flip bypasses 90 s cooldown → event spam |
| **P1** | Event state machine (§8.1) | 4 h | No candidate tier → false positives on noisy frames |
| **P1** | Frame-rate-aware persistence (§8.4) | 1 h | Slow inference on Jetson → "5 s" is actually 2 s of video |
| **P1** | `frame_queue` overflow logging (N1, §8.6) | 30 min | Silent frame loss undiagnosed |
| **P2** | Person↔object association (§8.2) | 6 h | Cannot answer "who has the phone" |
| **P2** | Evidence bundle upgrade (§8.3) | 2 h | Crops lack context; event.json missing |
| **P2** | Headless mode (N2, §8.7) | 1 h | `cv2.imshow` crashes on headless Jetson |
| **P3** | Log rotation (N3) | 1 h | Gigabyte log file over long exams |
| **P3** | Test skeleton (§8.9) | 4 h | Regression risk on any refactor |
| **P3** | Remove `original_backup/` (N4) | 0 min | Developer confusion; copy-paste risk |

---

## 10. Open Questions for the User

1. **Model format on Jetson:** Are the deployed weights `.pt` (PyTorch) or
   `.engine` (TensorRT)? This determines whether `ultralytics` alone is sufficient
   or TensorRT export is part of the next phase.
2. **Exam room camera geometry:** How many cameras per room? The live `main.py`
   is single-video. If multi-camera is required, a process-per-camera design
   (one `CheatingDetectionPipeline` per camera, one Flask app) is needed —
   the current single-thread model does not scale.
3. **What constitutes a "confirmed event" for the exam proctor?** Is 10 frames
   (0.33 s at 30 fps) enough candidate persistence, or does the task expect
   the original 5 s? This is the single most impactful tuning parameter.
4. **Is the Flask web feed the primary operator interface, or is there a
   separate dashboard?** The current `/video_feed` MJPEG endpoint drops frames
   silently; if this is the primary monitoring UI, the overflow logging and
   `/stats` endpoint are P0, not P1.
5. **What is the acceptable false-positive rate?** The task says
   false positives are worse than false negatives — but what concrete number?
   1 FP per hour? 1 FP per exam session? This determines `CONFIRM_CONF` and
   `CANDIDATE_FRAMES` calibration.

---

## Appendix A — File Line Reference (Current Source)

| File | Lines | Key Symbols |
|---|---|---|
| `inference/main.py` | 64 | `main()` — orchestrator |
| `inference/core/pipeline.py` | 304 | `CheatingDetectionPipeline`, `process_video`, `process_frame`, `detect_pose`, `detect_objects`, `process_detection` |
| `inference/core/nms.py` | 47 | `apply_nms` |
| `inference/config/constants.py` | 48 | All threshold constants |
| `inference/config/paths.py` | 63 | Path resolution |
| `inference/utils/image.py` | 28 | `expand_crop`, `is_crop_valid`, `save_crop` |
| `inference/utils/csv_log.py` | 32 | `append_csv_row` |
| `inference/utils/logging.py` | 12 | `log_info` |
| `inference/web/app.py` | 35 | Flask app, `generate_frames` |
| `inference/original_backup/main.py` | 482 | Dead monolith (do not reference) |
