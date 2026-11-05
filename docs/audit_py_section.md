## 11. Python / Software Engineering Audit

### PY-001: `log_file` opened in main thread, closed in processing thread
**Lokasi** — `main.py` line 48 (`log_file = open(...)`), line 49 (passed to thread); `pipeline.py` line 127 (`log_file.close()`).
**Temuan** — The file handle is opened on the main thread and handed to the daemon processing thread. Close lives at line 127 inside `process_video`. If the thread is killed or an exception occurs before line 127, the handle leaks. The main thread's except block (lines 56-60) joins the thread but never closes `log_file`.
**Teori / Prinsip** — File handles should be closed by the thread that owns them, or via `try/finally` / context manager. Ownership of the close must be unambiguous.
**Perbandingan** — Risky
**Dampak** — File-handle leak on abnormal shutdown; on Windows, `detection_log.txt` stays locked. Buffered log lines lost.
**Solusi** — Open the file inside `process_video`, or wrap in `try/finally`. Add `log_file.close()` after `join()` in main, guarded by `if not log_file.closed:`.
**Prioritas** — HIGH
**Status** — Defect

---

### PY-002: Daemon thread risks silent resource leak on exit
**Lokasi** — `main.py` line 50 (`processing_thread.daemon = True`); `pipeline.py` lines 125-127.
**Temuan** — As a daemon thread, it is killed when the main thread exits. `cap.release()`, `cv2.destroyAllWindows()`, and `log_file.close()` may never execute. The C++ VideoCapture object's destructor may not run.
**Teori / Prinsip** — Daemon threads are unsafe for resources requiring explicit cleanup. Use non-daemon thread + `stop_event` + `join()` + explicit resource release.
**Perbandingan** — Risky
**Dampak** — Video decoder resources leak; log buffer lost on every shutdown.
**Solusi** — Remove `daemon = True`. After `join()`, call an explicit `pipeline.release_resources()` method wrapping all cleanup in `try/except`.
**Prioritas** — HIGH
**Status** — Defect

---

### PY-003: No exception handling around model loading
**Lokasi** — `pipeline.py` lines 52-69, `initialize_models()`, lines 55 and 64.
**Temuan** — `YOLO(path, task=...)` called without `try/except`. Missing/corrupt `.pt` or sentinel path (`<missing-pose-model>`) raises an unhandled `RuntimeError`/`FileNotFoundError` from the constructor.
**Teori / Prinsip** — Resource initialization should catch and re-raise with a clear message, or validate the path first.
**Perbandingan** — Risky
**Dampak** — Raw PyTorch/ultralytics traceback; user cannot diagnose the problem.
**Solusi** — Wrap in `try/except (OSError, RuntimeError)`, log a human-readable error, check `os.path.isfile()` first.
**Prioritas** — MEDIUM
**Status** — Defect

---

### PY-004: `cap.read()` failure treated as "end of video"
**Lokasi** — `pipeline.py` lines 87-90.
**Temuan** — `success, frame = cap.read()`; `if not success: break`. A mid-stream decode error returns `(False, None)` — indistinguishable from clean EOF. Log says "End of video stream reached" even for corruption.
**Teori / Prinsip** — Compare `frame_idx` against `CAP_PROP_FRAME_COUNT` to distinguish clean end from decode error.
**Perbandingan** — Partially Correct
**Dampak** — Mid-stream decode error silently truncates analysis; no events after the corrupt frame.
**Solusi** — Track expected frame count; log `[WARN] Decode error at frame N/expected` when `success` is False before the last frame.
**Prioritas** — MEDIUM
**Status** — Defect

---

### PY-005: Non-atomic check-then-act on `frame_queue`
**Lokasi** — `pipeline.py` lines 107-111.
**Temuan** — `if self.frame_queue.empty(): put(block=False)` — two separate operations. A consumer could `get()` between check and put. The `except queue.Full: pass` silently swallows the error with no log.
**Teori / Prinsip** — Idiomatic single-slot pattern: `get_nowait()` before `put()`, or use a lock. Never silently swallow `queue.Full`.
**Perbandingan** — Partially Correct
**Dampak** — Low risk (single producer). Silent frame loss undiagnosable.
**Solusi** — Replace with `get_nowait()` before `put()`, or lock-guarded buffer. Log dropped-frame count.
**Prioritas** — LOW
**Status** — Optimization Opportunity

---

### PY-006: `generate_frames` never terminates when video ends naturally
**Lokasi** — `pipeline.py` lines 294-303; `main.py` line 58.
**Temuan** — `stop_event` is set only by `KeyboardInterrupt` or 'q' keypress. When the video ends naturally (loop at line 86 exits), `stop_event` is never set. The generator loops forever on an empty queue. Flask thread blocked indefinitely.
**Teori / Prinsip** — A MJPEG generator must have a defined termination condition for natural video end.
**Perbandingan** — Incorrect
**Dampak** — Frozen last frame in browser; zombie Flask worker thread; no indication video is done.
**Solusi** — Set `self.stop_event` at the end of `process_video` (after line 127). Or add a `video_done` flag checked in the `while` condition.
**Prioritas** — HIGH
**Status** — Defect

---

### PY-007: `cv2.waitKey(1)` called in a non-main thread
**Lokasi** — `pipeline.py` line 114 and line 292.
**Temuan** — `process_video` runs in the daemon thread. `cv2.waitKey` is a GUI function tied to the main thread's event loop. Calling it from another thread is undefined behavior — may hang, segfault, or silently return 0.
**Teori / Prinsip** — GUI calls must be made from the thread that owns the GUI event loop. Use cross-thread signals for stop control.
**Perbandingan** — Risky
**Dampak** — On Windows with opencv-python GUI build, may hang or crash the processing thread.
**Solusi** — Gate `waitKey` behind a platform/DISPLAY check. For web-only path, add a REST `/stop` endpoint.
**Prioritas** — HIGH
**Status** — Defect

---

### PY-008: `DISPLAY` env-var check makes GUI unreachable on Windows
**Lokasi** — `pipeline.py` line 283.
**Temuan** — `if not os.environ.get('DISPLAY'): return`. On Windows, `DISPLAY` is never set, so `cv2.imshow` (line 291) never executes. The 'q' keypress handler (line 114) is also unreachable. No user control mechanism on Windows.
**Teori / Prinsip** — Platform-specific GUI availability should use `sys.platform` or `cv2.namedWindow` probe, not a Unix-specific env var.
**Perbandingan** — Incorrect
**Dampak** — No detection window on Windows; no keypress stop; only `Ctrl+C` works.
**Solusi** — Use `sys.platform` check or `Config.ENABLE_GUI` flag. Add `/stop` REST endpoint.
**Prioritas** — MEDIUM
**Status** — Defect

---

### PY-009: `cv2.VideoCapture.set()` may be silently ignored
**Lokasi** — `pipeline.py` lines 80-81.
**Temuan** — `cap.set(CAP_PROP_FRAME_WIDTH, 640)` on a file-based FFmpeg capture is often silently ignored (returns `False`). Actual frame size comes from source video. `imgsz=640` in `track()` handles model input, but `expand_crop` uses actual `img_w/img_h`. The `set()` calls are misleading dead code.
**Teori / Prinsip** — Check the return value of `set()`; log a warning if `False`.
**Perbandingan** — Partially Correct
**Dampak** — No functional bug, but code implies 640x640 enforcement when it does not.
**Solusi** — Remove the lines or add a log: `if not cap.set(...): log_info("[WARN] Could not set frame size; using source resolution")`.
**Prioritas** — LOW
**Status** — Optimization Opportunity

---

### PY-010: `id_info_pose` and `id_info_obj` grow unboundedly
**Lokasi** — `pipeline.py` lines 29-30, 258-264.
**Temuan** — Every track ID creates a dict entry. With `persist=True`, IDs are never reused. Entries are never pruned. Over a long video, memory grows linearly with distinct tracked IDs. `display_results` (line 285) reports `len(id_info_pose)` as "Tracking persons", which inflates with stale entries.
**Teori / Prinsip** — Per-track state should have a TTL sweep or max-size cap.
**Perbandingan** — Risky
**Dampak** — Slow memory leak; inflated tracking count display.
**Solusi** — Periodically sweep entries older than 5 minutes. Use post-sweep count in display.
**Prioritas** — MEDIUM
**Status** — Optimization Opportunity

---

### PY-011: No input validation on video path
**Lokasi** — `config/paths.py` lines 27-34; `pipeline.py` lines 75-78.
**Temuan** — `resolve_input_video` returns sentinel `input/<missing-video>` when no video found. This is passed directly to `VideoCapture`. No check that the path is valid, exists, or is readable.
**Teori / Prinsip** — Validate with `os.path.isfile()` and `os.access()` before handing to a C library.
**Perbandingan** — Partially Correct
**Dampak** — Opaque FFmpeg error on broken paths; user must read OpenCV logs.
**Solusi** — Add `os.path.isfile()` check before `VideoCapture`; log a specific error.
**Prioritas** — LOW
**Status** — Optimization Opportunity

---

### PY-012: `requirements.txt` contains an invalid and duplicated entry
**Lokasi** — `requirements.txt` lines 1-6.
**Temuan** — Line 5 lists `cv2` (not a pip package; the distribution is `opencv-python` on line 3). `pip install cv2` fails. Line 6 duplicates `flask` from line 2. No version pinning for any dependency.
**Teori / Prinsip** — List installable distribution names (PEP 503). Pin versions for reproducibility.
**Perbandingan** — Incorrect
**Dampak** — `pip install -r requirements.txt` fails. Unpinned deps risk incompatible torch/opencv combinations.
**Solusi** — Remove `cv2` and duplicate `flask`. Pin: `ultralytics>=8.0`, `flask>=2.3`, `opencv-python>=4.8`, `numpy>=1.24`, `torch>=2.0`.
**Prioritas** — HIGH
**Status** — Defect

---

### PY-013: `Config` class has no `__init__` and uses class-level attributes
**Lokasi** — `config/constants.py` lines 5-48; `main.py` line 37.
**Temuan** — `Config()` creates an instance with no instance state. All constants are class attributes. Mutating `config.CONF_THRESH_POSE` shadows the class attribute for that instance only; other code reading `Config.CONF_THRESH_POSE` gets the original.
**Teori / Prinsip** — Use `@dataclass(frozen=True)` or `__init__` that copies defaults into instance attributes.
**Perbandingan** — Partially Correct
**Dampak** — Accidental instance-attribute write creates silent divergence; hard to debug.
**Solusi** — Convert to `@dataclass(frozen=True)` or `__slots__` + `__init__`.
**Prioritas** — LOW
**Status** — Optimization Opportunity

---

### PY-014: Mutable class-level dicts `POSE_COLORS` / `OBJECT_COLORS`
**Lokasi** — `config/constants.py` lines 40-48.
**Temuan** — Mutable `dict` at class level. `config.POSE_COLORS["x"] = v` mutates the class dict for all instances. Current code only calls `.get()`, so no mutation in practice, but the danger is latent.
**Teori / Prinsip** — Use `types.MappingProxyType` or `Mapping` to make read-only.
**Perbandingan** — Risky
**Dampak** — Future code mutating the dict silently affects all pipeline instances.
**Solusi** — Wrap in `types.MappingProxyType({...})` or move to `__init__` as instance copies.
**Prioritas** — LOW
**Status** — Optimization Opportunity

---

### PY-015: `log_file.write()` without explicit flush
**Lokasi** — `utils/logging.py` line 11; `main.py` line 48.
**Temuan** — File opened with `open(path, "w")` uses block buffering (4096 bytes) by default. On crash/kill, up to one buffer of log lines is lost. `close()` at pipeline.py line 127 flushes, but only if reached.
**Teori / Prinsip** — For crash-survivable logs, use `buffering=1` (line-buffered) or call `flush()` per write.
**Perbandingan** — Risky
**Dampak** — Last 4 KB of log entries lost on crash. Final detection events (most important) may be missing.
**Solusi** — Open with `open(path, "w", buffering=1)` or add `log_file.flush()` after each write.
**Prioritas** �� MEDIUM
**Status** — Defect

---

### PY-016: CSV log opened and closed per row
**Lokasi** — `utils/csv_log.py` lines 22-32.
**Temuan** — Every detection row triggers a full `open`/`write`/`close` cycle. On Windows, file-locking makes this expensive. Adds I/O jitter per detection.
**Teori / Prinsip** — Keep the file handle open for the process lifetime; flush periodically. Or use `logging.FileHandler`.
**Perbandingan** — Optimization Opportunity
**Dampak** — CPU/I/O overhead proportional to detection count; measurable frame-processing jitter on slow disks.
**Solusi** — Hold a module-level handle opened by `init_csv_log()`, close in `shutdown_csv_log()` at end of `process_video`.
**Prioritas** — LOW
**Status** — Optimization Opportunity

---

### PY-017: No exception handling in `process_detection` around crop save
**Lokasi** — `pipeline.py` lines 256-279, line 277 (`save_crop`); `utils/image.py` line 28 (`cv2.imwrite`).
**Temuan** — If `cv2.imwrite` fails (disk full, permissions), the exception propagates up through `detect_pose` → `process_frame` → `process_video` loop. No `try/except` anywhere in the chain. The entire pipeline crashes.
**Teori / Prinsip** — File I/O in a long-running loop should be wrapped in `try/except (OSError, cv2.error)`.
**Perbandingan** — Risky
**Dampak** — Full disk or permission change during exam crashes the pipeline; all subsequent frames lost.
**Solusi** — Wrap lines 272-279 in `try/except OSError as e: log_info(f"[WARN] Crop save failed: {e}")`.
**Prioritas** — MEDIUM
**Status** — Defect

---

### PY-018: `prev_time` initialized in `__init__`, inflating first-frame FPS
**Lokasi** — `pipeline.py` line 32, line 134.
**Temuan** — `__init__` runs before video starts. The gap includes model loading (seconds), directory setup, thread start. First FPS reading divides 1 by that entire gap, showing e.g. 0.1 instead of ~30.
**Teori / Prinsip** — Timing baselines should be set at the start of the measured region, not at construction.
**Perbandingan** — Partially Correct
**Dampak** — Cosmetic: first frame shows garbage FPS value. No functional impact.
**Solusi** — Set `self.prev_time = None`; guard the first frame: `if self.prev_time is None: self.prev_time = curr_time; display_fps = 0.0`.
**Prioritas** — LOW
**Status** — Cosmetic

---

### PY-019: `stop_event` and `frame_queue` set after pipeline construction
**Lokasi** — `pipeline.py` lines 35-36; `main.py` lines 44-45.
**Temuan** — `__init__` sets both to `None`. Real values assigned only after the constructor returns. If `process_video` is called before the assignment, `AttributeError` on `None.is_set()`.
**Teori / Prinsip** — An object should not be usable in a partial state. Initialize in `__init__` or use property guards.
**Perbandingan** — Partially Correct
**Dampak** — No bug today. Fragility for future refactors or test harnesses.
**Solusi** — Move `queue.Queue(maxsize=1)` and `threading.Event()` into `__init__`. Remove post-constructor assignment.
**Prioritas** — LOW
**Status** — Optimization Opportunity

---

### PY-020: No graceful Flask shutdown when video ends
**Lokasi** — `web/app.py` lines 30-33; `main.py` lines 54-55.
**Temuan** — When the processing thread exits, Flask keeps running. `generate_frames` loops on empty queue (PY-006). No notification to browser, no shutdown, no status endpoint.
**Teori / Prinsip** — Streaming service should signal completion to clients. Close the generator or add a `/status` endpoint.
**Perbandingan** — Incorrect
**Dampak** — Browser shows frozen frame indefinitely; zombie Flask thread; user must kill the process.
**Solusi** — Set `stop_event` when video ends (fixes PY-006). Add `/status` endpoint. Optionally shut down Flask after video completes.
**Prioritas** — MEDIUM
**Status** — Defect

---

### PY-021: NMS applied to BoT-SORT output that already performed NMS
**Lokasi** — `pipeline.py` lines 180-184, 230-234; `core/nms.py` lines 5-47.
**Temuan** — `YOLO.track()` already performs NMS internally. The manual `apply_nms` re-applies NMS with the same threshold (0.45). Redundant O(n^2) compute per frame. If thresholds diverge, the second pass may drop valid detections.
**Teori / Prinsip** — Avoid double NMS. If a second pass is intentional, use a different threshold and document why.
**Perbandingan** — Optimization Opportunity
**Dampak** — Wasted compute; potential loss of valid low-IoU detections.
**Solusi** — Remove the manual `apply_nms` calls since `track()` handles NMS. Or add a comment if intentionally different.
**Prioritas** — LOW
**Status** — Optimization Opportunity

---

### PY-022: `cv2.imwrite` return value not checked
**Lokasi** — `pipeline.py` lines 100, 139; `utils/image.py` line 28.
**Temuan** — `cv2.imwrite` returns `True`/`False`. None of the three call sites check the return value. Silent failure produces no log and no exception.
**Teori / Prinsip** — `cv2.imwrite` reports failure via return value, not exception. Check and log.
**Perbandingan** — Risky
**Dampak** — Evidence crops silently fail to save. CSV log records a detection with no corresponding image file.
**Solusi** — `if not cv2.imwrite(path, img): log_info(f"[WARN] imwrite failed: {path}")`.
**Prioritas** — MEDIUM
**Status** — Defect
