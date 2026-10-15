# Phase 2D — Jetson Hardware Validation Plan

**Status:** HARNESS READY — hardware validation NOT RUN
**Created:** 2026-10-15
**Owner:** A
**Jetson hardware:** NOT AVAILABLE

---

## 1. Purpose

Phase 2D provides the measurement schema and report structures for validating
the production pipeline on target hardware (Jetson Orin Nano or equivalent).
This document is the execution checklist for that validation run.

What the schema guarantees:
- All fields are Optional; a missing sensor produces `None`, never a fake zero.
- `NOT_RUN` validation states are explicit and visible in both JSON and the
  human summary.
- `BenchmarkReport` is fully JSON-serialisable for archival and diffing.
- No GPU framework (torch, cv2, ultralytics) is imported by
  `inference/evaluation/benchmark_schema.py` — it is safe to run anywhere.

What this document does NOT do:
- No production pipeline changes.
- No assumption that Jetson is available at write time.
- No interpretation of laptop benchmark results as Jetson results.

---

## 2. Files in Scope

| File | Role |
|------|------|
| `inference/evaluation/benchmark_schema.py` | All data structures + helpers |
| `tests/test_phase2d_benchmark.py` | 38 unit tests covering the schema |
| `docs/PHASE2D_JETSON_VALIDATION.md` | This document |

Production files (`inference/core/pipeline.py`, `inference/benchmark.py`,
`inference/utils/metrics.py`) are NOT modified by Phase 2D.

---

## 3. Schema Fields

See `inference/evaluation/benchmark_schema.py` for the authoritative source.
Summary of top-level `BenchmarkReport` fields:

| Field group | Key fields |
|-------------|-----------|
| Metadata | `run_id`, `timestamp_utc`, `hardware_name`, `hardware_info` |
| Input | `input_config` (video_id, fps, resolution, warmup/steady-state) |
| Runtime | `runtime_config` (backend, device, model_names, precision_mode) |
| Timing | `timing_components: List[ComponentTiming]`, `processing_fps` |
| Frames | `frame_accounting` (read/processed/dropped/output, intentional flag) |
| Resources | `resources: ResourceMetrics`, `thermal: ThermalMetrics` |
| Events | `event_comparison: EventComparison`, `evidence_latency: EvidenceLatency` |
| Status | `real_time_status`, `semantic_regression_status`, `thermal_status`, `validation_status` |
| Provenance | `limitations: List[str]`, `notes` |

`validation_status` values:
- `"NOT_RUN"` — no hardware validation executed (laptop-only run)
- `"HARNESS_READY"` — schema complete, hardware validation pending
- `"PASS"` — all critical criteria met on target hardware
- `"PASS_WITH_LIMITATIONS"` — passed with documented non-critical issues
- `"FAIL"` — a critical criterion failed

---

## 4. Jetson Execution Checklist

Run in order. Record actual values in the `BenchmarkReport` JSON after each step.

### 4.1 Environment

- [ ] Confirm Jetson model and JetPack version
  ```bash
  cat /etc/nv_tegra_release
  ```
- [ ] Confirm PyTorch + CUDA are installed and working
  ```bash
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  ```
- [ ] Record `hardware_name` (e.g. `"Jetson Orin Nano 8 GB"`)
- [ ] Record `runtime_config.backend = "pytorch"`, `device = "cuda:0"`

### 4.2 Warmup

- [ ] Run the pipeline on the canonical video for `warmup_frames` frames
      (recommend 60+ frames or 5 s at 30 fps)
- [ ] Discard warmup results; only steady-state results feed the report
- [ ] Record `warmup_duration_s`

### 4.3 Steady-State Run

- [ ] Run the pipeline on the canonical video for `steady_state_frames` frames
      (recommend >= 300 frames or 10 s at 30 fps)
- [ ] Collect per-frame timing into `timing_components` with `name="frame"`
- [ ] Collect per-component timings:
      - `name="pose_inference"` (PoseYOLO.forward)
      - `name="object_inference"` (ObjectYOLO.forward)
      - `name="matching"` (match_candidates)
      - `name="event_confirmation"` (confirmation logic)
      - `name="jpeg_encode"` (evidence snapshot)
      - `name="frame"` (end-to-end frame)
- [ ] Record `frame_accounting`:
      - `frames_read`
      - `frames_processed`
      - `frames_dropped`
      - `frames_output`
      - `intentional_sampling` = `False` for continuous capture
- [ ] Compute `processing_fps = frames_processed / steady_state_duration_s`
- [ ] Set `real_time_status = "PASS"` if `processing_fps >= source_fps`, else `"FAIL"`

### 4.4 Resources

- [ ] If available on Jetson, start a `ResourceMonitor` (subclass or wrapper)
      and call `monitor.snapshot()` at start and end of the run
- [ ] Populate `resources` with actual values (leave `None` for unavailable sensors)
- [ ] Populate `thermal` from `/sys/class/thermal/thermal_zone*/temp` or `jetson-stats`

### 4.5 Semantic Regression

- [ ] Run the reference `inference/benchmark.py` (PyTorch/CPU or GPU) to produce
      a reference event count
- [ ] Run the same pipeline on Jetson and count deployment events
- [ ] Fill `event_comparison`:
      - `reference_event_count`
      - `deployment_event_count`
      - `event_count_tolerance` (recommend +/-2 events for a 10 s clip)
- [ ] Set `semantic_regression_status = "PASS"` if `within_tolerance() is True`,
      else `"FAIL"`

### 4.6 Evidence Latency

- [ ] If evidence capture is enabled, measure `confirmation_to_evidence_ms`
      and `evidence_generation_ms` for each detected event
- [ ] Set `evidence_latency.asynchronous` = `True` when evidence is generated
      off the main loop
- [ ] Record `evidence_failure_count` / `evidence_total_count`

### 4.7 Thermal (sustained)

- [ ] If running for > 5 minutes, monitor temperature and
      `thermal_throttling_detected`
- [ ] Set `thermal_status = "PASS"` when no throttling, `"FAIL"` when
      throttling reduced processing FPS below `target_fps`

### 4.8 Final Status

- [ ] Set `validation_status` based on the above checks
- [ ] Populate `limitations` for any non-critical issues
- [ ] Save the full `BenchmarkReport` to a JSON file in `artifacts/phase2d/`

---

## 5. Acceptance Criteria

| Criterion | PASS requires |
|-----------|---------------|
| Real-time throughput | `processing_fps >= source_fps` (e.g. >= 30 fps for 30 fps video) |
| Semantic regression | Event count within tolerance of reference run |
| Evidence latency | Mean `confirmation_to_evidence_ms < 500 ms` (if applicable) |
| Thermal | No sustained throttling causing FPS drop below target |
| Resources | No `ResourceMetrics` field that should have a value is `None` |

A laptop run **never** satisfies these criteria for Jetson. The `limitations`
list must explicitly note `"Jetson validation NOT RUN"` when the run is on a
laptop.

---

## 6. Known Limitations

1. **`intentional_sampling` flag**: The pipeline does not currently emit an
   explicit tag for intentional frame drops. Callers must set
   `intentional_sampling=True` manually when using a downsampled input.
2. **`output_fps` in producer-only queue mode**: Documented in
   `inference/utils/metrics.py`; will remain `None` when the queue is the
   only output path.
3. **Resource monitor on Jetson**: A real `ResourceMonitor` implementation
   wrapping `jetson-stats` or `/sys/class/thermal/` is not yet in the repo.
   Until it is, `resources` fields in Jetson reports will show `None`.
4. **Laptop benchmark =/= Jetson benchmark**: CPU/GPU frequencies, thermal
   behaviour, and available cores differ substantially. Do not extrapolate.

---

## 7. Change Log

| Date | Entry |
|------|-------|
| 2026-10-15 | Created `inference/evaluation/benchmark_schema.py` and `tests/test_phase2d_benchmark.py`; 38 tests green; this document added |
