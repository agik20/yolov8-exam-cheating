# YOLOv8 Exam-Cheating Detection

Real-time detection of exam-cheating behavior — phone usage, passing objects, and looking at a friend's paper — using YOLOv8 pose and object-detection models. The system tracks detected candidates through a confirmation gate, applies a post-confirmation cooldown window, associates persons with objects, and saves timestamped evidence crops and CSV logs.

Built for deployment on Jetson Orin Nano; all inference paths are backend-agnostic (PyTorch CPU/GPU).

---

## Repository Structure

```
yolov8-exam-cheating/
├── inference/
│   ├── main.py                 # entry point: model bootstrap, threading, Flask server
│   ├── benchmark.py            # performance benchmark runner (Phase 2D)
│   ├── requirements.txt
│   ├── config/
│   │   ├── constants.py        # legacy frozen thresholds (5 s rule, 90 s cooldown, NMS IoU)
│   │   ├── paths.py            # CWD-anchored path resolution
│   │   └── shadow_config.py    # Phase 2B.6 — shadow-path typed configuration
│   ├── core/
│   │   ├── pipeline.py         # CheatingDetectionPipeline (video loop, detection, evidence)
│   │   ├── candidate_layer.py  # observation buffer per track
│   │   ├── confirmation.py     # shadow confirmation-gate policy
│   │   ├── cooldown.py         # post-confirmation cooldown window
│   │   ├── shadow_lifecycle.py # full shadow observation → confirm → cooldown lifecycle
│   │   ├── association.py      # person-object spatial + keypoint association
│   │   ├── events.py           # event data structures, Observation, CandidateState
│   │   └── nms.py              # Non-Maximum Suppression
│   ├── evaluation/
│   │   ├── ground_truth.py     # Phase 2C.1 — GT dataset, incidents, labels
│   │   ├── matching.py         # Phase 2C.2 — deterministic prediction ↔ GT matching
│   │   ├── runner.py           # Phase 2C.3 — offline evaluation runner
│   │   ├── calibration.py      # Phase 2C.4 — parameter grid sweep + operating-point selection
│   │   └── benchmark_schema.py # Phase 2D — JSON-serialisable benchmark report schema
│   ├── utils/
│   │   ├── image.py            # crop expansion, brightness validation, crop saving
│   │   ├── csv_log.py          # CSV detection log
│   │   ├── logging.py          # timestamped console + file logger
│   │   ├── metrics.py          # timing / frame accounting collector
│   │   └── telemetry.py        # optional performance telemetry
│   ├── web/
│   │   └── app.py              # Flask: MJPEG stream at /video_feed, HTML at /
│   ├── input/                  # video files (e.g. sk2.mp4)
│   ├── models/                 # .pt model weights (not committed to Git)
│   └── output/                 # generated artifacts (images, benchmark JSON)
│
├── tests/                      # 479 unit tests across all phases
│
├── docs/                       # phase deliverable documentation
│   ├── PHASE2B6_EVIDENCE_AND_CONFIG.md
│   ├── PHASE2C2_MATCHING.md
│   ├── PHASE2C3_EVALUATION.md
│   ├── PHASE2C4_CALIBRATION.md
│   └── PHASE2D_JETSON_VALIDATION.md
│
├── datasets-acquisition/       # scripts to build annotated datasets from raw video
├── datasets-annotation/        # label validation, deduplication, balancing scripts
├── processing-image/           # image augmentation and preprocessing
├── training-datasets/          # YOLOv8 training scripts and dataset config
└── evaluasi-model/             # model evaluation: mAP, OOKS, loss curves, scatter plots
```

---

## Prerequisites

- Python 3.10+
- PyTorch (CPU or CUDA)
- See `inference/requirements.txt` for pinned dependencies

```bash
pip install -r inference/requirements.txt
```

Model weights (`.pt`) and video inputs (`.mp4`) are **not** committed to Git.
Place them at:

```
inference/models/pose-70s.pt
inference/models/v8s-object.pt
inference/input/<your_video>.mp4
```

---

## Running the Pipeline

```bash
cd inference
python main.py
```

The pipeline reads the video, runs pose + object detection per frame, applies the
shadow confirmation gate and cooldown, saves evidence crops, and writes a CSV log.
A Flask server exposes the live MJPEG stream at `http://localhost:5000`.

---

## Running the Benchmark

```bash
cd inference
python benchmark.py
```

Produces a `BenchmarkReport` JSON (Phase 2D schema) in `output/benchmark_results.json`.
On a laptop the `validation_status` is `HARNESS_READY`; target-hardware PASS requires
Jetson validation per `docs/PHASE2D_JETSON_VALIDATION.md`.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

479 tests cover all phases from event structures (2B.1) through ShadowConfig DI (2E).
The test suite is fully offline — no GPU or video files required.

---

## Key Phases

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 2B.1  | Event data structures (`events.py`) | Complete |
| 2B.2  | Candidate observation layer (`candidate_layer.py`) | Complete |
| 2B.3  | Shadow confirmation gate (`confirmation.py`) | Complete |
| 2B.4  | Post-confirmation cooldown (`cooldown.py`) | Complete |
| 2B.5  | Person-object association (`association.py`) | Complete |
| 2B.6  | Shadow config + evidence schema (`shadow_config.py`) | Complete |
| 2C.1  | Ground-truth dataset schema (`ground_truth.py`) | Complete |
| 2C.2  | Deterministic matching (`matching.py`) | Complete |
| 2C.3  | Offline evaluation runner (`runner.py`) | Complete |
| 2C.4  | Calibration sweep (`calibration.py`) | Complete |
| 2D    | Benchmark schema (`benchmark_schema.py`) + Jetson validation doc | Harness ready |
| 2E    | ShadowConfig dependency injection into pipeline | In progress |

---

## License

Private — internal project.
