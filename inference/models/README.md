# Model Weights — `inference/models/`

Model files are **not committed** to Git. They are stored externally and must be
placed in this directory manually before running inference.

## Required Files

| File | Purpose | Expected Size |
|---|---|---|
| `pose-70s.pt` | YOLOv8-Pose keypoint model — detects cheating pose classes (`no-cheating`, `provide-object`, `see-friends-work`) | ~23 MB |
| `v8s-object.pt` | YOLOv8s object detection model — detects `smartphone` and `calculator` | ~50 MB |

## How to Place the Files

```
inference/models/
├── pose-70s.pt      # <— place here
├── v8s-object.pt    # <— place here
└── README.md
```

Both files are resolved at runtime via `inference/config/paths.py`:

```python
# inference/config/paths.py
POSE_MODEL_PATH   = PROJECT_ROOT / "inference/models/pose-70s.pt"
OBJECT_MODEL_PATH = PROJECT_ROOT / "inference/models/v8s-object.pt"
```

If a file is missing the pipeline will raise a `FileNotFoundError` at startup.

## Where to Get the Files

Model weights are produced during training. Two paths exist:

1. **Retrain locally** — run the training scripts in `training-datasets/exam-act/`
   (`train-keypoint-recognition.py` and `train-bounding-box.py`) and copy the
   best checkpoint (`.pt`) into this directory with the names above.
2. **Receive from team** — ask for the two files from whoever has the trained
   checkpoints and place them here.

> No external download links are provided. Files must be supplied by the team.

## Notes

- Files matching `*.pt`, `*.onnx`, `*.engine`, `*.bin`, `*.safetensors`,
  `*.weights` inside this directory are ignored by Git (see root `.gitignore`).
- Do not rename the files — the pipeline references them by exact name.
- If you need a different model variant, update `inference/config/paths.py`
  and this README together.
