# YOLOv8 Exam-Cheating Detection — Inference System

Real-time detection of exam-cheating behavior (phone usage, passing objects,
looking at friends' work) using YOLOv8 pose + object detection models.
Detected cheating triggers automated evidence cropping and CSV logging.

## Folder Structure

```
inference/
├── main.py          # entry point: bootstrap, pipeline, threading, Flask
├── requirements.txt # pinned dependencies
├── config/
│   ├── paths.py     # root-anchored path resolution (CWD-independent)
│   └── constants.py # detection thresholds, class lists, colors
├── core/
│   ├── nms.py       # Non-Maximum Suppression
│   └── pipeline.py  # video loop, pose/object detection, tracking,
│                    #  5-second detection rule, 90-second cooldown,
│                    #  brightness validation, evidence crop saving
├── utils/
│   ├── image.py     # crop expansion, brightness check, crop saving
│   ├── logging.py   # timestamped console + file logging
│   └── csv_log.py   # CSV detection log (anchored to inference root)
├── web/
│   └── app.py       # Flask app: / (index) and /video_feed (MJPEG stream)
├── input/           # video file to process  (e.g. sk2.mp4)
├── models/          # .pt model weights
│   ├── pose-70s.pt
│   └── v8s-object.pt
├── output/          # generated artifacts
│   ├── frames/        # full-frame snapshots (every 15 s)
│   ├── room-view/     # frame-1 snapshot
│   ├── pose/provide-object/
│   ├── pose/see-friends-work/
│   └── object/smartphone/
└── original_backup/ # pre-refactor monolithic source (do not modify)
```

## Running

```bash
cd inference
python main.py
```

Models load from `models/`, video reads from `input/`, and all output is
written under `output/` and to `detection_log.txt` / `YYYY-MM-DD_detection_log.csv`
at the `inference/` root.

## Flask Endpoints

| Endpoint        | Description                                      |
|-----------------|--------------------------------------------------|
| `GET /`         | HTML page embedding the MJPEG stream             |
| `GET /video_feed` | MJPEG stream (multipart/x-mixed-replace)        |

Server listens on `0.0.0.0:5000` while the video is being processed.

## Dependencies

See `requirements.txt`. Install with:

```bash
pip install -r inference/requirements.txt
```
