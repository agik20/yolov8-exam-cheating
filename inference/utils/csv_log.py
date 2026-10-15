"""CSV detection logging, anchored to the inference root (not the CWD)."""
import os
import csv
from datetime import datetime

from ..config import paths


def get_csv_log_filename():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(paths.INFER_ROOT, f"{today}_detection_log.csv")


def init_csv_log():
    filename = get_csv_log_filename()
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Filename', 'Jenis Kecurangan', 'Confidence Score', 'Waktu Deteksi'])


def log_detection_to_csv(detection_id, filename, detection_type, confidence, detection_time):
    csv_filename = get_csv_log_filename()
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            detection_id,
            filename,
            detection_type,
            f"{confidence:.2f}",
            detection_time.strftime("%Y-%m-%d %H:%M:%S")
        ])
