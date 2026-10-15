"""Image utilities: crop expansion, validation, and saving."""
import os

import cv2
import numpy as np


def expand_crop(x1, y1, x2, y2, img_w, img_h, scale=2.5):
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2
    new_w, new_h = int(w * scale), int(h * scale)
    new_x1 = max(0, cx - new_w // 2)
    new_y1 = max(0, cy - new_h // 2)
    new_x2 = min(img_w, cx + new_w // 2)
    new_y2 = min(img_h, cy + new_h // 2)
    return new_x1, new_y1, new_x2, new_y2


def is_valid_crop(crop, min_brightness=30):
    if crop is None or crop.size == 0:
        return False
    brightness = np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    return brightness >= min_brightness


def save_crop(folder_path, filename, crop):
    os.makedirs(folder_path, exist_ok=True)
    cv2.imwrite(os.path.join(folder_path, filename), crop)
