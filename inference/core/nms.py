"""Non-Maximum Suppression. Algorithm unchanged from the original pipeline."""
import numpy as np


def apply_nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []

    # Convert boxes to numpy array
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Get coordinates of the boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute area of each box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Get sorted indices by score (highest first)
    indices = np.argsort(scores)[::-1]

    keep = []
    while len(indices) > 0:
        # Get index of current box with highest score
        current = indices[0]
        keep.append(current)

        # Compute IoU between current box and all remaining boxes
        xx1 = np.maximum(x1[current], x1[indices[1:]])
        yy1 = np.maximum(y1[current], y1[indices[1:]])
        xx2 = np.minimum(x2[current], x2[indices[1:]])
        yy2 = np.minimum(y2[current], y2[indices[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        union = areas[current] + areas[indices[1:]] - intersection
        iou = intersection / union

        # Keep only boxes with IoU below threshold
        remaining_indices = np.where(iou <= iou_threshold)[0]
        indices = indices[remaining_indices + 1]

    return keep
