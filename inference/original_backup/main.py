import cv2
import os
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
from collections import deque
from flask import Flask, Response, render_template_string
import threading
import queue
import csv

# ========== CONFIGURATION ==========
class Config:
    # Path configurations
    VIDEO_PATH = "C:\\Users\\ardut\\projects\\yolov8-exam-cheating\\sk2.mp4"
    POSE_MODEL_PATH = "C:\\Users\\ardut\\projects\\exam-cheating-final\\configs\\model\\pose\\yolo-pose-70s\\weights\\best.pt"
    OBJECT_MODEL_PATH = "C:\\Users\\ardut\\projects\\exam-cheating-final\\configs\\model\\object\\yolov8m-object\\weights\\best.pt"

    OUTPUT_POSE_BASE = "inference/output/pose"
    OUTPUT_OBJ_BASE = "inference/output/object"
    OUTPUT_FULLROOM = "inference/output/room-view"
    OUTPUT_FRAMES = "inference/output/frames"

    # Detection parameters
    POSE_CLASSES = ["no-cheating", "provide-object", "see-friends-work"]
    OBJECT_CLASSES = ["smartphone", "calculator"]
    SAVE_POSE_CLASSES = ["provide-object", "see-friends-work"]
    SAVE_OBJECT_CLASSES = ["smartphone"]

    CONF_THRESH_POSE = 0.3
    CONF_THRESH_OBJ = 0.3
    DISPLAY_CROP_SCALE = 1.5
    POSE_SAVE_CROP_SCALE = 2.5
    OBJ_SAVE_CROP_SCALE = 4.5
    MIN_CROP_BRIGHTNESS = 30

    # NMS parameters
    NMS_IOU_THRESHOLD = 0.45

    # Display parameters
    POSE_BOX_THICKNESS = 1
    OBJECT_BOX_THICKNESS = 1
    TEXT_THICKNESS = 1

    # Cooldown parameters
    POSE_COOLDOWN = 90
    OBJECT_COOLDOWN = 90
    REQUIRED_DETECTION_TIME = 5

    FRAME_SAVE_INTERVAL = 15

    POSE_COLORS = {
        "no-cheating": (0, 255, 0),
        "provide-object": (255, 0, 0),
        "see-friends-work": (0, 255, 255)
    }
    OBJECT_COLORS = {
        "smartphone": (0, 0, 255),
        "calculator": (0, 255, 0)
    }

# ========== UTILITY FUNCTIONS ==========
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

def log_info(msg, log_file=None, print_to_console=True):
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    log_msg = f"[{timestamp}] {msg}"
    if print_to_console:
        print(log_msg)
    if log_file:
        log_file.write(log_msg + "\n")

def get_csv_log_filename():
    today = datetime.now().strftime("%Y-%m-%d")
    return f"{today}_detection_log.csv"

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
# ========== DETECTION PIPELINE ==========
class CheatingDetectionPipeline:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.initialize_models()
        self.id_info_pose = {}
        self.id_info_obj = {}
        self.frame_idx = 0
        self.prev_time = time.time()
        self.start_time = time.time()
        self.last_frame_save_time = time.time()
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        init_csv_log()

    def setup_directories(self):
        log_info("[INFO] Setting up output directories...", print_to_console=False)
        os.makedirs(self.config.OUTPUT_POSE_BASE, exist_ok=True)
        os.makedirs(self.config.OUTPUT_OBJ_BASE, exist_ok=True)
        os.makedirs(self.config.OUTPUT_FULLROOM, exist_ok=True)
        os.makedirs(self.config.OUTPUT_FRAMES, exist_ok=True)
        for cls in self.config.SAVE_POSE_CLASSES:
            os.makedirs(os.path.join(self.config.OUTPUT_POSE_BASE, cls), exist_ok=True)
        for cls in self.config.SAVE_OBJECT_CLASSES:
            os.makedirs(os.path.join(self.config.OUTPUT_OBJ_BASE, cls), exist_ok=True)
        log_info("[INFO] Output directories ready", print_to_console=False)

    def initialize_models(self):
        log_info("[INFO] Initializing pose estimation model...")
        self.pose_model = YOLO(self.config.POSE_MODEL_PATH, task='pose')
        self.pose_class_map = {
            0: 'no-cheating',
            1: 'provide-object',
            2: 'see-friends-work'
        }
        log_info("[INFO] Pose model loaded successfully")

        log_info("[INFO] Initializing object detection model...")
        self.object_model = YOLO(self.config.OBJECT_MODEL_PATH, task='detect')
        self.object_class_map = {
            0: 'smartphone',
            1: 'calculator'
        }
        log_info("[INFO] Object model loaded successfully")

    def process_video(self):
        log_file = open("detection_log.txt", "w")
        log_info("[INFO] Starting video processing pipeline", log_file)

        cap = cv2.VideoCapture(self.config.VIDEO_PATH)
        if not cap.isOpened():
            log_info("[ERROR] Could not open video file", log_file)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        log_info(f"[INFO] Video source: {self.config.VIDEO_PATH}", log_file)
        log_info(f"[INFO] Frame rate: {cap.get(cv2.CAP_PROP_FPS):.2f}", log_file)

        while cap.isOpened() and not self.stop_event.is_set():
            success, frame = cap.read()
            if not success:
                log_info("[INFO] End of video stream reached", log_file)
                break

            self.frame_idx += 1
            current_time = datetime.now()
            timestamp = current_time.strftime("%H%M%S")

            current_time_sec = time.time()
            if current_time_sec - self.last_frame_save_time >= self.config.FRAME_SAVE_INTERVAL:
                frame_filename = f"frame_{timestamp}_{self.frame_idx}.jpg"
                frame_path = os.path.join(self.config.OUTPUT_FRAMES, frame_filename)
                cv2.imwrite(frame_path, frame)
                log_info(f"[INFO] Saved full frame: {frame_path}", log_file, print_to_console=False)
                self.last_frame_save_time = current_time_sec

            log_info(f"[INFO] Processing frame {self.frame_idx}", log_file, print_to_console=False)
            annotated_frame = self.process_frame(frame, current_time, timestamp, log_file)

            if self.frame_queue.empty():
                try:
                    self.frame_queue.put(annotated_frame.copy(), block=False)
                except queue.Full:
                    pass

            self.display_results(annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log_info("Pipeline stopped by user", log_file)
                self.stop_event.set()
                break

        processing_time = time.time() - self.start_time
        avg_fps = self.frame_idx / processing_time
        log_info(f"[INFO] Processing completed. Total frames: {self.frame_idx}", log_file)
        log_info(f"[INFO] Total processing time: {processing_time:.2f} seconds", log_file)
        log_info(f"[INFO] Average FPS: {avg_fps:.2f}", log_file)

        cap.release()
        cv2.destroyAllWindows()
        log_file.close()

    def process_frame(self, frame, current_time, timestamp, log_file):
        img_h, img_w = frame.shape[:2]
        annotated = frame.copy()
        curr_time = time.time()
        display_fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
        self.prev_time = curr_time

        if self.frame_idx == 1:
            log_info("[INFO] Saving full room view snapshot", log_file, print_to_console=False)
            cv2.imwrite(os.path.join(self.config.OUTPUT_FULLROOM, f"room_{timestamp}.jpg"), frame)

        log_info("[INFO] Running pose detection...", log_file, print_to_console=False)
        annotated = self.detect_pose(annotated, frame, current_time, timestamp, img_w, img_h, log_file)

        log_info("[INFO] Running object detection...", log_file, print_to_console=False)
        annotated = self.detect_objects(annotated, frame, current_time, timestamp, img_w, img_h, log_file)

        log_info(
            f"[INFO] Frame {self.frame_idx} processed - Pose IDs: {len(self.id_info_pose)}, Object IDs: {len(self.id_info_obj)}",
            log_file, print_to_console=False)

        cv2.putText(annotated, f"FPS: {display_fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), self.config.TEXT_THICKNESS)
        cv2.putText(annotated, f"Frame: {self.frame_idx}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), self.config.TEXT_THICKNESS)
        return annotated

    def detect_pose(self, annotated, frame, current_time, timestamp, img_w, img_h, log_file):
        results = self.pose_model.track(
            source=frame, 
            conf=self.config.CONF_THRESH_POSE, 
            iou=self.config.NMS_IOU_THRESHOLD,  # Use our NMS threshold
            persist=True, 
            imgsz=640
        )
        
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None and boxes.cls is not None:
            # Prepare data for NMS
            all_boxes = []
            all_scores = []
            all_cls = []
            all_ids = []
            
            for box, cls_id, track_id, conf in zip(boxes.xyxy, boxes.cls, boxes.id, boxes.conf):
                all_boxes.append(box.tolist())
                all_scores.append(float(conf))
                all_cls.append(int(cls_id))
                all_ids.append(int(track_id))
            
            # Apply NMS
            if len(all_boxes) > 0:
                keep_indices = apply_nms(
                    all_boxes, 
                    all_scores, 
                    self.config.NMS_IOU_THRESHOLD
                )
                
                # Process only the kept detections
                for idx in keep_indices:
                    box = all_boxes[idx]
                    cls_id = all_cls[idx]
                    track_id = all_ids[idx]
                    conf = all_scores[idx]
                    
                    class_name = self.pose_class_map.get(cls_id, f"unknown_{cls_id}")
                    color = self.config.POSE_COLORS.get(class_name, (255, 255, 255))
                    x1, y1, x2, y2 = map(int, box)
                    dx1, dy1, dx2, dy2 = expand_crop(x1, y1, x2, y2, img_w, img_h, self.config.DISPLAY_CROP_SCALE)
                    cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), color, self.config.POSE_BOX_THICKNESS)
                    label = f"ID:{track_id} {class_name} {conf:.2f}"
                    cv2.putText(annotated, label, (dx1, dy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                self.config.TEXT_THICKNESS)
                    
                    if class_name in self.config.SAVE_POSE_CLASSES:
                        self.process_detection(track_id, class_name, x1, y1, x2, y2, frame, current_time, timestamp,
                                               img_w, img_h, self.id_info_pose, self.config.POSE_COOLDOWN,
                                               self.config.OUTPUT_POSE_BASE, "pose", log_file, conf, self.config.POSE_SAVE_CROP_SCALE)
        return annotated

    def detect_objects(self, annotated, frame, current_time, timestamp, img_w, img_h, log_file):
        results = self.object_model.track(
            source=frame, 
            conf=self.config.CONF_THRESH_OBJ, 
            iou=self.config.NMS_IOU_THRESHOLD,  # Use our NMS threshold
            persist=True, 
            imgsz=640
        )
        
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            # Prepare data for NMS
            all_boxes = []
            all_scores = []
            all_cls = []
            all_ids = []
            
            for box, cls_id, track_id, conf in zip(boxes.xyxy, boxes.cls, boxes.id, boxes.conf):
                all_boxes.append(box.tolist())
                all_scores.append(float(conf))
                all_cls.append(int(cls_id))
                all_ids.append(int(track_id))
            
            # Apply NMS
            if len(all_boxes) > 0:
                keep_indices = apply_nms(
                    all_boxes, 
                    all_scores, 
                    self.config.NMS_IOU_THRESHOLD,
                )
                
                # Process only the kept detections
                for idx in keep_indices:
                    box = all_boxes[idx]
                    cls_id = all_cls[idx]
                    track_id = all_ids[idx]
                    conf = all_scores[idx]
                    
                    class_name = self.object_class_map.get(cls_id, f"unknown_{cls_id}")
                    color = self.config.OBJECT_COLORS.get(class_name, (255, 255, 255))
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.config.OBJECT_BOX_THICKNESS)
                    label = f"ID:{track_id} {class_name} {conf:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                self.config.TEXT_THICKNESS)
                    
                    if class_name in self.config.SAVE_OBJECT_CLASSES:
                        self.process_detection(track_id, class_name, x1, y1, x2, y2, frame, current_time, timestamp,
                                               img_w, img_h, self.id_info_obj, self.config.OBJECT_COOLDOWN,
                                               self.config.OUTPUT_OBJ_BASE, "object", log_file, conf, self.config.OBJ_SAVE_CROP_SCALE)
        return annotated

    def process_detection(self, track_id, class_name, x1, y1, x2, y2, frame, current_time, timestamp,
                          img_w, img_h, id_info, cooldown, output_base, detection_type, log_file, conf=None, scale=2.5):
        if track_id not in id_info:
            id_info[track_id] = {
                'first_time': current_time,
                'last_saved': None,
                'last_class': class_name,
                'last_conf': conf
            }
        info = id_info[track_id]
        if info['last_saved'] and (current_time - info['last_saved']).total_seconds() < cooldown:
            return
        if info['last_class'] != class_name:
            info['first_time'] = current_time
            info['last_class'] = class_name
        elif (current_time - info['first_time']).total_seconds() >= self.config.REQUIRED_DETECTION_TIME:
            ex1, ey1, ex2, ey2 = expand_crop(x1, y1, x2, y2, img_w, img_h, scale)
            crop = frame[ey1:ey2, ex1:ex2]
            if is_valid_crop(crop, self.config.MIN_CROP_BRIGHTNESS):
                conf_str = f"{info['last_conf']:.2f}" if info['last_conf'] else "0.00"
                filename = f"{track_id}_{class_name}_{conf_str}_{timestamp}.jpg"
                save_crop(os.path.join(output_base, class_name), filename, crop)
                info['last_saved'] = current_time
                # Log to CSV
                log_detection_to_csv(track_id, filename, class_name, info['last_conf'], current_time)

    def display_results(self, annotated_frame):
        """Display annotated frame with additional information"""
        if not os.environ.get('DISPLAY'):
            return
        cv2.putText(annotated_frame, f"Tracking {len(self.id_info_pose)} persons",
                    (10, annotated_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    self.config.TEXT_THICKNESS)
        cv2.putText(annotated_frame, f"Tracking {len(self.id_info_obj)} objects",
                    (10, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    self.config.TEXT_THICKNESS)
        cv2.imshow("[INFO] Detection Results", annotated_frame)
        cv2.waitKey(1)

    def generate_frames(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except queue.Empty:
                continue

# ========== FLASK APP ==========
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Streaming</title>
    </head>
    <body>
        <h1>Video Streaming</h1>
        <img src="{{ url_for('video_feed') }}">
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(pipeline.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    log_info("[INFO] Initializing cheating detection pipeline")
    pipeline = CheatingDetectionPipeline(Config())

    processing_thread = threading.Thread(target=pipeline.process_video)
    processing_thread.daemon = True
    processing_thread.start()

    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        log_info("[INFO] Shutting down...")
        pipeline.stop_event.set()
        processing_thread.join()
        log_info("Pipeline execution completed gracefully.")