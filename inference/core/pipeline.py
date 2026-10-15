"""Pose + object detection, NMS, tracking/cooldown, and evidence/crop saving.

Logic, thresholds and class handling are carried over unchanged from the
original monolithic pipeline to preserve behavior exactly.
"""
import os
import queue
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

from ..config import paths as _paths
from .nms import apply_nms
from .candidate_layer import CandidateLayer, InvalidSourceFps
from .confirmation import ConfirmationPolicy, UnconfiguredConfirmationPolicy
from .cooldown import CooldownPolicy, UnconfiguredCooldownPolicy
from .shadow_lifecycle import ShadowLifecycleManager
from .events import ModelSource, Observation, CandidateState
from .association import (
    PersonObjectAssociator, PersonObservation, ObjectObservation,
    PersonKeypoints, AssociationResult, AssociationStatus,
)
from ..utils.image import expand_crop, is_valid_crop, save_crop
from ..utils.logging import log_info
from ..utils.csv_log import init_csv_log, log_detection_to_csv
from ..utils.metrics import MetricsCollector


class CheatingDetectionPipeline:
    def __init__(self, config, video_path, pose_model_path, object_model_path,
                 headless=False, shadow_config=None):
        self.config = config
        self.video_path = video_path
        self.pose_model_path = pose_model_path
        self.object_model_path = object_model_path
        self.headless = headless
        self.metrics: MetricsCollector = None   # set externally by benchmark runner
        self.setup_directories()
        self.initialize_models()
        self.id_info_pose = {}
        self.id_info_obj = {}
        self.frame_idx = 0
        self.prev_time = time.time()
        self.start_time = time.time()
        self.last_frame_save_time = time.time()
        self.stop_event = None
        self.frame_queue = None
        self._candidate_layer = None   # Phase 2B.2 shadow layer (additive, set in process_video)
        self._src_fps = None
        # Phase 2B.3 — shadow confirmation gate (additive, no effect on legacy path)
        self._confirmation_policy: ConfirmationPolicy = UnconfiguredConfirmationPolicy()
        self._shadow_results: list = []  # in-memory diagnostic results; cleared per video run
        # Phase 2B.4 — post-confirmation cooldown (additive, no effect on legacy path)
        self._cooldown_policy: CooldownPolicy = UnconfiguredCooldownPolicy()
        self._shadow_lifecycle: ShadowLifecycleManager = None  # set in process_video
        # Phase 2B.5 — person-object association (additive, no effect on legacy path)
        self._associator: PersonObjectAssociator = PersonObjectAssociator()
        self._association_results: list = []  # in-memory; cleared per video run
        self._frame_person_obs: list = []      # per-frame PersonObservation list
        self._frame_object_obs: list = []      # per-frame ObjectObservation list
        self.shadow_config = shadow_config

        # Phase 2E-R1 — optional ShadowConfig DI: wire calibrated policies when provided
        if shadow_config is not None:
            confirmation_policy, cooldown_policy = shadow_config.build_policies()
            self._confirmation_policy = confirmation_policy
            self._cooldown_policy = cooldown_policy
            self._associator = shadow_config.build_associator()

    def _record_candidate_observation(
        self,
        track_id: int,
        class_name: str,
        conf: float,
        box: tuple,
        model_source: ModelSource,
        keypoints: tuple = None,
    ) -> None:
        """Phase 2B.2 — record a detection into the shadow observation layer.

        Purely additive: no return value, no side effect on the legacy
        event path.  When the layer is None (invalid FPS or not yet
        initialised) this is a no-op.

        keypoints (Phase 2B.5) — optional tuple of (x, y, visibility)
        tuples in source-frame pixels.  None for object-model observations
        and when the pose model returns no keypoint data.

        After recording the observation, Phase 2B.3 evaluates the shadow
        confirmation gate.  The result is stored in-memory only.
        """
        if self._candidate_layer is None or self._src_fps is None:
            return
        video_time = self.frame_idx / self._src_fps
        obs = Observation(
            frame_index=self.frame_idx,
            video_time=video_time,
            class_name=class_name,
            confidence=conf,
            bbox=box,
            source=model_source,
            track_id=track_id,
            keypoints=keypoints,
        )
        if self._shadow_lifecycle is not None:
            # Phase 2B.4 — full lifecycle path (cooldown-aware)
            result = self._shadow_lifecycle.process_observation(obs)
            if result is not None:
                # Append to the 2B.3-compatible shadow_results list so
                # existing diagnostic access patterns continue to work.
                self._shadow_results.append(result.gate_result)
        else:
            # Fallback: 2B.3 path (no lifecycle manager initialised yet)
            cand = self._candidate_layer.record_observation(obs)
            self._evaluate_shadow_confirmation(cand)

    def _evaluate_shadow_confirmation(self, candidate) -> None:
        """Phase 2B.3 — evaluate the shadow confirmation gate on a candidate.

        Purely diagnostic: the result is appended to self._shadow_results
        (in-memory list).  No production event is created, no file is
        written, and the legacy process_detection path is not affected.

        The gate is only invoked for candidates that are still in
        CANDIDATE state.  Candidates already in CONFIRMED state are
        skipped (their first shadow confirmation is already recorded),
        preventing one shadow result per frame for the same incident.
        """
        if candidate.state == CandidateState.CONFIRMED:
            return
        result = self._confirmation_policy.evaluate(candidate)
        if result.confirmed:
            candidate.state = CandidateState.CONFIRMED
        self._shadow_results.append(result)

    def _run_shadow_association(
        self,
        person_observations: list,
        object_observations: list,
    ) -> list:
        """Phase 2B.5 — evaluate person-object association for this frame.

        Purely diagnostic: results are appended to self._association_results
        (in-memory list, bounded by the number of objects per frame).
        No production event is created, no file is written, and the legacy
        process_detection path is not affected.

        Parameters
        ----------
        person_observations : list[PersonObservation]
            Pose-model detections for this frame, with optional keypoints.
        object_observations : list[ObjectObservation]
            Object-model detections for this frame.

        Returns the list of AssociationResult for this frame (one per object).
        """
        if not object_observations:
            return []
        results = self._associator.associate(person_observations, object_observations)
        self._association_results.extend(results)
        return results

    def setup_directories(self):

        log_info("[INFO] Setting up output directories...", print_to_console=False)
        cfg = self.config
        os.makedirs(cfg.OUTPUT_POSE_BASE, exist_ok=True)
        os.makedirs(cfg.OUTPUT_OBJ_BASE, exist_ok=True)
        os.makedirs(cfg.OUTPUT_FULLROOM, exist_ok=True)
        os.makedirs(cfg.OUTPUT_FRAMES, exist_ok=True)
        for cls in cfg.SAVE_POSE_CLASSES:
            os.makedirs(os.path.join(cfg.OUTPUT_POSE_BASE, cls), exist_ok=True)
        for cls in cfg.SAVE_OBJECT_CLASSES:
            os.makedirs(os.path.join(cfg.OUTPUT_OBJ_BASE, cls), exist_ok=True)
        log_info("[INFO] Output directories ready", print_to_console=False)

    def initialize_models(self):
        
        log_info("[INFO] Initializing pose estimation model...")
        self.pose_model = YOLO(self.pose_model_path, task='pose')
        self.pose_class_map = {
            0: 'no-cheating',
            1: 'provide-object',
            2: 'see-friends-work'
        }
        log_info("[INFO] Pose model loaded successfully")

        log_info("[INFO] Initializing object detection model...")
        self.object_model = YOLO(self.object_model_path, task='detect')
        self.object_class_map = {
            0: 'smartphone',
            1: 'calculator'
        }
        log_info("[INFO] Object model loaded successfully")


    def process_video(self, log_file, metrics=None):
        """Process the video stream.

        metrics: optional MetricsCollector.  When provided, every timing
                 point is recorded.  When None (production default), all
                 timing gates are no-ops and the path is byte-identical
                 to the pre-instrumented version.
        """
        log_info("[INFO] Starting video processing pipeline", log_file)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            log_info("[ERROR] Could not open video file", log_file)
            if self.stop_event is not None:
                self.stop_event.set()
            log_file.close()
            return

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

            src_fps = cap.get(cv2.CAP_PROP_FPS)
            self._src_fps = src_fps
            log_info(f"[INFO] Video source: {self.video_path}", log_file)
            log_info(f"[INFO] Frame rate: {src_fps:.2f}", log_file)

            # Phase 2B.2 — construct the shadow observation layer.
            # If src_fps is invalid the layer stays None and all
            # observation-recording calls become no-ops; the legacy
            # event path is completely unaffected either way.
            try:
                self._candidate_layer = CandidateLayer(
                    src_fps=src_fps,
                    observation_buffer_maxlen=32,   # ~1 s at 30 fps — structural bound
                )
                # Phase 2B.4 — create the shadow lifecycle manager.
                # Additive: uses the same layer + existing confirmation policy
                # and a new cooldown policy.  No effect on the legacy path.
                self._shadow_lifecycle = ShadowLifecycleManager(
                    layer=self._candidate_layer,
                    confirmation_policy=self._confirmation_policy,
                    cooldown_policy=self._cooldown_policy,
                )
                log_info("[INFO] Shadow candidate layer active (with cooldown lifecycle)",
                         log_file, print_to_console=False)
            except InvalidSourceFps as _e:
                log_info(f"[WARN] Candidate layer disabled ({_e})", log_file, print_to_console=False)

            while cap.isOpened() and not self.stop_event.is_set():
                success, frame = cap.read()
                if not success:
                    log_info("[INFO] End of video stream reached", log_file)
                    break

                if metrics:
                    metrics.on_frame_read()

                self.frame_idx += 1
                current_time = datetime.now()
                timestamp = current_time.strftime('%H%M%S')

                current_time_sec = time.time()
                if current_time_sec - self.last_frame_save_time >= self.config.FRAME_SAVE_INTERVAL:
                    frame_filename = f'frame_{timestamp}_{self.frame_idx}.jpg'
                    frame_path = os.path.join(self.config.OUTPUT_FRAMES, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    log_info(f'[INFO] Saved full frame: {frame_path}', log_file, print_to_console=False)
                    self.last_frame_save_time = current_time_sec

                log_info(f'[INFO] Processing frame {self.frame_idx}', log_file, print_to_console=False)

                # FRAME-TIMING WRAPPER: wrap the whole process_frame call
                if metrics:
                    with metrics.timer('frame'):
                        annotated_frame = self.process_frame(
                            frame, current_time, timestamp, log_file, metrics)
                else:
                    annotated_frame = self.process_frame(
                        frame, current_time, timestamp, log_file, metrics)

                if metrics:
                    metrics.on_frame_processed()

                # QUEUE-INSTRUMENTATION WRAPPER: same semantics as before
                if self.frame_queue is not None:
                    success_put = False
                    if self.frame_queue.empty():
                        try:
                            self.frame_queue.put(annotated_frame.copy(), block=False)
                            success_put = True
                        except queue.Full:
                            pass
                    if metrics:
                        metrics.on_queue_insert(success_put)

                # HEADLESS GATE: skip display only when headless=True
                if not self.headless:
                    self.display_results(annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        log_info('Pipeline stopped by user', log_file)
                        self.stop_event.set()
                        break

            processing_time = time.time() - self.start_time
            avg_fps = self.frame_idx / processing_time if processing_time > 0 else 0
            log_info(f'[INFO] Processing completed. Total frames: {self.frame_idx}', log_file)
            log_info(f'[INFO] Total processing time: {processing_time:.2f} seconds', log_file)
            log_info(f'[INFO] Average FPS: {avg_fps:.2f}', log_file)

            # Signal the MJPEG consumer (generate_frames) that processing is done,
            # so it does not block indefinitely on an empty queue after EOF.
            if self.stop_event is not None:
                self.stop_event.set()

        except Exception:
            log_info("[ERROR] Processing thread exception — stopping pipeline", log_file)
            import traceback
            traceback.print_exc()
            if self.stop_event is not None:
                self.stop_event.set()
        finally:
            cap.release()
            if not self.headless:
                cv2.destroyAllWindows()
            log_file.close()

    def process_frame(self, frame, current_time, timestamp, log_file, metrics=None):

        img_h, img_w = frame.shape[:2]
        annotated = frame.copy()
        curr_time = time.time()
        display_fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
        self.prev_time = curr_time

        if self.frame_idx == 1:
            log_info("[INFO] Saving full room view snapshot", log_file, print_to_console=False)
            cv2.imwrite(os.path.join(self.config.OUTPUT_FULLROOM, f"room_{timestamp}.jpg"), frame)

        log_info("[INFO] Running pose detection...", log_file, print_to_console=False)
        annotated = self.detect_pose(annotated, frame, current_time, timestamp, img_w, img_h, log_file, metrics)

        log_info("[INFO] Running object detection...", log_file, print_to_console=False)
        annotated = self.detect_objects(annotated, frame, current_time, timestamp, img_w, img_h, log_file, metrics)

        # Phase 2B.5 — run person-object association (shadow branch only;
        # never touches the legacy process_detection path).
        self._run_shadow_association(
            self._frame_person_obs,
            self._frame_object_obs,
        )

        log_info(
            f"[INFO] Frame {self.frame_idx} processed - Pose IDs: {len(self.id_info_pose)}, Object IDs: {len(self.id_info_obj)}",
            log_file, print_to_console=False)

        cv2.putText(annotated, f"FPS: {display_fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), self.config.TEXT_THICKNESS)
        cv2.putText(annotated, f"Frame: {self.frame_idx}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), self.config.TEXT_THICKNESS)
        return annotated

    def detect_pose(self, annotated, frame, current_time, timestamp, img_w, img_h, log_file,
                   metrics=None):
        # POSE INFERENCE — timed
        if metrics:
            with metrics.timer('pose_inference'):
                results = self.pose_model.track(
                    source=frame,
                    conf=self.config.CONF_THRESH_POSE,
                    iou=self.config.NMS_IOU_THRESHOLD,
                    persist=True,
                    imgsz=640
                )
        else:
            results = self.pose_model.track(
                source=frame,
                conf=self.config.CONF_THRESH_POSE,
                iou=self.config.NMS_IOU_THRESHOLD,
                persist=True,
                imgsz=640
            )

        # POSE POSTPROCESSING (NMS + annotation) — timed
        if metrics:
            with metrics.timer('pose_postprocess'):
                annotated = self._pose_postprocess(
                    annotated, results, frame, img_w, img_h,
                    current_time, timestamp, log_file, metrics)
        else:
            annotated = self._pose_postprocess(
                annotated, results, frame, img_w, img_h,
                current_time, timestamp, log_file, metrics)

        return annotated

    def _pose_postprocess(self, annotated, results, frame, img_w, img_h,
                          current_time, timestamp, log_file, metrics=None):
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None and boxes.cls is not None:
            all_boxes = []
            all_scores = []
            all_cls = []
            all_ids = []

            for box, cls_id, track_id, conf in zip(boxes.xyxy, boxes.cls, boxes.id, boxes.conf):
                all_boxes.append(box.tolist())
                all_scores.append(float(conf))
                all_cls.append(int(cls_id))
                all_ids.append(int(track_id))

            if len(all_boxes) > 0:
                keep_indices = apply_nms(
                    all_boxes,
                    all_scores,
                    self.config.NMS_IOU_THRESHOLD
                )

                # Phase 2B.5 — capture per-person keypoint data if available.
                # results[0].keypoints is a Keypoints object from ultralytics;
                # .data is a torch tensor of shape (N, K, 3) in pixel coords.
                # We extract it here so it can be passed to the shadow layer.
                kpts = getattr(results[0], "keypoints", None)
                kpts_data = None
                if kpts is not None and hasattr(kpts, "data") and kpts.data is not None:
                    try:
                        kpts_data = kpts.data.cpu().numpy()
                    except Exception:
                        kpts_data = None

                person_obs_list = []
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

                    # Phase 2B.5 — extract keypoint tuples for this person.
                    kp_tuples = None
                    if kpts_data is not None and idx < len(kpts_data):
                        kp_arr = kpts_data[idx]
                        kp_tuples = tuple(
                            (float(kp[0]), float(kp[1]), float(kp[2]))
                            for kp in kp_arr
                        )

                    # Phase 2B.2 — record observation in the shadow layer
                    self._record_candidate_observation(
                        track_id=track_id,
                        class_name=class_name,
                        conf=conf,
                        box=(x1, y1, x2, y2),
                        model_source=ModelSource.POSE,
                        keypoints=kp_tuples,
                    )

                    # Phase 2B.5 — collect for association this frame.
                    if class_name in self.config.SAVE_POSE_CLASSES:
                        person_obs_list.append(PersonObservation(
                            track_id=track_id,
                            class_name=class_name,
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            keypoints=(
                                PersonKeypoints(
                                    x=tuple(kp[0] for kp in kp_tuples),
                                    y=tuple(kp[1] for kp in kp_tuples),
                                    visibility=tuple(kp[2] for kp in kp_tuples),
                                )
                                if kp_tuples is not None else None
                            ),
                        ))

                        # Legacy event path (unchanged):
                        if metrics:
                            with metrics.timer('pose_event'):
                                self.process_detection(track_id, class_name, x1, y1, x2, y2, frame, current_time, timestamp,
                                                       img_w, img_h, self.id_info_pose, self.config.POSE_COOLDOWN,
                                                       self.config.OUTPUT_POSE_BASE, "pose", log_file, conf, self.config.POSE_SAVE_CROP_SCALE,
                                                       metrics=metrics)
                        else:
                            self.process_detection(track_id, class_name, x1, y1, x2, y2, frame, current_time, timestamp,
                                                   img_w, img_h, self.id_info_pose, self.config.POSE_COOLDOWN,
                                                   self.config.OUTPUT_POSE_BASE, "pose", log_file, conf, self.config.POSE_SAVE_CROP_SCALE,
                                                   metrics=metrics)

                # Stash for use by process_frame association call.
                self._frame_person_obs = person_obs_list
        return annotated

    def detect_objects(self, annotated, frame, current_time, timestamp, img_w, img_h, log_file,
                       metrics=None):
        # OBJECT INFERENCE — timed
        if metrics:
            with metrics.timer('object_inference'):
                results = self.object_model.track(
                    source=frame,
                    conf=self.config.CONF_THRESH_OBJ,
                    iou=self.config.NMS_IOU_THRESHOLD,
                    persist=True,
                    imgsz=640
                )
        else:
            results = self.object_model.track(
                source=frame,
                conf=self.config.CONF_THRESH_OBJ,
                iou=self.config.NMS_IOU_THRESHOLD,
                persist=True,
                imgsz=640
            )

        # OBJECT POSTPROCESSING — timed
        if metrics:
            with metrics.timer('object_postprocess'):
                annotated = self._object_postprocess(
                    annotated, results, frame, img_w, img_h,
                    current_time, timestamp, log_file, metrics)
        else:
            annotated = self._object_postprocess(
                annotated, results, frame, img_w, img_h,
                current_time, timestamp, log_file, metrics)

        return annotated

    def _object_postprocess(self, annotated, results, frame, img_w, img_h,
                            current_time, timestamp, log_file, metrics=None):
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            all_boxes = []
            all_scores = []
            all_cls = []
            all_ids = []

            for box, cls_id, track_id, conf in zip(boxes.xyxy, boxes.cls, boxes.id, boxes.conf):
                all_boxes.append(box.tolist())
                all_scores.append(float(conf))
                all_cls.append(int(cls_id))
                all_ids.append(int(track_id))

            if len(all_boxes) > 0:
                keep_indices = apply_nms(
                    all_boxes,
                    all_scores,
                    self.config.NMS_IOU_THRESHOLD,
                )

                obj_obs_list = []
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

                    # Phase 2B.2 — record observation in the shadow layer
                    # for every kept object detection, before any legacy gate.
                    self._record_candidate_observation(
                        track_id=track_id,
                        class_name=class_name,
                        conf=conf,
                        box=(x1, y1, x2, y2),
                        model_source=ModelSource.OBJECT,
                    )

                    # Phase 2B.5 — collect for association this frame.
                    if class_name in self.config.SAVE_OBJECT_CLASSES:
                        obj_obs_list.append(ObjectObservation(
                            track_id=track_id,
                            class_name=class_name,
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                        ))

                    if class_name in self.config.SAVE_OBJECT_CLASSES:
                        if metrics:
                            with metrics.timer('object_event'):
                                self.process_detection(track_id, class_name, x1, y1, x2, y2, frame, current_time, timestamp,
                                                       img_w, img_h, self.id_info_obj, self.config.OBJECT_COOLDOWN,
                                                       self.config.OUTPUT_OBJ_BASE, "object", log_file, conf, self.config.OBJ_SAVE_CROP_SCALE,
                                                       metrics=metrics)
                        else:
                            self.process_detection(track_id, class_name, x1, y1, x2, y2, frame, current_time, timestamp,
                                                   img_w, img_h, self.id_info_obj, self.config.OBJECT_COOLDOWN,
                                                   self.config.OUTPUT_OBJ_BASE, "object", log_file, conf, self.config.OBJ_SAVE_CROP_SCALE,
                                                   metrics=metrics)

                # Stash for use by process_frame association call.
                self._frame_object_obs = obj_obs_list
        return annotated

    def process_detection(self, track_id, class_name, x1, y1, x2, y2, frame, current_time, timestamp,
                          img_w, img_h, id_info, cooldown, output_base, detection_type, log_file,
                          conf=None, scale=2.5, metrics=None):
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

                # EVIDENCE-WRITING — timed
                if metrics:
                    import time as _time
                    _ev_t0 = _time.perf_counter()
                    save_crop(os.path.join(output_base, class_name), filename, crop)
                    log_detection_to_csv(track_id, filename, class_name, info['last_conf'], current_time)
                    ev_ms = (_time.perf_counter() - _ev_t0) * 1000.0
                    metrics._evidence_save.append(ev_ms)
                    metrics.record_evidence_event(
                        frame_idx=self.frame_idx,
                        track_id=track_id,
                        class_name=class_name,
                        confidence=info['last_conf'],
                        save_duration_ms=ev_ms,
                        detection_type=detection_type,
                    )
                else:
                    save_crop(os.path.join(output_base, class_name), filename, crop)
                    log_detection_to_csv(track_id, filename, class_name, info['last_conf'], current_time)

                info['last_saved'] = current_time

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
        """Yield MJPEG-encoded frames.  JPEG-encode time is recorded when
        self.metrics is set (benchmark mode).  Production callers are unaffected."""
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                if self.metrics:
                    import time as _time
                    _j0 = _time.perf_counter()
                    ret, buffer = cv2.imencode('.jpg', frame)
                    self.metrics._jpeg_encode.append((_time.perf_counter() - _j0) * 1000.0)
                else:
                    ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except queue.Empty:
                continue
