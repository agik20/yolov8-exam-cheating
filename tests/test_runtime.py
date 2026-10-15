"""Phase 1 — Runtime correctness tests.

Deterministic, no Jetson hardware required, no performance claims.
Run:  python -m pytest tests/test_runtime.py -v
"""
import io
import os
import queue
import threading
import time
import unittest
from unittest.mock import patch, MagicMock

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from inference.config import Config
from inference.core.pipeline import CheatingDetectionPipeline


def _make_pipeline(tmpdir, video_path="test.mp4"):
    """Build a pipeline without loading real models (mock the YOLO calls)."""
    cfg = Config()
    cfg.OUTPUT_POSE_BASE = os.path.join(tmpdir, "pose")
    cfg.OUTPUT_OBJ_BASE = os.path.join(tmpdir, "obj")
    cfg.OUTPUT_FULLROOM = os.path.join(tmpdir, "room")
    cfg.OUTPUT_FRAMES = os.path.join(tmpdir, "frames")

    pipeline = CheatingDetectionPipeline.__new__(CheatingDetectionPipeline)
    pipeline.config = cfg
    pipeline.video_path = video_path
    pipeline.pose_model_path = "pose.pt"
    pipeline.object_model_path = "obj.pt"
    pipeline.headless = True
    pipeline.metrics = None
    pipeline.id_info_pose = {}
    pipeline.id_info_obj = {}
    pipeline.frame_idx = 0
    pipeline.prev_time = time.time()
    pipeline.start_time = time.time()
    pipeline.last_frame_save_time = time.time()
    pipeline.stop_event = threading.Event()
    pipeline.frame_queue = queue.Queue(maxsize=1)
    pipeline.pose_model = MagicMock()
    pipeline.object_model = MagicMock()
    pipeline.pose_class_map = {0: "no-cheating", 1: "provide-object", 2: "see-friends-work"}
    pipeline.object_class_map = {0: "smartphone", 1: "calculator"}
    # Phase 2B.2–2B.5 shadow attributes (normally set in __init__,
    # which this helper bypasses via __new__).
    pipeline._candidate_layer = None
    pipeline._src_fps = None
    pipeline._confirmation_policy = None
    pipeline._shadow_results = []
    pipeline._cooldown_policy = None
    pipeline._shadow_lifecycle = None
    pipeline._associator = None
    pipeline._association_results = []
    pipeline._frame_person_obs = []
    pipeline._frame_object_obs = []
    return pipeline


class TestHeadlessCorrectness(unittest.TestCase):
    """Inference must run without requiring an OpenCV GUI when headless=True."""

    def test_headless_does_not_call_imshow(self):
        tmpdir = "/tmp/_phase1_test_headless"
        os.makedirs(tmpdir, exist_ok=True)
        pipeline = _make_pipeline(tmpdir)

        with patch("inference.core.pipeline.cv2.imshow") as mock_imshow, \
             patch("inference.core.pipeline.cv2.waitKey") as mock_waitkey, \
             patch("inference.core.pipeline.cv2.destroyAllWindows") as mock_destroy:
            import numpy as np
            from datetime import datetime
            frame = np.zeros((640, 640, 3), dtype=np.uint8)
            log = open(os.devnull, "w")
            pipeline.process_frame(frame, datetime.now(), "120000", log)
            log.close()
            mock_imshow.assert_not_called()
            mock_waitkey.assert_not_called()
            mock_destroy.assert_not_called()


class TestEOFCleanup(unittest.TestCase):
    """When video reaches EOF, resources must be released and stop_event must be set."""

    def test_eof_releases_capture_and_sets_stop_event(self):
        tmpdir = "/tmp/_phase1_test_eof"
        os.makedirs(tmpdir, exist_ok=True)
        pipeline = _make_pipeline(tmpdir, video_path="nonexistent.mp4")
        log_file = io.StringIO()

        with patch("inference.core.pipeline.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (False, None)
            mock_vc.return_value = mock_cap

            pipeline.process_video(log_file)

            self.assertTrue(pipeline.stop_event.is_set(),
                            "stop_event not set after video EOF")
            mock_cap.release.assert_called()

    def test_failed_capture_opening_sets_stop_event(self):
        tmpdir = "/tmp/_phase1_test_failed_open"
        os.makedirs(tmpdir, exist_ok=True)
        pipeline = _make_pipeline(tmpdir, video_path="nonexistent.mp4")
        log_file = io.StringIO()

        with patch("inference.core.pipeline.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_vc.return_value = mock_cap

            pipeline.process_video(log_file)

            self.assertTrue(pipeline.stop_event.is_set(),
                            "stop_event not set when capture fails to open")
            self.assertTrue(log_file.closed)


class TestExceptionCleanup(unittest.TestCase):
    """If an exception occurs inside the processing loop, resources must still be released."""

    def test_exception_releases_capture_and_log(self):
        tmpdir = "/tmp/_phase1_test_exc"
        os.makedirs(tmpdir, exist_ok=True)
        pipeline = _make_pipeline(tmpdir)
        log_file = io.StringIO()

        with patch("inference.core.pipeline.cv2.VideoCapture") as mock_vc, \
             patch.object(pipeline, "process_frame", side_effect=RuntimeError("simulated")):
            import numpy as np
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            good_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cap.read.side_effect = [(True, good_frame), (False, None)]
            mock_vc.return_value = mock_cap

            # Should not raise — process_video catches and handles the exception.
            pipeline.process_video(log_file)

            self.assertTrue(pipeline.stop_event.is_set())
            mock_cap.release.assert_called()
            self.assertTrue(log_file.closed)


class TestShutdown(unittest.TestCase):
    """Processing thread must terminate cleanly; join must not block indefinitely."""

    def test_stop_event_terminates_generate_frames(self):
        tmpdir = "/tmp/_phase1_test_shutdown"
        os.makedirs(tmpdir, exist_ok=True)
        pipeline = _make_pipeline(tmpdir)
        pipeline.frame_queue = queue.Queue(maxsize=1)
        pipeline.stop_event = threading.Event()
        pipeline.stop_event.set()

        frames = list(pipeline.generate_frames())
        self.assertEqual(frames, [])


class TestLoggingAppendMode(unittest.TestCase):
    """Restarting the application must not truncate previous log content."""

    def test_main_uses_append_mode(self):
        import inspect
        import inference.main as main_mod
        src = inspect.getsource(main_mod.main)
        self.assertIn('open(_paths.DETECTION_LOG_TXT, "a")', src,
                      "main() must open log file in append mode")
        self.assertNotIn('open(_paths.DETECTION_LOG_TXT, "w")', src,
                         "main() must not open log file in write (truncate) mode")


class TestQueueTermination(unittest.TestCase):
    """After shutdown, the queue must not leave the consumer in an inconsistent state."""

    def test_queue_empty_after_stop_event_set(self):
        tmpdir = "/tmp/_phase1_test_queue"
        os.makedirs(tmpdir, exist_ok=True)
        pipeline = _make_pipeline(tmpdir)
        pipeline.frame_queue = queue.Queue(maxsize=1)
        pipeline.stop_event = threading.Event()

        import numpy as np
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        pipeline.frame_queue.put(frame)
        pipeline.stop_event.set()

        frames = list(pipeline.generate_frames())
        self.assertLessEqual(len(frames), 1)


if __name__ == "__main__":
    unittest.main()
