import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "component_best.pt"


def gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1920,
    capture_height: int = 1080,
    display_width: int = 1920,
    display_height: int = 1080,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
    )


class CameraService:
    def __init__(
        self,
        sensor_id: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        flip_method: int = 0,
        jpeg_quality: int = 80,
        detect_conf: float = 0.25,
        infer_size: int = 640,
    ):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method
        self.jpeg_quality = jpeg_quality
        self.detect_conf = detect_conf
        self.infer_size = infer_size

        self.cap: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False

        self.lock = threading.Lock()
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_raw_jpeg: Optional[bytes] = None
        self.latest_annotated_jpeg: Optional[bytes] = None
        self.latest_detections: list[dict] = []
        self.mode = "raw"
        self.last_error: Optional[str] = None

        self.kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )

        self.model = None
        if MODEL_PATH.exists():
            self.model = YOLO(str(MODEL_PATH))
        else:
            self.last_error = f"Model not found: {MODEL_PATH}"

    def start(self) -> None:
        if self.running:
            return

        pipeline = gstreamer_pipeline(
            sensor_id=self.sensor_id,
            capture_width=self.width,
            capture_height=self.height,
            display_width=self.width,
            display_height=self.height,
            framerate=self.fps,
            flip_method=self.flip_method,
        )

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open Jetson CSI camera.")

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.thread = None

    def set_mode(self, mode: str) -> None:
        mode = (mode or "").strip().lower()
        if mode in {"raw", "detection"}:
            self.mode = mode

    def get_mode(self) -> str:
        return self.mode

    def _encode_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if not ok:
            return None
        return buf.tobytes()

    def _run_detection(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        if self.model is None:
            return frame.copy(), []

        sharp = cv2.filter2D(frame, -1, self.kernel)
        h, w = sharp.shape[:2]

        infer = cv2.resize(sharp, (self.infer_size, self.infer_size))
        results = self.model(infer, verbose=False, conf=self.detect_conf)

        annotated = sharp.copy()
        detections: list[dict] = []

        scale_x = w / self.infer_size
        scale_y = h / self.infer_size

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = self.model.names[cls_id]

                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)

                detections.append(
                    {
                        "label": label,
                        "confidence": round(conf, 4),
                        "bbox": [x1, y1, x2, y2],
                    }
                )

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        return annotated, detections

    def _loop(self) -> None:
        while self.running:
            try:
                if self.cap is None:
                    time.sleep(0.05)
                    continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.last_error = "Failed to read frame from camera."
                    time.sleep(0.02)
                    continue

                annotated, detections = self._run_detection(frame)

                raw_jpeg = self._encode_jpeg(frame)
                ann_jpeg = self._encode_jpeg(annotated)

                with self.lock:
                    self.latest_raw_frame = frame
                    self.latest_annotated_frame = annotated
                    self.latest_raw_jpeg = raw_jpeg
                    self.latest_annotated_jpeg = ann_jpeg
                    self.latest_detections = detections
                    self.last_error = None

            except Exception as e:
                self.last_error = str(e)
                time.sleep(0.05)

    def get_jpeg(self, mode: str = "raw") -> Optional[bytes]:
        with self.lock:
            if mode == "detection":
                return self.latest_annotated_jpeg
            return self.latest_raw_jpeg

    def get_detections(self) -> list[dict]:
        with self.lock:
            return list(self.latest_detections)

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "camera_ready": self.cap is not None and self.cap.isOpened() if self.cap else False,
                "mode": self.mode,
                "model_path": str(MODEL_PATH),
                "model_loaded": self.model is not None,
                "raw_frame_ready": self.latest_raw_jpeg is not None,
                "annotated_frame_ready": self.latest_annotated_jpeg is not None,
                "detection_count": len(self.latest_detections),
                "last_error": self.last_error,
                "resolution": {"width": self.width, "height": self.height},
                "fps": self.fps,
            }


camera_service = CameraService()


def get_camera_status() -> Dict[str, Any]:
    return camera_service.get_status()