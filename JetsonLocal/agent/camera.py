import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "component_best.pt"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def argus_pipeline(
    sensor_id: int,
    width: int,
    height: int,
    fps: int,
    flip_method: int,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


class CameraService:
    def __init__(
        self,
        sensor_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        flip_method: int = 0,
        jpeg_quality: int = 70,
        detect_conf: float = 0.25,
        infer_size: int = 416,
        idle_timeout_seconds: int = 10,
    ):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method
        self.jpeg_quality = jpeg_quality
        self.detect_conf = detect_conf
        self.infer_size = infer_size
        self.idle_timeout_seconds = idle_timeout_seconds

        self.camera_backend = _env_str("CAMERA_BACKEND", "auto").lower()
        self.camera_device = _env_str("CAMERA_DEVICE", "/dev/video0")
        self.usb_index = _env_int("CAMERA_USB_INDEX", 0)

        self.cap: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False

        self.lock = threading.Lock()

        self.latest_raw_jpeg: Optional[bytes] = None
        self.latest_annotated_jpeg: Optional[bytes] = None
        self.latest_detections: list[dict] = []

        self.mode = "raw"
        self.enabled = False
        self.last_error: Optional[str] = None
        self.last_access_ts = 0.0
        self.stream_clients = 0

        self.consecutive_failures = 0
        self.capture_backend = "none"

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
            try:
                self.model = YOLO(str(MODEL_PATH))
            except Exception as e:
                self.model = None
                self.last_error = f"Failed to load model: {e}"
        else:
            self.last_error = f"Model not found: {MODEL_PATH}"

    def _read_probe_frame(self, cap: cv2.VideoCapture, tries: int = 30, delay_s: float = 0.10):
        for _ in range(tries):
            ok, frame = cap.read()
            if ok and frame is not None and getattr(frame, "size", 0) > 0:
                return frame
            time.sleep(delay_s)
        return None

    def _apply_capture_settings(self, cap: cv2.VideoCapture) -> None:
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

    def _open_argus_camera(self) -> cv2.VideoCapture:
        pipeline = argus_pipeline(
            sensor_id=self.sensor_id,
            width=self.width,
            height=self.height,
            fps=self.fps,
            flip_method=self.flip_method,
        )

        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError("Could not open Argus camera pipeline.")

        frame = self._read_probe_frame(cap)
        if frame is None:
            cap.release()
            raise RuntimeError("Argus pipeline opened but failed to read first frame.")

        return cap

    def _open_v4l2_path_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.camera_device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open V4L2 device path {self.camera_device}.")

        self._apply_capture_settings(cap)

        frame = self._read_probe_frame(cap)
        if frame is None:
            cap.release()
            raise RuntimeError(f"V4L2 device path opened but failed to read first frame from {self.camera_device}.")

        return cap

    def _open_usb_index_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.usb_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open V4L2 camera index {self.usb_index}.")

        self._apply_capture_settings(cap)

        frame = self._read_probe_frame(cap)
        if frame is None:
            cap.release()
            raise RuntimeError(f"V4L2 camera index opened but failed to read first frame from index {self.usb_index}.")

        return cap

    def _open_camera(self) -> cv2.VideoCapture:
        backend = self.camera_backend
        errors = []

        if backend == "argus":
            order = ["argus"]
        elif backend == "v4l2":
            order = ["v4l2_path"]
        elif backend == "usb":
            order = ["usb_index"]
        else:
            order = ["v4l2_path", "argus"]

        for candidate in order:
            try:
                if candidate == "argus":
                    cap = self._open_argus_camera()
                elif candidate == "v4l2_path":
                    cap = self._open_v4l2_path_camera()
                else:
                    cap = self._open_usb_index_camera()

                self.capture_backend = candidate
                self.last_error = None
                return cap
            except Exception as e:
                errors.append(f"{candidate}: {e}")

        self.capture_backend = "none"
        raise RuntimeError(" | ".join(errors))

    def _close_camera(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = None
        self.latest_raw_jpeg = None
        self.latest_annotated_jpeg = None
        self.latest_detections = []
        self.capture_backend = "none"

    def activate(self, mode: str = "raw") -> None:
        mode = (mode or "raw").strip().lower()
        if mode not in {"raw", "detection"}:
            mode = "raw"

        with self.lock:
            self.mode = mode
            self.enabled = True
            self.last_access_ts = time.time()

        if not self.running:
            self.start()

    def deactivate(self) -> None:
        with self.lock:
            self.enabled = False
            self.stream_clients = 0
            self.last_access_ts = 0.0

        self.stop()

    def set_mode(self, mode: str) -> None:
        mode = (mode or "").strip().lower()
        if mode not in {"raw", "detection"}:
            return

        with self.lock:
            self.mode = mode
            self.last_access_ts = time.time()

    def get_mode(self) -> str:
        with self.lock:
            return self.mode

    def mark_access(self) -> None:
        with self.lock:
            self.last_access_ts = time.time()

    def add_stream_client(self) -> None:
        with self.lock:
            self.stream_clients += 1
            self.last_access_ts = time.time()

    def remove_stream_client(self) -> None:
        with self.lock:
            self.stream_clients = max(0, self.stream_clients - 1)
            self.last_access_ts = time.time()

    def start(self) -> None:
        if self.running:
            return

        self.cap = self._open_camera()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self._close_camera()
        self.thread = None

    def restart_camera(self) -> None:
        self._close_camera()
        time.sleep(1.0)
        self.cap = self._open_camera()
        self.consecutive_failures = 0

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
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        return annotated, detections

    def _should_stop_for_idle(self) -> bool:
        with self.lock:
            if not self.enabled:
                return True
            if self.stream_clients > 0:
                return False
            if self.last_access_ts <= 0:
                return True
            return (time.time() - self.last_access_ts) > self.idle_timeout_seconds

    def _loop(self) -> None:
        while self.running:
            try:
                if self._should_stop_for_idle():
                    self.running = False
                    break

                if self.cap is None:
                    self.cap = self._open_camera()

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.consecutive_failures += 1
                    self.last_error = f"Failed to read frame ({self.consecutive_failures}) on backend={self.capture_backend}"
                    time.sleep(0.05)

                    if self.consecutive_failures >= 10:
                        self.last_error = f"Camera read timeout on backend={self.capture_backend}, restarting camera"
                        self.restart_camera()
                    continue

                self.consecutive_failures = 0

                raw_jpeg = self._encode_jpeg(frame)

                with self.lock:
                    current_mode = self.mode

                if current_mode == "detection":
                    annotated, detections = self._run_detection(frame)
                    annotated_jpeg = self._encode_jpeg(annotated)
                else:
                    annotated_jpeg = None
                    detections = []

                with self.lock:
                    self.latest_raw_jpeg = raw_jpeg
                    self.latest_annotated_jpeg = annotated_jpeg
                    self.latest_detections = detections
                    self.last_error = None

                time.sleep(0.01)

            except Exception as e:
                self.last_error = str(e)
                time.sleep(0.2)
                try:
                    self.restart_camera()
                except Exception as inner:
                    self.last_error = f"{e} | restart failed: {inner}"
                    time.sleep(1.0)

        self._close_camera()

    def get_jpeg(self) -> Optional[bytes]:
        self.mark_access()
        with self.lock:
            if self.mode == "detection":
                return self.latest_annotated_jpeg
            return self.latest_raw_jpeg

    def get_detections(self) -> list[dict]:
        self.mark_access()
        with self.lock:
            return list(self.latest_detections)

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "camera_ready": self.cap is not None and self.cap.isOpened() if self.cap else False,
                "enabled": self.enabled,
                "running": self.running,
                "mode": self.mode,
                "stream_clients": self.stream_clients,
                "model_path": str(MODEL_PATH),
                "model_loaded": self.model is not None,
                "raw_frame_ready": self.latest_raw_jpeg is not None,
                "annotated_frame_ready": self.latest_annotated_jpeg is not None,
                "detection_count": len(self.latest_detections),
                "last_error": self.last_error,
                "resolution": {"width": self.width, "height": self.height},
                "fps": self.fps,
                "idle_timeout_seconds": self.idle_timeout_seconds,
                "capture_backend": self.capture_backend,
                "camera_backend_requested": self.camera_backend,
                "camera_device": self.camera_device,
                "usb_index": self.usb_index,
            }


camera_service = CameraService(
    sensor_id=_env_int("CAMERA_SENSOR_ID", 0),
    width=_env_int("CAMERA_WIDTH", 1280),
    height=_env_int("CAMERA_HEIGHT", 720),
    fps=_env_int("CAMERA_FPS", 30),
    flip_method=_env_int("CAMERA_FLIP_METHOD", 0),
    jpeg_quality=_env_int("CAMERA_JPEG_QUALITY", 70),
    detect_conf=_env_float("CAMERA_DETECT_CONF", 0.25),
    infer_size=_env_int("CAMERA_INFER_SIZE", 416),
    idle_timeout_seconds=_env_int("CAMERA_IDLE_TIMEOUT_SECONDS", 10),
)


def get_camera_status() -> Dict[str, Any]:
    return camera_service.get_status()