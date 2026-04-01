import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "component_best.pt"

CAMERA_SENSOR_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_FLIP_METHOD = 0
CAMERA_JPEG_QUALITY = 70
CAMERA_IDLE_TIMEOUT_SECONDS = 10
CAMERA_DETECT_CONF = 0.25
CAMERA_INFER_SIZE = 416

# Set to an integer like 2 or 3 if you want to force a specific Argus sensor mode.
# Leaving this as None lets Argus choose automatically.
CAMERA_SENSOR_MODE: Optional[int] = None


def build_gstreamer_pipeline(
    sensor_id: int,
    capture_width: int,
    capture_height: int,
    display_width: int,
    display_height: int,
    framerate: int,
    flip_method: int,
    sensor_mode: Optional[int] = None,
    use_bufapi: bool = True,
) -> str:
    sensor_mode_part = f"sensor-mode={sensor_mode} " if sensor_mode is not None else ""
    bufapi_part = "bufapi-version=true " if use_bufapi else ""

    return (
        f"nvarguscamerasrc sensor-id={sensor_id} {sensor_mode_part}{bufapi_part}! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, "
        f"height=(int){capture_height}, "
        f"format=(string)NV12, "
        f"framerate=(fraction){framerate}/1 ! "
        f"queue max-size-buffers=1 leaky=downstream ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, "
        f"width=(int){display_width}, "
        f"height=(int){display_height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


class CameraService:
    def __init__(
        self,
        sensor_id: int = CAMERA_SENSOR_ID,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
        flip_method: int = CAMERA_FLIP_METHOD,
        jpeg_quality: int = CAMERA_JPEG_QUALITY,
        detect_conf: float = CAMERA_DETECT_CONF,
        infer_size: int = CAMERA_INFER_SIZE,
        idle_timeout_seconds: int = CAMERA_IDLE_TIMEOUT_SECONDS,
        sensor_mode: Optional[int] = CAMERA_SENSOR_MODE,
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
        self.sensor_mode = sensor_mode

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
        self.capture_backend = "argus"

        self.kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )

        self.model = None
        if YOLO is None:
            self.last_error = "ultralytics is not installed; detection disabled"
        elif MODEL_PATH.exists():
            try:
                self.model = YOLO(str(MODEL_PATH))
                self.last_error = None
            except Exception as e:
                self.model = None
                self.last_error = f"Failed to load model: {e}"
        else:
            self.last_error = f"Model not found: {MODEL_PATH}"

    def _pipeline_candidates(self) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []

        # Best option for JetPack 5/6. This commonly fixes NvBufSurfaceFromFd / dmabuf issues.
        candidates.append(
            (
                "argus-bufapi",
                build_gstreamer_pipeline(
                    sensor_id=self.sensor_id,
                    capture_width=self.width,
                    capture_height=self.height,
                    display_width=self.width,
                    display_height=self.height,
                    framerate=self.fps,
                    flip_method=self.flip_method,
                    sensor_mode=self.sensor_mode,
                    use_bufapi=True,
                ),
            )
        )

        # Fallback without bufapi-version, just in case a setup behaves differently.
        candidates.append(
            (
                "argus-legacy",
                build_gstreamer_pipeline(
                    sensor_id=self.sensor_id,
                    capture_width=self.width,
                    capture_height=self.height,
                    display_width=self.width,
                    display_height=self.height,
                    framerate=self.fps,
                    flip_method=self.flip_method,
                    sensor_mode=self.sensor_mode,
                    use_bufapi=False,
                ),
            )
        )

        return candidates

    def _read_probe_frame(
        self,
        cap: cv2.VideoCapture,
        warmup_frames: int = 10,
        tries: int = 40,
        delay_s: float = 0.08,
    ) -> Optional[np.ndarray]:
        for _ in range(warmup_frames):
            try:
                cap.read()
            except Exception:
                pass
            time.sleep(delay_s)

        for _ in range(tries):
            ok, frame = cap.read()
            if ok and frame is not None and getattr(frame, "size", 0) > 0:
                return frame
            time.sleep(delay_s)

        return None

    def _open_camera(self) -> cv2.VideoCapture:
        errors: list[str] = []

        for backend_name, pipeline in self._pipeline_candidates():
            cap = None
            try:
                print(f"[CAMERA] trying backend={backend_name}")
                print(f"[CAMERA] pipeline={pipeline}")

                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if not cap.isOpened():
                    raise RuntimeError("VideoCapture did not open")

                frame = self._read_probe_frame(cap)
                if frame is None:
                    raise RuntimeError("Opened camera but never received a usable frame")

                self.capture_backend = backend_name
                self.last_error = None
                print(f"[CAMERA] opened successfully on backend={backend_name}")
                return cap

            except Exception as e:
                err = f"{backend_name}: {e}"
                errors.append(err)
                print(f"[CAMERA] backend failed: {err}")
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                time.sleep(0.6)

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
        print("[CAMERA] restarting camera")
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
                if not ret or frame is None or getattr(frame, "size", 0) == 0:
                    self.consecutive_failures += 1
                    self.last_error = (
                        f"Failed to read frame ({self.consecutive_failures}) "
                        f"on backend={self.capture_backend}"
                    )
                    time.sleep(0.05)

                    if self.consecutive_failures >= 10:
                        self.last_error = (
                            f"Camera read timeout on backend={self.capture_backend}, "
                            f"restarting camera"
                        )
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
                print(f"[CAMERA] loop error: {e}")
                time.sleep(0.2)

                try:
                    self.restart_camera()
                except Exception as inner:
                    self.last_error = f"{e} | restart failed: {inner}"
                    print(f"[CAMERA] restart failed: {inner}")
                    time.sleep(1.0)

        self._close_camera()
        print("[CAMERA] stopped")

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
                "sensor_id": self.sensor_id,
                "sensor_mode": self.sensor_mode,
                "flip_method": self.flip_method,
            }


camera_service = CameraService()


def get_camera_status() -> Dict[str, Any]:
    return camera_service.get_status()