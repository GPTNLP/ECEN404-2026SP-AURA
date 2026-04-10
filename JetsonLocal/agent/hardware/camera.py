from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from core.config import (
    CAMERA_DEVICE_INDEX,
    CAMERA_DEVICE_PATH,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    CAMERA_JPEG_QUALITY,
    CAMERA_IDLE_TIMEOUT_SECONDS,
    CAMERA_DETECT_CONF,
    CAMERA_INFER_SIZE,
)

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


BASE_DIR = Path(__file__).resolve().parent
AGENT_DIR = BASE_DIR.parent
JETSONLOCAL_DIR = AGENT_DIR.parent
MODEL_DIR = JETSONLOCAL_DIR / "models"

COMPONENT_MODEL_PATH = MODEL_DIR / "component_best.pt"
COLORCODE_MODEL_PATH = MODEL_DIR / "colorcode_best.pt"
FACE_MODEL_PATH = MODEL_DIR / "yolov11s-face.pt"

SUPPORTED_MODES = {"raw", "detection", "colorcode", "face"}


class CameraService:
    def __init__(self):
        self.device_index = CAMERA_DEVICE_INDEX
        self.device_path = CAMERA_DEVICE_PATH
        self.width = CAMERA_WIDTH
        self.height = CAMERA_HEIGHT
        self.fps = CAMERA_FPS
        self.jpeg_quality = CAMERA_JPEG_QUALITY
        self.detect_conf = CAMERA_DETECT_CONF
        self.infer_size = CAMERA_INFER_SIZE
        self.idle_timeout_seconds = CAMERA_IDLE_TIMEOUT_SECONDS

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
        self.capture_backend = "usb"
        self.active_source = None

        self.kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )

        self.models: Dict[str, Optional[Any]] = {
            "detection": None,
            "colorcode": None,
            "face": None,
        }
        self.model_paths: Dict[str, Path] = {
            "detection": COMPONENT_MODEL_PATH,
            "colorcode": COLORCODE_MODEL_PATH,
            "face": FACE_MODEL_PATH,
        }

        self.last_annotated_frame: Optional[np.ndarray] = None
        self.last_mode_for_annotation: Optional[str] = None
        self.last_inference_time = 0.0
        self.inference_interval_s = 0.12
        self.last_frame_ts = 0.0

        if YOLO is None:
            self.last_error = "ultralytics is not installed; detection modes disabled"

    def _ensure_model_loaded(self, mode: str):
        if YOLO is None:
            self.last_error = "ultralytics is not installed; detection modes disabled"
            return None

        if mode not in {"detection", "colorcode", "face"}:
            return None

        if self.models.get(mode) is not None:
            return self.models[mode]

        model_path = self.model_paths[mode]
        if not model_path.exists():
            self.last_error = f"Model not found: {model_path}"
            return None

        try:
            self.models[mode] = YOLO(str(model_path))
            self.last_error = None
            print(f"[CAMERA] loaded model for mode='{mode}' from {model_path}")
            return self.models[mode]
        except Exception as e:
            self.models[mode] = None
            self.last_error = f"Failed to load model '{mode}': {e}"
            print(f"[CAMERA] failed to load model '{mode}': {e}")
            return None

    def _source_candidates(self) -> list[tuple[str, Any]]:
        candidates: list[tuple[str, Any]] = []
        if self.device_path:
            candidates.append(("v4l2-path", self.device_path))
        candidates.append(("v4l2-index", self.device_index))
        for idx in [0, 1, 2, 3]:
            if idx != self.device_index:
                candidates.append((f"v4l2-index-{idx}", idx))
        return candidates

    def _configure_cap(self, cap: cv2.VideoCapture) -> None:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def _read_probe_frame(
        self,
        cap: cv2.VideoCapture,
        warmup_frames: int = 8,
        tries: int = 30,
        delay_s: float = 0.04,
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

        for source_name, source in self._source_candidates():
            cap = None
            try:
                print(f"[CAMERA] trying source={source_name} value={source}")

                cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(source)

                if not cap.isOpened():
                    raise RuntimeError("VideoCapture did not open")

                self._configure_cap(cap)
                frame = self._read_probe_frame(cap)
                if frame is None:
                    raise RuntimeError("Opened camera but never received a usable frame")

                self.capture_backend = "usb-v4l2"
                self.active_source = str(source)
                self.last_error = None

                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)

                print(
                    f"[CAMERA] opened successfully source={source} "
                    f"resolution={actual_w}x{actual_h} fps={actual_fps:.2f}"
                )
                return cap

            except Exception as e:
                err = f"{source_name}: {e}"
                errors.append(err)
                print(f"[CAMERA] source failed: {err}")

                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass

                time.sleep(0.2)

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
        self.last_annotated_frame = None
        self.last_mode_for_annotation = None
        self.active_source = None

    def activate(self, mode: str = "raw") -> None:
        mode = (mode or "raw").strip().lower()
        if mode not in SUPPORTED_MODES:
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
        if mode not in SUPPORTED_MODES:
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
        time.sleep(0.4)
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

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    def _draw_tracking_overlay(self, annotated: np.ndarray, detections: list[dict]) -> None:
        h, w = annotated.shape[:2]
        cx = w // 2
        cy = h // 2

        cv2.line(annotated, (cx - 20, cy), (cx + 20, cy), (255, 255, 0), 2)
        cv2.line(annotated, (cx, cy - 20), (cx, cy + 20), (255, 255, 0), 2)

        if not detections:
            return

        best = max(detections, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = best["bbox"]
        tx = (x1 + x2) // 2
        ty = (y1 + y2) // 2

        cv2.line(annotated, (cx, cy), (tx, ty), (0, 255, 255), 2)
        cv2.circle(annotated, (tx, ty), 7, (0, 255, 255), -1)

    def _run_model_mode(self, frame: np.ndarray, mode: str) -> tuple[np.ndarray, list[dict]]:
        model = self._ensure_model_loaded(mode)
        if model is None:
            return frame.copy(), []

        sharp = cv2.filter2D(frame, -1, self.kernel)
        h, w = sharp.shape[:2]
        infer = cv2.resize(sharp, (self.infer_size, self.infer_size))

        results = model(infer, verbose=False, conf=self.detect_conf)
        annotated = sharp.copy()
        detections: list[dict] = []

        scale_x = w / self.infer_size
        scale_y = h / self.infer_size

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = model.names[cls_id]

                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)

                detections.append(
                    {
                        "label": label,
                        "confidence": round(conf, 4),
                        "bbox": [x1, y1, x2, y2],
                        "mode": mode,
                    }
                )

                color = (0, 255, 0)
                if mode == "colorcode":
                    color = (255, 200, 0)
                elif mode == "face":
                    color = (255, 0, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if mode == "face":
            self._draw_tracking_overlay(annotated, detections)

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
                    time.sleep(0.01)

                    if self.consecutive_failures >= 10:
                        self.last_error = (
                            f"Camera read timeout on backend={self.capture_backend}, "
                            f"restarting camera"
                        )
                        self.restart_camera()
                    continue

                self.consecutive_failures = 0
                frame = self._rotate_frame(frame)
                self.last_frame_ts = time.time()

                raw_jpeg = self._encode_jpeg(frame)

                with self.lock:
                    current_mode = self.mode

                annotated_jpeg = None
                detections: list[dict] = []

                if current_mode != "raw":
                    now = time.time()
                    should_run_inference = (
                        self.last_mode_for_annotation != current_mode
                        or self.last_annotated_frame is None
                        or (now - self.last_inference_time) >= self.inference_interval_s
                    )

                    if should_run_inference:
                        annotated_frame, detections = self._run_model_mode(frame, current_mode)
                        self.last_annotated_frame = annotated_frame
                        self.last_mode_for_annotation = current_mode
                        self.last_inference_time = now
                        annotated_jpeg = self._encode_jpeg(annotated_frame)
                        if annotated_jpeg is not None:
                            self.latest_annotated_jpeg = annotated_jpeg
                            self.latest_detections = detections
                    else:
                        if self.last_annotated_frame is not None:
                            annotated_jpeg = self._encode_jpeg(self.last_annotated_frame)
                        detections = self.latest_detections

                with self.lock:
                    self.latest_raw_jpeg = raw_jpeg
                    if current_mode == "raw":
                        self.latest_annotated_jpeg = None
                        self.latest_detections = []
                    elif annotated_jpeg is not None:
                        self.latest_annotated_jpeg = annotated_jpeg
                        self.latest_detections = detections
                    self.last_error = None

                time.sleep(0.001)

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
            if self.mode == "raw":
                return self.latest_raw_jpeg
            return self.latest_annotated_jpeg or self.latest_raw_jpeg

    def get_detections(self) -> list[dict]:
        self.mark_access()
        with self.lock:
            return list(self.latest_detections)

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            actual_width = 0
            actual_height = 0
            actual_fps = 0.0

            if self.cap is not None and self.cap.isOpened():
                try:
                    actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    actual_fps = float(cv2.CAP_PROP_FPS and self.cap.get(cv2.CAP_PROP_FPS) or 0)
                except Exception:
                    pass

            return {
                "camera_ready": self.cap is not None and self.cap.isOpened() if self.cap else False,
                "enabled": self.enabled,
                "running": self.running,
                "mode": self.mode,
                "available_modes": sorted(SUPPORTED_MODES),
                "stream_clients": self.stream_clients,
                "model_paths": {k: str(v) for k, v in self.model_paths.items()},
                "models_loaded": {k: self.models[k] is not None for k in self.models},
                "raw_frame_ready": self.latest_raw_jpeg is not None,
                "annotated_frame_ready": self.latest_annotated_jpeg is not None,
                "detection_count": len(self.latest_detections),
                "last_error": self.last_error,
                "resolution": {"width": self.width, "height": self.height},
                "actual_resolution": {"width": actual_width, "height": actual_height},
                "fps": self.fps,
                "actual_fps": actual_fps,
                "idle_timeout_seconds": self.idle_timeout_seconds,
                "capture_backend": self.capture_backend,
                "device_index": self.device_index,
                "device_path": self.device_path,
                "active_source": self.active_source,
                "last_frame_ts": self.last_frame_ts,
                "rotation": "90cw",
            }


camera_service = CameraService()


def get_camera_status() -> Dict[str, Any]:
    return camera_service.get_status()