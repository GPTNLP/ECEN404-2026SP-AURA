import cv2
import threading
import time


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class JetsonCamera:
    def __init__(
        self,
        sensor_id=0,
        width=1280,
        height=720,
        fps=30,
        flip_method=0,
    ):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method

        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
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
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_jpeg_bytes(self, quality=80):
        frame = self.get_frame()
        if frame is None:
            return None

        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            return None

        return buffer.tobytes()

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=1.0)

        if self.cap is not None:
            self.cap.release()

        self.cap = None
        self.thread = None