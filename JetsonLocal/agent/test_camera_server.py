import os
import time
import threading
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
import uvicorn


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def argus_pipeline(sensor_id: int, width: int, height: int, fps: int, flip_method: int) -> str:
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


class CameraTester:
    def __init__(self):
        self.sensor_id = env_int("CAMERA_SENSOR_ID", 0)
        self.width = env_int("CAMERA_WIDTH", 1280)
        self.height = env_int("CAMERA_HEIGHT", 720)
        self.fps = env_int("CAMERA_FPS", 30)
        self.flip_method = env_int("CAMERA_FLIP_METHOD", 0)
        self.jpeg_quality = env_int("CAMERA_JPEG_QUALITY", 80)

        self.cap: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_jpeg = None
        self.last_error = None
        self.frame_count = 0
        self.clients = 0

    def open(self):
        pipeline = argus_pipeline(
            self.sensor_id,
            self.width,
            self.height,
            self.fps,
            self.flip_method,
        )

        print("[INFO] Opening pipeline:")
        print(pipeline)

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open Argus pipeline")

        print("[INFO] Warming up camera...")
        for i in range(12):
            ret, frame = self.cap.read()
            print(f"[WARMUP {i+1}/12] ret={ret} frame_none={frame is None}")
            time.sleep(0.08)

        good = False
        for i in range(20):
            ret, frame = self.cap.read()
            ok = ret and frame is not None and getattr(frame, "size", 0) > 0
            print(f"[PROBE {i+1}/20] ret={ret} ok={ok}")
            if ok:
                good = True
                break
            time.sleep(0.05)

        if not good:
            self.cap.release()
            self.cap = None
            raise RuntimeError("Argus pipeline opened but no usable frames were received")

    def start(self):
        if self.running:
            return
        self.open()
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.thread = None

    def loop(self):
        while self.running:
            try:
                if self.cap is None:
                    self.open()

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.last_error = "Failed to read frame"
                    time.sleep(0.03)
                    continue

                ok, buf = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
                )
                if not ok:
                    self.last_error = "JPEG encode failed"
                    time.sleep(0.03)
                    continue

                with self.lock:
                    self.latest_frame = frame
                    self.latest_jpeg = buf.tobytes()
                    self.frame_count += 1
                    self.last_error = None

                time.sleep(0.01)

            except Exception as e:
                self.last_error = str(e)
                time.sleep(0.2)

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_status(self):
        with self.lock:
            return {
                "running": self.running,
                "camera_open": self.cap.isOpened() if self.cap is not None else False,
                "frame_count": self.frame_count,
                "has_jpeg": self.latest_jpeg is not None,
                "last_error": self.last_error,
                "clients": self.clients,
                "resolution": {
                    "width": self.width,
                    "height": self.height,
                },
                "fps_requested": self.fps,
                "sensor_id": self.sensor_id,
                "flip_method": self.flip_method,
            }

    def add_client(self):
        with self.lock:
            self.clients += 1

    def remove_client(self):
        with self.lock:
            self.clients = max(0, self.clients - 1)


camera = CameraTester()
app = FastAPI()


def mjpeg_generator():
    camera.add_client()
    try:
        while True:
            frame = camera.get_jpeg()
            if frame is None:
                time.sleep(0.03)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.03)
    finally:
        camera.remove_client()


@app.on_event("startup")
def startup():
    camera.start()


@app.on_event("shutdown")
def shutdown():
    camera.stop()


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html>
    <head>
        <title>AURA Camera Test</title>
        <style>
            body {
                background: #111;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 20px;
                margin: 0;
            }
            img {
                max-width: 95vw;
                max-height: 80vh;
                background: black;
                border: 2px solid #444;
            }
            .links {
                margin-top: 16px;
            }
            a {
                color: #7cc7ff;
                margin: 0 10px;
                text-decoration: none;
            }
        </style>
    </head>
    <body>
        <h1>AURA Camera Test</h1>
        <img src="/stream" alt="camera stream" />
        <div class="links">
            <a href="/snapshot" target="_blank">Snapshot</a>
            <a href="/status" target="_blank">Status</a>
        </div>
    </body>
    </html>
    """


@app.get("/stream")
def stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/snapshot")
def snapshot():
    frame = camera.get_jpeg()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available")
    return Response(content=frame, media_type="image/jpeg")


@app.get("/status")
def status():
    return camera.get_status()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)