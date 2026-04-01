import time
from typing import Optional

import cv2
import numpy as np


SENSOR_ID = 0
WIDTH = 1280
HEIGHT = 720
FPS = 30
FLIP_METHOD = 0


def build_argus_pipeline(
    sensor_id: int = SENSOR_ID,
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
    flip_method: int = FLIP_METHOD,
    sensor_mode: Optional[int] = None,
    use_bufapi: bool = True,
) -> str:
    sensor_mode_part = f"sensor-mode={sensor_mode} " if sensor_mode is not None else ""
    bufapi_part = "bufapi-version=true " if use_bufapi else ""

    return (
        f"nvarguscamerasrc sensor-id={sensor_id} {sensor_mode_part}{bufapi_part}! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){width}, "
        f"height=(int){height}, "
        f"format=(string)NV12, "
        f"framerate=(fraction){fps}/1 ! "
        f"queue max-size-buffers=1 leaky=downstream ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, "
        f"width=(int){width}, "
        f"height=(int){height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def build_v4l2_pipeline(
    device: str = "/dev/video0",
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
) -> str:
    return (
        f"v4l2src device={device} ! "
        f"video/x-raw, "
        f"width=(int){width}, "
        f"height=(int){height}, "
        f"framerate=(fraction){fps}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def make_waiting_frame(title: str, line2: str = "") -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        title,
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if line2:
        cv2.putText(
            frame,
            line2,
            (40, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        "Press q to quit",
        (40, HEIGHT - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def try_pipeline(name: str, pipeline: str, open_timeout_s: float = 8.0) -> bool:
    print(f"\nTRYING: {name}")
    print(pipeline)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    cv2.namedWindow("Jetson Camera Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Jetson Camera Test", 1280, 720)

    start = time.time()
    got_frame = False

    while True:
        ret, frame = cap.read()

        if ret and frame is not None and getattr(frame, "size", 0) > 0:
            got_frame = True
            cv2.imshow("Jetson Camera Test", frame)
        else:
            elapsed = time.time() - start
            wait_frame = make_waiting_frame(
                f"Trying pipeline: {name}",
                f"No frame yet... {elapsed:.1f}s",
            )
            cv2.imshow("Jetson Camera Test", wait_frame)

            if elapsed >= open_timeout_s:
                break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit(0)

        if got_frame:
            print(f"SUCCESS: {name}")
            while True:
                ret, frame = cap.read()
                if ret and frame is not None and getattr(frame, "size", 0) > 0:
                    cv2.imshow("Jetson Camera Test", frame)
                else:
                    cv2.imshow(
                        "Jetson Camera Test",
                        make_waiting_frame(f"{name} opened", "Frame temporarily lost"),
                    )

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return True

    cap.release()
    return False


def main() -> int:
    pipelines = [
        ("argus_bufapi_auto", build_argus_pipeline(sensor_mode=None, use_bufapi=True)),
        ("argus_legacy_auto", build_argus_pipeline(sensor_mode=None, use_bufapi=False)),
        ("argus_bufapi_mode3", build_argus_pipeline(sensor_mode=3, use_bufapi=True)),
        ("argus_legacy_mode3", build_argus_pipeline(sensor_mode=3, use_bufapi=False)),
        ("v4l2_video0", build_v4l2_pipeline("/dev/video0")),
        ("v4l2_video1", build_v4l2_pipeline("/dev/video1")),
    ]

    for name, pipeline in pipelines:
        ok = try_pipeline(name, pipeline)
        if ok:
            return 0

    cv2.namedWindow("Jetson Camera Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Jetson Camera Test", 1280, 720)

    while True:
        cv2.imshow(
            "Jetson Camera Test",
            make_waiting_frame("No pipeline worked", "Press q to quit"),
        )
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())