import cv2
import time

SENSOR_ID = 0
WIDTH = 1280
HEIGHT = 720
FPS = 30
FLIP_METHOD = 0


def build_pipeline(sensor_mode=None, use_bufapi=True):
    sensor_mode_part = f"sensor-mode={sensor_mode} " if sensor_mode is not None else ""
    bufapi_part = "bufapi-version=true " if use_bufapi else ""

    return (
        f"nvarguscamerasrc sensor-id={SENSOR_ID} {sensor_mode_part}{bufapi_part}! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){WIDTH}, height=(int){HEIGHT}, "
        f"format=(string)NV12, framerate=(fraction){FPS}/1 ! "
        f"nvvidconv flip-method={FLIP_METHOD} ! "
        f"video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def try_open(pipeline_name, pipeline):
    print("\n" + "=" * 60)
    print(f"TRYING: {pipeline_name}")
    print(pipeline)
    print("=" * 60)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("FAILED: could not open camera")
        return False

    print("Opened camera, starting stream... (press q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("No frame...")
            time.sleep(0.05)
            continue

        cv2.imshow("Jetson Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True


def main():
    pipelines = [
        ("argus_bufapi_auto", build_pipeline(sensor_mode=None, use_bufapi=True)),
        ("argus_legacy_auto", build_pipeline(sensor_mode=None, use_bufapi=False)),
        ("argus_bufapi_mode3", build_pipeline(sensor_mode=3, use_bufapi=True)),
        ("argus_legacy_mode3", build_pipeline(sensor_mode=3, use_bufapi=False)),
    ]

    for name, pipe in pipelines:
        if try_open(name, pipe):
            print(f"\nSUCCESS with: {name}")
            return

    print("\nNo pipeline worked.")


if __name__ == "__main__":
    main()