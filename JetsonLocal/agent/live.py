import cv2
import numpy as np
from ultralytics import YOLO

# 🔥 LOAD MODEL
model = YOLO("models/component_best.pt")  # change if needed

def gstreamer_pipeline():
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, width=1280, height=720, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )

def main():
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        raise RuntimeError("Could not open Jetson CSI camera.")

    print("🔥 Live detection running (press 'q' to quit)")

    # 🔥 sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5,-1],
        [0, -1, 0]
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # 🔥 sharpen image
        frame = cv2.filter2D(frame, -1, kernel)

        # 🔥 resize for YOLO
        resized = cv2.resize(frame, (640, 640))

        # 🔥 run detection
        results = model(resized, verbose=False)

        # 🔥 draw detections
        annotated = results[0].plot()

        # 🔥 show
        cv2.imshow("AURA Live Detection", annotated)

        # quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()