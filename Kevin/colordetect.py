import cv2
import torch
from ultralytics import YOLO

# Load trained model
model = YOLO("colorband.pt")

# Force GPU if available
if torch.cuda.is_available():
    model.to("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available, using CPU")

# GStreamer pipeline for Jetson CSI camera (CAM0 = sensor-id 0)
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true sync=false"
    )

pipeline = gstreamer_pipeline()
print("GStreamer pipeline:")
print(pipeline)

# Open Jetson CSI camera
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Cannot open Jetson CSI camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame, conf=0.25, device=0 if torch.cuda.is_available() else "cpu")

    # Draw detections with smaller box + text
    annotated_frame = results[0].plot(
        line_width=1,
        font_size=0.5
    )

    cv2.imshow("Resistor Color Band Detection", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
