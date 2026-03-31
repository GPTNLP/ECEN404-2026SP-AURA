import time
import cv2
from ultralytics import YOLO

from camera import JetsonCamera


def main():
    # ---- LOAD MODELS ----
    component_model = YOLO("../Kevin/component_best.pt")
    color_model = YOLO("../Kevin/colorcode_best.pt")

    print("Models loaded.")

    # ---- START CAMERA ----
    cam = JetsonCamera()
    cam.start()

    print("Camera started. Waiting for frame...")
    time.sleep(2)

    frame = cam.get_frame()

    if frame is None:
        print("ERROR: No frame from camera.")
        cam.stop()
        return

    print("Running component detection...")

    # ---- RUN COMPONENT DETECTION ----
    results = component_model(frame, conf=0.5)

    annotated = frame.copy()

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = component_model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append(label)

            # draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # ---- OPTIONAL: COLOR CODE MODEL (only for resistors) ----
            if "resistor" in label.lower():
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    color_results = color_model(crop, conf=0.3)

                    for cr in color_results:
                        for cbox in cr.boxes:
                            c_cls = int(cbox.cls[0])
                            c_label = color_model.names[c_cls]

                            print(f"  -> Color band: {c_label}")

    print("Detections:", detections)

    # ---- SAVE RESULT ----
    ok = cv2.imwrite("detection_result.jpg", annotated)

    if ok:
        print("Saved detection_result.jpg")
    else:
        print("Failed to save image")

    cam.stop()


if __name__ == "__main__":
    main()