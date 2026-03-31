import time
import cv2
from camera import JetsonCamera


def main():
    cam = JetsonCamera()
    cam.start()

    print("Starting camera...")
    time.sleep(2)

    frame = cam.get_frame()
    if frame is None:
        print("ERROR: No frame received.")
    else:
        ok = cv2.imwrite("test_frame.jpg", frame)
        if ok:
            print("Saved test_frame.jpg successfully.")
        else:
            print("ERROR: Failed to save test_frame.jpg.")

    cam.stop()


if __name__ == "__main__":
    main()