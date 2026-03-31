import cv2
import time
from camera import JetsonCamera


def main():
    cam = JetsonCamera()
    cam.start()

    print("Camera module started. Press q to quit.")

    try:
        while True:
            frame = cam.get_frame()
            if frame is not None:
                cv2.imshow("Jetson Camera Module Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            time.sleep(0.01)

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()