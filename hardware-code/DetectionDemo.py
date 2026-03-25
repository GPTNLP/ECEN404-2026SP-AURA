import cv2
import time
# import serial # Uncomment if integrating directly with ESP serial here
from ultralytics import YOLO

# --- Robot Control Setup ---
# esp_serial = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

def command_robot(action):
    """
    Sends movement commands to the robot. 
    Replace the print statements with your actual serial commands to the ESP.
    """
    print(f"[ACTION] Executing: {action}")
    # if esp_serial:
    #     esp_serial.write(f"MOVE:{action}\n".encode('utf-8'))

# --- Camera & Model Setup ---
model = YOLO("yolov11s-face.pt")
cap = cv2.VideoCapture(0)

# Value of how much of the screen should be used for the "centered" deadzone
pct = 50

# Track the current movement state to prevent serial port spam
current_state = "STOP"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
        
    h, w = frame.shape[:2]

    # Define the "in bounds" central column
    leftBound = int(w * ((100 - pct) / 2) / 100)
    rightBound = int(w * ((100 + pct) / 2) / 100)

    # Run YOLO detection
    results = model(frame, conf=0.4, verbose=False)
    res = results[0]
    out = frame.copy()

    # Draw central column for visualization
    cv2.rectangle(out, (leftBound, 0), (rightBound, h), (255, 0, 0), 2)
    cv2.putText(out, f"center column ({pct}%)", (leftBound + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    boxes = getattr(res, "boxes", None)

    # Default action is to do nothing / stop moving
    next_action = "STOP" 

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy
        
        # Variables to track the most prominent face
        largest_area = 0
        target_cx = -1
        
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            
            # Calculate center X and bounding box area
            cx = (x1 + x2) // 2
            area = (x2 - x1) * (y2 - y1)
            
            # Identify if it is inside the center bounds
            inside = leftBound <= cx <= rightBound
            color = (0, 255, 0) if inside else (0, 0, 255)
            
            # Draw bounding boxes for all faces
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.circle(out, (cx, (y1 + y2) // 2), 3, color, -1)
            
            # Update target if this is the largest face detected so far
            if area > largest_area:
                largest_area = area
                target_cx = cx

        # Decide movement based on the target face's position
        if target_cx != -1:
            if target_cx < leftBound:
                next_action = "TURN_LEFT"
            elif target_cx > rightBound:
                next_action = "TURN_RIGHT"
            else:
                # Face is within the bounds -> stay still
                next_action = "STOP"

    # Only send a new command to the robot if the desired action has changed
    if next_action != current_state:
        command_robot(next_action)
        current_state = next_action

    # Show the video feed
    cv2.imshow("AURA Face Tracking", out)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()