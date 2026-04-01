import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ultralytics import YOLO

class AuraFaceTracker(Node):
    def __init__(self):
        super().__init__('aura_face_tracker')
        # Create a ROS 2 publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Load YOLO model
        self.model = YOLO("yolov11s-face.pt")
        self.cap = cv2.VideoCapture(0)
        self.pct = 50
        
        # Run the tracking loop at roughly 30Hz
        self.timer = self.create_timer(0.033, self.tracking_loop)
        self.current_state = "STOP"

    def command_robot(self, action):
        """Creates and publishes a ROS 2 Twist message based on the action."""
        msg = Twist()
        
        if action == "TURN_LEFT":
            msg.angular.z = 1.0   # Positive Z is usually left/CCW
        elif action == "TURN_RIGHT":
            msg.angular.z = -1.0  # Negative Z is usually right/CW
        else:
            msg.angular.z = 0.0   # Stop
            
        self.cmd_vel_pub.publish(msg)
        self.get_logger().info(f'Publishing: {action}')

    def tracking_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to grab frame.")
            return

        h, w = frame.shape[:2]
        leftBound = int(w * ((100 - self.pct) / 2) / 100)
        rightBound = int(w * ((100 + self.pct) / 2) / 100)

        results = self.model(frame, conf=0.4, verbose=False)
        boxes = getattr(results[0], "boxes", None)
        
        next_action = "STOP"
        out = frame.copy()
        cv2.rectangle(out, (leftBound, 0), (rightBound, h), (255, 0, 0), 2)

        if boxes is not None and len(boxes) > 0:
            largest_area = 0
            target_cx = -1
            
            for (x1, y1, x2, y2) in boxes.xyxy:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cx = (x1 + x2) // 2
                area = (x2 - x1) * (y2 - y1)
                
                if area > largest_area:
                    largest_area = area
                    target_cx = cx

            if target_cx != -1:
                if target_cx < leftBound:
                    next_action = "TURN_LEFT"
                elif target_cx > rightBound:
                    next_action = "TURN_RIGHT"
                else:
                    next_action = "STOP"

        # Publish only if the desired action has changed
        if next_action != self.current_state:
            self.command_robot(next_action)
            self.current_state = next_action

        cv2.imshow("AURA ROS 2 Tracking", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = AuraFaceTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()