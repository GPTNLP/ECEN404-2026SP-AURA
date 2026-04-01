import threading
from flask import Flask, request, jsonify
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from flask_cors import CORS # Needed if your React app is hosted elsewhere

app = Flask(__name__)
CORS(app) # Allow the Azure website to ping this local Jetson API

class WebTeleopNode(Node):
    def __init__(self):
        super().__init__('web_teleop_node')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

    def send_velocity(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published Twist: Linear={linear_x}, Angular={angular_z}')

# Initialize ROS 2 in the background
rclpy.init(args=None)
ros_node = WebTeleopNode()

# Run ROS 2 spin in a separate thread so it doesn't block Flask
def spin_ros():
    rclpy.spin(ros_node)

threading.Thread(target=spin_ros, daemon=True).start()

@app.post("/move")
def move():
    data = request.json
    cmd = data.get("cmd", "").lower()
    print("ROBOT COMMAND RECEIVED:", cmd)

    # Translate Web Commands to ROS 2 Twist logic
    # Linear.x = Forward/Backward | Angular.z = Left/Right (Pan)
    if cmd == "forward":
        ros_node.send_velocity(1.0, 0.0)
    elif cmd == "backward":
        ros_node.send_velocity(-1.0, 0.0)
    elif cmd == "left":
        ros_node.send_velocity(0.0, 1.0)
    elif cmd == "right":
        ros_node.send_velocity(0.0, -1.0)
    elif cmd == "stop":
        ros_node.send_velocity(0.0, 0.0)
    else:
        return jsonify({"status": "error", "message": "Unknown command"}), 400

    return jsonify({"status": "ok", "received": cmd})

if __name__ == "__main__":
    try:
        # Run on 0.0.0.0 so the website can reach it over the TAMU network
        app.run(host="0.0.0.0", port=5001)
    except KeyboardInterrupt:
        pass
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()