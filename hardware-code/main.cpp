#include <Arduino.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Bluepad32.h>

"""
Updated with respect to the following article:
https://forums.developer.nvidia.com/t/seamless-communication-between-jetson-nano-and-esp32-with-microros/308910
"""

// --- MicroROS Includes ---
#include <micro_ros_arduino.h>
#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <geometry_msgs/msg/twist.h>

#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){}}

// ---------- I2C / PCA9685 CONFIG ----------
#define I2C_SDA 4
#define I2C_SCL 5
#define PCA9685_ADDR 0x40
Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(PCA9685_ADDR);

#define SERVOMIN  80
#define SERVOMAX  600
#define SERVO0_CH 0    
#define SERVO1_CH 1    
#define SERVO2_CH 2    // Pan / Tracking

// ---------- SERVO CONFIG ----------
const int HOME0_ANGLE = 134; 
const int HOME1_ANGLE = 70;  
const int HOME2_ANGLE = 90;  

// Assuming 4 main shoulder joints for the 4 TARS segments
const int PIN_OUTER_LEFT  = 18;
const int PIN_INNER_LEFT  = 19;
const int PIN_INNER_RIGHT = 21;
const int PIN_OUTER_RIGHT = 22;

Servo outerLeft;
Servo innerLeft;
Servo innerRight;
Servo outerRight;

const int BASE_ANGLE = 90;

int servo0Angle = HOME0_ANGLE;
int servo1Angle = HOME1_ANGLE;
int servo2Angle = HOME2_ANGLE;

// ---------- ROS 2 CONTROL STATE ----------
enum RosMoveState { ROS_STOP, ROS_LEFT, ROS_RIGHT };
RosMoveState currentRosMove = ROS_STOP;
uint32_t lastRosMoveMs = 0;
const int ROS_STEP_DEG = 1;   
const int ROS_STEP_MS = 20;   

// ---------- MICROROS GLOBALS ----------
rcl_subscription_t subscriber;
geometry_msgs__msg__Twist msg;
rclc_executor_t executor;
rcl_allocator_t allocator;
rclc_support_t support;
rcl_node_t node;

// ---------- BLUEPAD32 STATE ----------
ControllerPtr myControllers[BP32_MAX_GAMEPADS];
bool presetActive = false;
// ... (Keep your existing preset configuration variables here) ...

void error_loop(){
  while(1){ delay(100); }
}

void setServoAngle(uint8_t channel, int angleDeg) {
  angleDeg = constrain(angleDeg, 0, 180);
  uint16_t pulse = map(angleDeg, 0, 180, SERVOMIN, SERVOMAX);
  pca9685.setPWM(channel, 0, pulse);
}

// ---------- MICROROS CALLBACK ----------
// Update your enum to include forward and backward
enum RosMoveState { ROS_STOP, ROS_LEFT, ROS_RIGHT, ROS_FORWARD, ROS_BACKWARD };
RosMoveState currentRosMove = ROS_STOP;

void twist_callback(const void * msgin) {
  const geometry_msgs__msg__Twist * twist_msg = (const geometry_msgs__msg__Twist *)msgin;
  
  // Prioritize turning, then forward/backward
  if (twist_msg->angular.z > 0.1) {
    currentRosMove = ROS_LEFT;
  } else if (twist_msg->angular.z < -0.1) {
    currentRosMove = ROS_RIGHT;
  } else if (twist_msg->linear.x > 0.1) {
    currentRosMove = ROS_FORWARD;
  } else if (twist_msg->linear.x < -0.1) {
    currentRosMove = ROS_BACKWARD;
  } else {
    currentRosMove = ROS_STOP;
  }
}

void updateRosMotion() {
  if (presetActive) return; 
  if (currentRosMove == ROS_STOP) return;

  uint32_t now = millis();
  if (now - lastRosMoveMs < ROS_STEP_MS) return;
  lastRosMoveMs = now;

  // Servo 2 (Channel 2) handles Left/Right Panning
  if (currentRosMove == ROS_LEFT) {
    servo2Angle -= ROS_STEP_DEG; 
  } 
  else if (currentRosMove == ROS_RIGHT) {
    servo2Angle += ROS_STEP_DEG; 
  }
  // Servos 0 & 1 (Channels 0 & 1) handle Forward/Backward motion
  else if (currentRosMove == ROS_FORWARD) {
    servo0Angle += ROS_STEP_DEG;
    servo1Angle -= ROS_STEP_DEG; // Assuming inverse mounting, adjust if needed
  }
  else if (currentRosMove == ROS_BACKWARD) {
    servo0Angle -= ROS_STEP_DEG;
    servo1Angle += ROS_STEP_DEG;
  }

  // Constrain to prevent hardware damage
  servo0Angle = constrain(servo0Angle, 0, 180);
  servo1Angle = constrain(servo1Angle, 0, 180);
  servo2Angle = constrain(servo2Angle, 0, 180);
  
  // Apply movements
  setServoAngle(SERVO0_CH, servo0Angle);
  setServoAngle(SERVO1_CH, servo1Angle);
  setServoAngle(SERVO2_CH, servo2Angle);
}

// ... (Keep your existing processGamepad(), updatePresetMotion(), and Bluepad Callbacks) ...

void setup() {
  // 1. Init Serial for MicroROS (DO NOT use Serial.begin manually here)
  set_microros_transports();

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);
  pca9685.begin();
  pca9685.setPWMFreq(50);
  
  setServoAngle(SERVO0_CH, servo0Angle);
  setServoAngle(SERVO1_CH, servo1Angle);
  setServoAngle(SERVO2_CH, servo2Angle);

  // 2. Init Bluepad32
  BP32.setup(&onConnectedController, &onDisconnectedController);
  BP32.forgetBluetoothKeys();
  BP32.enableVirtualDevice(false);

  // 3. Init MicroROS
  delay(2000);
  allocator = rcl_get_default_allocator();
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "esp32_aura_base", "", &support));

  // Subscribe to /cmd_vel
  RCCHECK(rclc_subscription_init_default(
    &subscriber, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist),
    "cmd_vel"));

  RCCHECK(rclc_executor_init(&executor, &support.context, 1, &allocator));
  RCCHECK(rclc_executor_add_subscription(&executor, &subscriber, &msg, &twist_callback, ON_NEW_DATA));
}

void loop() {
  // 1. Process MicroROS callbacks
  RCSOFTCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10)));

  // 2. Update Bluetooth
  bool dataUpdated = BP32.update();
  // if (dataUpdated) { processControllers(); }

  // 3. Move hardware
  // updatePresetMotion();
  updateRosMotion();

  if (!dataUpdated) { vTaskDelay(1); }
}