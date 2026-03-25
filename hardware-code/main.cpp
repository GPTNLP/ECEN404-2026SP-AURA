#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Bluepad32.h>

// ---------- I2C / PCA9685 CONFIG ----------

#define I2C_SDA 4
#define I2C_SCL 5

#define PCA9685_ADDR 0x40
Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(PCA9685_ADDR);

#define SERVOMIN  80
#define SERVOMAX  600

// PCA9685 channels for servos
#define SERVO0_CH 0    // "servo 1" 
#define SERVO1_CH 1    // "servo 2"
#define SERVO2_CH 2    // "servo 3" (Pan / Tracking)

// ---------- SERVO / PRESET CONFIG ----------

// Homes
const int HOME0_ANGLE = 134;  // servo 1 home
const int HOME1_ANGLE = 70;   // servo 2 home
const int HOME2_ANGLE = 90;   // servo 3 home (adjust as needed)

int servo0Angle = HOME0_ANGLE;
int servo1Angle = HOME1_ANGLE;
int servo2Angle = HOME2_ANGLE;

const int JOYSTICK_DEADZONE = 32;

// Joystick ranges around home
const int MAX_DELTA0 = 60;
const int MAX_DELTA1 = 60;
const int MAX_DELTA2 = 60;

// Preset offsets
const int PRESET_OFFSET01_DEG   = 30;  // ±30° for servos 1 & 2 during preset
const int PRESET_OFFSET2_DEG    = 30;  // +30° CW for servo 3
const int PRESET_STEP_DEG       = 2;
const uint16_t PRESET_STEP_MS   = 15;

// Preset state machine
enum PresetPhase {
  PRESET_IDLE = 0,
  PRESET_MOVING_OUT_2,
  PRESET_MOVING_OUT_01,
  PRESET_MOVING_BACK_2,
  PRESET_MOVING_BACK_01
};

bool         presetActive      = false;
PresetPhase  presetPhase       = PRESET_IDLE;
int          presetTargetOut0  = HOME0_ANGLE;
int          presetTargetOut1  = HOME1_ANGLE;
int          presetTargetOut2  = HOME2_ANGLE;
uint32_t     lastPresetStepMs  = 0;

// ---------- SERIAL CONTROL STATE (NEW FOR JETSON) ----------
enum SerialMoveState { SERIAL_STOP, SERIAL_LEFT, SERIAL_RIGHT };
SerialMoveState currentSerialMove = SERIAL_STOP;
uint32_t lastSerialMoveMs = 0;
const int SERIAL_STEP_DEG = 1;   // How many degrees to move per step
const int SERIAL_STEP_MS = 20;   // Speed of the rotation (lower = faster)

// ---------- BLUEPAD32 STATE ----------

ControllerPtr myControllers[BP32_MAX_GAMEPADS];

// ---------- HELPERS ----------

void setServoAngle(uint8_t channel, int angleDeg) {
  angleDeg = constrain(angleDeg, 0, 180);
  uint16_t pulse = map(angleDeg, 0, 180, SERVOMIN, SERVOMAX);
  pca9685.setPWM(channel, 0, pulse);
}

int applyDeadzoneInt(int v, int dz) {
  if (abs(v) < dz) return 0;
  return v;
}

// ---------- SERIAL PROCESSING (NEW FOR JETSON) ----------

void processSerialCommands() {
  // Check if the Jetson Orin Nano has sent any data over USB
  while (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim(); // Remove any extra \r or spaces
    
    if (cmd == "MOVE:TURN_LEFT") {
      currentSerialMove = SERIAL_LEFT;
    } 
    else if (cmd == "MOVE:TURN_RIGHT") {
      currentSerialMove = SERIAL_RIGHT;
    } 
    else if (cmd == "MOVE:STOP") {
      currentSerialMove = SERIAL_STOP;
    }
  }
}

void updateSerialMotion() {
  // If a gamepad preset is running, let it finish first
  if (presetActive) return; 
  // If we are centered/stopped, do nothing
  if (currentSerialMove == SERIAL_STOP) return;

  uint32_t now = millis();
  if (now - lastSerialMoveMs < SERIAL_STEP_MS) return;
  lastSerialMoveMs = now;

  // Adjust Servo 3 (Pan) based on the Jetson's command
  if (currentSerialMove == SERIAL_LEFT) {
    servo2Angle -= SERIAL_STEP_DEG; // Decrease angle
  } 
  else if (currentSerialMove == SERIAL_RIGHT) {
    servo2Angle += SERIAL_STEP_DEG; // Increase angle
  }

  // Ensure we don't break the servo
  servo2Angle = constrain(servo2Angle, 0, 180);
  
  // Apply the movement
  setServoAngle(SERVO2_CH, servo2Angle);
}

// ---------- PRESET MOTION LOGIC ----------

void startPresetForDpad(int dpadValue) {
  if (presetActive) return;

  int offset0 = 0;
  int offset1 = 0;

  if (dpadValue == 1) { // Up
    offset0 = +PRESET_OFFSET01_DEG;
    offset1 = -PRESET_OFFSET01_DEG;
  } else if (dpadValue == 2) { // Right
    offset0 = +PRESET_OFFSET01_DEG;
    offset1 = +PRESET_OFFSET01_DEG;
  } else if (dpadValue == 8) { // Left
    offset0 = -PRESET_OFFSET01_DEG;
    offset1 = -PRESET_OFFSET01_DEG;
  } else {
    return;
  }

  servo0Angle = HOME0_ANGLE;
  servo1Angle = HOME1_ANGLE;
  servo2Angle = HOME2_ANGLE;
  setServoAngle(SERVO0_CH, servo0Angle);
  setServoAngle(SERVO1_CH, servo1Angle);
  setServoAngle(SERVO2_CH, servo2Angle);

  presetTargetOut0 = constrain(HOME0_ANGLE + offset0, 0, 180);
  presetTargetOut1 = constrain(HOME1_ANGLE + offset1, 0, 180);
  presetTargetOut2 = constrain(HOME2_ANGLE + PRESET_OFFSET2_DEG, 0, 180);

  presetPhase      = PRESET_MOVING_OUT_2;
  presetActive     = true;
  lastPresetStepMs = millis();
}

void updatePresetMotion() {
  if (!presetActive) return;

  uint32_t now = millis();
  if (now - lastPresetStepMs < PRESET_STEP_MS) return;
  lastPresetStepMs = now;

  switch (presetPhase) {

    case PRESET_MOVING_OUT_2:
      if (servo2Angle < presetTargetOut2)      servo2Angle += PRESET_STEP_DEG;
      else if (servo2Angle > presetTargetOut2) servo2Angle -= PRESET_STEP_DEG;

      servo2Angle = constrain(servo2Angle, 0, 180);
      setServoAngle(SERVO2_CH, servo2Angle);

      if (abs(servo2Angle - presetTargetOut2) <= PRESET_STEP_DEG) {
        presetPhase = PRESET_MOVING_OUT_01;
      }
      break;

    case PRESET_MOVING_OUT_01:
      if (servo0Angle < presetTargetOut0)      servo0Angle += PRESET_STEP_DEG;
      else if (servo0Angle > presetTargetOut0) servo0Angle -= PRESET_STEP_DEG;

      if (servo1Angle < presetTargetOut1)      servo1Angle += PRESET_STEP_DEG;
      else if (servo1Angle > presetTargetOut1) servo1Angle -= PRESET_STEP_DEG;

      servo0Angle = constrain(servo0Angle, 0, 180);
      servo1Angle = constrain(servo1Angle, 0, 180);

      setServoAngle(SERVO0_CH, servo0Angle);
      setServoAngle(SERVO1_CH, servo1Angle);

      if (abs(servo0Angle - presetTargetOut0) <= PRESET_STEP_DEG &&
          abs(servo1Angle - presetTargetOut1) <= PRESET_STEP_DEG) {
        presetPhase = PRESET_MOVING_BACK_2;
      }
      break;

    case PRESET_MOVING_BACK_2:
      if (servo2Angle < HOME2_ANGLE)      servo2Angle += PRESET_STEP_DEG;
      else if (servo2Angle > HOME2_ANGLE) servo2Angle -= PRESET_STEP_DEG;

      servo2Angle = constrain(servo2Angle, 0, 180);
      setServoAngle(SERVO2_CH, servo2Angle);

      if (abs(servo2Angle - HOME2_ANGLE) <= PRESET_STEP_DEG) {
        presetPhase = PRESET_MOVING_BACK_01;
      }
      break;

    case PRESET_MOVING_BACK_01:
      if (servo0Angle < HOME0_ANGLE)      servo0Angle += PRESET_STEP_DEG;
      else if (servo0Angle > HOME0_ANGLE) servo0Angle -= PRESET_STEP_DEG;

      if (servo1Angle < HOME1_ANGLE)      servo1Angle += PRESET_STEP_DEG;
      else if (servo1Angle > HOME1_ANGLE) servo1Angle -= PRESET_STEP_DEG;

      servo0Angle = constrain(servo0Angle, 0, 180);
      servo1Angle = constrain(servo1Angle, 0, 180);

      setServoAngle(SERVO0_CH, servo0Angle);
      setServoAngle(SERVO1_CH, servo1Angle);

      if (abs(servo0Angle - HOME0_ANGLE) <= PRESET_STEP_DEG &&
          abs(servo1Angle - HOME1_ANGLE) <= PRESET_STEP_DEG) {
        presetActive = false;
        presetPhase  = PRESET_IDLE;
      }
      break;

    case PRESET_IDLE:
    default:
      break;
  }
}

// ---------- BLUEPAD32 CALLBACKS ----------

void onConnectedController(ControllerPtr ctl) {
  for (int i = 0; i < BP32_MAX_GAMEPADS; i++) {
    if (myControllers[i] == nullptr) {
      ControllerProperties properties = ctl->getProperties();
      myControllers[i] = ctl;
      return;
    }
  }
}

void onDisconnectedController(ControllerPtr ctl) {
  for (int i = 0; i < BP32_MAX_GAMEPADS; i++) {
    if (myControllers[i] == ctl) {
      myControllers[i] = nullptr;
      return;
    }
  }
}

// ---------- GAMEPAD PROCESSING ----------

void processGamepad(ControllerPtr ctl) {
  int dpad = ctl->dpad();

  // Trigger preset on D-pad
  if (!presetActive && dpad != 0) {
    startPresetForDpad(dpad);
  }

  // During presets, ignore joystick control
  if (presetActive) {
    return;
  }

  int ly = ctl->axisY();
  int ry = ctl->axisRY();
  int rx = ctl->axisRX();

  ly = applyDeadzoneInt(ly, JOYSTICK_DEADZONE);
  ry = applyDeadzoneInt(ry, JOYSTICK_DEADZONE);
  rx = applyDeadzoneInt(rx, JOYSTICK_DEADZONE);

  if (ly != 0) {
    int delta0 = map(ly, -512, 512, -MAX_DELTA0, MAX_DELTA0);
    servo0Angle = HOME0_ANGLE + delta0;
  }

  if (ry != 0) {
    int delta1 = map(ry, -512, 512, -MAX_DELTA1, MAX_DELTA1);
    servo1Angle = HOME1_ANGLE + delta1;
  }

  if (rx != 0) {
    int delta2 = map(rx, -512, 512, -MAX_DELTA2, MAX_DELTA2);
    servo2Angle = HOME2_ANGLE + delta2;
  }

  servo0Angle = constrain(servo0Angle, 0, 180);
  servo1Angle = constrain(servo1Angle, 0, 180);
  servo2Angle = constrain(servo2Angle, 0, 180);

  setServoAngle(SERVO0_CH, servo0Angle);
  setServoAngle(SERVO1_CH, servo1Angle);
  setServoAngle(SERVO2_CH, servo2Angle);
}

void processControllers() {
  for (auto ctl : myControllers) {
    if (!ctl) continue;
    if (!ctl->isConnected()) continue;
    if (!ctl->hasData()) continue;

    if (ctl->isGamepad()) {
      processGamepad(ctl);
    }
  }
}

// ---------- SETUP & LOOP ----------

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  if (!pca9685.begin()) {
    while (1) delay(1000);
  }
  pca9685.setPWMFreq(50);
  delay(10);

  // Initialize to home positions
  setServoAngle(SERVO0_CH, servo0Angle);
  setServoAngle(SERVO1_CH, servo1Angle);
  setServoAngle(SERVO2_CH, servo2Angle);

  BP32.setup(&onConnectedController, &onDisconnectedController);
  BP32.forgetBluetoothKeys();
  BP32.enableVirtualDevice(false);

  for (int i = 0; i < BP32_MAX_GAMEPADS; i++) {
    myControllers[i] = nullptr;
  }
}

void loop() {
  // 1. Check for Bluetooth Gamepad updates
  bool dataUpdated = BP32.update();
  if (dataUpdated) {
    processControllers();
  }

  // 2. Check for Jetson USB Serial updates
  processSerialCommands();

  // 3. Execute any active animations/movements
  updatePresetMotion();
  updateSerialMotion();

  if (!dataUpdated) {
    vTaskDelay(1);
  }
}