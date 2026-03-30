#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define SDA_PIN 4
#define SCL_PIN 5

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

// ======================================================
// SERVO MAP
// ======================================================
// 0 = right shoulder
// 1 = left shoulder
// 2 = right leg
// 3 = left leg
//
// right shoulder: min 1600, max 2200, home 1900
// left shoulder : min  800, max 1400, home 1100
// right leg     : min  800, max 2200, home 1400
// left leg      : min  800, max 2200, home 1350
// ======================================================

const uint8_t NUM_SERVOS = 4;
const uint8_t servoChannels[NUM_SERVOS] = {0, 1, 2, 3};

const char* servoNames[NUM_SERVOS] = {
  "right shoulder",
  "left shoulder",
  "right leg",
  "left leg"
};

const uint16_t SERVO_MIN_US[NUM_SERVOS]  = {1600,  800,  800,  800};
const uint16_t SERVO_MAX_US[NUM_SERVOS]  = {2200, 1400, 2200, 2200};
const uint16_t SERVO_HOME_US[NUM_SERVOS] = {1900, 1100, 1400, 1350};

const int8_t SERVO_SIGN[NUM_SERVOS] = {
  +1,
  -1,
  +1,
  -1
};

const uint16_t SERVO_FREQ = 50;
const unsigned long UPDATE_INTERVAL_MS = 20;
const uint16_t STEP_US = 12;
const unsigned long GAIT_TOGGLE_MS = 350;

// safety timeout: if no motion command comes in for this long, stop
const unsigned long COMMAND_TIMEOUT_MS = 700;

const int SHOULDER_SWING_US = 120;
const int LEG_SWING_US = 160;
const int TURN_SHOULDER_US = 90;
const int TURN_LEG_US = 120;

// ======================================================
// GLOBAL STATE
// ======================================================

enum MotionMode {
  MODE_STOP,
  MODE_FORWARD,
  MODE_BACKWARD,
  MODE_LEFT,
  MODE_RIGHT
};

MotionMode currentMode = MODE_STOP;

uint16_t currentUs[NUM_SERVOS];
uint16_t targetUs[NUM_SERVOS];

unsigned long lastUpdateMs = 0;
unsigned long lastGaitToggleMs = 0;
unsigned long lastMotionCommandMs = 0;
bool gaitPhase = false;

// ======================================================
// HELPERS
// ======================================================

uint16_t usToTicks(uint16_t us) {
  return (uint16_t)((us * 4096.0) / 20000.0);
}

uint16_t clampServoUs(uint8_t idx, int valueUs) {
  if (valueUs < SERVO_MIN_US[idx]) valueUs = SERVO_MIN_US[idx];
  if (valueUs > SERVO_MAX_US[idx]) valueUs = SERVO_MAX_US[idx];
  return (uint16_t)valueUs;
}

void writeServoUs(uint8_t idx, uint16_t us) {
  us = clampServoUs(idx, us);
  pca.setPWM(servoChannels[idx], 0, usToTicks(us));
  currentUs[idx] = us;
}

void setTargetUs(uint8_t idx, int us) {
  targetUs[idx] = clampServoUs(idx, us);
}

void setTargetHomeAll() {
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    targetUs[i] = SERVO_HOME_US[i];
  }
}

int applySignedOffset(uint8_t idx, int offsetUs) {
  return (int)SERVO_HOME_US[idx] + (SERVO_SIGN[idx] * offsetUs);
}

void markMotionSeen() {
  lastMotionCommandMs = millis();
}

void printStatus() {
  Serial.println();
  Serial.println("===== CURRENT SERVO STATUS =====");
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    Serial.print("Servo ");
    Serial.print(i);
    Serial.print(" (");
    Serial.print(servoNames[i]);
    Serial.print(")  current=");
    Serial.print(currentUs[i]);
    Serial.print(" us  target=");
    Serial.print(targetUs[i]);
    Serial.print(" us  home=");
    Serial.print(SERVO_HOME_US[i]);
    Serial.println(" us");
  }
  Serial.print("Mode = ");
  Serial.println((int)currentMode);
  Serial.println("===============================");
  Serial.println();
}

void printHelp() {
  Serial.println();
  Serial.println("===== COMMANDS =====");
  Serial.println("Jetson / website commands:");
  Serial.println("  MOVE:forward");
  Serial.println("  MOVE:backward");
  Serial.println("  MOVE:left");
  Serial.println("  MOVE:right");
  Serial.println("  MOVE:stop");
  Serial.println();
  Serial.println("Debug commands:");
  Serial.println("  home");
  Serial.println("  status");
  Serial.println("  help");
  Serial.println("====================");
  Serial.println();
}

// ======================================================
// MODE SETTERS
// ======================================================

void applyStopPose() {
  setTargetHomeAll();
  currentMode = MODE_STOP;
  gaitPhase = false;
  Serial.println("POSE: stop/home");
}

void applyForwardPose() {
  currentMode = MODE_FORWARD;
  markMotionSeen();
  Serial.println("POSE: forward");
}

void applyBackwardPose() {
  currentMode = MODE_BACKWARD;
  markMotionSeen();
  Serial.println("POSE: backward");
}

void applyLeftPose() {
  currentMode = MODE_LEFT;
  markMotionSeen();
  Serial.println("POSE: left");
}

void applyRightPose() {
  currentMode = MODE_RIGHT;
  markMotionSeen();
  Serial.println("POSE: right");
}

// ======================================================
// SERIAL COMMAND HANDLER
// ======================================================

void handleMoveCommand(String moveCmd) {
  moveCmd.trim();
  moveCmd.toLowerCase();

  int colon = moveCmd.indexOf(':');
  if (colon != -1) {
    moveCmd = moveCmd.substring(0, colon);
    moveCmd.trim();
  }

  if (moveCmd == "forward") {
    applyForwardPose();
    Serial.println("ACK:MOVE:forward");
  }
  else if (moveCmd == "backward") {
    applyBackwardPose();
    Serial.println("ACK:MOVE:backward");
  }
  else if (moveCmd == "left") {
    applyLeftPose();
    Serial.println("ACK:MOVE:left");
  }
  else if (moveCmd == "right") {
    applyRightPose();
    Serial.println("ACK:MOVE:right");
  }
  else if (moveCmd == "stop") {
    applyStopPose();
    Serial.println("ACK:MOVE:stop");
  }
  else {
    Serial.print("ERR:UNKNOWN_MOVE:");
    Serial.println(moveCmd);
  }
}

void handleDebugCommand(String cmd) {
  cmd.trim();
  cmd.toLowerCase();

  if (cmd == "help") {
    printHelp();
  }
  else if (cmd == "status") {
    printStatus();
  }
  else if (cmd == "home") {
    applyStopPose();
    Serial.println("Moved to home pose");
  }
  else {
    Serial.println("Unknown command. Type help");
  }
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (cmd.startsWith("MOVE:")) {
    String moveCmd = cmd.substring(5);
    handleMoveCommand(moveCmd);
    return;
  }

  handleDebugCommand(cmd);
}

// ======================================================
// GAIT LOGIC
// ======================================================

void serviceGait() {
  unsigned long now = millis();

  if (currentMode == MODE_STOP) return;
  if (now - lastGaitToggleMs < GAIT_TOGGLE_MS) return;

  lastGaitToggleMs = now;
  gaitPhase = !gaitPhase;

  if (currentMode == MODE_FORWARD) {
    if (!gaitPhase) {
      setTargetUs(0, applySignedOffset(0, +SHOULDER_SWING_US));
      setTargetUs(1, applySignedOffset(1, -SHOULDER_SWING_US));
      setTargetUs(2, applySignedOffset(2, +LEG_SWING_US));
      setTargetUs(3, applySignedOffset(3, -LEG_SWING_US));
    } else {
      setTargetUs(0, applySignedOffset(0, -SHOULDER_SWING_US));
      setTargetUs(1, applySignedOffset(1, +SHOULDER_SWING_US));
      setTargetUs(2, applySignedOffset(2, -LEG_SWING_US));
      setTargetUs(3, applySignedOffset(3, +LEG_SWING_US));
    }
  }
  else if (currentMode == MODE_BACKWARD) {
    if (!gaitPhase) {
      setTargetUs(0, applySignedOffset(0, -SHOULDER_SWING_US));
      setTargetUs(1, applySignedOffset(1, +SHOULDER_SWING_US));
      setTargetUs(2, applySignedOffset(2, -LEG_SWING_US));
      setTargetUs(3, applySignedOffset(3, +LEG_SWING_US));
    } else {
      setTargetUs(0, applySignedOffset(0, +SHOULDER_SWING_US));
      setTargetUs(1, applySignedOffset(1, -SHOULDER_SWING_US));
      setTargetUs(2, applySignedOffset(2, +LEG_SWING_US));
      setTargetUs(3, applySignedOffset(3, -LEG_SWING_US));
    }
  }
  else if (currentMode == MODE_LEFT) {
    if (!gaitPhase) {
      setTargetUs(0, applySignedOffset(0, +TURN_SHOULDER_US));
      setTargetUs(1, applySignedOffset(1, -TURN_SHOULDER_US));
      setTargetUs(2, applySignedOffset(2, -TURN_LEG_US));
      setTargetUs(3, applySignedOffset(3, +TURN_LEG_US));
    } else {
      setTargetHomeAll();
    }
  }
  else if (currentMode == MODE_RIGHT) {
    if (!gaitPhase) {
      setTargetUs(0, applySignedOffset(0, -TURN_SHOULDER_US));
      setTargetUs(1, applySignedOffset(1, +TURN_SHOULDER_US));
      setTargetUs(2, applySignedOffset(2, +TURN_LEG_US));
      setTargetUs(3, applySignedOffset(3, -TURN_LEG_US));
    } else {
      setTargetHomeAll();
    }
  }
}

// ======================================================
// SMOOTH MOTION UPDATE
// ======================================================

void serviceMotion() {
  unsigned long now = millis();
  if (now - lastUpdateMs < UPDATE_INTERVAL_MS) return;
  lastUpdateMs = now;

  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    int cur = currentUs[i];
    int tgt = targetUs[i];

    if (cur == tgt) continue;

    if (abs(tgt - cur) <= STEP_US) {
      cur = tgt;
    } else if (tgt > cur) {
      cur += STEP_US;
    } else {
      cur -= STEP_US;
    }

    writeServoUs(i, (uint16_t)cur);
  }
}

void serviceCommandTimeout() {
  if (currentMode == MODE_STOP) return;

  unsigned long now = millis();
  if (now - lastMotionCommandMs > COMMAND_TIMEOUT_MS) {
    Serial.println("TIMEOUT: stopping motion");
    applyStopPose();
  }
}

// ======================================================
// SETUP / LOOP
// ======================================================

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("Starting AURA gait controller");

  Wire.begin(SDA_PIN, SCL_PIN, 100000);

  Wire.beginTransmission(0x40);
  byte err = Wire.endTransmission();
  if (err != 0) {
    Serial.print("PCA9685 not found. I2C error = ");
    Serial.println(err);
    while (1) delay(100);
  }

  Serial.println("PCA9685 found at 0x40");

  pca.begin();
  pca.setPWMFreq(SERVO_FREQ);
  delay(10);

  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    currentUs[i] = SERVO_HOME_US[i];
    targetUs[i] = SERVO_HOME_US[i];
    writeServoUs(i, SERVO_HOME_US[i]);
    delay(120);
  }

  applyStopPose();
  markMotionSeen();
  printHelp();
  printStatus();
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.length() > 0) {
      Serial.print("RAW CMD = [");
      Serial.print(cmd);
      Serial.println("]");
      handleCommand(cmd);
    }
  }

  serviceCommandTimeout();
  serviceGait();
  serviceMotion();
}