#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define SDA_PIN 4
#define SCL_PIN 5
#define PCA_ADDR 0x40

Adafruit_PWMServoDriver pca(PCA_ADDR);

const uint8_t NUM_SERVOS = 4;
const uint8_t CH[NUM_SERVOS] = {0, 1, 2, 3};

// PWM0 = Right shoulder
// PWM1 = Left shoulder
// PWM2 = Right arm
// PWM3 = Left arm
const uint16_t SERVO_MIN[NUM_SERVOS]  = {1650,  650, 1220, 1125};
const uint16_t SERVO_MAX[NUM_SERVOS]  = {2300, 1650, 1625, 1530};
const uint16_t SERVO_HOME[NUM_SERVOS] = {1850, 1100, 1400, 1350};

// Slow/safe motion settings for testing
const uint16_t SERVO_FREQ = 50;
const uint16_t STEP_US = 16;
const unsigned long MOTION_UPDATE_MS = 35;
const unsigned long PHASE_HOLD_MS = 350;

// Amount the arms move in the opposite direction before returning home
const uint16_t ARM_RETURN_OVERSHOOT_US = 150;

// Forward-only left shoulder trim.
// Positive value moves PWM1 farther downward.
// If this makes the robot turn more left, change it to a negative value.
const int16_t FORWARD_LEFT_SHOULDER_TRIM_US = 20;

// Tilt pose tuning
const uint16_t TILT_SHOULDER_MARGIN_US = 40;

// Near-maximum downward shoulder positions for tabletop-facing tilt
const uint16_t RIGHT_SHOULDER_TILT_US = SERVO_MIN[0] + TILT_SHOULDER_MARGIN_US;
const uint16_t LEFT_SHOULDER_TILT_US  = SERVO_MAX[1] - TILT_SHOULDER_MARGIN_US;

uint16_t currentUs[NUM_SERVOS];
uint16_t targetUs[NUM_SERVOS];

unsigned long lastMotionUpdate = 0;
String serialBuffer = "";

// -------------------- Calibrated sequence targets --------------------
// Shoulders: "max lift"
const uint16_t RIGHT_SHOULDER_UP_US = SERVO_MAX[0]; // 2300
const uint16_t LEFT_SHOULDER_UP_US  = SERVO_MIN[1]; // 650

// Arms: forward / backward
const uint16_t RIGHT_ARM_FORWARD_US  = SERVO_MAX[2]; // 1625
const uint16_t RIGHT_ARM_BACKWARD_US = SERVO_MIN[2]; // 1220
const uint16_t LEFT_ARM_FORWARD_US   = SERVO_MIN[3]; // 1125
const uint16_t LEFT_ARM_BACKWARD_US  = SERVO_MAX[3]; // 1530

// Final standby / ready pose after each movement
const uint16_t RIGHT_SHOULDER_READY_US = (SERVO_HOME[0] + RIGHT_SHOULDER_UP_US) / 2; // 2075
const uint16_t LEFT_SHOULDER_READY_US  = (SERVO_HOME[1] + LEFT_SHOULDER_UP_US) / 2;  // 875

// -------------------- Sequence engine --------------------
enum SequenceType {
  SEQ_NONE,
  SEQ_FORWARD,
  SEQ_BACKWARD,
  SEQ_TURN_LEFT,
  SEQ_TURN_RIGHT
};

enum SequencePhase {
  PHASE_IDLE,
  PHASE_SHOULDERS_UP,
  PHASE_ARMS_MOVE,
  PHASE_SHOULDERS_HOME,
  PHASE_ARMS_OPPOSITE,
  PHASE_ARMS_HOME,
  PHASE_READY_POSE,
  PHASE_FINAL_HOME,
  PHASE_DONE
};

SequenceType activeSequence = SEQ_NONE;
SequencePhase activePhase = PHASE_IDLE;
unsigned long phaseStartMs = 0;
uint8_t sequenceCyclesRemaining = 0;

bool tiltModeActive = false;

// -------------------- Utility --------------------
uint16_t usToTicks(uint16_t us) {
  return (uint16_t)((us * 4096.0) / 20000.0);
}

uint16_t clampUs(uint8_t i, int us) {
  if (us < SERVO_MIN[i]) us = SERVO_MIN[i];
  if (us > SERVO_MAX[i]) us = SERVO_MAX[i];
  return (uint16_t)us;
}

void writeServo(uint8_t i, uint16_t us) {
  us = clampUs(i, us);
  pca.setPWM(CH[i], 0, usToTicks(us));
  currentUs[i] = us;
}

void setTarget(uint8_t i, int us) {
  targetUs[i] = clampUs(i, us);
}

void setHomeTargets() {
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    targetUs[i] = SERVO_HOME[i];
  }
}

bool allTargetsReached() {
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    if (currentUs[i] != targetUs[i]) return false;
  }
  return true;
}

void cancelSequenceAndHome() {
  activeSequence = SEQ_NONE;
  activePhase = PHASE_IDLE;
  sequenceCyclesRemaining = 0;
  tiltModeActive = false;
  setHomeTargets();
  Serial.println("ACK:STOP:HOME");
}

void printStatus() {
  Serial.println("---- SERVO STATUS ----");

  Serial.print("PWM0 Right shoulder | current=");
  Serial.print(currentUs[0]);
  Serial.print(" target=");
  Serial.println(targetUs[0]);

  Serial.print("PWM1 Left shoulder  | current=");
  Serial.print(currentUs[1]);
  Serial.print(" target=");
  Serial.println(targetUs[1]);

  Serial.print("PWM2 Right arm      | current=");
  Serial.print(currentUs[2]);
  Serial.print(" target=");
  Serial.println(targetUs[2]);

  Serial.print("PWM3 Left arm       | current=");
  Serial.print(currentUs[3]);
  Serial.print(" target=");
  Serial.println(targetUs[3]);

  Serial.print("Sequence = ");
  Serial.println((int)activeSequence);
  Serial.print("Phase = ");
  Serial.println((int)activePhase);
  Serial.print("Cycles remaining = ");
  Serial.println(sequenceCyclesRemaining);
  Serial.print("Tilt mode = ");
  Serial.println(tiltModeActive ? "ON" : "OFF");
  Serial.println("----------------------");
}

void printHelp() {
  Serial.println();
  Serial.println("Predefined movement commands:");
  Serial.println("  forward       -> one slow forward step");
  Serial.println("  backward      -> one slow backward step");
  Serial.println("  left          -> one slow left turn step");
  Serial.println("  right         -> one slow right turn step");
  Serial.println("  left90        -> 4 left-step sequences");
  Serial.println("  right90       -> 4 right-step sequences");
  Serial.println("  left180       -> 8 left-step sequences");
  Serial.println("  right180      -> 8 right-step sequences");
  Serial.println("  left360       -> 16 left-step sequences");
  Serial.println("  right360      -> 16 right-step sequences");
  Serial.println("  tilt          -> tilt forward and hold until home/stop");
  Serial.println("  stop          -> cancel active sequence and return true home");
  Serial.println("  home          -> same as stop");
  Serial.println();

  Serial.println("Shoulder calibration commands:");
  Serial.println("  both_up:25");
  Serial.println("  both_down:25");
  Serial.println("  r_up:25");
  Serial.println("  r_down:25");
  Serial.println("  l_up:25");
  Serial.println("  l_down:25");
  Serial.println("  set0:1900");
  Serial.println("  set1:1100");
  Serial.println();

  Serial.println("Arm calibration commands:");
  Serial.println("  both_arm_fwd:25");
  Serial.println("  both_arm_back:25");
  Serial.println("  ra_fwd:25");
  Serial.println("  ra_back:25");
  Serial.println("  la_fwd:25");
  Serial.println("  la_back:25");
  Serial.println("  set2:1400");
  Serial.println("  set3:1350");
  Serial.println();

  Serial.println("Other:");
  Serial.println("  status");
  Serial.println("  help");
  Serial.println();
}

int parseValueAfterColon(String cmd) {
  int colon = cmd.indexOf(':');
  if (colon < 0) return -1;

  String valueText = cmd.substring(colon + 1);
  valueText.trim();
  if (valueText.length() == 0) return -1;

  return valueText.toInt();
}

void moveServoRaw(uint8_t idx, int signedDeltaUs) {
  setTarget(idx, targetUs[idx] + signedDeltaUs);

  Serial.print("ACK:SERVO");
  Serial.print(idx);
  Serial.print(":TARGET=");
  Serial.println(targetUs[idx]);
}

// -------------------- Manual calibration helpers --------------------
// Shoulders
void moveRightShoulderUp(int deltaUs)    { moveServoRaw(0, +deltaUs); }
void moveRightShoulderDown(int deltaUs)  { moveServoRaw(0, -deltaUs); }
void moveLeftShoulderUp(int deltaUs)     { moveServoRaw(1, -deltaUs); }
void moveLeftShoulderDown(int deltaUs)   { moveServoRaw(1, +deltaUs); }

// Arms
void moveRightArmForward(int deltaUs)    { moveServoRaw(2, +deltaUs); }
void moveRightArmBackward(int deltaUs)   { moveServoRaw(2, -deltaUs); }
void moveLeftArmForward(int deltaUs)     { moveServoRaw(3, -deltaUs); }
void moveLeftArmBackward(int deltaUs)    { moveServoRaw(3, +deltaUs); }

// -------------------- Sequence targets --------------------
void applyShouldersUpTargets() {
  setTarget(0, RIGHT_SHOULDER_UP_US);
  setTarget(1, LEFT_SHOULDER_UP_US);
}

void applyShouldersHomeTargets() {
  setTarget(0, SERVO_HOME[0]);
  setTarget(1, SERVO_HOME[1]);
}

void applyForwardShouldersHomeTargets() {
  setTarget(0, SERVO_HOME[0]);
  setTarget(1, SERVO_HOME[1] + FORWARD_LEFT_SHOULDER_TRIM_US);
}

void applyArmsHomeTargets() {
  setTarget(2, SERVO_HOME[2]);
  setTarget(3, SERVO_HOME[3]);
}

void applyReadyPoseTargets() {
  // Ready pose only affects shoulders.
  setTarget(0, RIGHT_SHOULDER_READY_US);
  setTarget(1, LEFT_SHOULDER_READY_US);
}

void applyShouldersDownTargets() {
  // Max downward depth
  setTarget(0, SERVO_MIN[0]);  // right shoulder down
  setTarget(1, SERVO_MAX[1]);  // left shoulder down
}

void applyForwardArmTargets() {
  setTarget(2, RIGHT_ARM_FORWARD_US);
  setTarget(3, LEFT_ARM_FORWARD_US);
}

void applyBackwardArmTargets() {
  setTarget(2, RIGHT_ARM_BACKWARD_US);
  setTarget(3, LEFT_ARM_BACKWARD_US);
}

void applyLeftTurnArmTargets() {
  // Left turn = right arm forward, left arm backward
  setTarget(2, RIGHT_ARM_FORWARD_US);
  setTarget(3, LEFT_ARM_BACKWARD_US);
}

void applyRightTurnArmTargets() {
  // Right turn = right arm backward, left arm forward
  setTarget(2, RIGHT_ARM_BACKWARD_US);
  setTarget(3, LEFT_ARM_FORWARD_US);
}

void applyOppositeArmTargets() {
  if (activeSequence == SEQ_FORWARD) {
    setTarget(2, RIGHT_ARM_FORWARD_US - ARM_RETURN_OVERSHOOT_US);
    setTarget(3, LEFT_ARM_FORWARD_US + ARM_RETURN_OVERSHOOT_US);
    Serial.println("ACK:PHASE:ARMS_OPPOSITE_AFTER_FORWARD");
  }
  else if (activeSequence == SEQ_BACKWARD) {
    setTarget(2, RIGHT_ARM_FORWARD_US - ARM_RETURN_OVERSHOOT_US);
    setTarget(3, LEFT_ARM_FORWARD_US + ARM_RETURN_OVERSHOOT_US);
    Serial.println("ACK:PHASE:ARMS_OPPOSITE_AFTER_BACKWARD_FORWARD_STYLE");
  }
  else if (activeSequence == SEQ_TURN_LEFT) {
    setTarget(2, RIGHT_ARM_FORWARD_US - ARM_RETURN_OVERSHOOT_US);
    setTarget(3, LEFT_ARM_BACKWARD_US - ARM_RETURN_OVERSHOOT_US);
    Serial.println("ACK:PHASE:ARMS_OPPOSITE_AFTER_LEFT");
  }
  else if (activeSequence == SEQ_TURN_RIGHT) {
    setTarget(2, RIGHT_ARM_BACKWARD_US + ARM_RETURN_OVERSHOOT_US);
    setTarget(3, LEFT_ARM_FORWARD_US + ARM_RETURN_OVERSHOOT_US);
    Serial.println("ACK:PHASE:ARMS_OPPOSITE_AFTER_RIGHT");
  }
}

void applyTiltPoseTargets() {
  // Lower shoulders deeply and move arms forward to face the tabletop
  setTarget(0, RIGHT_SHOULDER_TILT_US);
  setTarget(1, LEFT_SHOULDER_TILT_US);
  setTarget(2, RIGHT_ARM_FORWARD_US);
  setTarget(3, LEFT_ARM_FORWARD_US);
}

void beginSequenceCycle() {
  activePhase = PHASE_SHOULDERS_UP;
  phaseStartMs = millis();

  // Forward/backward start by pushing shoulders down deeply.
  // Turns keep the original lifted-shoulder behavior.
  if (activeSequence == SEQ_BACKWARD) {
    applyShouldersDownTargets();
  } else {
    applyShouldersUpTargets();
  }

  applyArmsHomeTargets();
}

void startSequence(SequenceType seq, uint8_t cycles = 1) {
  if (cycles == 0) cycles = 1;

  tiltModeActive = false;
  activeSequence = seq;
  sequenceCyclesRemaining = cycles;

  beginSequenceCycle();

  if (seq == SEQ_FORWARD) {
    Serial.print("ACK:SEQ:FORWARD:START:CYCLES=");
    Serial.println(sequenceCyclesRemaining);
  } else if (seq == SEQ_BACKWARD) {
    Serial.print("ACK:SEQ:BACKWARD:START:CYCLES=");
    Serial.println(sequenceCyclesRemaining);
  } else if (seq == SEQ_TURN_LEFT) {
    Serial.print("ACK:SEQ:LEFT:START:CYCLES=");
    Serial.println(sequenceCyclesRemaining);
  } else if (seq == SEQ_TURN_RIGHT) {
    Serial.print("ACK:SEQ:RIGHT:START:CYCLES=");
    Serial.println(sequenceCyclesRemaining);
  }
}

void advanceSequencePhase() {
  activePhase = (SequencePhase)((int)activePhase + 1);
  phaseStartMs = millis();

  if (activePhase == PHASE_ARMS_MOVE) {
    if (activeSequence == SEQ_FORWARD) {
      applyForwardArmTargets();
      Serial.println("ACK:PHASE:ARMS_FORWARD");
    }
    else if (activeSequence == SEQ_BACKWARD) {
      applyForwardArmTargets();
      Serial.println("ACK:PHASE:ARMS_BACKWARD_USING_FORWARD_TARGETS");
    }
    else if (activeSequence == SEQ_TURN_LEFT) {
      applyLeftTurnArmTargets();
      Serial.println("ACK:PHASE:ARMS_LEFT_TURN");
    }
    else if (activeSequence == SEQ_TURN_RIGHT) {
      applyRightTurnArmTargets();
      Serial.println("ACK:PHASE:ARMS_RIGHT_TURN");
    }
  }
  else if (activePhase == PHASE_SHOULDERS_HOME) {
    if (activeSequence == SEQ_FORWARD || activeSequence == SEQ_BACKWARD) {
      applyForwardShouldersHomeTargets();
      Serial.println("ACK:PHASE:SHOULDERS_HOME_FORWARD_TRIM");
    } else {
      applyShouldersHomeTargets();
      Serial.println("ACK:PHASE:SHOULDERS_HOME");
    }
  }
  else if (activePhase == PHASE_ARMS_OPPOSITE) {
    applyOppositeArmTargets();
  }
  else if (activePhase == PHASE_ARMS_HOME) {
    applyArmsHomeTargets();
    Serial.println("ACK:PHASE:ARMS_HOME");
  }
  else if (activePhase == PHASE_READY_POSE) {
    applyReadyPoseTargets();
    Serial.println("ACK:PHASE:READY_POSE");
  }
  else if (activePhase == PHASE_FINAL_HOME) {
    setHomeTargets();
    Serial.println("ACK:PHASE:FINAL_HOME");
  }
  else if (activePhase == PHASE_DONE) {
    if (sequenceCyclesRemaining > 1) {
      sequenceCyclesRemaining--;
      beginSequenceCycle();
      Serial.print("ACK:SEQ:REPEAT_REMAINING=");
      Serial.println(sequenceCyclesRemaining);
    } else {
      Serial.println("ACK:SEQ:DONE");
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
    }
  }
}

void serviceSequence() {
  if (activeSequence == SEQ_NONE || activePhase == PHASE_IDLE) return;
  if (!allTargetsReached()) return;
  if (millis() - phaseStartMs < PHASE_HOLD_MS) return;

  advanceSequencePhase();
}

// -------------------- Serial command parsing --------------------
String normalizeCommand(String cmd) {
  cmd.trim();
  cmd.toLowerCase();

  while (cmd.startsWith("move:")) {
    cmd = cmd.substring(5);
    cmd.trim();
  }

  while (cmd.startsWith(":")) {
    cmd = cmd.substring(1);
    cmd.trim();
  }

  return cmd;
}

void handleCommand(String raw) {
  String cmd = normalizeCommand(raw);

  Serial.print("CMD: ");
  Serial.println(cmd);

  if (cmd == "forward") {
    startSequence(SEQ_FORWARD, 1);
  }
  else if (cmd == "backward") {
    startSequence(SEQ_BACKWARD, 1);
  }
  else if (cmd == "left") {
    startSequence(SEQ_TURN_LEFT, 1);
  }
  else if (cmd == "right") {
    startSequence(SEQ_TURN_RIGHT, 1);
  }
  else if (cmd == "left90") {
    startSequence(SEQ_TURN_LEFT, 4);
  }
  else if (cmd == "right90") {
    startSequence(SEQ_TURN_RIGHT, 2;
  }
  else if (cmd == "left180") {
    startSequence(SEQ_TURN_LEFT, 8);
  }
  else if (cmd == "right180") {
    startSequence(SEQ_TURN_RIGHT, 4);
  }
  else if (cmd == "left360") {
  startSequence(SEQ_TURN_LEFT, 16);
  }
  else if (cmd == "right360") {
  startSequence(SEQ_TURN_RIGHT, 8);
  }
  else if (cmd == "tilt") {
    activeSequence = SEQ_NONE;
    activePhase = PHASE_IDLE;
    sequenceCyclesRemaining = 0;
    tiltModeActive = true;
    applyTiltPoseTargets();
    Serial.println("ACK:POSE:TILT");
  }
  else if (cmd == "stop" || cmd == "home") {
    cancelSequenceAndHome();
  }
  else if (cmd == "status") {
    printStatus();
  }
  else if (cmd == "help") {
    printHelp();
  }

  // -------- Shoulder calibration --------
  else if (cmd.startsWith("both_up:")) {
    int delta = parseValueAfterColon(cmd);
    if (delta > 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      moveRightShoulderUp(delta);
      moveLeftShoulderUp(delta);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }
  else if (cmd.startsWith("both_down:")) {
    int delta = parseValueAfterColon(cmd);
    if (delta > 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      moveRightShoulderDown(delta);
      moveLeftShoulderDown(delta);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }
  else if (cmd.startsWith("r_up:")) {
    int delta = parseValueAfterColon(cmd);
    if (delta > 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      moveRightShoulderUp(delta);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }
  else if (cmd.startsWith("r_down:")) {
    int delta = parseValueAfterColon(cmd);
    if (delta > 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      moveRightShoulderDown(delta);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }
  else if (cmd.startsWith("l_up:")) {
    int delta = parseValueAfterColon(cmd);
    if (delta > 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      moveLeftShoulderUp(delta);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }
  else if (cmd.startsWith("l_down:")) {
    int delta = parseValueAfterColon(cmd);
    if (delta > 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      moveLeftShoulderDown(delta);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }

  // -------- Arm calibration --------
  else if (cmd.startsWith("both_arm_fwd:")) {
    int delta = parseValueAfterColon(cmd);
    if (delta > 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      moveRightArmForward(delta);
      moveLeftArmForward(delta);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }
  else if (cmd.startsWith("both_arm_back:")) {
    int delta = parseValueAfterColon(cmd);
    if (delta > 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      moveRightArmBackward(delta);
      moveLeftArmBackward(delta);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }
  else if (cmd.startsWith("ra_fwd:")) {
    int delta = parseValueAfterColon(cmd);
    if (delta > 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      moveRightArmForward(delta);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }
  else if (cmd.startsWith("ra_back:")) {
    int delta = parseValueAfterColon(cmd);
    if (delta > 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      moveRightArmBackward(delta);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }
  else if (cmd.startsWith("la_fwd:")) {
    int delta = parseValueAfterColon(cmd);
    if (delta > 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      moveLeftArmForward(delta);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }
  else if (cmd.startsWith("la_back:")) {
    int delta = parseValueAfterColon(cmd);
    if (delta > 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      moveLeftArmBackward(delta);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }

  // -------- Direct set commands --------
  else if (cmd.startsWith("set0:")) {
    int value = parseValueAfterColon(cmd);
    if (value >= 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      setTarget(0, value);
      Serial.print("ACK:SERVO0:TARGET=");
      Serial.println(targetUs[0]);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }
  else if (cmd.startsWith("set1:")) {
    int value = parseValueAfterColon(cmd);
    if (value >= 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      setTarget(1, value);
      Serial.print("ACK:SERVO1:TARGET=");
      Serial.println(targetUs[1]);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }
  else if (cmd.startsWith("set2:")) {
    int value = parseValueAfterColon(cmd);
    if (value >= 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      setTarget(2, value);
      Serial.print("ACK:SERVO2:TARGET=");
      Serial.println(targetUs[2]);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }
  else if (cmd.startsWith("set3:")) {
    int value = parseValueAfterColon(cmd);
    if (value >= 0) {
      activeSequence = SEQ_NONE;
      activePhase = PHASE_IDLE;
      sequenceCyclesRemaining = 0;
      tiltModeActive = false;
      setTarget(3, value);
      Serial.print("ACK:SERVO3:TARGET=");
      Serial.println(targetUs[3]);
    } else {
      Serial.println("ERR:BAD_VALUE");
    }
  }
  else if (cmd.length() == 0) {
    Serial.println("ERR:EMPTY");
  }
  else {
    Serial.print("ERR:UNKNOWN:");
    Serial.println(cmd);
  }
}

void serviceSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        handleCommand(serialBuffer);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
      if (serialBuffer.length() > 120) {
        serialBuffer = "";
      }
    }
  }
}

void serviceMotion() {
  unsigned long now = millis();
  if (now - lastMotionUpdate < MOTION_UPDATE_MS) return;
  lastMotionUpdate = now;

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

    writeServo(i, (uint16_t)cur);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println("BOOT");
  Wire.begin(SDA_PIN, SCL_PIN);

  Wire.beginTransmission(PCA_ADDR);
  byte err = Wire.endTransmission();

  if (err != 0) {
    Serial.print("ERR:PCA9685_NOT_FOUND:");
    Serial.println(err);
  } else {
    Serial.println("PCA9685 OK");
  }

  pca.begin();
  pca.setPWMFreq(SERVO_FREQ);
  delay(10);

  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    currentUs[i] = SERVO_HOME[i];
    targetUs[i] = SERVO_HOME[i];
    writeServo(i, SERVO_HOME[i]);
    delay(60);
  }

  setHomeTargets();

  Serial.println("READY");
  printHelp();
}

void loop() {
  serviceSerial();
  serviceSequence();
  serviceMotion();
}