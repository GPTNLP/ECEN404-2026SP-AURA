#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define SDA_PIN 4
#define SCL_PIN 5
#define PCA_ADDR 0x40

Adafruit_PWMServoDriver pca(PCA_ADDR);

const uint8_t NUM_SERVOS = 4;
const uint8_t CH[NUM_SERVOS] = {0, 1, 2, 3};

const uint16_t SERVO_MIN[NUM_SERVOS]  = {1600,  800,  800,  800};
const uint16_t SERVO_MAX[NUM_SERVOS]  = {2200, 1400, 2200, 2200};
const uint16_t SERVO_HOME[NUM_SERVOS] = {1900, 1100, 1400, 1350};

const int8_t SIGN[NUM_SERVOS] = {+1, -1, +1, -1};

const uint16_t SERVO_FREQ = 50;
const uint16_t STEP_US = 20;
const unsigned long MOTION_UPDATE_MS = 10;
const unsigned long GAIT_UPDATE_MS = 180;
const unsigned long COMMAND_TIMEOUT_MS = 500;

const int SHOULDER_SWING = 120;
const int LEG_SWING = 160;
const int TURN_SHOULDER = 90;
const int TURN_LEG = 120;

enum Mode {
  STOP_MODE,
  FORWARD_MODE,
  BACKWARD_MODE,
  LEFT_MODE,
  RIGHT_MODE
};

Mode currentMode = STOP_MODE;

uint16_t currentUs[NUM_SERVOS];
uint16_t targetUs[NUM_SERVOS];

unsigned long lastMotionUpdate = 0;
unsigned long lastGaitUpdate = 0;
unsigned long lastCommandTime = 0;
String serialBuffer = "";
bool gaitPhase = false;

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

int offsetFromHome(uint8_t i, int offset) {
  return SERVO_HOME[i] + SIGN[i] * offset;
}

void touchCommand() {
  lastCommandTime = millis();
}

void stopMotion() {
  currentMode = STOP_MODE;
  gaitPhase = false;
  setHomeTargets();
  Serial.println("ACK:MOVE:stop");
}

void setModeForward() {
  currentMode = FORWARD_MODE;
  touchCommand();
  Serial.println("ACK:MOVE:forward");
}

void setModeBackward() {
  currentMode = BACKWARD_MODE;
  touchCommand();
  Serial.println("ACK:MOVE:backward");
}

void setModeLeft() {
  currentMode = LEFT_MODE;
  touchCommand();
  Serial.println("ACK:MOVE:left");
}

void setModeRight() {
  currentMode = RIGHT_MODE;
  touchCommand();
  Serial.println("ACK:MOVE:right");
}

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

  int colon = cmd.indexOf(':');
  if (colon != -1) {
    cmd = cmd.substring(0, colon);
    cmd.trim();
  }

  return cmd;
}

void handleCommand(String raw) {
  String cmd = normalizeCommand(raw);

  Serial.print("CMD: ");
  Serial.println(cmd);

  if (cmd == "forward") {
    setModeForward();
  } else if (cmd == "backward") {
    setModeBackward();
  } else if (cmd == "left") {
    setModeLeft();
  } else if (cmd == "right") {
    setModeRight();
  } else if (cmd == "stop" || cmd == "home") {
    stopMotion();
  } else if (cmd == "status") {
    Serial.print("Mode: ");
    Serial.println((int)currentMode);
  } else if (cmd == "help") {
    Serial.println("Commands: forward backward left right stop home status help");
    Serial.println("Also accepts MOVE:forward and MOVE::forward");
  } else if (cmd.length() == 0) {
    Serial.println("ERR:EMPTY");
  } else {
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

void serviceGait() {
  if (currentMode == STOP_MODE) return;

  unsigned long now = millis();
  if (now - lastGaitUpdate < GAIT_UPDATE_MS) return;
  lastGaitUpdate = now;
  gaitPhase = !gaitPhase;

  if (currentMode == FORWARD_MODE) {
    if (!gaitPhase) {
      setTarget(0, offsetFromHome(0, +SHOULDER_SWING));
      setTarget(1, offsetFromHome(1, -SHOULDER_SWING));
      setTarget(2, offsetFromHome(2, +LEG_SWING));
      setTarget(3, offsetFromHome(3, -LEG_SWING));
    } else {
      setTarget(0, offsetFromHome(0, -SHOULDER_SWING));
      setTarget(1, offsetFromHome(1, +SHOULDER_SWING));
      setTarget(2, offsetFromHome(2, -LEG_SWING));
      setTarget(3, offsetFromHome(3, +LEG_SWING));
    }
  } else if (currentMode == BACKWARD_MODE) {
    if (!gaitPhase) {
      setTarget(0, offsetFromHome(0, -SHOULDER_SWING));
      setTarget(1, offsetFromHome(1, +SHOULDER_SWING));
      setTarget(2, offsetFromHome(2, -LEG_SWING));
      setTarget(3, offsetFromHome(3, +LEG_SWING));
    } else {
      setTarget(0, offsetFromHome(0, +SHOULDER_SWING));
      setTarget(1, offsetFromHome(1, -SHOULDER_SWING));
      setTarget(2, offsetFromHome(2, +LEG_SWING));
      setTarget(3, offsetFromHome(3, -LEG_SWING));
    }
  } else if (currentMode == LEFT_MODE) {
    if (!gaitPhase) {
      setTarget(0, offsetFromHome(0, +TURN_SHOULDER));
      setTarget(1, offsetFromHome(1, -TURN_SHOULDER));
      setTarget(2, offsetFromHome(2, -TURN_LEG));
      setTarget(3, offsetFromHome(3, +TURN_LEG));
    } else {
      setHomeTargets();
    }
  } else if (currentMode == RIGHT_MODE) {
    if (!gaitPhase) {
      setTarget(0, offsetFromHome(0, -TURN_SHOULDER));
      setTarget(1, offsetFromHome(1, +TURN_SHOULDER));
      setTarget(2, offsetFromHome(2, +TURN_LEG));
      setTarget(3, offsetFromHome(3, -TURN_LEG));
    } else {
      setHomeTargets();
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

void serviceTimeout() {
  if (currentMode == STOP_MODE) return;

  if (millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
    Serial.println("TIMEOUT");
    stopMotion();
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
  touchCommand();

  Serial.println("READY");
  Serial.println("Commands: forward backward left right stop");
}

void loop() {
  serviceSerial();
  serviceTimeout();
  serviceGait();
  serviceMotion();
}