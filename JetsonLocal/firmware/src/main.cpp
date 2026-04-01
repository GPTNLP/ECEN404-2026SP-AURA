#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define I2C_SDA 4
#define I2C_SCL 5
#define PCA9685_ADDR 0x40

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(PCA9685_ADDR);

// Actuator mapping (Channels 0-3)
const uint8_t L_LEG = 0;
const uint8_t R_LEG = 1;
const uint8_t L_ARM = 2;
const uint8_t R_ARM = 3;

// Base home angles
const int HOME_ANGLES[4] = {90, 90, 90, 90};
int currentAngles[4] = {90, 90, 90, 90};
int targetAngles[4] = {90, 90, 90, 90};

int pitchOffset = 0;

enum Mode { STOP, FORWARD, BACKWARD, LEFT, RIGHT };
Mode currentMode = STOP;

unsigned long lastGaitTime = 0;
bool gaitPhase = false;

void setServoAngle(uint8_t channel, int angle) {
    angle = constrain(angle, 0, 180);
    // Typical pulse widths for standard servos (adjust if using 270 deg or custom)
    uint16_t pulse = map(angle, 0, 180, 150, 600); 
    pca.setPWM(channel, 0, pulse);
}

void updateStance() {
    if (currentMode == STOP) {
        // Pitch tilts all servos uniformly 
        targetAngles[L_LEG] = HOME_ANGLES[L_LEG] + pitchOffset;
        targetAngles[R_LEG] = HOME_ANGLES[R_LEG] + pitchOffset;
        targetAngles[L_ARM] = HOME_ANGLES[L_ARM] + pitchOffset;
        targetAngles[R_ARM] = HOME_ANGLES[R_ARM] + pitchOffset;
    }
}

void processCommand(String cmd) {
    cmd.trim();
    if (cmd.startsWith("MOVE:")) {
        String action = cmd.substring(5);
        if (action.startsWith("pitch:")) {
            pitchOffset = action.substring(6).toInt();
            updateStance();
            Serial.println("ACK:MOVE:pitch");
        } else if (action == "forward") {
            currentMode = FORWARD;
            Serial.println("ACK:MOVE:forward");
        } else if (action == "backward") {
            currentMode = BACKWARD;
            Serial.println("ACK:MOVE:backward");
        } else if (action == "stop") {
            currentMode = STOP;
            updateStance();
            Serial.println("ACK:MOVE:stop");
        }
    }
}

void handleSerial() {
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        processCommand(cmd);
    }
}

void handleGait() {
    if (currentMode == STOP) return;
    
    unsigned long now = millis();
    if (now - lastGaitTime > 250) { // Gait transition speed
        lastGaitTime = now;
        gaitPhase = !gaitPhase;

        int legSwing = 30;
        int armSwing = 25; // Counter-balance

        if (currentMode == FORWARD) {
            // Legs move opposite to each other, arms swing opposite to legs
            targetAngles[L_LEG] = HOME_ANGLES[L_LEG] + pitchOffset + (gaitPhase ? legSwing : -legSwing);
            targetAngles[R_LEG] = HOME_ANGLES[R_LEG] + pitchOffset + (gaitPhase ? -legSwing : legSwing);
            targetAngles[L_ARM] = HOME_ANGLES[L_ARM] + pitchOffset + (gaitPhase ? -armSwing : armSwing);
            targetAngles[R_ARM] = HOME_ANGLES[R_ARM] + pitchOffset + (gaitPhase ? armSwing : -armSwing);
        } else if (currentMode == BACKWARD) {
            targetAngles[L_LEG] = HOME_ANGLES[L_LEG] + pitchOffset + (gaitPhase ? -legSwing : legSwing);
            targetAngles[R_LEG] = HOME_ANGLES[R_LEG] + pitchOffset + (gaitPhase ? legSwing : -legSwing);
            targetAngles[L_ARM] = HOME_ANGLES[L_ARM] + pitchOffset + (gaitPhase ? armSwing : -armSwing);
            targetAngles[R_ARM] = HOME_ANGLES[R_ARM] + pitchOffset + (gaitPhase ? -armSwing : armSwing);
        }
    }
}

void moveServos() {
    for (int i = 0; i < 4; i++) {
        if (currentAngles[i] != targetAngles[i]) {
            currentAngles[i] = targetAngles[i];
            setServoAngle(i, currentAngles[i]);
        }
    }
}

void setup() {
    Serial.begin(115200);
    Wire.begin(I2C_SDA, I2C_SCL);
    pca.begin();
    pca.setPWMFreq(50);
    
    updateStance();
    moveServos();
    
    Serial.println("READY");
}

void loop() {
    handleSerial();
    handleGait();
    moveServos();
    delay(10);
}