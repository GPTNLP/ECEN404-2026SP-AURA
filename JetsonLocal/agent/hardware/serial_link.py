import time
from typing import Optional

import serial

from core.config import (
    SERIAL_PORT,
    SERIAL_BAUDRATE,
    SERIAL_TIMEOUT,
    SERIAL_DRY_RUN,
)


class SerialLink:
    def __init__(self):
        self.esp_serial: Optional[serial.Serial] = None
        self.dry_run = SERIAL_DRY_RUN
        self.MOVEMENT_COMMANDS = {
            "forward",
            "backward",
            "left",
            "right",
            "stop",
            "pitch",
            "yaw",
        }

    def connect(self) -> bool:
        if self.dry_run:
            print("[SERIAL] DRY RUN enabled - skipping real ESP32 connection")
            return True

        try:
            if self.esp_serial and self.esp_serial.is_open:
                return True

            self.esp_serial = serial.Serial(
                SERIAL_PORT,
                SERIAL_BAUDRATE,
                timeout=SERIAL_TIMEOUT,
            )
            time.sleep(2.0)

            self.esp_serial.reset_input_buffer()
            self.esp_serial.reset_output_buffer()

            print(f"[SERIAL] Connected to {SERIAL_PORT}")
            return True

        except Exception as e:
            self.esp_serial = None
            print(f"[SERIAL] Connect failed: {e}")
            return False

    def send_command(self, cmd: str, val: str = "") -> str:
        cmd = cmd.strip().lower()

        if cmd not in self.MOVEMENT_COMMANDS:
            raise ValueError(f"Unsupported serial command: {cmd}")

        if self.dry_run:
            serial_msg = f"MOVE:{cmd}:{val}" if val else f"MOVE:{cmd}"
            print(f"[SERIAL][DRY RUN] would send -> {serial_msg}")
            return f"SENT:{serial_msg}"

        if not self.connect():
            raise RuntimeError("ESP serial is not connected")

        serial_msg = f"MOVE:{cmd}:{val}\n" if val else f"MOVE:{cmd}\n"

        try:
            self.esp_serial.reset_input_buffer()
        except Exception:
            pass

        self.esp_serial.write(serial_msg.encode("utf-8"))
        self.esp_serial.flush()

        print(f"[SERIAL] sent -> {serial_msg.strip()}")

        # Optional debug read only.
        # We do NOT require a legacy ACK:MOVE:<cmd> anymore because
        # the ESP32 firmware now emits sequence/status messages instead.
        deadline = time.time() + 0.20
        while time.time() < deadline:
            try:
                line = self.esp_serial.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                break

            if not line:
                continue

            print(f"[SERIAL][ESP] {line}")

            if line.startswith("ERR:"):
                raise RuntimeError(line)

        return f"SENT:{cmd}"


serial_link = SerialLink()