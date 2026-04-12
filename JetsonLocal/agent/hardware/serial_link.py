import time
from typing import Optional

import os
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
        self.last_connect_ok = False
        self.last_error = ""
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
            self.last_connect_ok = True
            self.last_error = ""
            print("[SERIAL] DRY RUN enabled - skipping real ESP32 connection")
            return True

        try:
            if self.esp_serial and self.esp_serial.is_open:
                self.last_connect_ok = True
                self.last_error = ""
                return True

            self.esp_serial = serial.Serial(
                SERIAL_PORT,
                SERIAL_BAUDRATE,
                timeout=SERIAL_TIMEOUT,
            )
            time.sleep(2.0)

            self.esp_serial.reset_input_buffer()
            self.esp_serial.reset_output_buffer()

            self.last_connect_ok = True
            self.last_error = ""
            print(f"[SERIAL] Connected to {SERIAL_PORT}")
            return True

        except Exception as e:
            self.esp_serial = None
            self.last_connect_ok = False
            self.last_error = str(e)
            print(f"[SERIAL] Connect failed: {e}")
            return False

    def disconnect(self) -> None:
        try:
            if self.esp_serial and self.esp_serial.is_open:
                self.esp_serial.close()
        except Exception:
            pass
        finally:
            self.esp_serial = None

    def _ensure_live_connection(self) -> None:
        if self.esp_serial is None or not self.esp_serial.is_open:
            if not self.connect():
                raise RuntimeError("ESP serial is not connected")
            return

        try:
            if hasattr(self.esp_serial, "in_waiting"):
                _ = self.esp_serial.in_waiting
        except Exception:
            self.disconnect()
            if not self.connect():
                raise RuntimeError("ESP serial reconnect failed")

    def send_command(self, cmd: str, val: str = "") -> str:
        cmd = (cmd or "").strip().lower()

        if cmd not in self.MOVEMENT_COMMANDS:
            raise ValueError(f"Unsupported serial command: {cmd}")

        if self.dry_run:
            serial_msg = f"MOVE:{cmd}:{val}" if val else f"MOVE:{cmd}"
            print(f"[SERIAL][DRY RUN] would send -> {serial_msg}")
            return f"SENT:{serial_msg}"

        self._ensure_live_connection()

        serial_msg = f"MOVE:{cmd}:{val}\n" if val else f"MOVE:{cmd}\n"

        try:
            try:
                self.esp_serial.reset_input_buffer()
            except Exception:
                pass

            self.esp_serial.write(serial_msg.encode("utf-8"))
            self.esp_serial.flush()
            print(f"[SERIAL] sent -> {serial_msg.strip()}")

            # Debug-read only. We no longer require a strict legacy ACK line.
            # The ESP firmware now emits sequence/status messages like:
            # ACK:SEQ:FORWARD:START, ACK:SEQ:DONE, ACK:STOP:HOME, etc.
            deadline = time.time() + 0.25
            debug_lines = []

            while time.time() < deadline:
                try:
                    raw = self.esp_serial.readline()
                except Exception as read_err:
                    raise RuntimeError(f"ESP serial read failed: {read_err}")

                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                debug_lines.append(line)
                print(f"[SERIAL][ESP] {line}")

                if line.startswith("ERR:"):
                    raise RuntimeError(line)

            return f"SENT:{cmd}"

        except Exception:
            self.disconnect()
            raise

    def get_health(self) -> dict:
        port_exists = os.path.exists(SERIAL_PORT)
        is_open = bool(self.esp_serial and self.esp_serial.is_open)

        return {
            "port": SERIAL_PORT,
            "port_exists": port_exists,
            "connected": is_open,
            "last_connect_ok": self.last_connect_ok,
            "last_error": self.last_error or "",
        }    

serial_link = SerialLink()