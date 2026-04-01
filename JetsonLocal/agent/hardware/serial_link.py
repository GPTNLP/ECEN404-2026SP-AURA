import time
import serial
from core.config import SERIAL_PORT

class SerialLink:
    def __init__(self):
        self.esp_serial = None
        self.MOVEMENT_COMMANDS = {"forward", "backward", "left", "right", "stop", "pitch", "yaw"}

    def connect(self) -> bool:
        try:
            if self.esp_serial and self.esp_serial.is_open:
                return True
            self.esp_serial = serial.Serial(SERIAL_PORT, 115200, timeout=1)
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
        if not self.connect():
            raise RuntimeError("ESP serial is not connected")

        cmd = cmd.strip().lower()
        if cmd not in self.MOVEMENT_COMMANDS:
            raise ValueError(f"Unsupported serial command: {cmd}")

        serial_msg = f"MOVE:{cmd}:{val}\n" if val else f"MOVE:{cmd}\n"
        expected_ack = f"ACK:MOVE:{cmd}"

        self.esp_serial.reset_input_buffer()
        self.esp_serial.write(serial_msg.encode("utf-8"))
        self.esp_serial.flush()
        
        # Wait for ACK
        deadline = time.time() + 2.0
        while time.time() < deadline:
            line = self.esp_serial.readline().decode("utf-8", errors="ignore").strip()
            if line == expected_ack:
                return line
            if line.startswith("ERR:"):
                raise RuntimeError(line)
                
        raise RuntimeError(f"No ACK from ESP for {cmd}")

serial_link = SerialLink()