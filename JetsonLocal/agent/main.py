import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from api_client import ApiClient
from config import (
    AGENT_LOG_FILE,
    CAMERA_READY_DEFAULT,
    DEVICE_ID,
    DEVICE_NAME,
    DEVICE_SOFTWARE_VERSION,
    DEVICE_TYPE,
    HEARTBEAT_SECONDS,
    INPUT_MODE,
    OFFLINE_RETRY_SECONDS,
    PENDING_LOGS_FILE,
    PENDING_STATUS_FILE,
    RUNTIME_FILE,
    SERIAL_PORT,
    STATUS_SECONDS,
)

try:
    import serial
except Exception:
    serial = None


def _now_ts() -> int:
    return int(time.time())


def _safe_read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _safe_write_json(path: Path, data: Any):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def _append_jsonl(path: Path, row: Dict[str, Any]):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


FAST_MOVE_MAP = {
    "forward": "F",
    "backward": "B",
    "left": "L",
    "right": "R",
    "stop": "S",
    "f": "F",
    "b": "B",
    "l": "L",
    "r": "R",
    "s": "S",
}


class AuraJetsonAgent:
    def __init__(self):
        self.api = ApiClient()

        self.serial_conn = None
        self.serial_ready = False
        self.last_serial_error = ""

        self.command_poll_seconds = float(os.getenv("DEVICE_COMMAND_POLL_SECONDS", "0.08"))
        self.serial_baud = int(os.getenv("SERIAL_BAUD", "115200"))

        self.last_heartbeat_ts = 0.0
        self.last_status_ts = 0.0
        self.last_command_ts = 0.0

        self.last_command_id: Optional[str] = None
        self.registered = False

        self.runtime = _safe_read_json(Path(RUNTIME_FILE), {})
        self.runtime.setdefault("started_at", _now_ts())
        self.runtime.setdefault("last_command", None)
        self.runtime.setdefault("last_command_at", None)
        self.runtime.setdefault("serial_port", SERIAL_PORT)
        self._save_runtime()

    def _save_runtime(self):
        _safe_write_json(Path(RUNTIME_FILE), self.runtime)

    def log_local(self, level: str, event: str, message: str, **extra):
        row = {
            "ts": _now_ts(),
            "level": level,
            "event": event,
            "message": message,
            **extra,
        }
        _append_jsonl(Path(AGENT_LOG_FILE), row)
        print(f"[{level.upper()}] {event}: {message}")

    def queue_status_fallback(self, payload: Dict[str, Any]):
        _append_jsonl(Path(PENDING_STATUS_FILE), payload)

    def queue_log_fallback(self, payload: Dict[str, Any]):
        _append_jsonl(Path(PENDING_LOGS_FILE), payload)

    def connect_serial(self):
        if serial is None:
            self.serial_ready = False
            self.last_serial_error = "pyserial not installed"
            self.log_local("error", "serial", self.last_serial_error)
            return

        try:
            self.serial_conn = serial.Serial(
                SERIAL_PORT,
                self.serial_baud,
                timeout=0,
                write_timeout=0.25,
            )
            time.sleep(0.2)
            self.serial_ready = True
            self.last_serial_error = ""
            self.log_local("info", "serial", f"Connected to serial {SERIAL_PORT} @ {self.serial_baud}")
        except Exception as e:
            self.serial_conn = None
            self.serial_ready = False
            self.last_serial_error = str(e)
            self.log_local("error", "serial", f"Failed to open serial {SERIAL_PORT}: {e}")

    def read_serial_lines(self, max_lines: int = 20):
        if not self.serial_conn or not self.serial_ready:
            return

        try:
            count = 0
            while self.serial_conn.in_waiting and count < max_lines:
                raw = self.serial_conn.readline()
                if not raw:
                    break
                try:
                    line = raw.decode("utf-8", errors="ignore").strip()
                except Exception:
                    line = ""
                if line:
                    self.log_local("info", "esp32_rx", line)
                count += 1
        except Exception as e:
            self.serial_ready = False
            self.last_serial_error = str(e)
            self.log_local("error", "serial", f"Serial read error: {e}")

    def send_serial_line(self, line: str) -> bool:
        if not line:
            return False
        if not self.serial_conn or not self.serial_ready:
            return False

        try:
            payload = (line.strip() + "\n").encode("utf-8")
            self.serial_conn.write(payload)
            self.serial_conn.flush()
            self.runtime["last_command"] = line.strip()
            self.runtime["last_command_at"] = _now_ts()
            self._save_runtime()
            self.log_local("info", "esp32_tx", line.strip())
            return True
        except Exception as e:
            self.serial_ready = False
            self.last_serial_error = str(e)
            self.log_local("error", "serial", f"Serial write error: {e}")
            return False

    def make_register_payload(self) -> Dict[str, Any]:
        return {
            "device_id": DEVICE_ID,
            "name": DEVICE_NAME,
            "device_type": DEVICE_TYPE,
            "software_version": DEVICE_SOFTWARE_VERSION,
            "input_mode": INPUT_MODE,
            "serial_port": SERIAL_PORT,
        }

    def make_heartbeat_payload(self) -> Dict[str, Any]:
        return {
            "device_id": DEVICE_ID,
            "ts": _now_ts(),
            "software_version": DEVICE_SOFTWARE_VERSION,
        }

    def make_status_payload(self) -> Dict[str, Any]:
        return {
            "device_id": DEVICE_ID,
            "ts": _now_ts(),
            "status": "online",
            "input_mode": INPUT_MODE,
            "serial_ready": bool(self.serial_ready),
            "serial_port": SERIAL_PORT,
            "serial_error": self.last_serial_error,
            "camera_ready": bool(CAMERA_READY_DEFAULT),
            "last_command": self.runtime.get("last_command"),
            "last_command_at": self.runtime.get("last_command_at"),
        }

    def do_register(self):
        try:
            self.api.register(self.make_register_payload())
            self.registered = True
            self.log_local("info", "register", "Device registered")
        except Exception as e:
            self.registered = False
            self.log_local("error", "register", f"Register failed: {e}")

    def do_heartbeat(self):
        payload = self.make_heartbeat_payload()
        try:
            self.api.heartbeat(payload)
            self.last_heartbeat_ts = time.time()
        except Exception as e:
            self.log_local("error", "heartbeat", f"Heartbeat failed: {e}")

    def do_status(self):
        payload = self.make_status_payload()
        try:
            self.api.status(payload)
            self.last_status_ts = time.time()
        except Exception as e:
            self.queue_status_fallback(payload)
            self.log_local("error", "status", f"Status failed: {e}")

    def normalize_move_text(self, raw: str) -> Optional[str]:
        if not raw:
            return None

        low = str(raw).strip().lower()

        if low in ("f", "b", "l", "r", "s"):
            return low.upper()

        if low.startswith("move:"):
            move = low.split(":", 1)[1].strip()
            return FAST_MOVE_MAP.get(move)

        return FAST_MOVE_MAP.get(low)

    def send_move_command(self, move: str) -> bool:
        key = self.normalize_move_text(move)
        if not key:
            return False
        return self.send_serial_line(key)

    def try_handle_command_payload(self, cmd: Dict[str, Any]) -> bool:
        """
        Handles actual queued entry shape:
          {
            "id": "...",
            "device_id": "jetson-001",
            "command": "forward",
            "payload": {},
            "status": "delivered"
          }
        """
        command_id = cmd.get("command_id") or cmd.get("id") or cmd.get("uuid")
        command_name = cmd.get("command")
        payload = cmd.get("payload") or {}

        handled = False
        note = None
        result = None

        if isinstance(command_name, str):
            move = self.normalize_move_text(command_name)
            if move:
                handled = self.send_move_command(move)
                note = f"movement {move}"
            elif command_name in ("pitch", "yaw"):
                # not implemented yet, but do not crash
                handled = False
                note = f"{command_name} not implemented yet"

        if command_id:
            try:
                self.api.ack_command(
                    {
                        "device_id": DEVICE_ID,
                        "command_id": command_id,
                        "status": "completed" if handled else "failed",
                        "note": note,
                        "result": result,
                    }
                )
            except Exception as e:
                self.log_local("error", "command_ack", f"Ack failed: {e}")

        if handled:
            self.last_command_id = str(command_id or "")
            self.last_command_ts = time.time()
            self.log_local("info", "command", f"Handled command payload: {cmd}")
        else:
            self.log_local("warning", "command", f"Unhandled command payload: {cmd}")

        return handled

    def poll_command_once(self):
        try:
            data = self.api.get_next_command(DEVICE_ID)
            if not isinstance(data, dict):
                return

            # backend returns {"ok": True, "command": next_cmd}
            queued = data.get("command")
            if not isinstance(queued, dict):
                return

            self.try_handle_command_payload(queued)
        except Exception as e:
            self.log_local("error", "command_poll", f"Command poll failed: {e}")

    def boot(self):
        self.connect_serial()
        self.do_register()
        self.do_status()
        self.do_heartbeat()

    def run_forever(self):
        self.boot()

        while True:
            now = time.time()

            if not self.serial_ready and serial is not None:
                self.connect_serial()

            self.read_serial_lines()

            if (now - self.last_heartbeat_ts) >= HEARTBEAT_SECONDS:
                self.do_heartbeat()

            if (now - self.last_status_ts) >= STATUS_SECONDS:
                self.do_status()

            self.poll_command_once()
            time.sleep(self.command_poll_seconds)


def main():
    agent = AuraJetsonAgent()

    while True:
        try:
            agent.run_forever()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            agent.log_local("error", "main", f"Agent crashed: {e}")
            time.sleep(OFFLINE_RETRY_SECONDS)


if __name__ == "__main__":
    main()