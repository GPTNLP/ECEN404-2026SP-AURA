import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from config import STORAGE_DIR
from security import require_auth

# Load Website/.env if it exists
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(env_path)

router = APIRouter(prefix="/device", tags=["device"])

DEVICES_FILE = STORAGE_DIR / "devices.json"
DEVICE_LOGS_FILE = STORAGE_DIR / "device_logs.jsonl"

DEVICE_SHARED_SECRET = os.getenv("DEVICE_SHARED_SECRET", "").strip()
DEVICE_DEFAULT_POLL_SECONDS = int(os.getenv("DEVICE_DEFAULT_POLL_SECONDS", "1"))
DEVICE_DEFAULT_HEARTBEAT_SECONDS = int(os.getenv("DEVICE_DEFAULT_HEARTBEAT_SECONDS", "10"))
DEVICE_DEFAULT_STATUS_SECONDS = int(os.getenv("DEVICE_DEFAULT_STATUS_SECONDS", "2"))

STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class DeviceRegisterReq(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=100)
    device_name: Optional[str] = Field(default=None, max_length=120)
    device_type: Optional[str] = Field(default="jetson", max_length=50)
    software_version: Optional[str] = Field(default=None, max_length=100)
    hostname: Optional[str] = Field(default=None, max_length=120)
    local_ip: Optional[str] = Field(default=None, max_length=120)


class DeviceHeartbeatReq(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=100)
    software_version: Optional[str] = Field(default=None, max_length=100)
    uptime_seconds: Optional[int] = None
    hostname: Optional[str] = Field(default=None, max_length=120)
    local_ip: Optional[str] = Field(default=None, max_length=120)
    ollama_ready: Optional[bool] = None
    vector_db_ready: Optional[bool] = None
    camera_ready: Optional[bool] = None


class DeviceStatusReq(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=100)

    battery_percent: Optional[float] = None
    battery_voltage: Optional[float] = None
    charging: Optional[bool] = None

    cpu_percent: Optional[float] = None
    ram_percent: Optional[float] = None
    disk_percent: Optional[float] = None
    temperature_c: Optional[float] = None

    camera_ready: Optional[bool] = None
    mic_ready: Optional[bool] = None
    speaker_ready: Optional[bool] = None

    ollama_ready: Optional[bool] = None
    vector_db_ready: Optional[bool] = None

    current_mode: Optional[str] = Field(default=None, max_length=100)
    current_task: Optional[str] = Field(default=None, max_length=200)

    extra: Optional[Dict[str, Any]] = None


class DeviceLogReq(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=100)
    level: str = Field(default="info", max_length=20)
    event: str = Field(..., min_length=1, max_length=120)
    message: str = Field(..., min_length=1, max_length=4000)
    meta: Optional[Dict[str, Any]] = None


class DeviceListResponse(BaseModel):
    ok: bool
    count: int
    items: List[Dict[str, Any]]


def _now() -> int:
    return int(time.time())


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _require_device_secret(request: Request) -> None:
    if not DEVICE_SHARED_SECRET:
        raise HTTPException(status_code=500, detail="Server missing DEVICE_SHARED_SECRET")

    got = (request.headers.get("x-device-secret", "") or "").strip()
    if not got or got != DEVICE_SHARED_SECRET:
        raise HTTPException(status_code=403, detail="Bad device secret")


def _read_devices() -> Dict[str, Any]:
    if not DEVICES_FILE.exists():
        return {"devices": {}}

    try:
        raw = DEVICES_FILE.read_text(encoding="utf-8").strip()
        if not raw:
            return {"devices": {}}
        data = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Devices store unreadable: {e}")

    if not isinstance(data, dict):
        return {"devices": {}}

    devices = data.get("devices")
    if not isinstance(devices, dict):
        data["devices"] = {}

    return data


def _write_devices(data: Dict[str, Any]) -> None:
    tmp = DEVICES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(DEVICES_FILE)


def _get_or_create_device_slot(data: Dict[str, Any], device_id: str) -> Dict[str, Any]:
    devices = data.setdefault("devices", {})
    rec = devices.get(device_id)
    if not isinstance(rec, dict):
        rec = {
            "device_id": device_id,
            "created_at": _now(),
            "status": {},
        }
        devices[device_id] = rec
    return rec


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _sorted_device_items() -> List[Dict[str, Any]]:
    data = _read_devices()
    devices = data.get("devices", {})

    items: List[Dict[str, Any]] = []
    for rec in devices.values():
        if isinstance(rec, dict):
            items.append(rec)

    items.sort(key=lambda x: int(x.get("last_seen_at", 0) or 0), reverse=True)
    return items


@router.post("/register")
def device_register(body: DeviceRegisterReq, request: Request):
    _require_device_secret(request)

    device_id = body.device_id.strip()
    if not device_id:
        raise HTTPException(status_code=400, detail="device_id is required")

    data = _read_devices()
    rec = _get_or_create_device_slot(data, device_id)

    rec.update({
        "device_id": device_id,
        "device_name": (body.device_name or device_id).strip(),
        "device_type": (body.device_type or "jetson").strip(),
        "software_version": (body.software_version or "").strip(),
        "hostname": (body.hostname or "").strip(),
        "local_ip": (body.local_ip or "").strip(),
        "last_register_at": _now(),
        "last_seen_at": _now(),
        "last_seen_ip": _client_ip(request),
        "online": True,
    })

    _write_devices(data)

    return {
        "ok": True,
        "device": {
            "device_id": rec.get("device_id"),
            "device_name": rec.get("device_name"),
            "online": rec.get("online"),
            "last_seen_at": rec.get("last_seen_at"),
        }
    }


@router.post("/heartbeat")
def device_heartbeat(body: DeviceHeartbeatReq, request: Request):
    _require_device_secret(request)

    device_id = body.device_id.strip()
    if not device_id:
        raise HTTPException(status_code=400, detail="device_id is required")

    data = _read_devices()
    rec = _get_or_create_device_slot(data, device_id)

    rec.update({
        "software_version": (body.software_version or rec.get("software_version") or "").strip(),
        "hostname": (body.hostname or rec.get("hostname") or "").strip(),
        "local_ip": (body.local_ip or rec.get("local_ip") or "").strip(),
        "uptime_seconds": body.uptime_seconds,
        "ollama_ready": body.ollama_ready,
        "vector_db_ready": body.vector_db_ready,
        "camera_ready": body.camera_ready,
        "last_seen_at": _now(),
        "last_seen_ip": _client_ip(request),
        "online": True,
    })

    _write_devices(data)

    return {
        "ok": True,
        "server_time": _now(),
        "device_id": device_id,
    }


@router.post("/status")
def device_status(body: DeviceStatusReq, request: Request):
    _require_device_secret(request)

    device_id = body.device_id.strip()
    if not device_id:
        raise HTTPException(status_code=400, detail="device_id is required")

    data = _read_devices()
    rec = _get_or_create_device_slot(data, device_id)

    rec["status"] = {
        "battery_percent": body.battery_percent,
        "battery_voltage": body.battery_voltage,
        "charging": body.charging,
        "cpu_percent": body.cpu_percent,
        "ram_percent": body.ram_percent,
        "disk_percent": body.disk_percent,
        "temperature_c": body.temperature_c,
        "camera_ready": body.camera_ready,
        "mic_ready": body.mic_ready,
        "speaker_ready": body.speaker_ready,
        "ollama_ready": body.ollama_ready,
        "vector_db_ready": body.vector_db_ready,
        "current_mode": body.current_mode,
        "current_task": body.current_task,
        "extra": body.extra or {},
        "updated_at": _now(),
    }

    rec["last_seen_at"] = _now()
    rec["last_seen_ip"] = _client_ip(request)
    rec["online"] = True

    _write_devices(data)

    return {
        "ok": True,
        "device_id": device_id,
        "status_updated_at": rec["status"]["updated_at"],
    }


@router.post("/logs")
def device_log(body: DeviceLogReq, request: Request):
    _require_device_secret(request)

    device_id = body.device_id.strip()
    if not device_id:
        raise HTTPException(status_code=400, detail="device_id is required")

    entry = {
        "ts": _now(),
        "device_id": device_id,
        "level": (body.level or "info").strip().lower(),
        "event": body.event.strip(),
        "message": body.message.strip(),
        "meta": body.meta or {},
        "ip": _client_ip(request),
    }

    _append_jsonl(DEVICE_LOGS_FILE, entry)

    data = _read_devices()
    rec = _get_or_create_device_slot(data, device_id)
    rec["last_log_at"] = entry["ts"]
    rec["last_seen_at"] = entry["ts"]
    rec["last_seen_ip"] = entry["ip"]
    rec["online"] = True
    _write_devices(data)

    return {
        "ok": True,
        "logged": True,
        "ts": entry["ts"],
    }


@router.get("/config")
def device_config(request: Request, device_id: str):
    _require_device_secret(request)

    device_id = (device_id or "").strip()
    if not device_id:
        raise HTTPException(status_code=400, detail="device_id is required")

    data = _read_devices()
    devices = data.get("devices", {})
    rec = devices.get(device_id)

    if not isinstance(rec, dict):
        raise HTTPException(status_code=404, detail="Unknown device")

    return {
        "ok": True,
        "device_id": device_id,
        "poll_seconds": DEVICE_DEFAULT_POLL_SECONDS,
        "heartbeat_seconds": DEVICE_DEFAULT_HEARTBEAT_SECONDS,
        "status_seconds": DEVICE_DEFAULT_STATUS_SECONDS,
        "camera_mode": "reserved",
        "commands_enabled": True,
        "ollama_enabled": False,
        "vector_sync_enabled": False,
    }


@router.get("/list", response_model=DeviceListResponse)
def list_devices(request: Request):
    require_auth(request)
    items = _sorted_device_items()
    return {
        "ok": True,
        "count": len(items),
        "items": items,
    }


@router.get("/admin/list", response_model=DeviceListResponse)
def admin_list_devices(request: Request):
    payload = require_auth(request)
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    items = _sorted_device_items()
    return {
        "ok": True,
        "count": len(items),
        "items": items,
    }
