from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal, Dict, Any, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from security import require_role

router = APIRouter(tags=["device_commands"])

ALLOWED_COMMANDS = {
    "forward",
    "backward",
    "left",
    "right",
    "stop",
    "build_rag",
    "chat_prompt",
    "sync_vectors",
    "pitch",
    "yaw",
    "camera_activate_raw",
    "camera_activate_detection",
    "camera_deactivate",
}

MOVEMENT_COMMANDS = {"forward", "backward", "left", "right", "stop"}

STORAGE_DIR = Path(
    os.getenv("AURA_STORAGE_DIR")
    or (Path(__file__).resolve().parent / "storage")
)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

COMMANDS_FILE = STORAGE_DIR / "device_commands.json"

print(f"[DEVICE_COMMANDS] STORAGE_DIR = {STORAGE_DIR}")
print(f"[DEVICE_COMMANDS] COMMANDS_FILE = {COMMANDS_FILE}")


def _load_commands() -> list[dict]:
    if not COMMANDS_FILE.exists():
        return []
    try:
        return json.loads(COMMANDS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_commands(commands: list[dict]) -> None:
    COMMANDS_FILE.write_text(json.dumps(commands, indent=2), encoding="utf-8")


def _require_admin(request: Request) -> None:
    require_role(request, "admin")


def _require_device_secret(x_device_secret: str | None) -> None:
    expected = os.getenv("DEVICE_SHARED_SECRET", "").strip()
    provided = (x_device_secret or "").strip()

    if not expected:
        raise HTTPException(status_code=500, detail="DEVICE_SHARED_SECRET is not configured on backend")

    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid device secret")


def _is_active_status(status: str | None) -> bool:
    return status in {"pending", "delivered"}


class DeviceCommandIn(BaseModel):
    device_id: str
    command: str
    payload: Optional[Dict[str, Any]] = {}


class DeviceCommandAckIn(BaseModel):
    command_id: str
    device_id: str
    status: Literal["completed", "failed"] = "completed"
    note: str | None = None
    result: Optional[Dict[str, Any]] = None


@router.post("/device/admin/command")
def queue_device_command(payload: DeviceCommandIn, request: Request):
    _require_admin(request)

    if payload.command not in ALLOWED_COMMANDS:
        raise HTTPException(status_code=400, detail=f"Invalid command. Allowed: {ALLOWED_COMMANDS}")

    commands = _load_commands()
    now_ms = int(time.time() * 1000)
    now_s = int(time.time())

    if payload.command in MOVEMENT_COMMANDS:
        filtered: list[dict] = []
        for item in commands:
            same_device = item.get("device_id") == payload.device_id
            same_kind = item.get("command") in MOVEMENT_COMMANDS
            active = _is_active_status(item.get("status"))

            if same_device and same_kind and active:
                continue
            filtered.append(item)
        commands = filtered

    entry = {
        "id": f"{payload.device_id}-{now_ms}",
        "device_id": payload.device_id,
        "command": payload.command,
        "payload": payload.payload or {},
        "created_at": now_s,
        "status": "pending",
    }

    commands.append(entry)
    _save_commands(commands)

    print(f"[DEVICE_COMMANDS] queued {payload.command} for {payload.device_id} -> {COMMANDS_FILE}")

    return {"ok": True, "queued": entry}


@router.get("/device/command/next")
def get_next_device_command(
    device_id: str,
    x_device_secret: str | None = Header(default=None, alias="X-Device-Secret"),
):
    _require_device_secret(x_device_secret)

    commands = _load_commands()
    next_cmd = None

    for item in commands:
        if item.get("device_id") == device_id and item.get("status") == "pending":
            item["status"] = "delivered"
            item["delivered_at"] = int(time.time())
            next_cmd = item
            break

    _save_commands(commands)

    return {
        "ok": True,
        "command": next_cmd,
    }


@router.post("/device/command/ack")
def ack_device_command(
    payload: DeviceCommandAckIn,
    x_device_secret: str | None = Header(default=None, alias="X-Device-Secret"),
):
    _require_device_secret(x_device_secret)

    commands = _load_commands()
    updated = None

    for item in commands:
        if item.get("id") == payload.command_id and item.get("device_id") == payload.device_id:
            item["status"] = payload.status
            item["acked_at"] = int(time.time())
            item["note"] = payload.note
            item["result"] = payload.result
            updated = item
            break

    _save_commands(commands)

    if not updated:
        raise HTTPException(status_code=404, detail="Command not found")

    return {
        "ok": True,
        "command": updated,
    }

import time

@router.post("/device/admin/chat")
def chat_via_jetson(payload: DeviceCommandIn, request: Request):
    _require_admin(request)

    if payload.command != "chat_prompt":
        raise HTTPException(status_code=400, detail="Only chat_prompt supported here")

    commands = _load_commands()
    now_ms = int(time.time() * 1000)
    now_s = int(time.time())

    command_id = f"{payload.device_id}-{now_ms}"

    entry = {
        "id": command_id,
        "device_id": payload.device_id,
        "command": "chat_prompt",
        "payload": payload.payload or {},
        "created_at": now_s,
        "status": "pending",
    }

    commands.append(entry)
    _save_commands(commands)

    print(f"[CHAT] queued chat_prompt -> {command_id}")

    # wait for response (poll for ack)
    timeout = 60  # seconds
    start = time.time()

    while time.time() - start < timeout:
        commands = _load_commands()

        for item in commands:
            if item.get("id") == command_id:
                if item.get("status") in ("completed", "failed"):
                    result = item.get("result") or {}
                    answer = result.get("answer", "")

                    return {
                        "ok": True,
                        "answer": answer,
                        "status": item.get("status"),
                    }

        time.sleep(0.3)

    raise HTTPException(status_code=504, detail="Jetson did not respond in time")