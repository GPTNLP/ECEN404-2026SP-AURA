# backend/device_commands_api.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from security import require_role

router = APIRouter(tags=["device_commands"])

ALLOWED_COMMANDS = {"forward", "backward", "left", "right", "stop", "build_rag", "chat_prompt", "sync_vectors", "pitch", "yaw"}

STORAGE_DIR = Path(os.getenv("LOG_DIR", Path(__file__).resolve().parent / "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

COMMANDS_FILE = STORAGE_DIR / "device_commands.json"


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


class DeviceCommandIn(BaseModel):
    device_id: str
    command: Literal["forward", "backward", "left", "right", "stop"]


class DeviceCommandAckIn(BaseModel):
    command_id: str
    device_id: str
    status: Literal["completed", "failed"] = "completed"
    note: str | None = None


@router.post("/device/admin/command")
def queue_device_command(payload: DeviceCommandIn, request: Request):
    _require_admin(request)

    commands = _load_commands()

    entry = {
        "id": f"{payload.device_id}-{int(time.time() * 1000)}",
        "device_id": payload.device_id,
        "command": payload.command,
        "created_at": int(time.time()),
        "status": "pending",
    }

    commands.append(entry)
    _save_commands(commands)

    return {
        "ok": True,
        "queued": entry,
    }


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
            updated = item
            break

    _save_commands(commands)

    if not updated:
        raise HTTPException(status_code=404, detail="Command not found")

    return {
        "ok": True,
        "command": updated,
    }