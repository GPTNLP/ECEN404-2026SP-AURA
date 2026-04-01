from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse

from security import require_role

router = APIRouter(tags=["camera_bridge"])

BACKEND_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(os.getenv("AURA_STORAGE_DIR") or (BACKEND_DIR / "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

FRAMES_DIR = STORAGE_DIR / "camera_frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

COMMANDS_FILE = STORAGE_DIR / "device_commands.json"
CAMERA_FRAME_STALE_SECONDS = int(os.getenv("CAMERA_FRAME_STALE_SECONDS", "3"))


def _load_commands() -> list[dict]:
    if not COMMANDS_FILE.exists():
        return []
    try:
        return json.loads(COMMANDS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_commands(commands: list[dict]) -> None:
    COMMANDS_FILE.write_text(json.dumps(commands, indent=2), encoding="utf-8")


def _require_device_secret(x_device_secret: str | None) -> None:
    expected = os.getenv("DEVICE_SHARED_SECRET", "").strip()
    provided = (x_device_secret or "").strip()

    if not expected:
        raise HTTPException(status_code=500, detail="DEVICE_SHARED_SECRET is not configured on backend")
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid device secret")


def _frame_dir(device_id: str) -> Path:
    p = FRAMES_DIR / device_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _latest_frame_path(device_id: str) -> Path:
    return _frame_dir(device_id) / "latest.jpg"


def _latest_meta_path(device_id: str) -> Path:
    return _frame_dir(device_id) / "latest.json"


def _read_latest_meta(device_id: str) -> dict:
    meta_path = _latest_meta_path(device_id)
    if not meta_path.exists():
        return {}

    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _frame_is_fresh(device_id: str) -> bool:
    meta = _read_latest_meta(device_id)
    updated_at = int(meta.get("updated_at", 0) or 0)
    if updated_at <= 0:
        return False
    return (int(time.time()) - updated_at) <= CAMERA_FRAME_STALE_SECONDS


def _queue_command(device_id: str, command: str, payload: dict | None = None) -> dict:
    commands = _load_commands()
    now_ms = int(time.time() * 1000)
    now_s = int(time.time())

    entry = {
        "id": f"{device_id}-{now_ms}",
        "device_id": device_id,
        "command": command,
        "payload": payload or {},
        "created_at": now_s,
        "status": "pending",
    }

    commands.append(entry)
    _save_commands(commands)
    return entry


@router.post("/camera/control/activate")
def activate_camera(
    request: Request,
    device_id: str = Query(...),
    mode: str = Query("raw", pattern="^(raw|detection)$"),
):
    require_role(request, "admin")

    command = "camera_activate_detection" if mode == "detection" else "camera_activate_raw"
    queued = _queue_command(device_id=device_id, command=command)

    return {
        "ok": True,
        "queued": queued,
        "device_id": device_id,
        "mode": mode,
    }


@router.post("/camera/control/deactivate")
def deactivate_camera(
    request: Request,
    device_id: str = Query(...),
):
    require_role(request, "admin")

    queued = _queue_command(device_id=device_id, command="camera_deactivate")

    return {
        "ok": True,
        "queued": queued,
        "device_id": device_id,
        "mode": "off",
    }


@router.post("/device/camera/frame")
async def upload_camera_frame(
    request: Request,
    device_id: str = Query(...),
    mode: str = Query("raw", pattern="^(raw|detection)$"),
    x_device_secret: str | None = Header(default=None, alias="X-Device-Secret"),
):
    _require_device_secret(x_device_secret)

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty frame body")

    frame_path = _latest_frame_path(device_id)
    meta_path = _latest_meta_path(device_id)

    frame_path.write_bytes(body)
    meta_path.write_text(
        json.dumps(
            {
                "device_id": device_id,
                "mode": mode,
                "updated_at": int(time.time()),
                "bytes": len(body),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "device_id": device_id,
        "mode": mode,
        "bytes": len(body),
    }


@router.get("/camera/latest")
def get_latest_camera_frame(
    device_id: str = Query(...),
):
    frame_path = _latest_frame_path(device_id)
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="No camera frame available yet")

    if not _frame_is_fresh(device_id):
        raise HTTPException(status_code=404, detail="Camera frame is stale")

    return FileResponse(frame_path, media_type="image/jpeg")


@router.get("/camera/latest/meta")
def get_latest_camera_meta(
    device_id: str = Query(...),
):
    meta = _read_latest_meta(device_id)
    if not meta:
        return {
            "ok": True,
            "device_id": device_id,
            "available": False,
            "fresh": False,
            "stale_after_seconds": CAMERA_FRAME_STALE_SECONDS,
        }

    updated_at = int(meta.get("updated_at", 0) or 0)
    age_seconds = int(time.time()) - updated_at if updated_at > 0 else None
    fresh = _frame_is_fresh(device_id)

    return {
        "ok": True,
        "device_id": device_id,
        "available": True,
        "fresh": fresh,
        "age_seconds": age_seconds,
        "stale_after_seconds": CAMERA_FRAME_STALE_SECONDS,
        **meta,
    }