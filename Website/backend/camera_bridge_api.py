from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from security import require_role

router = APIRouter(tags=["camera_bridge"])

BACKEND_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(os.getenv("AURA_STORAGE_DIR") or (BACKEND_DIR / "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

FRAMES_DIR = STORAGE_DIR / "camera_frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

COMMANDS_FILE = STORAGE_DIR / "device_commands.json"

CAMERA_MODE_PATTERN = "^(raw|detection|colorcode|face)$"


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


def _camera_activate_command_for_mode(mode: str) -> str:
    mode = (mode or "raw").strip().lower()

    if mode == "raw":
        return "camera_activate_raw"
    if mode == "detection":
        return "camera_activate_detection"
    if mode == "colorcode":
        return "camera_activate_colorcode"
    if mode == "face":
        return "camera_activate_face"

    raise HTTPException(status_code=400, detail=f"Unsupported camera mode: {mode}")


@router.post("/camera/control/activate")
def activate_camera(
    request: Request,
    device_id: str = Query(...),
    mode: str = Query("raw", pattern=CAMERA_MODE_PATTERN),
):
    require_role(request, "admin")

    command = _camera_activate_command_for_mode(mode)
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
    mode: str = Query("raw", pattern=CAMERA_MODE_PATTERN),
    x_device_secret: str | None = Header(default=None, alias="X-Device-Secret"),
):
    _require_device_secret(x_device_secret)

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty frame body")

    frame_path = _latest_frame_path(device_id)
    meta_path = _latest_meta_path(device_id)

    tmp_path = frame_path.with_suffix(".tmp")
    tmp_path.write_bytes(body)
    tmp_path.replace(frame_path)

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
def get_latest_camera_frame(device_id: str = Query(...)):
    frame_path = _latest_frame_path(device_id)
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="No camera frame available yet")

    return FileResponse(
        frame_path,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/camera/stream")
def stream_camera(device_id: str = Query(...)):
    frame_path = _latest_frame_path(device_id)

    boundary = "frame"

    def gen():
        last_mtime_ns = -1

        while True:
            try:
                if not frame_path.exists():
                    time.sleep(0.03)
                    continue

                stat = frame_path.stat()
                if stat.st_mtime_ns == last_mtime_ns:
                    time.sleep(0.005)
                    continue

                jpg = frame_path.read_bytes()
                last_mtime_ns = stat.st_mtime_ns

                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n"
                    b"Cache-Control: no-store\r\n\r\n" +
                    jpg + b"\r\n"
                )

            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.03)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/camera/latest/meta")
def get_latest_camera_meta(device_id: str = Query(...)):
    meta_path = _latest_meta_path(device_id)

    if not meta_path.exists():
        return {
            "ok": True,
            "device_id": device_id,
            "available": False,
        }

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    return {
        "ok": True,
        "device_id": device_id,
        "available": True,
        **meta,
    }