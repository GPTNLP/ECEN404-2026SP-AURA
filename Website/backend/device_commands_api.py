from __future__ import annotations

import asyncio
import json
import os
import queue as _queue
import time
from pathlib import Path
from typing import Literal, Dict, Any, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from security import require_role

router = APIRouter(tags=["device_commands"])

ALLOWED_COMMANDS = {
    "forward",
    "backward",
    "left",
    "right",
    "stop",
    "left90",
    "right90",
    "left180",
    "right180",
    "left360",
    "right360",
    "build_rag",
    "chat_prompt",
    "sync_vectors",
    "delete_vectors",
    "camera_activate_raw",
    "camera_activate_detection",
    "camera_deactivate",
    "flush_models",
    "reload_llm",
}

MOVEMENT_COMMANDS = {
    "forward",
    "backward",
    "left",
    "right",
    "stop",
    "left90",
    "right90",
    "left180",
    "right180",
    "left360",
    "right360",
}

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


def _require_admin_or_ta(request: Request) -> None:
    require_role(request, "admin", "ta")


def _require_device_secret(x_device_secret: str | None) -> None:
    expected = os.getenv("DEVICE_SHARED_SECRET", "").strip()
    provided = (x_device_secret or "").strip()

    if not expected:
        raise HTTPException(status_code=500, detail="DEVICE_SHARED_SECRET is not configured on backend")

    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid device secret")


def _is_active_status(status: str | None) -> bool:
    return status in {"pending", "delivered"}


def _cancel_pending_movement_commands(
    commands: list[dict],
    device_id: str,
    reason: str,
) -> None:
    now_s = int(time.time())
    for item in commands:
        if item.get("device_id") != device_id:
            continue
        if item.get("command") not in MOVEMENT_COMMANDS:
            continue
        if item.get("status") != "pending":
            continue

        item["status"] = "cancelled"
        item["acked_at"] = now_s
        item["note"] = reason



def _select_next_command(commands: list[dict], device_id: str) -> dict | None:
    now_s = int(time.time())

    latest_stop_index: int | None = None
    for idx in range(len(commands) - 1, -1, -1):
        item = commands[idx]
        if item.get("device_id") != device_id:
            continue
        if item.get("status") != "pending":
            continue
        if item.get("command") == "stop":
            latest_stop_index = idx
            break

    if latest_stop_index is not None:
        next_cmd = commands[latest_stop_index]

        for idx, item in enumerate(commands):
            if idx == latest_stop_index:
                continue
            if item.get("device_id") != device_id:
                continue
            if item.get("command") not in MOVEMENT_COMMANDS:
                continue
            if item.get("status") != "pending":
                continue

            item["status"] = "cancelled"
            item["acked_at"] = now_s
            item["note"] = "Superseded by stop override"

        next_cmd["status"] = "delivered"
        next_cmd["delivered_at"] = now_s
        return next_cmd

    for item in commands:
        if item.get("device_id") == device_id and item.get("status") == "pending":
            item["status"] = "delivered"
            item["delivered_at"] = now_s
            return item

    return None


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


class DeviceCommandPartialIn(BaseModel):
    command_id: str
    device_id: str
    text: str  # one sentence chunk from the Jetson


# In-memory queues keyed by command_id.  The SSE streaming endpoint creates a
# queue when the chat command is queued; the Jetson pushes sentences via
# /device/command/partial; ack_device_command signals done.
# Works correctly for a single-process deployment (Azure App Service default).
_stream_queues: Dict[str, _queue.SimpleQueue] = {}

# Active WebSocket connections from Jetson devices (device_id → WebSocket).
# Used to push {"notify": "command"} when a command is queued, allowing the
# Jetson to fetch it immediately instead of waiting for the 5s HTTP poll.
_ws_connections: Dict[str, WebSocket] = {}


async def _ws_push(device_id: str, msg: dict) -> None:
    """Push a JSON message to the Jetson's WebSocket if connected (best-effort)."""
    ws = _ws_connections.get(device_id)
    if ws is None:
        return
    try:
        await ws.send_json(msg)
    except Exception:
        _ws_connections.pop(device_id, None)


@router.post("/device/admin/command")
async def queue_device_command(payload: DeviceCommandIn, request: Request):
    _require_admin(request)

    if payload.command not in ALLOWED_COMMANDS:
        raise HTTPException(status_code=400, detail=f"Invalid command. Allowed: {ALLOWED_COMMANDS}")

    commands = _load_commands()
    now_ms = int(time.time() * 1000)
    now_s = int(time.time())

    if payload.command in MOVEMENT_COMMANDS:
        _cancel_pending_movement_commands(
            commands,
            payload.device_id,
            f"Superseded by new movement command: {payload.command}",
        )

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

    # Wake the Jetson immediately over WebSocket so it doesn't wait for the 5s poll.
    await _ws_push(payload.device_id, {"notify": "command"})

    return {"ok": True, "queued": entry}


@router.get("/device/command/next")
def get_next_device_command(
    device_id: str,
    x_device_secret: str | None = Header(default=None, alias="X-Device-Secret"),
):
    _require_device_secret(x_device_secret)

    commands = _load_commands()
    next_cmd = _select_next_command(commands, device_id)
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

    # Signal the SSE stream generator that the answer is complete
    q = _stream_queues.get(payload.command_id)
    if q is not None:
        answer = (payload.result or {}).get("answer", "") if payload.result else ""
        event_type = "done" if payload.status == "completed" else "error"
        q.put({"type": event_type, "text": answer})

    return {
        "ok": True,
        "command": updated,
    }


@router.post("/device/command/partial")
def push_partial_result(
    payload: DeviceCommandPartialIn,
    x_device_secret: str | None = Header(default=None, alias="X-Device-Secret"),
):
    """Receive a sentence chunk from the Jetson during generation and forward it
    to any active SSE stream for this command."""
    _require_device_secret(x_device_secret)
    q = _stream_queues.get(payload.command_id)
    if q is not None:
        q.put({"type": "token", "text": payload.text})
    return {"ok": True}


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

    timeout = 180
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
                        "note": item.get("note"),
                        "result": result,
                    }

        time.sleep(0.3)

    raise HTTPException(status_code=504, detail="Jetson did not respond in time")


@router.post("/device/admin/chat/stream")
async def chat_via_jetson_stream(payload: DeviceCommandIn, request: Request):
    """SSE endpoint that streams sentence chunks as the Jetson generates them.

    Flow:
      1. Queue a chat_prompt command (same as /device/admin/chat).
      2. Create an in-memory SimpleQueue for this command_id.
      3. Jetson picks up the command, calls /device/command/partial per sentence.
      4. /device/command/partial puts each sentence in the queue.
      5. Jetson calls /device/command/ack when done → "done" event is enqueued.
      6. This generator dequeues events and sends them as SSE to the browser.

    SSE event format:
      data: {"text": "<sentence>"}   — partial sentence
      data: {"done": true, "answer": "<full answer>"}  — terminal event
      data: {"error": "<message>"}   — error terminal event
    """
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

    # Wake the Jetson immediately so it doesn't wait for the 5-second HTTP poll.
    await _ws_push(payload.device_id, {"notify": "command"})

    # Register queue before returning so ack_partial writes never miss
    q: _queue.SimpleQueue = _queue.SimpleQueue()
    _stream_queues[command_id] = q

    async def event_generator():
        try:
            deadline = time.time() + 180
            keepalive_ticks = 0
            while time.time() < deadline:
                try:
                    event = q.get_nowait()
                    if event["type"] == "token":
                        yield f"data: {json.dumps({'text': event['text']})}\n\n"
                    elif event["type"] == "done":
                        yield f"data: {json.dumps({'done': True, 'answer': event['text']})}\n\n"
                        return
                    elif event["type"] == "error":
                        yield f"data: {json.dumps({'error': event['text']})}\n\n"
                        return
                except _queue.Empty:
                    await asyncio.sleep(0.1)
                    keepalive_ticks += 1
                    if keepalive_ticks >= 30:  # every ~3 s
                        yield ": keepalive\n\n"
                        keepalive_ticks = 0
            yield f"data: {json.dumps({'error': 'Jetson did not respond in time'})}\n\n"
        finally:
            _stream_queues.pop(command_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/device/admin/flush_models")
def flush_jetson_models(payload: DeviceCommandIn, request: Request):
    """Queue a flush_models command to clear LLM + STT from Jetson VRAM/RAM.
    Accessible by admin and TA roles.
    """
    _require_admin_or_ta(request)

    commands = _load_commands()
    now_ms = int(time.time() * 1000)
    now_s = int(time.time())

    entry = {
        "id": f"{payload.device_id}-{now_ms}",
        "device_id": payload.device_id,
        "command": "flush_models",
        "payload": {},
        "created_at": now_s,
        "status": "pending",
    }

    commands.append(entry)
    _save_commands(commands)

    print(f"[DEVICE_COMMANDS] queued flush_models for {payload.device_id}")
    return {"ok": True, "queued": entry}


@router.websocket("/device/ws/{device_id}")
async def device_websocket(
    device_id: str,
    websocket: WebSocket,
    secret: str = Query(default=None),
):
    """Persistent WebSocket for cloud → Jetson command push.

    The Jetson connects once on startup and holds the connection open.
    When a command is queued via POST /device/admin/command, the cloud sends
    {"notify": "command"} so the Jetson fetches it immediately rather than
    waiting for the 5-second HTTP poll fallback.

    Authentication: ?secret=<DEVICE_SHARED_SECRET> query param (encrypted in TLS).
    """
    expected = os.getenv("DEVICE_SHARED_SECRET", "").strip()
    if not expected or (secret or "").strip() != expected:
        await websocket.close(code=1008)  # Policy Violation
        return

    await websocket.accept()
    _ws_connections[device_id] = websocket
    print(f"[WS] Jetson '{device_id}' connected via WebSocket command channel")

    try:
        # Keep-alive: the Jetson sends periodic pings; websockets library handles
        # them automatically. We just need to keep recv() running to detect disconnect.
        while True:
            await asyncio.wait_for(websocket.receive_text(), timeout=90.0)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception:
        pass
    finally:
        if _ws_connections.get(device_id) is websocket:
            _ws_connections.pop(device_id, None)
        print(f"[WS] Jetson '{device_id}' disconnected from WebSocket command channel")


@router.post("/device/admin/reload_llm")
def reload_jetson_llm(payload: DeviceCommandIn, request: Request):
    """Queue a reload_llm command: unloads the LLM then reloads it onto the GPU.
    Use this when Ollama silently fell back to CPU inference.
    Accessible by admin and TA roles.
    """
    _require_admin_or_ta(request)

    commands = _load_commands()
    now_ms = int(time.time() * 1000)
    now_s = int(time.time())

    entry = {
        "id": f"{payload.device_id}-{now_ms}",
        "device_id": payload.device_id,
        "command": "reload_llm",
        "payload": {},
        "created_at": now_s,
        "status": "pending",
    }

    commands.append(entry)
    _save_commands(commands)

    print(f"[DEVICE_COMMANDS] queued reload_llm for {payload.device_id}")
    return {"ok": True, "queued": entry}