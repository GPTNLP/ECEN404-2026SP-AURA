import os
import sys
import asyncio
from pathlib import Path

import serial
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
JETSONLOCAL_DIR = BASE_DIR.parent
if str(JETSONLOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(JETSONLOCAL_DIR))

from config import (
    STATIC_DIR,
    STORAGE_DIR,
    SERIAL_PORT,
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    LOCAL_DB_NAME,
    DEVICE_ID,
    DEVICE_NAME,
    DEVICE_TYPE,
    DEVICE_SOFTWARE_VERSION,
    HEARTBEAT_SECONDS,
    STATUS_SECONDS,
    CONFIG_REFRESH_SECONDS,
    OFFLINE_RETRY_SECONDS,
)
from api_client import ApiClient
from logger import write_local_log
from offline_queue import queue_log, queue_status, flush_logs, flush_statuses
from heartbeat import build_heartbeat_payload
from status import build_status_payload
from device_info import collect_device_info

from lightrag_local import LightRAG, OllamaClient

app = FastAPI(title="AURA Edge API (Jetson Orin Nano)")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)


ui_manager = ConnectionManager()
api = ApiClient()
rag_system = None
esp_serial = None
runtime_config = {
    "poll_seconds": 5,
    "heartbeat_seconds": HEARTBEAT_SECONDS,
    "status_seconds": STATUS_SECONDS,
}


def send_or_queue_log(level: str, event: str, message: str, meta=None):
    entry = write_local_log(level, event, message, meta)
    payload = {
        "device_id": entry["device_id"],
        "level": entry["level"],
        "event": entry["event"],
        "message": entry["message"],
        "meta": entry["meta"],
    }
    try:
        api.log(payload)
    except Exception:
        queue_log(payload)


def init_hardware():
    global esp_serial
    try:
        esp_serial = serial.Serial(SERIAL_PORT, 115200, timeout=1)
        send_or_queue_log("info", "serial_connected", f"Connected to serial port {SERIAL_PORT}")
    except serial.SerialException:
        esp_serial = None
        send_or_queue_log("warning", "serial_unavailable", f"Serial port unavailable: {SERIAL_PORT}")


def init_rag():
    global rag_system
    db_path = os.path.join(str(STORAGE_DIR), LOCAL_DB_NAME)
    os.makedirs(db_path, exist_ok=True)
    try:
        rag_system = LightRAG(
            working_dir=db_path,
            llm_model_name=DEFAULT_MODEL,
            embed_model_name=EMBEDDING_MODEL,
        )
        send_or_queue_log("info", "rag_initialized", f"RAG initialized at {db_path}")
    except Exception as e:
        rag_system = None
        send_or_queue_log("warning", "rag_init_failed", f"RAG init failed: {e}")


def build_register_payload():
    info = collect_device_info()
    return {
        "device_id": DEVICE_ID,
        "device_name": DEVICE_NAME,
        "device_type": DEVICE_TYPE,
        "software_version": DEVICE_SOFTWARE_VERSION,
        "hostname": info["hostname"],
        "local_ip": info["local_ip"],
    }


async def register_device():
    payload = build_register_payload()
    result = await asyncio.to_thread(api.register, payload)
    send_or_queue_log("info", "device_registered", "Device registered with backend", result)
    return result


async def refresh_config():
    global runtime_config
    try:
        result = await asyncio.to_thread(api.get_config, DEVICE_ID)
        runtime_config["poll_seconds"] = int(result.get("poll_seconds", runtime_config["poll_seconds"]))
        runtime_config["heartbeat_seconds"] = int(result.get("heartbeat_seconds", runtime_config["heartbeat_seconds"]))
        runtime_config["status_seconds"] = int(result.get("status_seconds", runtime_config["status_seconds"]))
    except Exception as e:
        send_or_queue_log("warning", "config_refresh_failed", f"Failed to refresh device config: {e}")


async def heartbeat_loop():
    while True:
        try:
            payload = build_heartbeat_payload()
            await asyncio.to_thread(api.heartbeat, payload)
        except Exception as e:
            send_or_queue_log("warning", "heartbeat_failed", f"Heartbeat failed: {e}")
        await asyncio.sleep(runtime_config["heartbeat_seconds"])


async def status_loop():
    while True:
        payload = build_status_payload()

        connection_text = "Connected to website"
        try:
            await asyncio.to_thread(api.status, payload)
        except Exception as e:
            queue_status(payload)
            send_or_queue_log("warning", "status_failed", f"Status upload failed: {e}")
            connection_text = "Robot only / website offline"

        await ui_manager.broadcast({
            "type": "telemetry",
            "cpu_percent": payload.get("cpu_percent"),
            "gpu_percent": (payload.get("extra") or {}).get("gpu_percent"),
            "db_name": (payload.get("extra") or {}).get("db_name", LOCAL_DB_NAME),
            "connection": connection_text,
        })

        await asyncio.sleep(runtime_config["status_seconds"])


async def flush_loop():
    while True:
        try:
            await asyncio.to_thread(flush_logs, api.log)
            await asyncio.to_thread(flush_statuses, api.status)
        except Exception:
            pass
        await asyncio.sleep(OFFLINE_RETRY_SECONDS)


async def config_loop():
    while True:
        await refresh_config()
        await asyncio.sleep(CONFIG_REFRESH_SECONDS)


async def parse_intent(user_msg: str):
    client = OllamaClient("http://127.0.0.1:11434", EMBEDDING_MODEL, DEFAULT_MODEL)
    system = "Classify user input as 'MOVEMENT' or 'QUESTION'. Reply with one word."
    prompt = f"Input: '{user_msg}'"
    try:
        res = await client.generate(prompt, system=system, timeout_s=5.0)
        return "MOVEMENT" if "MOVEMENT" in res.upper() else "QUESTION"
    except Exception:
        return "QUESTION"


async def handle_user_message(user_msg: str):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return

    await ui_manager.broadcast({"type": "chat", "sender": "user", "text": user_msg})
    await ui_manager.broadcast({"type": "status", "data": "Processing..."})

    intent = await parse_intent(user_msg)

    if intent == "MOVEMENT":
        if esp_serial:
            try:
                esp_serial.write(f"MOVE:{user_msg}\n".encode("utf-8"))
                ai_reply = "Movement command routed to ESP rotors."
            except Exception:
                ai_reply = "Failed to send movement command to ESP."
        else:
            ai_reply = "ESP serial is not connected."
    else:
        if rag_system:
            try:
                res = await rag_system.aquery(user_msg)
                ai_reply = res.get("answer", "No answer found.")
            except Exception as e:
                ai_reply = f"RAG query failed: {e}"
        else:
            ai_reply = "RAG database offline."

    send_or_queue_log("info", "assistant_interaction", "Assistant handled user message", {
        "user_message": user_msg,
        "reply_preview": ai_reply[:200],
    })

    await ui_manager.broadcast({"type": "chat", "sender": "ai", "text": ai_reply})
    await ui_manager.broadcast({"type": "status", "data": "Ready"})


@app.on_event("startup")
async def startup_event():
    init_hardware()
    init_rag()

    try:
        await register_device()
    except Exception as e:
        send_or_queue_log("warning", "register_failed", f"Initial register failed: {e}")

    asyncio.create_task(config_loop())
    asyncio.create_task(heartbeat_loop())
    asyncio.create_task(status_loop())
    asyncio.create_task(flush_loop())

    send_or_queue_log("info", "startup_complete", "Jetson agent startup complete")


@app.get("/")
async def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ui_manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "status",
            "data": "Ready"
        })
        await websocket.send_json({
            "type": "telemetry",
            "cpu_percent": None,
            "gpu_percent": None,
            "db_name": LOCAL_DB_NAME,
            "connection": "Connected to robot"
        })

        while True:
            raw = await websocket.receive_text()
            msg = raw.strip()
            if not msg:
                continue
            await handle_user_message(msg)
    except WebSocketDisconnect:
        ui_manager.disconnect(websocket)
    except Exception:
        ui_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)