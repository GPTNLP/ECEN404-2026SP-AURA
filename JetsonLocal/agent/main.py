import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader

import serial
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
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
from camera import camera_service, get_camera_status

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
    "poll_seconds": 0.05,
    "heartbeat_seconds": HEARTBEAT_SECONDS,
    "status_seconds": STATUS_SECONDS,
}

MOVEMENT_COMMANDS = {"forward", "backward", "left", "right", "stop"}


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
        time.sleep(2.0)
        try:
            esp_serial.reset_input_buffer()
            esp_serial.reset_output_buffer()
        except Exception:
            pass
        print(f"[SERIAL] connected to {SERIAL_PORT}")
        send_or_queue_log("info", "serial_connected", f"Connected to serial port {SERIAL_PORT}")
    except Exception as e:
        esp_serial = None
        print(f"[SERIAL] unavailable: {SERIAL_PORT} | error: {e}")
        send_or_queue_log("warning", "serial_unavailable", f"Serial port unavailable: {SERIAL_PORT} | error: {e}")


def ensure_serial():
    global esp_serial

    try:
        if esp_serial is not None and esp_serial.is_open:
            return True
    except Exception:
        pass

    try:
        esp_serial = serial.Serial(SERIAL_PORT, 115200, timeout=1)
        time.sleep(2.0)
        try:
            esp_serial.reset_input_buffer()
            esp_serial.reset_output_buffer()
        except Exception:
            pass
        print(f"[SERIAL] reconnected to {SERIAL_PORT}")
        send_or_queue_log("info", "serial_reconnected", f"Reconnected to serial port {SERIAL_PORT}")
        return True
    except Exception as e:
        esp_serial = None
        print(f"[SERIAL] reconnect failed: {e}")
        return False


def build_serial_message(cmd: str) -> str:
    cmd = (cmd or "").strip().lower()
    if cmd in MOVEMENT_COMMANDS:
        return f"MOVE:{cmd}\n"
    raise ValueError(f"Unsupported serial movement command: {cmd}")


def send_serial_command(cmd: str) -> str:
    global esp_serial

    if not ensure_serial():
        raise RuntimeError("ESP serial is not connected")

    serial_msg = build_serial_message(cmd)
    expected_ack = f"ACK:MOVE:{cmd}"

    try:
        esp_serial.reset_input_buffer()
    except Exception:
        pass

    esp_serial.write(serial_msg.encode("utf-8"))
    esp_serial.flush()
    print(f"[SERIAL_TX] {serial_msg.strip()}")

    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            line = esp_serial.readline().decode("utf-8", errors="ignore").strip()
        except Exception as e:
            raise RuntimeError(f"Serial read failed: {e}")

        if not line:
            continue

        print(f"[SERIAL_RX] {line}")

        if line.startswith("ERR:"):
            raise RuntimeError(line)

        if line == expected_ack:
            return line

    raise RuntimeError(f"No ACK from ESP for {cmd}")


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
        print(f"[RAG] initialized at {db_path}")
        send_or_queue_log("info", "rag_initialized", f"RAG initialized at {db_path}")
    except Exception as e:
        rag_system = None
        print(f"[RAG] init failed: {e}")
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


def _read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        return "\n\n".join(parts)
    except Exception as e:
        print(f"[PDF_READER] Error reading {path}: {e}")
        return ""


async def register_device():
    payload = build_register_payload()
    print(f"[REGISTER] sending: {payload}")
    result = await asyncio.to_thread(api.register, payload)
    print(f"[REGISTER] success: {result}")
    send_or_queue_log("info", "device_registered", "Device registered with backend", result)
    return result


async def refresh_config():
    global runtime_config
    try:
        result = await asyncio.to_thread(api.get_config, DEVICE_ID)
        runtime_config["poll_seconds"] = float(result.get("poll_seconds", runtime_config["poll_seconds"]))
        runtime_config["heartbeat_seconds"] = int(result.get("heartbeat_seconds", runtime_config["heartbeat_seconds"]))
        runtime_config["status_seconds"] = int(result.get("status_seconds", runtime_config["status_seconds"]))
        print(f"[CONFIG] success: {result}")
    except Exception as e:
        print(f"[CONFIG] failed: {e}")
        send_or_queue_log("warning", "config_refresh_failed", f"Failed to refresh device config: {e}")


async def heartbeat_loop():
    while True:
        try:
            payload = build_heartbeat_payload()
            await asyncio.to_thread(api.heartbeat, payload)
            print(f"[HEARTBEAT] sent for {payload.get('device_id')}")
        except Exception as e:
            print(f"[HEARTBEAT] failed: {e}")
            send_or_queue_log("warning", "heartbeat_failed", f"Heartbeat failed: {e}")
        await asyncio.sleep(runtime_config["heartbeat_seconds"])


async def status_loop():
    while True:
        payload = build_status_payload()

        connection_text = "Connected to website"
        try:
            await asyncio.to_thread(api.status, payload)
            print(
                f"[STATUS] sent cpu={payload.get('cpu_percent')} "
                f"gpu={(payload.get('extra') or {}).get('gpu_percent')}"
            )
        except Exception as e:
            queue_status(payload)
            print(f"[STATUS] failed: {e}")
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
            print("[FLUSH] attempted queued log/status flush")
        except Exception as e:
            print(f"[FLUSH] failed: {e}")
        await asyncio.sleep(OFFLINE_RETRY_SECONDS)


async def config_loop():
    while True:
        await refresh_config()
        await asyncio.sleep(CONFIG_REFRESH_SECONDS)


async def command_loop():
    last_cmd = None
    last_cmd_time = 0.0

    while True:
        try:
            result = await asyncio.to_thread(api.get_next_command, DEVICE_ID)
            command = result.get("command")

            if command:
                command_id = command.get("id")
                cmd = (command.get("command") or "").strip().lower()
                payload = command.get("payload") or {}
                now = asyncio.get_event_loop().time()

                print(f"[COMMAND] received: {cmd}")

                if cmd in {"forward", "backward", "left", "right", "pitch", "yaw", "stop"}:
                    if esp_serial:
                        try:
                            if cmd == last_cmd and (now - last_cmd_time) < 0.04:
                                await asyncio.to_thread(api.ack_command, {
                                    "command_id": command_id,
                                    "device_id": DEVICE_ID,
                                    "status": "completed",
                                    "note": f"Duplicate command skipped: {cmd}",
                                })
                            else:
                                val = payload.get("value", "")

                                if cmd in {"forward", "backward", "left", "right", "stop"}:
                                    serial_msg = f"MOVE:{cmd}\n"
                                else:
                                    serial_msg = f"MOVE:{cmd}:{val}\n"

                                esp_serial.write(serial_msg.encode("utf-8"))
                                esp_serial.flush()
                                print(f"[COMMAND] sent to ESP: {serial_msg.strip()}")

                                last_cmd = cmd
                                last_cmd_time = now

                                await asyncio.to_thread(api.ack_command, {
                                    "command_id": command_id,
                                    "device_id": DEVICE_ID,
                                    "status": "completed",
                                    "note": f"Sent to ESP as {serial_msg.strip()}",
                                })

                                send_or_queue_log(
                                    "info",
                                    "device_command_executed",
                                    f"Executed command: {cmd}",
                                    {"command_id": command_id, "serial_message": serial_msg.strip()},
                                )

                        except Exception as e:
                            print(f"[COMMAND] serial send failed: {e}")
                            await asyncio.to_thread(api.ack_command, {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"Serial send failed: {e}",
                            })
                    else:
                        print("[COMMAND] ESP serial not connected")
                        await asyncio.to_thread(api.ack_command, {
                            "command_id": command_id,
                            "device_id": DEVICE_ID,
                            "status": "failed",
                            "note": "ESP serial is not connected",
                        })

                elif cmd == "chat_prompt":
                    query = payload.get("query", "")
                    if rag_system and query:
                        try:
                            res = await rag_system.aquery(query)
                            ai_reply = res.get("answer", "No answer found.")

                            await asyncio.to_thread(api.ack_command, {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "result": {"answer": ai_reply, "sources": res.get("sources", [])}
                            })
                        except Exception as e:
                            await asyncio.to_thread(api.ack_command, {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": str(e)
                            })
                    else:
                        await asyncio.to_thread(api.ack_command, {
                            "command_id": command_id,
                            "device_id": DEVICE_ID,
                            "status": "failed",
                            "note": "RAG system offline or empty query"
                        })

                else:
                    print(f"[COMMAND] invalid command ignored: {cmd}")
                    if command_id:
                        await asyncio.to_thread(api.ack_command, {
                            "command_id": command_id,
                            "device_id": DEVICE_ID,
                            "status": "failed",
                            "note": f"Invalid command: {cmd}",
                        })

        except Exception as e:
            print(f"[COMMAND] poll failed: {e}")

        await asyncio.sleep(runtime_config["poll_seconds"])


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
        cmd = user_msg.lower().strip()
        if cmd not in MOVEMENT_COMMANDS:
            ai_reply = f"Unsupported movement command: {cmd}"
        else:
            try:
                await asyncio.to_thread(send_serial_command, cmd)
                ai_reply = f"Movement command '{cmd}' sent to ESP."
            except Exception as e:
                ai_reply = f"Failed to send movement command to ESP: {e}"
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


def mjpeg_generator(mode: str) -> Iterator[bytes]:
    while True:
        frame = camera_service.get_jpeg(mode)
        if frame is None:
            time.sleep(0.03)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.on_event("startup")
async def startup_event():
    print("[STARTUP] initializing hardware")
    init_hardware()

    print("[STARTUP] initializing rag")
    init_rag()

    print("[STARTUP] initializing camera")
    try:
        camera_service.start()
        print("[CAMERA] started")
        send_or_queue_log("info", "camera_started", "Camera service started")
    except Exception as e:
        print(f"[CAMERA] failed to start: {e}")
        send_or_queue_log("warning", "camera_start_failed", f"Camera failed to start: {e}")

    try:
        await register_device()
    except Exception as e:
        print(f"[REGISTER] initial register failed: {e}")
        send_or_queue_log("warning", "register_failed", f"Initial register failed: {e}")

    asyncio.create_task(config_loop())
    asyncio.create_task(heartbeat_loop())
    asyncio.create_task(status_loop())
    asyncio.create_task(flush_loop())
    asyncio.create_task(command_loop())

    print("[STARTUP] background loops started")
    send_or_queue_log("info", "startup_complete", "Jetson agent startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    try:
        camera_service.stop()
        print("[CAMERA] stopped")
    except Exception as e:
        print(f"[CAMERA] stop error: {e}")


@app.get("/")
async def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/camera/status")
async def camera_status():
    return get_camera_status()


@app.get("/camera/detections")
async def camera_detections():
    return {
        "mode": camera_service.get_mode(),
        "detections": camera_service.get_detections(),
    }


@app.post("/camera/mode")
async def set_camera_mode(mode: str = Query(..., pattern="^(raw|detection)$")):
    camera_service.set_mode(mode)
    return {"ok": True, "mode": camera_service.get_mode()}


@app.get("/camera/stream")
async def camera_stream():
    status = camera_service.get_status()
    if not status["camera_ready"]:
        raise HTTPException(status_code=503, detail="Camera not ready")

    return StreamingResponse(
        mjpeg_generator("raw"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/camera/detection-stream")
async def camera_detection_stream():
    status = camera_service.get_status()
    if not status["camera_ready"]:
        raise HTTPException(status_code=503, detail="Camera not ready")

    return StreamingResponse(
        mjpeg_generator("detection"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ui_manager.connect(websocket)
    print("[WS] client connected")
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
        print("[WS] client disconnected")
        ui_manager.disconnect(websocket)
    except Exception as e:
        print(f"[WS] error: {e}")
        ui_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)