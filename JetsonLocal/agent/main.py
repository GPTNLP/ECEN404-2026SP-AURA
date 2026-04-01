import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Iterator, Optional

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
from stt_faster import (
    SpeechToText,
    detect_last_movement_command,
    remove_wake_phrase,
    censor_text,
    contains_bad_language,
)

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
stt_service = None

runtime_config = {
    "poll_seconds": 0.05,
    "heartbeat_seconds": HEARTBEAT_SECONDS,
    "status_seconds": STATUS_SECONDS,
}

CAMERA_UPLOAD_TARGET_FPS = 10.0
CAMERA_UPLOAD_MIN_INTERVAL = 1.0 / CAMERA_UPLOAD_TARGET_FPS
camera_upload_lock = asyncio.Lock()
camera_upload_latest_jpeg: Optional[bytes] = None
camera_upload_latest_mode = "raw"
camera_upload_latest_seq = 0
camera_upload_sent_seq = -1
camera_upload_last_send_ts = 0.0

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


def init_stt():
    global stt_service
    try:
        stt_service = SpeechToText(
            model_size="base",
            input_device=4,
            device_sample_rate=48000,
            target_sample_rate=16000,
            channels=2,
            device="cpu",
            compute_type="int8",
            language="en",
            task="transcribe",
        )
        print("[STT] initialized")
        send_or_queue_log("info", "stt_initialized", "Speech-to-text initialized")
    except Exception as e:
        stt_service = None
        print(f"[STT] init failed: {e}")
        send_or_queue_log("warning", "stt_init_failed", f"STT init failed: {e}")


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

        await ui_manager.broadcast(
            {
                "type": "telemetry",
                "cpu_percent": payload.get("cpu_percent"),
                "gpu_percent": (payload.get("extra") or {}).get("gpu_percent"),
                "db_name": (payload.get("extra") or {}).get("db_name", LOCAL_DB_NAME),
                "connection": connection_text,
            }
        )

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


async def camera_upload_loop():
    global camera_upload_latest_jpeg
    global camera_upload_latest_mode
    global camera_upload_latest_seq

    while True:
        try:
            status = camera_service.get_status()
            if status.get("enabled") and status.get("running"):
                jpeg = camera_service.get_jpeg()
                if jpeg:
                    mode = camera_service.get_mode()
                    async with camera_upload_lock:
                        camera_upload_latest_jpeg = jpeg
                        camera_upload_latest_mode = mode
                        camera_upload_latest_seq += 1
        except Exception as e:
            print(f"[CAMERA_CAPTURE] failed: {e}")

        await asyncio.sleep(CAMERA_UPLOAD_MIN_INTERVAL)


async def camera_upload_sender_loop():
    global camera_upload_sent_seq
    global camera_upload_last_send_ts

    while True:
        seq_to_send = None
        mode_to_send = None
        jpeg_to_send = None

        try:
            async with camera_upload_lock:
                if camera_upload_latest_seq != camera_upload_sent_seq and camera_upload_latest_jpeg:
                    seq_to_send = camera_upload_latest_seq
                    mode_to_send = camera_upload_latest_mode
                    jpeg_to_send = camera_upload_latest_jpeg

            if seq_to_send is not None and jpeg_to_send is not None:
                now = time.time()
                since_last = now - camera_upload_last_send_ts
                if since_last < CAMERA_UPLOAD_MIN_INTERVAL:
                    await asyncio.sleep(CAMERA_UPLOAD_MIN_INTERVAL - since_last)

                started = time.time()
                await asyncio.to_thread(api.upload_camera_frame, DEVICE_ID, mode_to_send, jpeg_to_send)
                elapsed = time.time() - started

                camera_upload_last_send_ts = time.time()
                camera_upload_sent_seq = seq_to_send

                print(
                    f"[CAMERA_UPLOAD] sent seq={seq_to_send} "
                    f"bytes={len(jpeg_to_send)} mode={mode_to_send} dt={elapsed:.3f}s"
                )
            else:
                await asyncio.sleep(0.02)

        except Exception as e:
            print(f"[CAMERA_UPLOAD] failed: {e}")
            await asyncio.sleep(0.10)


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
                                await asyncio.to_thread(
                                    api.ack_command,
                                    {
                                        "command_id": command_id,
                                        "device_id": DEVICE_ID,
                                        "status": "completed",
                                        "note": f"Duplicate command skipped: {cmd}",
                                    },
                                )
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

                                await asyncio.to_thread(
                                    api.ack_command,
                                    {
                                        "command_id": command_id,
                                        "device_id": DEVICE_ID,
                                        "status": "completed",
                                        "note": f"Sent to ESP as {serial_msg.strip()}",
                                    },
                                )

                                send_or_queue_log(
                                    "info",
                                    "device_command_executed",
                                    f"Executed command: {cmd}",
                                    {"command_id": command_id, "serial_message": serial_msg.strip()},
                                )
                        except Exception as e:
                            print(f"[COMMAND] serial send failed: {e}")
                            await asyncio.to_thread(
                                api.ack_command,
                                {
                                    "command_id": command_id,
                                    "device_id": DEVICE_ID,
                                    "status": "failed",
                                    "note": f"Serial send failed: {e}",
                                },
                            )
                    else:
                        print("[COMMAND] ESP serial not connected")
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": "ESP serial is not connected",
                            },
                        )

                elif cmd == "camera_activate_raw":
                    try:
                        camera_service.activate("raw")
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "note": "Camera activated in raw mode",
                            },
                        )
                    except Exception as e:
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"Camera raw activation failed: {e}",
                            },
                        )

                elif cmd == "camera_activate_detection":
                    try:
                        camera_service.activate("detection")
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "note": "Camera activated in detection mode",
                            },
                        )
                    except Exception as e:
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"Camera detection activation failed: {e}",
                            },
                        )

                elif cmd == "camera_deactivate":
                    try:
                        camera_service.deactivate()
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "note": "Camera deactivated",
                            },
                        )
                    except Exception as e:
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"Camera deactivation failed: {e}",
                            },
                        )

                elif cmd == "chat_prompt":
                    query = payload.get("query", "")
                    if rag_system and query:
                        try:
                            res = await rag_system.aquery(query)
                            ai_reply = res.get("answer", "No answer found.")
                            await asyncio.to_thread(
                                api.ack_command,
                                {
                                    "command_id": command_id,
                                    "device_id": DEVICE_ID,
                                    "status": "completed",
                                    "result": {"answer": ai_reply, "sources": res.get("sources", [])},
                                },
                            )
                        except Exception as e:
                            await asyncio.to_thread(
                                api.ack_command,
                                {
                                    "command_id": command_id,
                                    "device_id": DEVICE_ID,
                                    "status": "failed",
                                    "note": str(e),
                                },
                            )
                    else:
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": "RAG system offline or empty query",
                            },
                        )

                else:
                    print(f"[COMMAND] invalid command ignored: {cmd}")
                    if command_id:
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"Invalid command: {cmd}",
                            },
                        )

        except Exception as e:
            print(f"[COMMAND] poll failed: {e}")

        await asyncio.sleep(runtime_config["poll_seconds"])


async def execute_user_request(user_msg: str, source: str = "websocket"):
    clean_msg = (user_msg or "").strip()
    if not clean_msg:
        return

    await ui_manager.broadcast({"type": "chat", "sender": "user", "text": clean_msg})
    await ui_manager.broadcast({"type": "status", "data": "Processing..."})

    movement_cmd = detect_last_movement_command(clean_msg)

    if movement_cmd:
        if esp_serial:
            try:
                ack = send_serial_command(movement_cmd)
                ai_reply = f"Movement command sent: {movement_cmd}. {ack}"

                send_or_queue_log(
                    "info",
                    "voice_or_text_movement",
                    f"Executed movement command from {source}: {movement_cmd}",
                    {"source": source, "raw_text": clean_msg, "command": movement_cmd},
                )
            except Exception as e:
                ai_reply = f"Movement failed: {e}"
                send_or_queue_log(
                    "warning",
                    "voice_or_text_movement_failed",
                    f"Movement failed from {source}: {e}",
                    {"source": source, "raw_text": clean_msg, "command": movement_cmd},
                )
        else:
            ai_reply = "ESP serial is not connected."
    else:
        if rag_system:
            try:
                res = await rag_system.aquery(clean_msg)
                ai_reply = res.get("answer", "No answer found.")

                send_or_queue_log(
                    "info",
                    "voice_or_text_rag_query",
                    f"Handled RAG query from {source}",
                    {"source": source, "query": clean_msg},
                )
            except Exception as e:
                ai_reply = f"RAG query failed: {e}"
                send_or_queue_log(
                    "warning",
                    "voice_or_text_rag_failed",
                    f"RAG query failed from {source}: {e}",
                    {"source": source, "query": clean_msg},
                )
        else:
            ai_reply = "RAG Database offline."

    await ui_manager.broadcast({"type": "chat", "sender": "ai", "text": ai_reply})
    await ui_manager.broadcast({"type": "status", "data": "Ready"})


async def handle_user_message(user_msg: str):
    await execute_user_request(user_msg, source="websocket")


async def voice_loop():
    global stt_service

    while True:
        try:
            if stt_service is None:
                await asyncio.sleep(2.0)
                continue

            woke, wake_text, leftover, reason = await asyncio.to_thread(
                stt_service.listen_for_wake_word
            )

            if not woke:
                await asyncio.sleep(0.05)
                continue

            print(f"[VOICE] wake detected: reason={reason} wake_text={wake_text!r}")
            await ui_manager.broadcast({"type": "status", "data": "Listening..."})

            if leftover:
                print(f"[VOICE] immediate speech after wake: {leftover}")
                final_text = leftover.strip()
            else:
                final_text = await asyncio.to_thread(stt_service.listen_until_done)
                final_text = (final_text or "").strip()

            if not final_text:
                print("[VOICE] no speech heard after wake")
                await ui_manager.broadcast({"type": "status", "data": "Ready"})
                await asyncio.sleep(0.1)
                continue

            final_text = remove_wake_phrase(final_text).strip()

            if not final_text:
                print("[VOICE] only wake phrase heard, no command/query")
                await ui_manager.broadcast({"type": "status", "data": "Ready"})
                await asyncio.sleep(0.1)
                continue

            try:
                stt_service.log_transcript(censor_text(final_text))
            except Exception as e:
                print(f"[VOICE] transcript log failed: {e}")

            if contains_bad_language(final_text):
                print("[VOICE] bad language detected in transcript")

            print(f"[VOICE] final_text={final_text!r}")
            await execute_user_request(final_text, source="voice")

        except Exception as e:
            print(f"[VOICE] loop failed: {e}")
            send_or_queue_log("warning", "voice_loop_failed", f"Voice loop failed: {e}")
            await ui_manager.broadcast({"type": "status", "data": "Ready"})
            await asyncio.sleep(1.0)


@app.on_event("startup")
async def startup_event():
    print("[STARTUP] initializing hardware")
    init_hardware()

    print("[STARTUP] initializing rag")
    init_rag()

    print("[STARTUP] initializing stt")
    init_stt()

    print("[STARTUP] camera service idle until activated")
    send_or_queue_log("info", "camera_idle", "Camera service is idle until activated")

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
    asyncio.create_task(camera_upload_loop())
    asyncio.create_task(camera_upload_sender_loop())
    asyncio.create_task(voice_loop())

    print("[STARTUP] background loops started")
    send_or_queue_log("info", "startup_complete", "Jetson agent startup complete")


@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"ok": True, "service": "AURA Edge API", "device_id": DEVICE_ID}


@app.get("/health")
async def health():
    return {
        "ok": True,
        "device_id": DEVICE_ID,
        "rag_online": rag_system is not None,
        "serial_connected": bool(esp_serial is not None),
        "camera": get_camera_status(),
        "stt_online": stt_service is not None,
    }


@app.get("/camera/status")
async def camera_status():
    return get_camera_status()


@app.post("/camera/activate")
async def camera_activate(mode: str = Query("raw")):
    if mode not in {"raw", "detection"}:
        raise HTTPException(status_code=400, detail="mode must be 'raw' or 'detection'")
    camera_service.activate(mode)
    return {"ok": True, "mode": mode}


@app.post("/camera/deactivate")
async def camera_deactivate():
    camera_service.deactivate()
    return {"ok": True}


def mjpeg_frame_generator() -> Iterator[bytes]:
    boundary = b"--frame\r\n"
    content_type = b"Content-Type: image/jpeg\r\n\r\n"

    while True:
        jpeg = camera_service.get_jpeg()
        if jpeg:
            yield boundary + content_type + jpeg + b"\r\n"
        time.sleep(0.03)


@app.get("/camera/stream")
async def camera_stream():
    if not camera_service.get_status().get("running"):
        camera_service.activate(camera_service.get_mode() or "raw")
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/camera/snapshot")
async def camera_snapshot():
    jpeg = camera_service.get_jpeg()
    if not jpeg:
        raise HTTPException(status_code=404, detail="No frame available")
    return StreamingResponse(iter([jpeg]), media_type="image/jpeg")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ui_manager.connect(websocket)
    await websocket.send_json({"type": "status", "data": "Connected"})
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "chat":
                user_msg = (data.get("text") or "").strip()
                await handle_user_message(user_msg)

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "camera_activate":
                mode = (data.get("mode") or "raw").strip().lower()
                if mode not in {"raw", "detection"}:
                    mode = "raw"
                camera_service.activate(mode)
                await websocket.send_json({"type": "camera_status", "data": get_camera_status()})

            elif msg_type == "camera_deactivate":
                camera_service.deactivate()
                await websocket.send_json({"type": "camera_status", "data": get_camera_status()})

            elif msg_type == "movement":
                cmd = (data.get("command") or "").strip().lower()
                if cmd not in MOVEMENT_COMMANDS:
                    await websocket.send_json({"type": "error", "data": f"Invalid movement command: {cmd}"})
                    continue

                try:
                    ack = send_serial_command(cmd)
                    await websocket.send_json(
                        {
                            "type": "movement_result",
                            "ok": True,
                            "command": cmd,
                            "ack": ack,
                        }
                    )
                except Exception as e:
                    await websocket.send_json(
                        {
                            "type": "movement_result",
                            "ok": False,
                            "command": cmd,
                            "error": str(e),
                        }
                    )

            else:
                await websocket.send_json({"type": "error", "data": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        ui_manager.disconnect(websocket)
    except Exception:
        ui_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )