import sys
import time
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Iterator, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
import uvicorn
import requests

# -------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
JETSONLOCAL_DIR = BASE_DIR.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if str(JETSONLOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(JETSONLOCAL_DIR))

# -------------------------------------------------------------------
# IMPORTS - MATCHED TO CURRENT REORG
# -------------------------------------------------------------------
from core.config import (
    DEVICE_ID,
    DEVICE_NAME,
    DEVICE_TYPE,
    DEVICE_SOFTWARE_VERSION,
    DEVICE_SHARED_SECRET,
    API_BASE_URL,
    HEARTBEAT_SECONDS,
    STATUS_SECONDS,
    CONFIG_REFRESH_SECONDS,
    OFFLINE_RETRY_SECONDS,
    LOCAL_DB_NAME,
)

from cloud.api_client import ApiClient
from cloud.heartbeat import build_heartbeat_payload
from cloud.status import build_status_payload

from core.logger import write_local_log
from core.offline_queue import queue_log, queue_status, flush_logs, flush_statuses

from hardware.serial_link import serial_link
from hardware.camera import camera_service, get_camera_status

from stt_faster import STTService

# -------------------------------------------------------------------
# APP
# -------------------------------------------------------------------
app = FastAPI(title="AURA Jetson Agent")
api = ApiClient()

runtime_config = {
    "poll_seconds": 0.10,
    "heartbeat_seconds": int(HEARTBEAT_SECONDS),
    "status_seconds": int(STATUS_SECONDS),
}

MOVEMENT_COMMANDS = {"forward", "backward", "left", "right", "stop"}

_last_messages = {}
_last_uploaded_signature: Optional[str] = None

stt_service: Optional[STTService] = None
stt_task: Optional[asyncio.Task] = None
voice_enabled = True


def quiet_print(key: str, message: str) -> None:
    if _last_messages.get(key) != message:
        print(message)
        _last_messages[key] = message


# -------------------------------------------------------------------
# LOGGING HELPERS
# -------------------------------------------------------------------
def send_or_queue_log(level: str, event: str, message: str, meta=None):
    entry = write_local_log(level, event, message, meta)
    payload = {
        "device_id": entry.get("device_id"),
        "level": entry.get("level"),
        "event": entry.get("event"),
        "message": entry.get("message"),
        "meta": entry.get("meta"),
    }

    try:
        api.log(payload)
    except Exception:
        queue_log(payload)


# -------------------------------------------------------------------
# VOICE HELPERS
# -------------------------------------------------------------------
async def handle_voice_text(text: str, intent: str, movement: Optional[str]) -> None:
    if intent == "movement" and movement in MOVEMENT_COMMANDS:
        try:
            serial_link.send_command(movement, "")
            send_or_queue_log(
                "info",
                "voice_movement_command",
                f"Voice movement command executed: {movement}",
                {"text": text, "movement": movement},
            )
            quiet_print("voice_action", f"[VOICE] movement -> {movement}")
        except Exception as e:
            send_or_queue_log(
                "warning",
                "voice_movement_failed",
                f"Voice movement command failed: {e}",
                {"text": text, "movement": movement},
            )
            quiet_print("voice_action", f"[VOICE] movement failed -> {movement}: {e}")
        return

    if intent == "llm":
        send_or_queue_log(
            "info",
            "voice_llm_query",
            "Voice query captured",
            {"text": text},
        )
        quiet_print("voice_action", f"[VOICE] llm -> {text}")
        return

    send_or_queue_log(
        "info",
        "voice_unclassified",
        "Voice input captured but not classified",
        {"text": text, "intent": intent, "movement": movement},
    )
    quiet_print("voice_action", f"[VOICE] unclassified -> {text}")


def build_stt_service() -> STTService:
    return STTService(
        callback=handle_voice_text,
        model_size="tiny.en",
        input_device=None,
        device_sample_rate=None,
        target_sample_rate=16000,
        channels=None,
        device="cpu",
        compute_type="int8",
        language="en",
        task="transcribe",
        log_path="~/SDP/AURA/JetsonLocal/storage/transcriptions.log",
    )


async def start_voice_loop() -> None:
    global stt_service, stt_task

    if stt_task and not stt_task.done():
        return

    stt_service = build_stt_service()
    stt_task = asyncio.create_task(stt_service.continuous_stt_loop())
    quiet_print("voice", "[VOICE] started")


async def stop_voice_loop() -> None:
    global stt_service, stt_task

    if stt_service is not None:
        stt_service.stop()

    if stt_task is not None:
        try:
            await asyncio.wait_for(stt_task, timeout=2.0)
        except Exception:
            stt_task.cancel()
            try:
                await stt_task
            except Exception:
                pass

    stt_service = None
    stt_task = None
    quiet_print("voice", "[VOICE] stopped")


# -------------------------------------------------------------------
# DEVICE REGISTRATION / CONFIG
# -------------------------------------------------------------------
def build_register_payload():
    import socket

    local_ip = "127.0.0.1"
    hostname = "jetson"

    try:
        hostname = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    return {
        "device_id": DEVICE_ID,
        "device_name": DEVICE_NAME,
        "device_type": DEVICE_TYPE,
        "software_version": DEVICE_SOFTWARE_VERSION,
        "hostname": hostname,
        "local_ip": local_ip,
    }


async def register_device():
    payload = build_register_payload()
    result = await asyncio.to_thread(api.register, payload)
    send_or_queue_log("info", "device_registered", "Device registered with backend", result)
    quiet_print("register", f"[REGISTER] ok device_id={DEVICE_ID}")
    return result


async def refresh_config():
    global runtime_config

    try:
        result = await asyncio.to_thread(api.get_config, DEVICE_ID)
        runtime_config["poll_seconds"] = float(result.get("poll_seconds", runtime_config["poll_seconds"]))
        runtime_config["heartbeat_seconds"] = int(result.get("heartbeat_seconds", runtime_config["heartbeat_seconds"]))
        runtime_config["status_seconds"] = 1
    except Exception as e:
        send_or_queue_log("warning", "config_refresh_failed", f"Failed to refresh config: {e}")
        quiet_print("config", f"[CONFIG] using local defaults ({e})")


# -------------------------------------------------------------------
# BACKGROUND LOOPS
# -------------------------------------------------------------------
async def heartbeat_loop():
    while True:
        try:
            payload = build_heartbeat_payload()
            await asyncio.to_thread(api.heartbeat, payload)
        except Exception as e:
            send_or_queue_log("warning", "heartbeat_failed", f"Heartbeat failed: {e}")
            quiet_print("heartbeat", f"[HEARTBEAT] failed: {e}")

        await asyncio.sleep(runtime_config["heartbeat_seconds"])


async def status_loop():
    while True:
        payload = build_status_payload()

        try:
            await asyncio.to_thread(api.status, payload)
            cpu = payload.get("cpu_percent")
            ram = payload.get("ram_percent")
            gpu = payload.get("gpu_percent")
            batt = payload.get("battery_percent")
            temp = payload.get("temperature_c")
            quiet_print(
                "status",
                f"[STATUS] ok cpu={cpu} gpu={gpu} ram={ram} batt={batt} temp={temp}",
            )
        except Exception as e:
            queue_status(payload)
            send_or_queue_log("warning", "status_failed", f"Status upload failed: {e}")
            quiet_print("status", f"[STATUS] failed: {e}")

        await asyncio.sleep(runtime_config["status_seconds"])


async def flush_loop():
    while True:
        try:
            sent_logs = await asyncio.to_thread(flush_logs, api.log)
            sent_status = await asyncio.to_thread(flush_statuses, api.status)

            if sent_logs or sent_status:
                quiet_print("flush", f"[FLUSH] logs={sent_logs} statuses={sent_status}")
        except Exception:
            pass

        await asyncio.sleep(int(OFFLINE_RETRY_SECONDS))


async def config_loop():
    while True:
        await refresh_config()
        await asyncio.sleep(int(CONFIG_REFRESH_SECONDS))


async def command_loop():
    while True:
        try:
            result = await asyncio.to_thread(api.get_next_command, DEVICE_ID)
            command = result.get("command")

            if command:
                command_id = command.get("id")
                cmd = (command.get("command") or "").strip().lower()
                payload = command.get("payload") or {}

                if cmd in MOVEMENT_COMMANDS:
                    try:
                        serial_link.send_command(cmd, payload.get("value", ""))
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "note": f"Sent movement command: {cmd}",
                            },
                        )
                        quiet_print("command", f"[COMMAND] ok {cmd}")
                    except Exception as e:
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"Movement failed: {e}",
                            },
                        )
                        quiet_print("command", f"[COMMAND] failed {cmd}: {e}")

                elif cmd == "camera_activate_raw":
                    try:
                        camera_service.activate("raw")
                        _reset_uploaded_signature()
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "note": "Camera activated in raw mode",
                            },
                        )
                        quiet_print("camera_cmd", "[COMMAND] camera raw")
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
                        quiet_print("camera_cmd", f"[COMMAND] camera raw failed: {e}")

                elif cmd == "camera_activate_detection":
                    try:
                        camera_service.activate("detection")
                        _reset_uploaded_signature()
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "note": "Camera activated in detection mode",
                            },
                        )
                        quiet_print("camera_cmd", "[COMMAND] camera detection")
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
                        quiet_print("camera_cmd", f"[COMMAND] camera detection failed: {e}")

                elif cmd == "camera_deactivate":
                    try:
                        camera_service.deactivate()
                        _reset_uploaded_signature()
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "note": "Camera deactivated",
                            },
                        )
                        quiet_print("camera_cmd", "[COMMAND] camera off")
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
                        quiet_print("camera_cmd", f"[COMMAND] camera off failed: {e}")

                else:
                    await asyncio.to_thread(
                        api.ack_command,
                        {
                            "command_id": command_id,
                            "device_id": DEVICE_ID,
                            "status": "failed",
                            "note": f"Invalid command: {cmd}",
                        },
                    )
                    quiet_print("command", f"[COMMAND] invalid {cmd}")

        except Exception as e:
            quiet_print("command_poll", f"[COMMAND] poll failed: {e}")

        await asyncio.sleep(runtime_config["poll_seconds"])


# -------------------------------------------------------------------
# CAMERA UPLOAD BRIDGE
# -------------------------------------------------------------------
def _reset_uploaded_signature():
    global _last_uploaded_signature
    _last_uploaded_signature = None


def upload_latest_frame_once():
    global _last_uploaded_signature

    if not API_BASE_URL:
        return

    status = camera_service.get_status()
    if not status.get("enabled") or not status.get("running"):
        return

    frame = camera_service.get_jpeg()
    if not frame:
        return

    mode = camera_service.get_mode()
    signature = f"{mode}:{len(frame)}:{frame[:32]!r}"

    if signature == _last_uploaded_signature:
        return

    url = f"{API_BASE_URL.rstrip('/')}/device/camera/frame"
    headers = {
        "X-Device-Secret": DEVICE_SHARED_SECRET,
        "Content-Type": "image/jpeg",
    }
    params = {
        "device_id": DEVICE_ID,
        "mode": mode,
    }

    resp = requests.post(url, params=params, headers=headers, data=frame, timeout=3)
    resp.raise_for_status()
    _last_uploaded_signature = signature


async def camera_upload_loop():
    while True:
        try:
            await asyncio.to_thread(upload_latest_frame_once)
        except Exception as e:
            quiet_print("camera_upload", f"[CAMERA_UPLOAD] failed: {e}")

        await asyncio.sleep(0.10)


# -------------------------------------------------------------------
# CAMERA STREAM
# -------------------------------------------------------------------
def mjpeg_generator() -> Iterator[bytes]:
    camera_service.add_stream_client()
    try:
        empty_count = 0

        while True:
            frame = camera_service.get_jpeg()

            if frame is None:
                empty_count += 1
                if empty_count % 50 == 0:
                    quiet_print("camera_stream_wait", "[CAMERA] waiting for first frame")
                time.sleep(0.03)
                continue

            empty_count = 0

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
            )
    finally:
        camera_service.remove_stream_client()


# -------------------------------------------------------------------
# LIFESPAN
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    try:
        serial_link.connect()
    except Exception as e:
        send_or_queue_log("warning", "serial_unavailable", f"Serial unavailable: {e}")
        quiet_print("serial", f"[SERIAL] unavailable: {e}")

    send_or_queue_log("info", "camera_idle", "Camera service idle until activated")

    try:
        await register_device()
    except Exception as e:
        send_or_queue_log("warning", "register_failed", f"Initial register failed: {e}")
        quiet_print("register", f"[REGISTER] failed: {e}")

    asyncio.create_task(config_loop())
    asyncio.create_task(heartbeat_loop())
    asyncio.create_task(status_loop())
    asyncio.create_task(flush_loop())
    asyncio.create_task(command_loop())
    asyncio.create_task(camera_upload_loop())

    if voice_enabled:
        try:
            await start_voice_loop()
        except Exception as e:
            send_or_queue_log("warning", "voice_start_failed", f"Voice loop failed to start: {e}")
            quiet_print("voice", f"[VOICE] failed to start: {e}")

    quiet_print("startup", "[STARTUP] telemetry agent running")
    yield

    try:
        await stop_voice_loop()
    except Exception:
        pass

    try:
        camera_service.deactivate()
        _reset_uploaded_signature()
    except Exception:
        pass


app.router.lifespan_context = lifespan


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "ok": True,
        "device_id": DEVICE_ID,
        "device_name": DEVICE_NAME,
        "device_type": DEVICE_TYPE,
        "db_name": LOCAL_DB_NAME,
        "camera": camera_service.get_status(),
        "voice": {
            "enabled": voice_enabled,
            "running": bool(stt_task and not stt_task.done()),
            "device_index": stt_service.input_device if stt_service else None,
            "device_sample_rate": stt_service.device_sample_rate if stt_service else None,
            "channels": stt_service.channels if stt_service else None,
            "noise_floor": stt_service.noise_floor if stt_service else None,
        },
    }


@app.get("/status")
async def status():
    return build_status_payload()


@app.get("/voice/status")
async def voice_status():
    return {
        "ok": True,
        "enabled": voice_enabled,
        "running": bool(stt_task and not stt_task.done()),
        "device_index": stt_service.input_device if stt_service else None,
        "device_sample_rate": stt_service.device_sample_rate if stt_service else None,
        "channels": stt_service.channels if stt_service else None,
        "noise_floor": stt_service.noise_floor if stt_service else None,
        "model_size": stt_service.model_size if stt_service else None,
    }


@app.post("/voice/start")
async def voice_start():
    await start_voice_loop()
    return {
        "ok": True,
        "running": bool(stt_task and not stt_task.done()),
    }


@app.post("/voice/stop")
async def voice_stop():
    await stop_voice_loop()
    return {
        "ok": True,
        "running": False,
    }


@app.get("/camera/status")
async def camera_status():
    return get_camera_status()


@app.get("/camera/detections")
async def camera_detections():
    return {
        "mode": camera_service.get_mode(),
        "detections": camera_service.get_detections(),
    }


@app.get("/camera/frame.jpg")
async def camera_frame():
    status = camera_service.get_status()

    if not status.get("enabled"):
        camera_service.activate("raw")
        _reset_uploaded_signature()
        time.sleep(0.4)

    frame = camera_service.get_jpeg()
    if frame is None:
        raise HTTPException(status_code=503, detail="No camera frame available yet")

    return StreamingResponse(
        iter([frame]),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.post("/camera/activate")
async def activate_camera(mode: str = Query("raw", pattern="^(raw|detection)$")):
    try:
        camera_service.activate(mode)
        _reset_uploaded_signature()
        return {"ok": True, "enabled": True, "mode": camera_service.get_mode()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate camera: {e}")


@app.post("/camera/deactivate")
async def deactivate_camera():
    camera_service.deactivate()
    _reset_uploaded_signature()
    return {"ok": True, "enabled": False}


@app.post("/camera/mode")
async def set_camera_mode(mode: str = Query(..., pattern="^(raw|detection)$")):
    status = camera_service.get_status()

    if not status.get("enabled"):
        camera_service.activate(mode)
    else:
        camera_service.set_mode(mode)

    _reset_uploaded_signature()
    return {"ok": True, "enabled": True, "mode": camera_service.get_mode()}


@app.get("/camera/stream")
async def camera_stream(mode: str = Query("raw", pattern="^(raw|detection)$")):
    try:
        status = camera_service.get_status()

        if not status.get("enabled"):
            camera_service.activate(mode)
            _reset_uploaded_signature()
            time.sleep(0.4)
        elif camera_service.get_mode() != mode:
            camera_service.set_mode(mode)
            _reset_uploaded_signature()
            time.sleep(0.2)

        return StreamingResponse(
            mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "keep-alive",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start camera stream: {e}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        access_log=False,
        log_level="warning",
    )