import sys
import time
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
import os
from typing import Iterator, List, Optional

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
# IMPORTS
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

from ai.rag_manager import rag_manager
from ai.chat_manager import chat_manager

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

                elif cmd == "sync_vectors":
                    db_name = payload.get("db_name", LOCAL_DB_NAME)
                    try:
                        ok = await rag_manager.load_remote_db(db_name, api)
                        status_note = f"Loaded vector DB '{db_name}'" if ok else f"Failed to load '{db_name}'"
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed" if ok else "failed",
                                "note": status_note,
                                "result": rag_manager.stats(),
                            },
                        )
                        quiet_print("rag_cmd", f"[COMMAND] sync_vectors: {status_note}")
                    except Exception as e:
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"sync_vectors failed: {e}",
                            },
                        )

                elif cmd == "chat_prompt":
                    print(f"[CHAT] received command: {payload}")

                    db_name = (payload.get("db_name") or LOCAL_DB_NAME or "").strip()
                    query = payload.get("query", "")
                    session_id = payload.get("session_id", chat_manager.active_session_id)

                    try:
                        if session_id != chat_manager.active_session_id:
                            chat_manager.set_session(session_id)

                        # Make sure the requested DB is actually active before querying.
                        if db_name and rag_manager.active_db_name != db_name:
                            print(f"[CHAT] loading DB: {db_name}")
                            ok = await rag_manager.load_remote_db(db_name, api)
                            if not ok:
                                raise RuntimeError(f"Failed to load DB '{db_name}'")

                        chat_manager.add_message("user", query, api, DEVICE_ID)

                        print(f"[CHAT] running RAG query on db='{rag_manager.active_db_name}': {query}")

                        answer = await rag_manager.query(query)

                        if isinstance(answer, dict):
                            answer = answer.get("answer") or answer.get("response") or str(answer)

                        print(f"[CHAT] raw answer: {answer}")

                        if not answer or not str(answer).strip():
                            answer = "No response generated from model."

                        chat_manager.add_message("assistant", answer, api, DEVICE_ID)

                        ack_payload = {
                            "command_id": command_id,
                            "device_id": DEVICE_ID,
                            "status": "completed",
                            "note": "Chat answered",
                            "result": {
                                "answer": answer,
                                "session_id": session_id,
                                "db_name": rag_manager.active_db_name,
                            },
                        }

                        print(f"[CHAT] sending ack: {ack_payload}")

                        await asyncio.to_thread(
                            api.ack_command,
                            ack_payload,
                        )

                        print("[CHAT] ack sent successfully")

                    except Exception as e:
                        print(f"[CHAT ERROR] {e}")

                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"chat_prompt failed: {e}",
                            },
                        )

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


async def rag_build_loop():
    while True:
        try:
            result = await asyncio.to_thread(api.get_next_rag_build_job, DEVICE_ID)
            job = (result or {}).get("job")

            if not job:
                await asyncio.sleep(2.0)
                continue

            job_id = str(job.get("job_id") or "").strip()
            db_name = str(job.get("db_name") or "").strip()
            document_paths = list(job.get("document_paths") or [])

            if not job_id or not db_name:
                await asyncio.sleep(2.0)
                continue

            quiet_print(
                "rag_job_claimed",
                f"[RAG JOB] claimed '{db_name}' with {len(document_paths)} PDF(s)",
            )

            await asyncio.to_thread(
                api.ack_rag_build_job,
                job_id,
                DEVICE_ID,
                "running",
                f"Jetson started vectorizing '{db_name}'",
                {
                    "db_name": db_name,
                    "file_count": len(document_paths),
                },
            )

            try:
                build_result = await rag_manager.build_database_from_document_paths(
                    db_name=db_name,
                    document_paths=document_paths,
                    api_client=api,
                )

                await asyncio.to_thread(
                    api.ack_rag_build_job,
                    job_id,
                    DEVICE_ID,
                    "completed",
                    f"Jetson finished vectorizing '{db_name}' and synced it back to website",
                    build_result,
                )

                quiet_print(
                    "rag_job_completed",
                    f"[RAG JOB] completed '{db_name}'",
                )

            except Exception as build_error:
                await asyncio.to_thread(
                    api.ack_rag_build_job,
                    job_id,
                    DEVICE_ID,
                    "failed",
                    f"Jetson build failed for '{db_name}': {build_error}",
                    {
                        "db_name": db_name,
                        "error": str(build_error),
                    },
                )

                quiet_print(
                    "rag_job_failed",
                    f"[RAG JOB] failed '{db_name}': {build_error}",
                )

        except Exception as poll_error:
            quiet_print("rag_job_poll_error", f"[RAG JOB] poll failed: {poll_error}")
            await asyncio.sleep(int(OFFLINE_RETRY_SECONDS))


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
        rag_manager.initialize()
        quiet_print("rag_ready", "[STARTUP] RAG build worker ready")
    except Exception as e:
        quiet_print("rag", f"[RAG] init warning: {e}")

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
    asyncio.create_task(rag_build_loop())
    asyncio.create_task(camera_upload_loop())

    quiet_print("startup", "[STARTUP] telemetry agent running")
    yield

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
        "rag": rag_manager.stats(),
        "camera": camera_service.get_status(),
    }


# -------------------------------------------------------------------
# RAG endpoints
# -------------------------------------------------------------------
from pydantic import BaseModel as _BaseModel


class _ChatRequest(_BaseModel):
    query: str
    session_id: Optional[str] = None


class _SyncVectorsRequest(_BaseModel):
    db_name: str


class _BuildRagRequest(_BaseModel):
    db_name: str
    document_paths: List[str] = []


@app.get("/rag/stats")
async def rag_stats():
    return rag_manager.stats()


@app.post("/rag/load_db")
async def rag_load_db(req: _SyncVectorsRequest):
    ok = await rag_manager.load_remote_db(req.db_name, api)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Failed to load DB '{req.db_name}'")
    return {"ok": True, "db_name": req.db_name, "stats": rag_manager.stats()}


@app.post("/rag/build")
async def rag_build(req: _BuildRagRequest):
    result = await rag_manager.build_database_from_document_paths(
        db_name=req.db_name,
        document_paths=req.document_paths,
        api_client=api,
    )
    return {"ok": True, **result, "stats": rag_manager.stats()}


@app.post("/rag/chat")
async def rag_chat(req: _ChatRequest):
    session_id = req.session_id or chat_manager.active_session_id
    if session_id != chat_manager.active_session_id:
        chat_manager.set_session(session_id)

    chat_manager.add_message("user", req.query, api)
    answer = await rag_manager.query(req.query)
    chat_manager.add_message("assistant", answer, api)

    return {
        "ok": True,
        "answer": answer,
        "session_id": session_id,
    }


# -------------------------------------------------------------------
# Session endpoints
# -------------------------------------------------------------------
@app.get("/sessions")
async def list_sessions():
    return {"sessions": chat_manager.list_local_sessions()}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    prev = chat_manager.active_session_id
    chat_manager.set_session(session_id)
    history = chat_manager.get_history()
    chat_manager.set_session(prev)
    return {"session_id": session_id, "history": history}


@app.post("/sessions/load/{session_id}")
async def load_session_from_cloud(session_id: str):
    try:
        data = await asyncio.to_thread(api.get_chat_session, session_id)
        history = data.get("history", [])
        chat_manager.set_session(session_id, remote_history=history)
        return {"ok": True, "session_id": session_id, "messages": len(history)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not pull session: {e}")


@app.get("/status")
async def status():
    return build_status_payload()


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