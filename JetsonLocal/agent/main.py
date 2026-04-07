import sys
import time
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
import os
import re
import shutil
import subprocess
import threading
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

# -------------------------------------------------------------------
# PUBLIC TUNNEL
# -------------------------------------------------------------------
_PUBLIC_URL: Optional[str] = None
_TUNNEL_PROC: Optional[subprocess.Popen] = None

_CF_URL_RE = re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com")
_NGROK_URL_RE = re.compile(r"https://[a-zA-Z0-9.-]+\.ngrok(?:-free)?\.(?:app|io)")

AURA_ENABLE_PUBLIC_TUNNEL = os.getenv("AURA_ENABLE_PUBLIC_TUNNEL", "0").strip() == "1"
AURA_PUBLIC_TUNNEL_PROVIDER = os.getenv("AURA_PUBLIC_TUNNEL_PROVIDER", "cloudflared").strip().lower()
AURA_PUBLIC_TUNNEL_PORT = int(os.getenv("AURA_PUBLIC_TUNNEL_PORT", "8000"))


def get_public_url() -> str:
    return _PUBLIC_URL or ""


def _set_public_url(url: str) -> None:
    global _PUBLIC_URL
    url = (url or "").strip()
    if not url:
        return
    if _PUBLIC_URL != url:
        _PUBLIC_URL = url
        quiet_print("public_url", f"[TUNNEL] public url: {_PUBLIC_URL}")


def _consume_tunnel_stdout(proc: subprocess.Popen, provider: str) -> None:
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue

            quiet_print(f"tunnel_line_{provider}", f"[TUNNEL:{provider}] {line}")

            if provider == "cloudflared":
                m = _CF_URL_RE.search(line)
                if m:
                    _set_public_url(m.group(0))
            elif provider == "ngrok":
                m = _NGROK_URL_RE.search(line)
                if m:
                    _set_public_url(m.group(0))
    except Exception as e:
        quiet_print("tunnel_reader_err", f"[TUNNEL] stdout reader failed: {e}")


def start_public_tunnel(local_port: int = 8000) -> str:
    global _TUNNEL_PROC

    if not AURA_ENABLE_PUBLIC_TUNNEL:
        quiet_print("tunnel_disabled", "[TUNNEL] disabled")
        return ""

    if _TUNNEL_PROC and _TUNNEL_PROC.poll() is None:
        quiet_print("tunnel_running", "[TUNNEL] already running")
        return get_public_url()

    provider = AURA_PUBLIC_TUNNEL_PROVIDER
    target = f"http://127.0.0.1:{int(local_port)}"

    if provider == "cloudflared":
        exe = shutil.which("cloudflared")
        if not exe:
            quiet_print("tunnel_missing", "[TUNNEL] cloudflared not found in PATH")
            return ""
        cmd = [exe, "tunnel", "--url", target, "--no-autoupdate"]

    elif provider == "ngrok":
        exe = shutil.which("ngrok")
        if not exe:
            quiet_print("tunnel_missing", "[TUNNEL] ngrok not found in PATH")
            return ""
        cmd = [exe, "http", str(int(local_port)), "--log", "stdout"]

    else:
        quiet_print("tunnel_bad_provider", f"[TUNNEL] unsupported provider: {provider}")
        return ""

    try:
        _TUNNEL_PROC = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        t = threading.Thread(target=_consume_tunnel_stdout, args=(_TUNNEL_PROC, provider), daemon=True)
        t.start()
        quiet_print("tunnel_start", f"[TUNNEL] started {provider} for {target}")
    except Exception as e:
        quiet_print("tunnel_start_fail", f"[TUNNEL] failed to start {provider}: {e}")
        return ""

    return ""


def stop_public_tunnel() -> None:
    global _TUNNEL_PROC
    try:
        if _TUNNEL_PROC and _TUNNEL_PROC.poll() is None:
            _TUNNEL_PROC.terminate()
            quiet_print("tunnel_stop", "[TUNNEL] stopped")
    except Exception as e:
        quiet_print("tunnel_stop_fail", f"[TUNNEL] stop failed: {e}")
    finally:
        _TUNNEL_PROC = None


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
        "public_url": get_public_url(),
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
            payload["public_url"] = get_public_url()
            await asyncio.to_thread(api.heartbeat, payload)
        except Exception as e:
            send_or_queue_log("warning", "heartbeat_failed", f"Heartbeat failed: {e}")
            quiet_print("heartbeat", f"[HEARTBEAT] failed: {e}")

        await asyncio.sleep(runtime_config["heartbeat_seconds"])


async def status_loop():
    while True:
        payload = build_status_payload()
        payload["public_url"] = get_public_url()

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

                elif cmd == "build_rag":
                    import tempfile
                    import shutil as _shutil

                    db_name = payload.get("db_name", LOCAL_DB_NAME)
                    doc_paths = payload.get("document_paths", [])
                    try:
                        tmp_dir = tempfile.mkdtemp(prefix="aura_rag_")
                        local_pdfs = []
                        for rel_path in doc_paths:
                            dest = os.path.join(tmp_dir, os.path.basename(rel_path))
                            await asyncio.to_thread(api.download_document, rel_path, dest)
                            local_pdfs.append(dest)

                        result = await rag_manager.ingest_pdfs_and_sync(local_pdfs, api, db_name)
                        _shutil.rmtree(tmp_dir, ignore_errors=True)

                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "note": f"RAG built and synced: {result}",
                                "result": {**result, **rag_manager.stats()},
                            },
                        )
                        quiet_print("rag_cmd", f"[COMMAND] build_rag complete: {result}")
                    except Exception as e:
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"build_rag failed: {e}",
                            },
                        )
                        quiet_print("rag_cmd", f"[COMMAND] build_rag failed: {e}")

                elif cmd == "chat_prompt":
                    query = payload.get("query", "")
                    session_id = payload.get("session_id", chat_manager.active_session_id)
                    try:
                        if session_id != chat_manager.active_session_id:
                            chat_manager.set_session(session_id)

                        chat_manager.add_message("user", query, api)
                        answer = await rag_manager.query(query)
                        chat_manager.add_message("assistant", answer, api)

                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "note": "Chat answered",
                                "result": {"answer": answer, "session_id": session_id},
                            },
                        )
                        quiet_print("rag_cmd", f"[COMMAND] chat_prompt answered")
                    except Exception as e:
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
    except Exception as e:
        quiet_print("rag", f"[RAG] init warning: {e}")

    start_public_tunnel(local_port=AURA_PUBLIC_TUNNEL_PORT)

    # give the tunnel a moment to print a URL, but don't block long
    for _ in range(20):
        if get_public_url():
            break
        await asyncio.sleep(0.25)

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

    quiet_print("startup", "[STARTUP] telemetry agent running")
    yield

    try:
        camera_service.deactivate()
        _reset_uploaded_signature()
    except Exception:
        pass

    stop_public_tunnel()


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
        "public_url": get_public_url(),
        "rag": rag_manager.stats(),
        "camera": camera_service.get_status(),
    }


@app.get("/public_url")
async def public_url():
    return {
        "ok": True,
        "public_url": get_public_url(),
        "provider": AURA_PUBLIC_TUNNEL_PROVIDER if AURA_ENABLE_PUBLIC_TUNNEL else "",
        "enabled": AURA_ENABLE_PUBLIC_TUNNEL,
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
    import tempfile
    import shutil as _shutil

    tmp_dir = tempfile.mkdtemp(prefix="aura_rag_")
    local_pdfs = []
    try:
        for rel_path in req.document_paths:
            dest = os.path.join(tmp_dir, os.path.basename(rel_path))
            await asyncio.to_thread(api.download_document, rel_path, dest)
            local_pdfs.append(dest)

        result = await rag_manager.ingest_pdfs_and_sync(local_pdfs, api, req.db_name)
        return {"ok": True, **result, "stats": rag_manager.stats()}
    finally:
        _shutil.rmtree(tmp_dir, ignore_errors=True)


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
    payload = build_status_payload()
    payload["public_url"] = get_public_url()
    return payload


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