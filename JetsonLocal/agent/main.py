import sys
import time
import asyncio
import os
from pathlib import Path
from contextlib import asynccontextmanager
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

from stt_faster import STTService
from tts import TTSService

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
CAMERA_MODE_PATTERN = "^(raw|detection|colorcode|face)$"

VOICE_RAG_TIMEOUT_SECONDS = 45.0
VOICE_TTS_TIMEOUT_SECONDS = 20.0
CHAT_RAG_TIMEOUT_SECONDS = 60.0

_last_messages = {}
_last_uploaded_signature: Optional[str] = None

# -------------------------------------------------------------------
# VOICE / TTS STATE
# -------------------------------------------------------------------
stt_task: Optional[asyncio.Task] = None
stt_service: Optional[STTService] = None
voice_enabled = True
voice_running = False
voice_loaded = False
voice_idle_seconds = 0
tts_service = TTSService()


def quiet_print(key: str, message: str) -> None:
    if _last_messages.get(key) != message:
        print(message)
        _last_messages[key] = message


_current_ui_state: Optional[str] = None


def set_ui_state(state: str, detail: str = "") -> None:
    global _current_ui_state

    state = (state or "READY").strip().upper()
    detail = (detail or "").strip()

    payload = f"[UI_STATE] {state}"
    if detail:
        payload += f" | {detail}"

    if _current_ui_state != payload:
        print(payload)
        _current_ui_state = payload


def truncate_for_ui(text: str, limit: int = 80) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


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
        runtime_config["poll_seconds"] = float(
            result.get("poll_seconds", runtime_config["poll_seconds"])
        )
        runtime_config["heartbeat_seconds"] = int(
            result.get("heartbeat_seconds", runtime_config["heartbeat_seconds"])
        )
        runtime_config["status_seconds"] = int(
            result.get("status_seconds", runtime_config["status_seconds"])
        )
    except Exception as e:
        send_or_queue_log("warning", "config_refresh_failed", f"Failed to refresh config: {e}")
        quiet_print("config", f"[CONFIG] using local defaults ({e})")


# -------------------------------------------------------------------
# VOICE HELPERS
# -------------------------------------------------------------------
def extract_speak_payload(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    lowered = raw.lower()
    idx = lowered.find("speak")
    if idx == -1:
        return ""

    payload = raw[idx + len("speak"):].strip(" ,.:;!-")
    fillers = (
        "this",
        "that",
        "the following",
        "out loud",
        "please",
    )

    changed = True
    while changed and payload:
        changed = False
        lowered_payload = payload.lower()
        for filler in fillers:
            prefix = filler + " "
            if lowered_payload.startswith(prefix):
                payload = payload[len(prefix):].strip(" ,.:;!-")
                changed = True
                break

    return payload


async def speak_with_timeout(text: str, ui_detail: str = "") -> bool:
    text = (text or "").strip()
    if not text:
        return False

    detail = truncate_for_ui(ui_detail or text)
    set_ui_state("SPEAKING", detail)
    quiet_print("voice_speaking", f"[VOICE] speaking: {truncate_for_ui(text, 120)}")

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(tts_service.speak, text),
            timeout=VOICE_TTS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        quiet_print("tts_timeout", "[TTS] timeout waiting for playback")
        send_or_queue_log(
            "warning",
            "tts_timeout",
            "TTS playback timed out",
            {"text": truncate_for_ui(text, 160)},
        )
        return False
    except Exception as e:
        quiet_print("tts_fail", f"[TTS] failed: {e}")
        send_or_queue_log(
            "warning",
            "tts_failed",
            f"TTS failed: {e}",
            {"text": truncate_for_ui(text, 160)},
        )
        return False


async def query_rag_with_timeout(query: str, timeout_seconds: float):
    return await asyncio.wait_for(rag_manager.query(query), timeout=timeout_seconds)


async def handle_voice_text(text: str, intent: str, movement: Optional[str]) -> None:
    global voice_idle_seconds

    text = (text or "").strip()
    if not text:
        return

    voice_idle_seconds = 0
    lowered = text.lower()

    set_ui_state("THINKING", truncate_for_ui(text))
    quiet_print("voice_question_received", f"[VOICE] question received: {text}")

    try:
        if "speak" in lowered:
            speak_payload = extract_speak_payload(text)

            if speak_payload:
                send_or_queue_log(
                    "info",
                    "voice_tts_command",
                    "Voice TTS command executed",
                    {"text": text, "spoken_text": speak_payload},
                )
                quiet_print("voice_action", f"[VOICE] tts -> {speak_payload}")
                await speak_with_timeout(speak_payload, speak_payload)
            else:
                send_or_queue_log(
                    "warning",
                    "voice_tts_missing_payload",
                    "Voice TTS command requested but no payload found",
                    {"text": text},
                )
                quiet_print("voice_action", "[VOICE] tts requested but no payload found")
                await speak_with_timeout(
                    "Please tell me what you want me to say.",
                    "Prompting for speech text",
                )
            return

        if movement and movement in MOVEMENT_COMMANDS:
            serial_link.send_command(movement, "")
            set_ui_state("COMMAND", movement)
            quiet_print("voice_move", f"[VOICE] movement command: {movement}")
            send_or_queue_log(
                "info",
                "voice_movement",
                f"Voice movement command: {movement}",
                {"text": text, "intent": intent, "movement": movement},
            )
            return

        chat_manager.add_message("user", text, api, DEVICE_ID)
        set_ui_state("THINKING", "Running local RAG query")

        try:
            answer = await query_rag_with_timeout(text, VOICE_RAG_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            answer = "I timed out while thinking. Please ask again."

        if isinstance(answer, dict):
            answer = answer.get("answer") or answer.get("response") or str(answer)

        if not answer or not str(answer).strip():
            answer = "No response generated from model."

        answer = str(answer)
        chat_manager.add_message("assistant", answer, api, DEVICE_ID)

        await speak_with_timeout(answer, truncate_for_ui(answer))

        quiet_print("voice_chat", f"[VOICE] answered: {text}")
        send_or_queue_log(
            "info",
            "voice_chat_answered",
            "Voice question answered",
            {"text": text, "intent": intent},
        )

    except Exception as e:
        set_ui_state("ERROR", truncate_for_ui(str(e)))
        quiet_print("voice_chat_fail", f"[VOICE] failed: {e}")
        send_or_queue_log(
            "warning",
            "voice_chat_failed",
            f"Voice chat failed: {e}",
            {"text": text, "intent": intent, "movement": movement},
        )
    finally:
        if voice_running:
            set_ui_state("LISTENING", "Wake word active")
            quiet_print("voice_listening", "[VOICE] listening")


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
        log_path=os.path.expanduser("~/SDP/AURA/JetsonLocal/storage/transcriptions.log"),
        unload_after_idle_seconds=60.0,
        auto_reload_model=True,
    )


async def start_voice_loop():
    global stt_task, stt_service, voice_running, voice_loaded

    if stt_task and not stt_task.done():
        quiet_print("voice_already_running", "[VOICE] already running")
        set_ui_state("LISTENING", "Wake word active")
        return

    stt_service = build_stt_service()
    voice_loaded = True

    async def _runner():
        global voice_running, voice_loaded
        try:
            voice_running = True
            quiet_print("voice_start", "[VOICE] started")
            set_ui_state("LISTENING", "Wake word active")
            quiet_print("voice_listening", "[VOICE] listening")
            await stt_service.continuous_stt_loop()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            set_ui_state("ERROR", truncate_for_ui(str(e)))
            quiet_print("voice_loop_error", f"[VOICE] loop failed: {e}")
            raise
        finally:
            voice_running = False
            voice_loaded = False
            quiet_print("voice_stop", "[VOICE] stopped")
            set_ui_state("READY", "Voice stopped")

    stt_task = asyncio.create_task(_runner())


async def stop_voice_loop():
    global stt_task, stt_service, voice_running, voice_loaded

    if stt_service is not None:
        try:
            stt_service.stop()
        except Exception as e:
            quiet_print("voice_stop_fail", f"[VOICE] stop failed: {e}")

    if stt_task is not None:
        try:
            await asyncio.wait_for(stt_task, timeout=3.0)
        except asyncio.TimeoutError:
            stt_task.cancel()
            try:
                await stt_task
            except Exception:
                pass
        except Exception:
            pass
        finally:
            stt_task = None

    stt_service = None
    voice_running = False
    voice_loaded = False
    quiet_print("voice_stop", "[VOICE] stopped")
    set_ui_state("READY", "Voice stopped")


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
    global voice_idle_seconds

    while True:
        payload = build_status_payload()

        if stt_service is not None and getattr(stt_service, "last_audio_activity_ts", None):
            voice_idle_seconds = max(0, int(time.time() - stt_service.last_audio_activity_ts))
        else:
            voice_idle_seconds = 0

        payload.setdefault("extra", {})
        payload["extra"]["voice"] = {
            "enabled": voice_enabled,
            "running": voice_running,
            "model_loaded": voice_loaded,
            "idle_seconds": voice_idle_seconds,
            "device_index": stt_service.input_device if stt_service else None,
            "device_sample_rate": stt_service.device_sample_rate if stt_service else None,
            "channels": stt_service.channels if stt_service else None,
            "noise_floor": stt_service.noise_floor if stt_service else None,
            "unload_after_idle_seconds": getattr(stt_service, "unload_after_idle_seconds", None) if stt_service else None,
            "tts_ready": True,
            "tts_device": tts_service.device,
        }

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
                        set_ui_state("COMMAND", cmd)
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
                        set_ui_state("READY", "Ready")
                    except Exception as e:
                        set_ui_state("ERROR", truncate_for_ui(str(e)))
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
                        set_ui_state("ERROR", truncate_for_ui(str(e)))
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
                        set_ui_state("ERROR", truncate_for_ui(str(e)))
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

                elif cmd == "camera_activate_colorcode":
                    try:
                        camera_service.activate("colorcode")
                        _reset_uploaded_signature()
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "note": "Camera activated in colorcode mode",
                            },
                        )
                        quiet_print("camera_cmd", "[COMMAND] camera colorcode")
                    except Exception as e:
                        set_ui_state("ERROR", truncate_for_ui(str(e)))
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"Camera colorcode activation failed: {e}",
                            },
                        )
                        quiet_print("camera_cmd", f"[COMMAND] camera colorcode failed: {e}")

                elif cmd == "camera_activate_face":
                    try:
                        camera_service.activate("face")
                        _reset_uploaded_signature()
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "note": "Camera activated in face mode",
                            },
                        )
                        quiet_print("camera_cmd", "[COMMAND] camera face")
                    except Exception as e:
                        set_ui_state("ERROR", truncate_for_ui(str(e)))
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"Camera face activation failed: {e}",
                            },
                        )
                        quiet_print("camera_cmd", f"[COMMAND] camera face failed: {e}")

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
                        set_ui_state("ERROR", truncate_for_ui(str(e)))
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

                elif cmd == "voice_start":
                    try:
                        await start_voice_loop()
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "note": "Voice loop started",
                            },
                        )
                        quiet_print("voice_cmd", "[COMMAND] voice start")
                    except Exception as e:
                        set_ui_state("ERROR", truncate_for_ui(str(e)))
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"voice_start failed: {e}",
                            },
                        )
                        quiet_print("voice_cmd", f"[COMMAND] voice start failed: {e}")

                elif cmd == "voice_stop":
                    try:
                        await stop_voice_loop()
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "note": "Voice loop stopped",
                            },
                        )
                        quiet_print("voice_cmd", "[COMMAND] voice stop")
                    except Exception as e:
                        set_ui_state("ERROR", truncate_for_ui(str(e)))
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"voice_stop failed: {e}",
                            },
                        )
                        quiet_print("voice_cmd", f"[COMMAND] voice stop failed: {e}")

                elif cmd == "sync_vectors":
                    db_name = (payload.get("db_name") or LOCAL_DB_NAME or "").strip()
                    try:
                        print(f"[JETSON DB] loading database '{db_name}' from website")
                        set_ui_state("VECTORIZING", f"Loading {db_name}")
                        ok = await rag_manager.load_remote_db(db_name, api)
                        status_note = f"Loaded vector DB '{db_name}'" if ok else f"Failed to load '{db_name}'"

                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed" if ok else "failed",
                                "note": status_note,
                                "result": {
                                    **rag_manager.stats(),
                                    "db_name": db_name,
                                },
                            },
                        )

                        if ok:
                            print(f"[JETSON DB] loaded database '{db_name}' successfully")
                            set_ui_state("READY", f"Loaded {db_name}")
                        else:
                            print(f"[JETSON DB] failed loading database '{db_name}'")
                            set_ui_state("ERROR", f"Failed to load {db_name}")
                    except Exception as e:
                        set_ui_state("ERROR", truncate_for_ui(str(e)))
                        print(f"[JETSON DB] sync_vectors failed for '{db_name}': {e}")
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"sync_vectors failed: {e}",
                            },
                        )

                elif cmd == "delete_vectors":
                    db_name = (payload.get("db_name") or "").strip()

                    try:
                        if not db_name:
                            raise RuntimeError("delete_vectors missing db_name")

                        db_dir = rag_manager.get_db_dir(db_name)

                        print(f"[JETSON DB] deleting database '{db_name}' from Jetson")
                        set_ui_state("VECTORIZING", f"Deleting {db_name}")

                        if db_dir.exists():
                            import shutil
                            shutil.rmtree(db_dir, ignore_errors=True)

                        if rag_manager.active_db_name == db_name:
                            rag_manager.rag_system = None
                            rag_manager.active_db_name = None
                            rag_manager.active_db_path = None
                            print(f"[JETSON DB] active database '{db_name}' cleared from memory")

                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "completed",
                                "note": f"Deleted DB '{db_name}' from Jetson",
                                "result": {
                                    "db_name": db_name,
                                    "deleted": True,
                                },
                            },
                        )

                        print(f"[JETSON DB] deleted database '{db_name}' from Jetson")
                        set_ui_state("READY", f"Deleted {db_name}")

                    except Exception as e:
                        set_ui_state("ERROR", truncate_for_ui(str(e)))
                        print(f"[JETSON DB] delete_vectors failed for '{db_name}': {e}")
                        await asyncio.to_thread(
                            api.ack_command,
                            {
                                "command_id": command_id,
                                "device_id": DEVICE_ID,
                                "status": "failed",
                                "note": f"delete_vectors failed: {e}",
                            },
                        )

                elif cmd == "chat_prompt":
                    print(f"[CHAT] received command: {payload}")

                    db_name = (payload.get("db_name") or LOCAL_DB_NAME or "").strip()
                    query = payload.get("query", "")
                    session_id = payload.get("session_id", chat_manager.active_session_id)

                    try:
                        set_ui_state("THINKING", "Chat request received")

                        if session_id != chat_manager.active_session_id:
                            chat_manager.set_session(session_id)

                        if db_name:
                            if rag_manager.active_db_name != db_name or rag_manager.rag_system is None:
                                print(f"[CHAT] loading DB: {db_name}")
                                ok = await rag_manager.load_remote_db(db_name, api)
                                if not ok:
                                    raise RuntimeError(f"Failed to load DB '{db_name}'")
                            else:
                                print(f"[CHAT] reusing already loaded DB: {db_name}")

                        chat_manager.add_message("user", query, api, DEVICE_ID)

                        print(f"[CHAT] running RAG query on db='{rag_manager.active_db_name}': {query}")
                        set_ui_state("THINKING", truncate_for_ui(query))

                        try:
                            answer = await query_rag_with_timeout(query, CHAT_RAG_TIMEOUT_SECONDS)
                        except asyncio.TimeoutError:
                            answer = "I timed out while generating a response."

                        if isinstance(answer, dict):
                            answer = answer.get("answer") or answer.get("response") or str(answer)

                        print(f"[CHAT] raw answer: {answer}")

                        if not answer or not str(answer).strip():
                            answer = "No response generated from model."

                        answer = str(answer)
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
                        set_ui_state("READY", "Chat response ready")

                        await asyncio.to_thread(api.ack_command, ack_payload)

                        print("[CHAT] ack sent successfully")

                    except Exception as e:
                        set_ui_state("ERROR", truncate_for_ui(str(e)))
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
            set_ui_state("VECTORIZING", f"{db_name} ({len(document_paths)} PDF(s))")

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
                set_ui_state("READY", f"{db_name} ready")

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
                set_ui_state("ERROR", truncate_for_ui(str(build_error)))

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

    resp = requests.post(url, params=params, headers=headers, data=frame, timeout=2.0)
    resp.raise_for_status()
    _last_uploaded_signature = signature


async def camera_upload_loop():
    while True:
        try:
            await asyncio.to_thread(upload_latest_frame_once)
        except Exception as e:
            quiet_print("camera_upload", f"[CAMERA_UPLOAD] failed: {e}")

        await asyncio.sleep(0.03)


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
                time.sleep(0.02)
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

    try:
        camera_service.deactivate()
        _reset_uploaded_signature()
        quiet_print("camera_idle", "[CAMERA] idle until website activates raw mode")
    except Exception as e:
        quiet_print("camera_idle", f"[CAMERA] idle init warning: {e}")

    quiet_print("tts", f"[TTS] ready device={tts_service.device}")

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

    if voice_enabled:
        try:
            await start_voice_loop()
        except Exception as e:
            send_or_queue_log("warning", "voice_start_failed", f"Voice loop failed to start: {e}")
            quiet_print("voice", f"[VOICE] failed to start: {e}")

    set_ui_state("READY", "AURA online")
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
    global voice_idle_seconds

    if stt_service is not None and getattr(stt_service, "last_audio_activity_ts", None):
        voice_idle_seconds = max(0, int(time.time() - stt_service.last_audio_activity_ts))
    else:
        voice_idle_seconds = 0

    return {
        "ok": True,
        "device_id": DEVICE_ID,
        "device_name": DEVICE_NAME,
        "device_type": DEVICE_TYPE,
        "db_name": LOCAL_DB_NAME,
        "rag": rag_manager.stats(),
        "camera": camera_service.get_status(),
        "voice": {
            "enabled": voice_enabled,
            "running": voice_running,
            "model_loaded": voice_loaded,
            "device_index": stt_service.input_device if stt_service else None,
            "device_sample_rate": stt_service.device_sample_rate if stt_service else None,
            "channels": stt_service.channels if stt_service else None,
            "noise_floor": stt_service.noise_floor if stt_service else None,
            "idle_seconds": voice_idle_seconds,
            "unload_after_idle_seconds": getattr(stt_service, "unload_after_idle_seconds", None) if stt_service else None,
            "tts_device": tts_service.device,
        },
    }


@app.get("/voice/status")
async def voice_status():
    global voice_idle_seconds

    if stt_service is not None and getattr(stt_service, "last_audio_activity_ts", None):
        voice_idle_seconds = max(0, int(time.time() - stt_service.last_audio_activity_ts))
    else:
        voice_idle_seconds = 0

    return {
        "ok": True,
        "enabled": voice_enabled,
        "running": voice_running,
        "model_loaded": voice_loaded,
        "idle_seconds": voice_idle_seconds,
        "device_index": stt_service.input_device if stt_service else None,
        "device_sample_rate": stt_service.device_sample_rate if stt_service else None,
        "channels": stt_service.channels if stt_service else None,
        "noise_floor": stt_service.noise_floor if stt_service else None,
        "unload_after_idle_seconds": getattr(stt_service, "unload_after_idle_seconds", None) if stt_service else None,
        "tts_device": tts_service.device,
    }


@app.post("/voice/start")
async def voice_start():
    try:
        await start_voice_loop()
        return {"ok": True, "running": voice_running}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start voice loop: {e}")


@app.post("/voice/stop")
async def voice_stop():
    try:
        await stop_voice_loop()
        return {"ok": True, "running": voice_running}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop voice loop: {e}")


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

    chat_manager.add_message("user", req.query, api, DEVICE_ID)

    try:
        answer = await query_rag_with_timeout(req.query, CHAT_RAG_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        answer = "I timed out while generating a response."

    if isinstance(answer, dict):
        answer = answer.get("answer") or answer.get("response") or str(answer)

    answer = str(answer or "").strip() or "No response generated from model."
    chat_manager.add_message("assistant", answer, api, DEVICE_ID)

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
    list_fn = getattr(chat_manager, "list_local_sessions", None)
    if callable(list_fn):
        return {"sessions": list_fn()}
    return {"sessions": []}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    prev = chat_manager.active_session_id
    chat_manager.set_session(session_id)

    get_history_fn = getattr(chat_manager, "get_history", None)
    if callable(get_history_fn):
        history = get_history_fn()
    else:
        history = getattr(chat_manager, "history", [])

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
    global voice_idle_seconds

    payload = build_status_payload()

    if stt_service is not None and getattr(stt_service, "last_audio_activity_ts", None):
        voice_idle_seconds = max(0, int(time.time() - stt_service.last_audio_activity_ts))
    else:
        voice_idle_seconds = 0

    payload.setdefault("extra", {})
    payload["extra"]["voice"] = {
        "enabled": voice_enabled,
        "running": voice_running,
        "model_loaded": voice_loaded,
        "idle_seconds": voice_idle_seconds,
        "tts_device": tts_service.device,
    }
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
async def activate_camera(mode: str = Query("raw", pattern=CAMERA_MODE_PATTERN)):
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
async def set_camera_mode(mode: str = Query(..., pattern=CAMERA_MODE_PATTERN)):
    status = camera_service.get_status()

    if not status.get("enabled"):
        camera_service.activate(mode)
    else:
        camera_service.set_mode(mode)

    _reset_uploaded_signature()
    return {"ok": True, "enabled": True, "mode": camera_service.get_mode()}


@app.get("/camera/stream")
async def camera_stream(mode: str = Query("raw", pattern=CAMERA_MODE_PATTERN)):
    try:
        status = camera_service.get_status()

        if not status.get("enabled"):
            raise HTTPException(status_code=503, detail="Camera is not active")
        elif camera_service.get_mode() != mode:
            camera_service.set_mode(mode)
            _reset_uploaded_signature()
            time.sleep(0.1)

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
    except HTTPException:
        raise
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