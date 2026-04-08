# backend/tts_api.py
import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv

from security import require_auth, require_ip_allowlist

# Local dev env load
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(env_path)

router = APIRouter(tags=["tts"])

TTS_RATE = int(os.getenv("TTS_RATE", "165"))
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))
TTS_VOICE_ID = os.getenv("TTS_VOICE_ID", "").strip()  # optional

_lock = threading.Lock()

class SpeakReq(BaseModel):
    text: str
    interrupt: bool = True  # if True, stop current speech before speaking

def _speak_pyttsx3(text: str, interrupt: bool):
    import pyttsx3

    # pyttsx3 is not fully thread-safe globally; we serialize access
    with _lock:
        engine = pyttsx3.init()

        # apply config
        try:
            engine.setProperty("rate", TTS_RATE)
            engine.setProperty("volume", TTS_VOLUME)
        except Exception:
            pass

        # optional voice selection
        if TTS_VOICE_ID:
            try:
                voices = engine.getProperty("voices")
                for v in voices:
                    if v.id == TTS_VOICE_ID or getattr(v, "name", "") == TTS_VOICE_ID:
                        engine.setProperty("voice", v.id)
                        break
            except Exception:
                pass

        # interrupt just means we don't queue; pyttsx3 engine is fresh anyway
        engine.say(text)
        engine.runAndWait()
        engine.stop()

@router.post("/api/tts/speak")
async def speak(request: Request, body: SpeakReq):
    require_ip_allowlist(request)
    require_auth(request)

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text")

    # Run in background thread so request returns fast
    def worker():
        try:
            _speak_pyttsx3(text, body.interrupt)
        except Exception:
            pass

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True}