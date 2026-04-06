import os
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
JETSONLOCAL_DIR = BASE_DIR.parent
ENV_PATH = JETSONLOCAL_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


class TTSService:
    def __init__(
        self,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.api_key = (os.getenv("ELEVENLABS_API_KEY", "")).strip()
        self.voice_id = (voice_id or os.getenv("ELEVENLABS_VOICE_ID", "")).strip()
        self.model_id = (model_id or os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")).strip()
        self.device = (device or os.getenv("AURA_TTS_DEVICE", "plughw:0,0")).strip()

    # ---------------------------
    # ONLINE (ElevenLabs)
    # ---------------------------
    def _speak_elevenlabs(self, text: str) -> bool:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.75,
                "similarity_boost": 0.85,
            },
        }

        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tmp_path = f.name
            f.write(response.content)

        try:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
                check=True,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        print(f"[TTS] ElevenLabs: {text}")
        return True

    # ---------------------------
    # OFFLINE (espeak)
    # ---------------------------
    def _speak_offline(self, text: str) -> bool:
        try:
            subprocess.run(["espeak-ng", text], check=True)
            print(f"[TTS] Offline fallback: {text}")
            return True
        except Exception as e:
            print(f"[TTS ERROR] offline failed: {e}")
            return False

    # ---------------------------
    # MAIN ENTRY
    # ---------------------------
    def speak(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False

        # Try ElevenLabs first
        if self.api_key and self.voice_id:
            try:
                return self._speak_elevenlabs(text)
            except Exception as e:
                print(f"[TTS] ElevenLabs failed → fallback: {e}")

        # Fallback to offline
        return self._speak_offline(text)