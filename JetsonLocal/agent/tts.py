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

        if not self.api_key:
            raise RuntimeError(f"ELEVENLABS_API_KEY is missing from environment. Looked for .env at: {ENV_PATH}")
        if not self.voice_id:
            raise RuntimeError(f"ELEVENLABS_VOICE_ID is missing from environment. Looked for .env at: {ENV_PATH}")

    def _play_file(self, audio_path: str) -> None:
        attempts = [
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path],
            ["mpg123", "-q", audio_path],
            ["mpv", "--no-video", "--really-quiet", audio_path],
        ]

        for cmd in attempts:
            try:
                subprocess.run(cmd, check=True)
                return
            except FileNotFoundError:
                continue
            except Exception as exc:
                print(f"[TTS] playback attempt failed: {' '.join(cmd)} -> {exc}")

        raise RuntimeError(
            "No working MP3 playback command found. Install ffmpeg (ffplay), mpg123, or mpv."
        )

    def speak(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            print("[TTS] empty text, skipping")
            return False

        tmp_path = None

        try:
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

            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tmp_path = f.name
                f.write(response.content)

            self._play_file(tmp_path)

            print(f"[TTS] spoke with ElevenLabs: {text}")
            return True

        except Exception as exc:
            print(f"[TTS ERROR] {exc}")
            return False

        finally:
            if tmp_path and Path(tmp_path).exists():
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass