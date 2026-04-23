import os
import json
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
JETSONLOCAL_DIR = BASE_DIR.parent
ENV_PATH = JETSONLOCAL_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)

TTS_SETTINGS_PATH = Path(
    os.path.expanduser(
        os.getenv(
            "AURA_TTS_SETTINGS_PATH",
            str(JETSONLOCAL_DIR / "storage" / "tts_settings.json"),
        )
    )
)


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

        self.ffmpeg_bin = shutil.which("ffmpeg")
        self.aplay_bin = shutil.which("aplay")
        self.espeak_bin = shutil.which("espeak-ng")

    def _read_volume_percent(self) -> int:
        try:
            with open(TTS_SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            percent = int(float(data.get("volume_percent", 50)))
            return max(0, min(100, percent))
        except Exception:
            return 50

    def _volume_multiplier(self, volume_percent: int) -> float:
        volume_percent = max(0, min(100, int(volume_percent)))
        return max(0.0, volume_percent / 100.0)

    # ---------------------------
    # INTERNAL PLAYBACK HELPERS
    # ---------------------------
    def _play_wav_on_device(self, wav_path: str) -> None:
        if not self.aplay_bin:
            raise RuntimeError("aplay is not installed or not found in PATH.")

        cmd = [self.aplay_bin]
        if self.device:
            cmd.extend(["-D", self.device])
        cmd.append(wav_path)

        subprocess.run(cmd, check=True)

    def _decode_mp3_to_wav(self, mp3_path: str, wav_path: str, volume_percent: int = 50) -> None:
        if not self.ffmpeg_bin:
            raise RuntimeError("ffmpeg is not installed or not found in PATH.")

        multiplier = self._volume_multiplier(volume_percent)

        subprocess.run(
            [
                self.ffmpeg_bin,
                "-y",
                "-loglevel",
                "quiet",
                "-i",
                mp3_path,
                "-filter:a",
                f"volume={multiplier:.3f}",
                wav_path,
            ],
            check=True,
        )

    # ---------------------------
    # ONLINE (ElevenLabs)
    # ---------------------------
    def _speak_elevenlabs(self, text: str, volume_percent: int = 50) -> bool:
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is missing.")
        if not self.voice_id:
            raise RuntimeError("ELEVENLABS_VOICE_ID is missing.")

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

        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()

        mp3_tmp = None
        wav_tmp = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                mp3_tmp = f.name
                f.write(response.content)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                wav_tmp = f.name

            self._decode_mp3_to_wav(mp3_tmp, wav_tmp, volume_percent=volume_percent)
            self._play_wav_on_device(wav_tmp)

            print(f"[TTS] ElevenLabs: {text} | volume={volume_percent}%")
            return True

        finally:
            if mp3_tmp:
                Path(mp3_tmp).unlink(missing_ok=True)
            if wav_tmp:
                Path(wav_tmp).unlink(missing_ok=True)

    # ---------------------------
    # OFFLINE (espeak-ng -> aplay)
    # ---------------------------
    def _speak_offline(self, text: str, volume_percent: int = 50) -> bool:
        if not self.espeak_bin:
            print("[TTS ERROR] offline failed: espeak-ng not found in PATH")
            return False

        if not self.aplay_bin:
            print("[TTS ERROR] offline failed: aplay not found in PATH")
            return False

        try:
            amplitude = max(0, min(200, int(round((max(0, min(100, volume_percent)) / 100.0) * 200))))
            espeak_cmd = [self.espeak_bin, "-a", str(amplitude), "--stdout", text]
            aplay_cmd = [self.aplay_bin]

            if self.device:
                aplay_cmd.extend(["-D", self.device])

            espeak_proc = subprocess.Popen(
                espeak_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            aplay_proc = subprocess.Popen(
                aplay_cmd,
                stdin=espeak_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            if espeak_proc.stdout is not None:
                espeak_proc.stdout.close()

            espeak_stderr = espeak_proc.stderr.read().decode("utf-8", errors="ignore") if espeak_proc.stderr else ""
            aplay_stderr = aplay_proc.stderr.read().decode("utf-8", errors="ignore") if aplay_proc.stderr else ""

            espeak_rc = espeak_proc.wait()
            aplay_rc = aplay_proc.wait()

            if espeak_rc != 0:
                raise RuntimeError(f"espeak-ng failed: {espeak_stderr.strip()}")
            if aplay_rc != 0:
                raise RuntimeError(f"aplay failed: {aplay_stderr.strip()}")

            print(f"[TTS] Offline fallback: {text} | volume={volume_percent}%")
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

        volume = self._read_volume_percent()

        if self.api_key and self.voice_id:
            try:
                return self._speak_elevenlabs(text, volume_percent=volume)
            except Exception as e:
                print(f"[TTS] ElevenLabs failed -> fallback: {e}")

        return self._speak_offline(text, volume_percent=volume)