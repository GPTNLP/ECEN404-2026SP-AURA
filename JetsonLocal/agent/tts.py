import os
import shutil
import subprocess
import tempfile
from typing import Optional


class TTSService:
    def __init__(self, voice: str = "en-us", device: Optional[str] = "hw:3,0"):
        self.voice = voice
        self.device = device or "hw:3,0"

    def _find_tts_binary(self) -> str:
        for candidate in ("espeak-ng", "espeak"):
            path = shutil.which(candidate)
            if path:
                return path
        raise FileNotFoundError("Neither 'espeak-ng' nor 'espeak' was found in PATH.")

    def speak(self, text: str) -> bool:
        if not text or not text.strip():
            print("[TTS] empty text, skipping")
            return False

        wav_path = None

        try:
            tts_bin = self._find_tts_binary()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                wav_path = f.name

            subprocess.run(
                [
                    tts_bin,
                    "-v",
                    self.voice,
                    "-w",
                    wav_path,
                    text,
                ],
                check=True,
            )

            subprocess.run(
                [
                    "aplay",
                    "-D",
                    self.device,
                    wav_path,
                ],
                check=True,
            )

            return True

        except Exception as exc:
            print(f"[TTS ERROR] {exc}")
            return False

        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass