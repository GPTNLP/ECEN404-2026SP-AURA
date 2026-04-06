import os
import shutil
import subprocess
import tempfile
import threading
from typing import Optional


class TTSService:
    def __init__(
        self,
        voice: Optional[str] = None,
        device: Optional[str] = None,
        rate: Optional[int] = None,
    ):
        self.voice = (voice or os.getenv("AURA_TTS_VOICE", "en-us")).strip()
        self.device = (device or os.getenv("AURA_TTS_DEVICE", "default")).strip()
        self.rate = int(os.getenv("AURA_TTS_RATE", str(rate if rate is not None else 140)))
        self._lock = threading.Lock()

    def _find_tts_binary(self) -> str:
        for candidate in ("espeak-ng", "espeak"):
            path = shutil.which(candidate)
            if path:
                return path
        raise FileNotFoundError("Neither 'espeak-ng' nor 'espeak' was found in PATH.")

    def _play_wav(self, wav_path: str) -> None:
        play_attempts = [
            ["aplay", "-D", self.device, wav_path],
            ["aplay", wav_path],
        ]

        for cmd in play_attempts:
            try:
                subprocess.run(cmd, check=True)
                return
            except Exception as exc:
                print(f"[TTS] playback attempt failed: {' '.join(cmd)} -> {exc}")

        ffplay_path = shutil.which("ffplay")
        if ffplay_path:
            try:
                subprocess.run(
                    [
                        ffplay_path,
                        "-nodisp",
                        "-autoexit",
                        "-loglevel",
                        "quiet",
                        wav_path,
                    ],
                    check=True,
                )
                return
            except Exception as exc:
                print(f"[TTS] ffplay fallback failed: {exc}")

        paplay_path = shutil.which("paplay")
        if paplay_path:
            try:
                subprocess.run([paplay_path, wav_path], check=True)
                return
            except Exception as exc:
                print(f"[TTS] paplay fallback failed: {exc}")

        raise RuntimeError("All playback methods failed")

    def _apply_tars_effect(self, input_wav: str) -> str:
        sox_path = shutil.which("sox")
        if not sox_path:
            print("[TTS] sox not found, skipping TARS effect")
            return input_wav

        output_wav = input_wav.replace(".wav", "_tars.wav")

        try:
            subprocess.run(
                [
                    sox_path,
                    input_wav,
                    output_wav,
                    "pitch",
                    "-180",
                    "treble",
                    "-8",
                    "bass",
                    "+2",
                    "echo",
                    "0.8",
                    "0.88",
                    "20",
                    "0.25",
                    "25",
                    "0.15",
                ],
                check=True,
            )
            return output_wav
        except Exception as exc:
            print(f"[TTS] sox effect failed, using original wav: {exc}")
            return input_wav

    def speak(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            print("[TTS] empty text, skipping")
            return False

        wav_path = None
        final_wav_path = None

        with self._lock:
            try:
                tts_bin = self._find_tts_binary()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    wav_path = f.name

                subprocess.run(
                    [
                        tts_bin,
                        "-v",
                        self.voice,
                        "-s",
                        str(self.rate),
                        "-p",
                        "25",
                        "-g",
                        "6",
                        "-w",
                        wav_path,
                        text,
                    ],
                    check=True,
                )

                final_wav_path = self._apply_tars_effect(wav_path)
                self._play_wav(final_wav_path)

                print(f"[TTS] spoke: {text}")
                return True

            except Exception as exc:
                print(f"[TTS ERROR] {exc}")
                return False

            finally:
                for path in (wav_path, final_wav_path):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass