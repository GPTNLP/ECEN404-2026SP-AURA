import subprocess
import tempfile
import os

class TTSService:
    def __init__(self, voice="gmw/en-us", device="hw:1,0"):
        self.voice = voice
        self.device = device

    def speak(self, text: str):
        try:
            # Create temp wav file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                wav_path = f.name

            # Generate speech using espeak-ng
            cmd_tts = [
                "espeak-ng",
                "-v", self.voice,
                "-w", wav_path,
                text
            ]
            subprocess.run(cmd_tts, check=True)

            # Play audio using aplay
            cmd_play = [
                "aplay",
                "-D", self.device,
                wav_path
            ]
            subprocess.run(cmd_play, check=True)

        except Exception as e:
            print(f"[TTS ERROR] {e}")

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)