import os
import requests
import tempfile
import subprocess

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

ELEVENLABS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

# CHANGE THIS IF NEEDED
AUDIO_DEVICE = os.getenv("AUDIO_DEVICE", "default")  
# examples:
# "default"
# "hw:0,0"
# "plughw:0,0"


def speak(text: str):
    if not ELEVENLABS_API_KEY or not VOICE_ID:
        print("[TTS ERROR] Missing ElevenLabs config")
        return

    try:
        print(f"[TTS] generating -> {text}")

        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }

        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.75,
                "similarity_boost": 0.9
            }
        }

        response = requests.post(ELEVENLABS_URL, json=payload, headers=headers)

        if response.status_code != 200:
            print("[TTS ERROR] ElevenLabs failed:", response.text)
            return

        # Save temp audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(response.content)
            temp_path = f.name

        print(f"[TTS] playing on device: {AUDIO_DEVICE}")

        # Play audio
        subprocess.run([
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel", "quiet",
            "-af", "volume=1.0",
            temp_path
        ])

        os.remove(temp_path)

    except Exception as e:
        print("[TTS ERROR]", e)