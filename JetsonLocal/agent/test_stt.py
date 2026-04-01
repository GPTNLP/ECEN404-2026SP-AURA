import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tempfile
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5

def main():
    print("[INFO] Loading whisper model...")
    model = WhisperModel("base", device="cpu", compute_type="int8")

    print(f"[INFO] Recording for {RECORD_SECONDS} seconds... speak now.")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32"
    )
    sd.wait()

    print("[INFO] Recording finished. Transcribing...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name

    try:
        audio_int16 = (audio.flatten() * 32767).astype(np.int16)

        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        segments, info = model.transcribe(
            temp_path,
            language="en",
            task="transcribe"
        )

        text = " ".join(seg.text for seg in segments).strip()

        print("\n[RESULT]")
        print(text if text else "(no speech detected)")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()