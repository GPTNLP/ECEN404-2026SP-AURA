import queue
import sys
import threading
import time

import numpy as np

try:
    import sounddevice as sd
except Exception as e:
    print(f"[ERROR] sounddevice import failed: {e}")
    print("Install with: pip install sounddevice")
    sys.exit(1)

try:
    from faster_whisper import WhisperModel
except Exception as e:
    print(f"[ERROR] faster_whisper import failed: {e}")
    print("Install with: pip install faster-whisper")
    sys.exit(1)


SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SECONDS = 0.5
CHUNK_SAMPLES = int(SAMPLE_RATE * BLOCK_SECONDS)

AUDIO_QUEUE = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[AUDIO STATUS] {status}", file=sys.stderr)
    AUDIO_QUEUE.put(indata.copy())


def rms_level(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))


def print_level_bar(level: float):
    level = min(level * 50.0, 1.0)
    bars = int(level * 30)
    bar = "#" * bars + "-" * (30 - bars)
    print(f"\rMIC [{bar}]", end="", flush=True)


def main():
    print("[INFO] loading Whisper model...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print("[INFO] model loaded")
    print("[INFO] speak into the mic. Press Ctrl+C to stop.\n")

    rolling_audio = np.zeros((0,), dtype=np.float32)
    last_transcribed_len = 0
    last_print_time = 0.0

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
        blocksize=CHUNK_SAMPLES,
    )

    try:
        with stream:
            while True:
                chunk = AUDIO_QUEUE.get()
                mono = chunk[:, 0].astype(np.float32)

                level = rms_level(mono)
                print_level_bar(level)

                rolling_audio = np.concatenate([rolling_audio, mono])

                max_seconds = 6
                max_samples = SAMPLE_RATE * max_seconds
                if len(rolling_audio) > max_samples:
                    rolling_audio = rolling_audio[-max_samples:]

                now = time.time()
                enough_audio = len(rolling_audio) >= SAMPLE_RATE * 2

                if enough_audio and (now - last_print_time) >= 1.2:
                    segments, info = model.transcribe(
                        rolling_audio,
                        language="en",
                        vad_filter=True,
                        beam_size=1,
                    )

                    text = " ".join(seg.text.strip() for seg in segments).strip()

                    if text:
                        print("\nYOU:", text)
                        last_print_time = now

    except KeyboardInterrupt:
        print("\n[INFO] stopped")


if __name__ == "__main__":
    main()