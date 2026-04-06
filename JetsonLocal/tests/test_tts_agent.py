from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent.tts import TTSService


if __name__ == "__main__":
    tts = TTSService(
        voice="en-us",
        device="default",
    )
    print(f"[TEST] using device: {tts.device}")
    ok = tts.speak("Hello. Aura text to speech is now integrated.")
    print(f"[TEST] success={ok}")