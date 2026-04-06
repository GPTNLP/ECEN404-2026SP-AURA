from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
AGENT_DIR = BASE_DIR / "agent"

if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from tts import TTSService


if __name__ == "__main__":
    text = "Hello. This is Aura voice test."
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:]).strip() or text

    tts = TTSService()
    print(f"[TEST] using device: {tts.device}")
    print(f"[TEST] voice_id: {tts.voice_id}")
    print(f"[TEST] model_id: {tts.model_id}")
    print(f"[TEST] speaking: {text}")

    ok = tts.speak(text)
    print(f"[TEST] success={ok}")