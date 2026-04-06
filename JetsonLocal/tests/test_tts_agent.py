from pathlib import Path
import sys
import subprocess
import re

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent.tts import TTSService


def detect_usb_playback_device() -> str:
    try:
        result = subprocess.run(
            ["aplay", "-l"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if "USB PnP Audio Device" in line or "USB Audio" in line:
                m = re.search(r"card (\d+): .*device (\d+):", line)
                if m:
                    return f"plughw:{m.group(1)},{m.group(2)}"
    except Exception as exc:
        print(f"[TEST] device detection failed: {exc}")

    return "default"


if __name__ == "__main__":
    text = "Hello. This is Aura voice test."
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:]).strip() or text

    device = detect_usb_playback_device()
    print(f"[TEST] using device: {device}")
    print(f"[TEST] speaking: {text}")

    tts = TTSService(
        voice="en-us",
        device=device,
        rate=132,
    )

    ok = tts.speak(text)
    print(f"[TEST] success={ok}")