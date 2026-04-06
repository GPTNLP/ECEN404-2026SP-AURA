import os
import shutil
import subprocess
import sys
from pathlib import Path


TEST_TEXT = "Hello. This is an AURA speaker output test."
WAV_PATH = Path("/tmp/aura_tts_test.wav")


def run_cmd(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result


def main() -> int:
    print("=== AURA TTS OUTPUT TEST ===")

    espeak_path = shutil.which("espeak")
    aplay_path = shutil.which("aplay")

    print(f"espeak: {espeak_path}")
    print(f"aplay : {aplay_path}")

    if not espeak_path:
        print("ERROR: espeak is not installed.")
        print("Install with: sudo apt-get update && sudo apt-get install -y espeak")
        return 1

    if not aplay_path:
        print("ERROR: aplay is not installed.")
        print("Install with: sudo apt-get update && sudo apt-get install -y alsa-utils")
        return 1

    print("\n=== ALSA PLAYBACK DEVICES ===")
    run_cmd(["aplay", "-l"])

    print("\n=== ALSA PCM DEVICES ===")
    run_cmd(["aplay", "-L"])

    if WAV_PATH.exists():
        WAV_PATH.unlink()

    print("\n=== GENERATING TEST WAV WITH ESPEAK ===")
    run_cmd([
        "espeak",
        "-v", "en-us",
        "-s", "160",
        "-w", str(WAV_PATH),
        TEST_TEXT,
    ], check=True)

    if not WAV_PATH.exists():
        print("ERROR: WAV file was not created.")
        return 1

    size = WAV_PATH.stat().st_size
    print(f"Generated WAV: {WAV_PATH} ({size} bytes)")

    if size <= 44:
        print("ERROR: WAV file looks empty.")
        return 1

    device = sys.argv[1] if len(sys.argv) > 1 else None

    print("\n=== PLAYING TEST WAV ===")
    if device:
        print(f"Using explicit ALSA device: {device}")
        result = run_cmd(["aplay", "-D", device, str(WAV_PATH)])
    else:
        print("Using default ALSA device")
        result = run_cmd(["aplay", str(WAV_PATH)])

    print("\n=== RESULT ===")
    if result.returncode == 0:
        print("Playback command completed.")
        print("If you still heard nothing, the issue is likely the selected output device, volume, or wiring.")
        return 0

    print("Playback failed.")
    print("Try running this script again with an explicit device from 'aplay -l'.")
    print("Example:")
    print("  python tests/test_tts_output.py hw:0,0")
    print("  python tests/test_tts_output.py hw:1,0")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())