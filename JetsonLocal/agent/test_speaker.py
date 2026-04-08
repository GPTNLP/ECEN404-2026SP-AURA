import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
JETSONLOCAL_DIR = BASE_DIR.parent
ENV_PATH = JETSONLOCAL_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


def run_cmd(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"[EXIT] {result.returncode}")

    if result.stdout.strip():
        print("[STDOUT]")
        print(result.stdout.strip())

    if result.stderr.strip():
        print("[STDERR]")
        print(result.stderr.strip())

    return result


def main():
    device = os.getenv("AURA_TTS_DEVICE", "default").strip()

    print("=" * 60)
    print("AURA Speaker Test")
    print(f"Configured output device: {device}")
    print("=" * 60)

    aplay_bin = shutil.which("aplay")
    espeak_bin = shutil.which("espeak-ng")
    speaker_test_bin = shutil.which("speaker-test")

    print(f"aplay: {aplay_bin}")
    print(f"espeak-ng: {espeak_bin}")
    print(f"speaker-test: {speaker_test_bin}")

    if not aplay_bin:
        print("\n[ERROR] aplay is not installed.")
        return

    print("\nAvailable ALSA playback devices:")
    run_cmd([aplay_bin, "-L"])

    print("\nHardware playback devices:")
    run_cmd([aplay_bin, "-l"])

    if speaker_test_bin:
        print("\nTesting speaker with speaker-test...")
        print("You should hear noise for a few seconds.")
        run_cmd([speaker_test_bin, "-D", device, "-c", "2", "-t", "wav", "-l", "1"])
    else:
        print("\n[WARN] speaker-test not found, skipping raw speaker test.")

    if espeak_bin:
        print("\nTesting TTS path with espeak-ng -> aplay ...")
        print("You should hear: Hello, this is an AURA speaker test.")
        espeak_cmd = [espeak_bin, "--stdout", "Hello, this is an AURA speaker test."]
        aplay_cmd = [aplay_bin, "-D", device]

        espeak_proc = subprocess.Popen(espeak_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        aplay_proc = subprocess.Popen(aplay_cmd, stdin=espeak_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if espeak_proc.stdout is not None:
            espeak_proc.stdout.close()

        espeak_stdout, espeak_stderr = espeak_proc.communicate()
        aplay_stdout, aplay_stderr = aplay_proc.communicate()

        print(f"[espeak-ng exit] {espeak_proc.returncode}")
        if espeak_stderr:
            print("[espeak-ng stderr]")
            print(espeak_stderr.decode(errors="ignore").strip())

        print(f"[aplay exit] {aplay_proc.returncode}")
        if aplay_stderr:
            print("[aplay stderr]")
            print(aplay_stderr.decode(errors="ignore").strip())
    else:
        print("\n[WARN] espeak-ng not found, skipping speech test.")

    print("\nDone.")


if __name__ == "__main__":
    main()