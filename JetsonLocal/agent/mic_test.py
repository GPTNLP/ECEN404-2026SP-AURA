import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "storage" / "audio_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stdout:
        print("[STDOUT]")
        print(result.stdout.strip())
    if result.stderr:
        print("[STDERR]")
        print(result.stderr.strip())
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def command_exists(name: str) -> bool:
    return subprocess.run(
        ["bash", "-lc", f"command -v {name}"],
        text=True,
        capture_output=True
    ).returncode == 0


def list_audio_devices():
    print("\n========== ALSA CAPTURE DEVICES ==========")
    if command_exists("arecord"):
        run_cmd(["arecord", "-l"])
        print("\n========== ALSA CAPTURE PCM NAMES ==========")
        run_cmd(["arecord", "-L"])
    else:
        print("arecord not found. Install ALSA utils:")
        print("  sudo apt update && sudo apt install -y alsa-utils")

    print("\n========== ALSA PLAYBACK DEVICES ==========")
    if command_exists("aplay"):
        run_cmd(["aplay", "-l"])
    else:
        print("aplay not found. Install ALSA utils:")
        print("  sudo apt update && sudo apt install -y alsa-utils")

    print("\n========== USB DEVICES ==========")
    if command_exists("lsusb"):
        run_cmd(["lsusb"])
    else:
        print("lsusb not found. Install usbutils if needed.")


def record_test(device: str | None, seconds: int, rate: int, channels: int) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    wav_path = OUTPUT_DIR / f"mic_test_{timestamp}.wav"

    cmd = [
        "arecord",
        "-f", "S16_LE",
        "-r", str(rate),
        "-c", str(channels),
        "-d", str(seconds),
        str(wav_path),
    ]

    if device:
        cmd[1:1] = ["-D", device]

    run_cmd(cmd, check=True)

    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise RuntimeError("Recording finished but file is empty.")

    print(f"\n[OK] Recording saved to: {wav_path}")
    print(f"[INFO] File size: {wav_path.stat().st_size} bytes")
    return wav_path


def playback_test(wav_path: Path):
    if not command_exists("aplay"):
        print("\n[WARN] aplay not found, skipping playback.")
        return

    print("\n========== PLAYBACK TEST ==========")
    run_cmd(["aplay", str(wav_path)])


def main():
    parser = argparse.ArgumentParser(description="Simple Jetson mic test for AURA.")
    parser.add_argument("--list", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--device", type=str, default=None, help="ALSA device name, ex: plughw:2,0")
    parser.add_argument("--seconds", type=int, default=5, help="How long to record")
    parser.add_argument("--rate", type=int, default=48000, help="Sample rate")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels")
    parser.add_argument("--no-playback", action="store_true", help="Skip playback after recording")
    args = parser.parse_args()

    if not command_exists("arecord"):
        print("arecord is not installed.")
        print("Run:")
        print("  sudo apt update && sudo apt install -y alsa-utils")
        sys.exit(1)

    if args.list:
        list_audio_devices()
        return

    print("========== AURA MIC TEST ==========")
    print(f"device   : {args.device or 'default'}")
    print(f"seconds  : {args.seconds}")
    print(f"rate     : {args.rate}")
    print(f"channels : {args.channels}")

    wav_path = record_test(
        device=args.device,
        seconds=args.seconds,
        rate=args.rate,
        channels=args.channels,
    )

    if not args.no_playback:
        playback_test(wav_path)

    print("\nDone.")
    print(f"Test file: {wav_path}")


if __name__ == "__main__":
    main()