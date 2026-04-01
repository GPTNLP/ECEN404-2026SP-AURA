from stt_service import SpeechToTextService
from voice_commands import detect_command


def main():
    stt = SpeechToTextService(
        input_device=4,
        device_sample_rate=48000,
        target_sample_rate=16000,
        model_size="base",
        device="cpu",
        compute_type="int8",
    )

    while True:
        user_in = input("\nPress Enter to record, or type q to quit: ").strip().lower()
        if user_in == "q":
            break

        print("[TEST] Wait 1 second, then speak.")
        text = stt.listen_once(seconds=6)

        print("\n[TRANSCRIPT]")
        print(text if text else "(no speech detected)")

        command = detect_command(text or "")

        print("\n[COMMAND DETECTED]")
        print(command if command else "None")


if __name__ == "__main__":
    main()