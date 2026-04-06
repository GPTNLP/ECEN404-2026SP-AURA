import sys
import pyttsx3


def main() -> int:
    text = "Hello. This is AURA text to speech test."

    try:
        engine = pyttsx3.init()

        # Slightly slower and clearer
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)

        voices = engine.getProperty("voices")
        print("Available voices:")
        for i, voice in enumerate(voices):
            print(f"[{i}] name={voice.name} id={voice.id}")

        print(f"\nSpeaking: {text}")
        engine.say(text)
        engine.runAndWait()
        engine.stop()

        print("TTS test finished.")
        return 0

    except Exception as exc:
        print(f"TTS test failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())