from tts import speak

if __name__ == "__main__":
    print("=== TTS TEST ===")

    while True:
        text = input("Enter text (or 'exit'): ").strip()

        if text.lower() == "exit":
            break

        if text:
            speak(text)