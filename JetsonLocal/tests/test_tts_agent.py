from agent.tts import TTSService

if __name__ == "__main__":
    tts = TTSService(
        voice="gmw/en-us",
        device="hw:1,0"  # change if needed
    )

    tts.speak("Hello. Aura text to speech is now integrated.")