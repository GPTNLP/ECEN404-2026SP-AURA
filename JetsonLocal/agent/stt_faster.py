import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import time
import wave
import tempfile
from datetime import datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


WAKE_PHRASES = [
    "hey aura",
    "hi aura",
    "okay aura",
    "ok aura",
    "yo aura",
    "hey ora",
    "hey arua",
]

MOVEMENT_PATTERNS = {
    "forward": [
        "move forward",
        "go forward",
        "forward",
    ],
    "backward": [
        "move backward",
        "go backward",
        "move back",
        "go back",
        "backward",
        "back",
    ],
    "left": [
        "go to the left",
        "move to the left",
        "turn left",
        "go left",
        "move left",
        "left",
    ],
    "right": [
        "go to the right",
        "move to the right",
        "turn right",
        "go right",
        "move right",
        "right",
    ],
    "stop": [
        "stop moving",
        "stop",
        "halt",
        "pause",
    ],
}

BAD_WORD_PATTERNS = [
    r"fuck\w*",
    r"shit\w*",
    r"bitch\w*",
    r"ass",
    r"asshole\w*",
]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def censor_text(text: str) -> str:
    for pattern in BAD_WORD_PATTERNS:
        text = re.sub(
            rf"(?i)\b({pattern})\b",
            lambda m: "█" * len(m.group(0)),
            text
        )
    return text


def contains_bad_language(text: str) -> bool:
    for pattern in BAD_WORD_PATTERNS:
        if re.search(rf"(?i)\b({pattern})\b", text):
            return True
    return False


def contains_wake_phrase(text: str) -> bool:
    norm = normalize_text(text)
    for phrase in WAKE_PHRASES:
        if normalize_text(phrase) in norm:
            return True
    return False


def remove_wake_phrase(text: str) -> str:
    norm = normalize_text(text)
    for phrase in WAKE_PHRASES:
        phrase_norm = normalize_text(phrase)
        norm = re.sub(rf"\b{re.escape(phrase_norm)}\b", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def detect_last_movement_command(text: str):
    norm = normalize_text(text)

    # Only trigger if there's an action verb present
    ACTION_WORDS = ["move", "go", "turn", "stop", "halt"]

    has_action_word = any(word in norm for word in ACTION_WORDS)

    # If no action word AND not a short command → ignore
    if not has_action_word and len(norm.split()) > 2:
        return None

    matches = []
    for command, phrases in MOVEMENT_PATTERNS.items():
        for phrase in phrases:
            phrase_norm = normalize_text(phrase)
            pattern = rf"\b{re.escape(phrase_norm)}\b"

            for match in re.finditer(pattern, norm):
                matches.append((match.start(), command))

    if not matches:
        return None

    matches.sort(key=lambda x: x[0])
    return matches[-1][1]

class SpeechToText:
    def __init__(
        self,
        model_size: str = "base",
        input_device: int = 4,
        device_sample_rate: int = 48000,
        target_sample_rate: int = 16000,
        channels: int = 1,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        task: str = "transcribe",
        log_path: str = "~/SDP/AURA/JetsonLocal/storage/transcriptions.log",
        silence_threshold: float = 0.015,
    ):
        print(f"[STT] Loading faster-whisper model '{model_size}'...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("[STT] Model loaded successfully!")

        self.input_device = input_device
        self.device_sample_rate = device_sample_rate
        self.target_sample_rate = target_sample_rate
        self.channels = channels
        self.language = language
        self.task = task
        self.log_path = os.path.expanduser(log_path)
        self.silence_threshold = silence_threshold

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio

        audio = audio.flatten()
        duration = len(audio) / orig_sr

        old_times = np.linspace(0, duration, num=len(audio), endpoint=False)
        new_length = int(duration * target_sr)
        new_times = np.linspace(0, duration, num=new_length, endpoint=False)

        resampled = np.interp(new_times, old_times, audio).astype(np.float32)
        return resampled.reshape(-1, 1)

    def _save_wav(self, path: str, audio: np.ndarray, sample_rate: int) -> None:
        audio = np.clip(audio.flatten(), -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

    def _transcribe_audio_array(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) == 0:
            return ""

        audio_16k = self._resample_audio(audio, self.device_sample_rate, self.target_sample_rate)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_filename = tmp_file.name

        try:
            self._save_wav(temp_filename, audio_16k, self.target_sample_rate)

            segments, info = self.model.transcribe(
                temp_filename,
                task=self.task,
                language=self.language,
                vad_filter=True,
                beam_size=5,
            )

            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text

        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def record_fixed(self, seconds: float) -> np.ndarray:
        audio = sd.rec(
            int(seconds * self.device_sample_rate),
            samplerate=self.device_sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.input_device,
        )
        sd.wait()
        return audio

    def listen_for_wake_word(self, chunk_seconds: float = 1.8) -> bool:
        audio = self.record_fixed(chunk_seconds)

        peak = float(np.max(np.abs(audio)))
        if peak < self.silence_threshold:
            return False

        text = self._transcribe_audio_array(audio)
        if not text:
            return False

        print(f"[WAKE CHECK] {text}")

        if contains_wake_phrase(text):
            return True

        return False

    def listen_until_done(self, timeout_seconds: float = 10.0, end_silence_seconds: float = 1.2) -> str:
        print("[AURA] Listening...")

        start_time = time.time()
        chunk_seconds = 0.4
        collected = []

        speech_started = False
        silence_after_speech = 0.0

        while time.time() - start_time < timeout_seconds:
            audio = self.record_fixed(chunk_seconds)
            collected.append(audio)

            level = float(np.mean(np.abs(audio)))

            if level > self.silence_threshold:
                speech_started = True
                silence_after_speech = 0.0
            else:
                if speech_started:
                    silence_after_speech += chunk_seconds

            if speech_started and silence_after_speech >= end_silence_seconds:
                break

        if not collected:
            return ""

        full_audio = np.concatenate(collected, axis=0)

        peak = float(np.max(np.abs(full_audio)))
        mean = float(np.mean(np.abs(full_audio)))
        print(f"[AURA] Peak level: {peak:.6f}")
        print(f"[AURA] Mean level: {mean:.6f}")

        if not speech_started:
            return ""

        return self._transcribe_audio_array(full_audio)

    def log_transcript(self, text: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {text}\n")


def handle_user_text(text: str):
    cleaned = remove_wake_phrase(text)
    command = detect_last_movement_command(cleaned)

    print("\n" + "=" * 60)
    print(f"RAW TEXT: {text}")
    print(f"CLEANED TEXT: {cleaned}")

    if command:
        print(f"COMMAND: {command}")
        print("ACTION TYPE: MOVEMENT")
    else:
        print(f"LLM QUERY: {cleaned if cleaned else text}")
        print("ACTION TYPE: LLM")
    print("=" * 60)


def main():
    stt = SpeechToText(
        model_size="base",
        input_device=4,
        device_sample_rate=48000,
        target_sample_rate=16000,
        channels=1,
        device="cpu",
        compute_type="int8",
        language="en",
        task="transcribe",
        silence_threshold=0.015,
    )

    print("[STT] Device info:")
    print(sd.query_devices(stt.input_device))
    print()
    print("=" * 60)
    print(" AURA VOICE MODE")
    print("=" * 60)
    print("Say 'Hey AURA' to activate.")
    print("Then it will listen for up to 10 seconds.")
    print("If you start speaking, it will stay active until you finish.")
    print("If no speech is heard, it returns to wake mode.")
    print("Press Ctrl+C to exit.")
    print("-" * 60)

    try:
        while True:
            woke = stt.listen_for_wake_word(chunk_seconds=1.8)
            if not woke:
                continue

            print("[AURA] Wake word detected.")

            text = stt.listen_until_done(timeout_seconds=10.0, end_silence_seconds=1.2)

            if not text:
                print("[AURA] No speech heard. Returning to wake mode.")
                print("-" * 60)
                continue

            censored = censor_text(text)
            stt.log_transcript(censored)

            if contains_bad_language(text):
                print("[AURA] Warning: inappropriate language detected.")

            handle_user_text(text)
            print("[AURA] Returning to wake mode.")
            print("-" * 60)

    except KeyboardInterrupt:
        print("\n[STT] Exiting cleanly.")


if __name__ == "__main__":
    main()