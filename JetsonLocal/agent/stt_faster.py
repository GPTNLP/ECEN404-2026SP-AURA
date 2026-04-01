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


# ============================================================
# TUNING
# ============================================================
WAKE_AUDIO_THRESHOLD = 0.035
COMMAND_AUDIO_THRESHOLD = 0.020
END_SILENCE_SECONDS = 1.1
WAKE_CHUNK_SECONDS = 2.0
COMMAND_CHUNK_SECONDS = 0.35
COMMAND_TIMEOUT_SECONDS = 10.0
NEAR_FIELD_BOOST = 1.0
USE_ONLY_LEFT_CHANNEL = True

# how many wake words must appear in the wake chunk
# 1 = more permissive
# 2 = stricter
WAKE_MATCH_MODE = 1


# ============================================================
# PHRASES / RULES
# ============================================================
MOVEMENT_PATTERNS = {
    "forward": [
        "move forward",
        "go forward",
        "drive forward",
        "forward",
    ],
    "backward": [
        "move backward",
        "go backward",
        "move back",
        "go back",
        "drive backward",
        "backward",
        "back",
    ],
    "left": [
        "turn left",
        "go left",
        "move left",
        "move to the left",
        "go to the left",
        "left",
    ],
    "right": [
        "turn right",
        "go right",
        "move right",
        "move to the right",
        "go to the right",
        "right",
    ],
    "stop": [
        "stop moving",
        "stop",
        "halt",
        "pause",
        "freeze",
    ],
}

ACTION_WORDS = [
    "move", "go", "turn", "drive", "stop", "halt", "pause", "freeze"
]

BAD_WORD_PATTERNS = [
    r"fuck\w*",
    r"shit\w*",
    r"bitch\w*",
    r"asshole\w*",
    r"\bass\b",
]

# very forgiving wake aliases
# since whisper is hearing "aura" as "or a", "ora", "or", etc.
WAKE_PREFIXES = ["hey", "hi", "ok", "okay", "yo"]
WAKE_AURA_ALIASES = [
    "aura",
    "ora",
    "or a",
    "or",
    "oura",
    "arua",
]


# ============================================================
# TEXT HELPERS
# ============================================================
def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("a u r a", "aura")
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


def wake_score(text: str):
    """
    Returns:
      (matched: bool, cleaned_text_after_wake: str, debug_reason: str)

    Very forgiving logic:
    - exact-ish pair like "hey aura", "hey or a", "okay ora"
    - also supports one-shot like "hey or a move forward"
    """
    norm = normalize_text(text)
    if not norm:
        return False, "", "empty"

    tokens = norm.split()

    # Build possible wake phrases
    candidates = []
    for prefix in WAKE_PREFIXES:
        for alias in WAKE_AURA_ALIASES:
            candidates.append(f"{prefix} {alias}")

    # Strong match: exact candidate appears anywhere
    for cand in candidates:
        cand_norm = normalize_text(cand)
        pattern = rf"\b{re.escape(cand_norm)}\b"
        m = re.search(pattern, norm)
        if m:
            leftover = norm[m.end():].strip()
            return True, leftover, f"exact:{cand_norm}"

    # Token-based fuzzy-ish match:
    # e.g. ["hey", "or", "a", "move", "forward"]
    # or  ["hey", "ora", "move", "forward"]
    for i in range(len(tokens)):
        if tokens[i] in WAKE_PREFIXES:
            # prefix + one-token alias
            if i + 1 < len(tokens):
                two = f"{tokens[i]} {tokens[i+1]}"
                if tokens[i+1] in {"aura", "ora", "or", "oura", "arua"}:
                    leftover = " ".join(tokens[i+2:]).strip()
                    return True, leftover, f"token_pair:{two}"

            # prefix + two-token alias ("or a")
            if i + 2 < len(tokens):
                three = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                if f"{tokens[i+1]} {tokens[i+2]}" == "or a":
                    leftover = " ".join(tokens[i+3:]).strip()
                    return True, leftover, f"token_triplet:{three}"

    # More permissive fallback:
    # allow wake if both a prefix and aura-ish alias exist anywhere
    has_prefix = any(tok in WAKE_PREFIXES for tok in tokens)
    has_auraish = (
        "aura" in tokens or
        "ora" in tokens or
        "or" in tokens or
        "oura" in tokens or
        "arua" in tokens or
        "or a" in norm
    )

    if WAKE_MATCH_MODE == 1 and has_prefix and has_auraish:
        # remove the first prefix and first aura-ish fragment loosely
        cleaned = norm
        cleaned = re.sub(r"\b(hey|hi|ok|okay|yo)\b", " ", cleaned, count=1)
        cleaned = re.sub(r"\b(aura|ora|or|oura|arua)\b", " ", cleaned, count=1)
        cleaned = re.sub(r"\bor a\b", " ", cleaned, count=1)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return True, cleaned, "fallback_prefix+auraish"

    return False, "", "no_match"


def remove_wake_phrase(text: str) -> str:
    matched, leftover, _ = wake_score(text)
    if matched:
        return leftover
    return normalize_text(text)


def classify_intent(text: str) -> str:
    norm = normalize_text(text)
    if not norm:
        return "empty"

    movement = detect_last_movement_command(norm)
    if movement:
        return "movement"

    return "llm"


def detect_last_movement_command(text: str):
    norm = normalize_text(text)
    if not norm:
        return None

    has_action_word = any(re.search(rf"\b{re.escape(word)}\b", norm) for word in ACTION_WORDS)
    word_count = len(norm.split())

    matches = []

    for command, phrases in MOVEMENT_PATTERNS.items():
        for phrase in phrases:
            phrase_norm = normalize_text(phrase)

            is_single_word_direction = phrase_norm in {"left", "right", "forward", "back", "backward", "stop"}

            if is_single_word_direction and word_count > 2 and not has_action_word:
                continue

            pattern = rf"\b{re.escape(phrase_norm)}\b"
            for match in re.finditer(pattern, norm):
                matches.append((match.start(), command, phrase_norm))

    if not matches:
        return None

    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


# ============================================================
# AUDIO / STT
# ============================================================
class SpeechToText:
    def __init__(
        self,
        model_size: str = "base",
        input_device: int = 4,
        device_sample_rate: int = 48000,
        target_sample_rate: int = 16000,
        channels: int = 2,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        task: str = "transcribe",
        log_path: str = "~/SDP/AURA/JetsonLocal/storage/transcriptions.log",
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

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            mono = audio.astype(np.float32)
        else:
            if USE_ONLY_LEFT_CHANNEL:
                mono = audio[:, 0].astype(np.float32)
            else:
                mono = np.mean(audio, axis=1).astype(np.float32)

        mono = mono * NEAR_FIELD_BOOST
        mono = np.clip(mono, -1.0, 1.0)
        return mono.reshape(-1, 1)

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

        mono = self._prepare_audio(audio)
        audio_16k = self._resample_audio(mono, self.device_sample_rate, self.target_sample_rate)

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

    def analyze_level(self, audio: np.ndarray):
        mono = self._prepare_audio(audio).flatten()
        peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
        mean = float(np.mean(np.abs(mono))) if len(mono) else 0.0
        return peak, mean

    def listen_for_wake_word(self, chunk_seconds: float = WAKE_CHUNK_SECONDS):
        audio = self.record_fixed(chunk_seconds)
        peak, mean = self.analyze_level(audio)

        if peak < WAKE_AUDIO_THRESHOLD:
            return False, "", "", "too_quiet"

        text = self._transcribe_audio_array(audio)
        if not text:
            return False, "", "", "no_text"

        matched, leftover, reason = wake_score(text)
        print(f"[WAKE CHECK] {text}  | reason={reason}")

        return matched, text, leftover, reason

    def listen_until_done(
        self,
        timeout_seconds: float = COMMAND_TIMEOUT_SECONDS,
        end_silence_seconds: float = END_SILENCE_SECONDS,
        chunk_seconds: float = COMMAND_CHUNK_SECONDS,
    ) -> str:
        print("[AURA] Listening...")

        start_time = time.time()
        collected = []

        speech_started = False
        silence_after_speech = 0.0

        while time.time() - start_time < timeout_seconds:
            audio = self.record_fixed(chunk_seconds)
            collected.append(audio)

            peak, mean = self.analyze_level(audio)

            if peak >= COMMAND_AUDIO_THRESHOLD:
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
        peak, mean = self.analyze_level(full_audio)

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


# ============================================================
# ACTION ROUTING
# ============================================================
def handle_user_text(text: str):
    cleaned = remove_wake_phrase(text)
    movement = detect_last_movement_command(cleaned)
    intent = classify_intent(cleaned)

    print("\n" + "=" * 60)
    print(f"RAW TEXT: {text}")
    print(f"CLEANED TEXT: {cleaned}")

    if intent == "movement" and movement:
        print("ACTION TYPE: MOVEMENT")
        print(f"COMMAND: {movement}")
    elif intent == "llm":
        print("ACTION TYPE: LLM")
        print(f"LLM QUERY: {cleaned if cleaned else text}")
    else:
        print("ACTION TYPE: NONE")
        print("No usable command or query detected.")
    print("=" * 60)


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    stt = SpeechToText(
        model_size="base",
        input_device=4,
        device_sample_rate=48000,
        target_sample_rate=16000,
        channels=2,
        device="cpu",
        compute_type="int8",
        language="en",
        task="transcribe",
    )

    print("[STT] Device info:")
    print(sd.query_devices(stt.input_device))
    print()
    print("=" * 60)
    print(" AURA VOICE MODE")
    print("=" * 60)
    print("Say 'Hey AURA' to activate.")
    print("It now accepts misheard versions like 'hey ora' or 'hey or a'.")
    print("Press Ctrl+C to exit.")
    print("-" * 60)

    try:
        while True:
            woke, wake_text, leftover, reason = stt.listen_for_wake_word()

            if not woke:
                continue

            print(f"[AURA] Wake word detected. reason={reason}")

            if leftover:
                print("[AURA] Immediate speech detected after wake phrase.")
                final_text = leftover
            else:
                final_text = stt.listen_until_done()

            if not final_text:
                print("[AURA] No speech heard. Returning to wake mode.")
                print("-" * 60)
                continue

            censored = censor_text(final_text)
            stt.log_transcript(censored)

            if contains_bad_language(final_text):
                print("[AURA] Warning: inappropriate language detected.")

            handle_user_text(final_text)

            print("[AURA] Returning to wake mode.")
            print("-" * 60)

    except KeyboardInterrupt:
        print("\n[STT] Exiting cleanly.")


if __name__ == "__main__":
    main()