import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import gc
import time
import asyncio
from datetime import datetime
from typing import Awaitable, Callable, Optional, Tuple

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

WAKE_AUDIO_THRESHOLD = 0.020
COMMAND_AUDIO_THRESHOLD = 0.012
END_SILENCE_SECONDS = 1.3
WAKE_CHUNK_SECONDS = 1.8
COMMAND_CHUNK_SECONDS = 0.50
COMMAND_TIMEOUT_SECONDS = 10.0
NEAR_FIELD_BOOST = 1.15
USE_ONLY_LEFT_CHANNEL = True
WAKE_MATCH_MODE = 1

NOISE_FLOOR_SAMPLES = 5
NOISE_FLOOR_MULTIPLIER = 1.8
MIN_DYNAMIC_THRESHOLD = 0.008

DEFAULT_MODEL_SIZE = "base.en"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"

DEFAULT_LANGUAGE = "en"
DEFAULT_TASK = "transcribe"

WAKE_COOLDOWN_SECONDS = 1.0


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
    "move",
    "go",
    "turn",
    "drive",
    "stop",
    "halt",
    "pause",
    "freeze",
]

BAD_WORD_PATTERNS = [
    r"fuck\w*",
    r"shit\w*",
    r"bitch\w*",
    r"asshole\w*",
    r"\bass\b",
]

WAKE_PREFIXES = ["hey", "hi", "ok", "okay", "yo"]
WAKE_AURA_ALIASES = [
    "aura",
    "ora",
    "or a",
    "or",
    "oura",
    "arua",
]

WAKE_ONLY_PHRASES = {
    "aura",
    "hey aura",
    "hi aura",
    "ok aura",
    "okay aura",
    "yo aura",
    "ora",
    "hey ora",
    "hi ora",
    "or a",
    "hey or a",
    "oura",
    "arua",
}


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
            text,
        )
    return text


def contains_bad_language(text: str) -> bool:
    for pattern in BAD_WORD_PATTERNS:
        if re.search(rf"(?i)\b({pattern})\b", text):
            return True
    return False


def wake_score(text: str) -> Tuple[bool, str, str]:
    norm = normalize_text(text)
    if not norm:
        return False, "", "empty"

    tokens = norm.split()

    candidates = []
    for prefix in WAKE_PREFIXES:
        for alias in WAKE_AURA_ALIASES:
            candidates.append(f"{prefix} {alias}")

    for cand in candidates:
        cand_norm = normalize_text(cand)
        pattern = rf"\b{re.escape(cand_norm)}\b"
        m = re.search(pattern, norm)
        if m:
            leftover = norm[m.end():].strip()
            return True, leftover, f"exact:{cand_norm}"

    for i in range(len(tokens)):
        if tokens[i] in WAKE_PREFIXES:
            if i + 1 < len(tokens):
                two = f"{tokens[i]} {tokens[i + 1]}"
                if tokens[i + 1] in {"aura", "ora", "or", "oura", "arua"}:
                    leftover = " ".join(tokens[i + 2:]).strip()
                    return True, leftover, f"token_pair:{two}"

            if i + 2 < len(tokens):
                three = f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}"
                if f"{tokens[i + 1]} {tokens[i + 2]}" == "or a":
                    leftover = " ".join(tokens[i + 3:]).strip()
                    return True, leftover, f"token_triplet:{three}"

    has_prefix = any(tok in WAKE_PREFIXES for tok in tokens)
    has_auraish = (
        "aura" in tokens
        or "ora" in tokens
        or "or" in tokens
        or "oura" in tokens
        or "arua" in tokens
        or "or a" in norm
    )

    if WAKE_MATCH_MODE == 1 and has_prefix and has_auraish:
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


def detect_last_movement_command(text: str) -> Optional[str]:
    norm = normalize_text(text)
    if not norm:
        return None

    has_action_word = any(
        re.search(rf"\b{re.escape(word)}\b", norm) for word in ACTION_WORDS
    )
    word_count = len(norm.split())
    matches = []

    for command, phrases in MOVEMENT_PATTERNS.items():
        for phrase in phrases:
            phrase_norm = normalize_text(phrase)
            is_single_word_direction = phrase_norm in {
                "left",
                "right",
                "forward",
                "back",
                "backward",
                "stop",
            }

            if is_single_word_direction and word_count > 2 and not has_action_word:
                continue

            pattern = rf"\b{re.escape(phrase_norm)}\b"
            for match in re.finditer(pattern, norm):
                matches.append((match.start(), command, phrase_norm))

    if not matches:
        return None

    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def classify_intent(text: str) -> str:
    norm = normalize_text(text)
    if not norm:
        return "empty"

    movement = detect_last_movement_command(norm)
    if movement:
        return "movement"

    return "llm"


def looks_like_weak_transcript(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return True

    if norm in WAKE_ONLY_PHRASES:
        return True

    words = norm.split()

    if len(words) == 1 and norm not in {"forward", "backward", "back", "left", "right", "stop"}:
        return True

    weak_two_word_phrases = {
        "speed food",
        "speak food",
        "feed food",
        "big food",
        "good food",
    }

    if norm in weak_two_word_phrases:
        return True

    if len(words) <= 2:
        movement = detect_last_movement_command(norm)
        if movement is None and "speak" not in words:
            return True

    return False


# ============================================================
# STT SERVICE
# ============================================================
class STTService:
    def __init__(
        self,
        callback: Callable[[str, str, Optional[str]], Awaitable[None]],
        model_size: str = DEFAULT_MODEL_SIZE,
        input_device: Optional[int] = None,
        device_sample_rate: Optional[int] = None,
        target_sample_rate: int = 16000,
        channels: Optional[int] = None,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        language: str = DEFAULT_LANGUAGE,
        task: str = DEFAULT_TASK,
        log_path: str = "~/SDP/AURA/JetsonLocal/storage/transcriptions.log",
        unload_after_idle_seconds: float = 60.0,
        auto_reload_model: bool = True,
    ):
        self.callback = callback
        self.model_size = model_size
        self.input_device = input_device
        self.device_sample_rate = device_sample_rate
        self.target_sample_rate = target_sample_rate
        self.channels = channels
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.task = task
        self.log_path = os.path.expanduser(log_path)

        self.unload_after_idle_seconds = float(unload_after_idle_seconds)
        self.auto_reload_model = bool(auto_reload_model)

        self.model: Optional[WhisperModel] = None
        self.is_running = False
        self.noise_floor = 0.0
        self.last_wake_time = 0.0
        self.last_audio_activity_ts = time.time()
        self.last_transcribe_ts = 0.0

        self._resolve_input_device()

    # --------------------------------------------------------
    # DEVICE SETUP
    # --------------------------------------------------------
    def _resolve_input_device(self) -> None:
        devices = sd.query_devices()

        if self.input_device is not None:
            info = sd.query_devices(self.input_device, "input")
            if self.device_sample_rate is None:
                self.device_sample_rate = int(info["default_samplerate"])
            if self.channels is None:
                self.channels = max(1, min(2, int(info["max_input_channels"])))
            print(f"[STT] Using configured input device #{self.input_device}: {info['name']}")
            return

        best_index = None
        best_score = -1

        for idx, dev in enumerate(devices):
            max_in = int(dev.get("max_input_channels", 0))
            if max_in <= 0:
                continue

            name = str(dev.get("name", "")).lower()
            score = 0

            if "usb" in name:
                score += 4
            if "mic" in name or "microphone" in name:
                score += 3
            if "nano" in name:
                score += 2
            if "default" in name:
                score += 1

            if score > best_score:
                best_score = score
                best_index = idx

        if best_index is None:
            default_input = sd.default.device[0]
            if default_input is None or default_input < 0:
                raise RuntimeError("No usable audio input device found.")
            best_index = int(default_input)

        info = sd.query_devices(best_index, "input")
        self.input_device = best_index

        if self.device_sample_rate is None:
            self.device_sample_rate = int(info["default_samplerate"])

        if self.channels is None:
            self.channels = max(1, min(2, int(info["max_input_channels"])))

        print(f"[STT] Selected input device #{self.input_device}: {info['name']}")
        print(f"[STT] Device sample rate: {self.device_sample_rate}")
        print(f"[STT] Channels: {self.channels}")

    # --------------------------------------------------------
    # MODEL LIFECYCLE
    # --------------------------------------------------------
    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return

        print(f"[STT] Loading faster-whisper model '{self.model_size}'...")
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=4,
        )
        print("[STT] Model loaded successfully!")

    def unload_model(self) -> None:
        if self.model is None:
            return

        print("[STT] Unloading whisper model due to idle timeout...")
        try:
            del self.model
        except Exception:
            pass

        self.model = None
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        print("[STT] Whisper model unloaded.")

    def maybe_unload_model_for_idle(self) -> None:
        if self.model is None:
            return

        idle_for = time.time() - self.last_audio_activity_ts
        if idle_for >= self.unload_after_idle_seconds:
            self.unload_model()

    # --------------------------------------------------------
    # AUDIO HELPERS
    # --------------------------------------------------------
    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            mono = audio.astype(np.float32)
        else:
            if USE_ONLY_LEFT_CHANNEL and audio.shape[1] >= 1:
                mono = audio[:, 0].astype(np.float32)
            else:
                mono = np.mean(audio, axis=1).astype(np.float32)

        mono = mono * NEAR_FIELD_BOOST
        mono = np.clip(mono, -1.0, 1.0)
        return mono

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio.astype(np.float32)

        duration = len(audio) / orig_sr
        old_times = np.linspace(0, duration, num=len(audio), endpoint=False)
        new_length = int(duration * target_sr)
        new_times = np.linspace(0, duration, num=new_length, endpoint=False)

        return np.interp(new_times, old_times, audio).astype(np.float32)

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

    def analyze_level(self, audio: np.ndarray) -> Tuple[float, float]:
        mono = self._prepare_audio(audio)
        peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
        mean = float(np.mean(np.abs(mono))) if len(mono) else 0.0
        return peak, mean

    def _dynamic_threshold(self, base_threshold: float) -> float:
        dynamic = max(
            base_threshold,
            MIN_DYNAMIC_THRESHOLD,
            self.noise_floor * NOISE_FLOOR_MULTIPLIER,
        )
        return float(dynamic)

    def calibrate_noise_floor(self) -> None:
        print("[STT] Calibrating noise floor...")
        samples = []

        for _ in range(NOISE_FLOOR_SAMPLES):
            audio = self.record_fixed(0.25)
            _, mean = self.analyze_level(audio)
            samples.append(mean)

        self.noise_floor = float(np.median(samples)) if samples else 0.0
        print(f"[STT] Noise floor: {self.noise_floor:.6f}")

    def _transcribe_audio_array(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) == 0:
            return ""

        if self.model is None:
            if not self.auto_reload_model:
                print("[STT] Model is unloaded and auto reload is disabled.")
                return ""
            self._ensure_model_loaded()

        mono = self._prepare_audio(audio)
        audio_16k = self._resample_audio(
            mono,
            orig_sr=self.device_sample_rate,
            target_sr=self.target_sample_rate,
        )

        self.last_audio_activity_ts = time.time()
        self.last_transcribe_ts = self.last_audio_activity_ts

        segments, _ = self.model.transcribe(
            audio_16k,
            task=self.task,
            language=self.language,
            vad_filter=False,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text

    # --------------------------------------------------------
    # LISTENING
    # --------------------------------------------------------
    def listen_for_wake_word(self, chunk_seconds: float = WAKE_CHUNK_SECONDS):
        audio = self.record_fixed(chunk_seconds)
        peak, mean = self.analyze_level(audio)

        threshold = self._dynamic_threshold(WAKE_AUDIO_THRESHOLD)
        if peak < threshold:
            return False, "", "", f"too_quiet peak={peak:.4f} threshold={threshold:.4f}"

        self.last_audio_activity_ts = time.time()

        text = self._transcribe_audio_array(audio)
        if not text:
            return False, "", "", "no_text"

        matched, leftover, reason = wake_score(text)
        print(f"[WAKE CHECK] {text} | reason={reason} | peak={peak:.4f} | thr={threshold:.4f}")
        return matched, text, leftover, reason

    def listen_until_done(
        self,
        timeout_seconds: float = COMMAND_TIMEOUT_SECONDS,
        end_silence_seconds: float = END_SILENCE_SECONDS,
        chunk_seconds: float = COMMAND_CHUNK_SECONDS,
    ) -> str:
        print("[AURA] Listening for command...")

        start_time = time.time()
        collected = []
        speech_started = False
        silence_after_speech = 0.0

        threshold = self._dynamic_threshold(COMMAND_AUDIO_THRESHOLD)

        while time.time() - start_time < timeout_seconds:
            audio = self.record_fixed(chunk_seconds)
            collected.append(audio)

            peak, _ = self.analyze_level(audio)

            if peak >= threshold:
                self.last_audio_activity_ts = time.time()
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
        print(f"[AURA] Command threshold: {threshold:.6f}")

        if not speech_started:
            return ""

        return self._transcribe_audio_array(full_audio)

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------
    def log_transcript(self, text: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {text}\n")

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------
    async def continuous_stt_loop(self) -> None:
        self._ensure_model_loaded()
        self.calibrate_noise_floor()

        print("[STT] Voice loop active. Say 'Hey AURA' to activate.")
        print("[AURA] Waiting for wake word...")

        self.is_running = True

        while self.is_running:
            try:
                self.maybe_unload_model_for_idle()

                woke, wake_text, leftover, reason = await asyncio.to_thread(
                    self.listen_for_wake_word
                )

                if not woke:
                    await asyncio.sleep(0.05)
                    continue

                now = time.time()
                if now - self.last_wake_time < WAKE_COOLDOWN_SECONDS:
                    await asyncio.sleep(0.05)
                    continue

                self.last_wake_time = now
                print(f"[AURA] Wake word detected. reason={reason}")

                if leftover:
                    print("[AURA] Immediate speech detected after wake phrase.")
                    final_text = leftover
                else:
                    final_text = await asyncio.to_thread(self.listen_until_done)

                final_text = normalize_text(final_text)

                if not final_text:
                    print("[AURA] No speech heard. Returning to wake mode.")
                    print("[AURA] Waiting for wake word...")
                    print("-" * 60)
                    await asyncio.sleep(0.05)
                    continue

                if looks_like_weak_transcript(final_text):
                    print(f"[AURA] Ignoring weak transcript: {final_text}")
                    print("[AURA] Waiting for wake word...")
                    print("-" * 60)
                    await asyncio.sleep(0.05)
                    continue

                movement = detect_last_movement_command(final_text)
                intent = classify_intent(final_text)

                censored = censor_text(final_text)
                self.log_transcript(censored)

                if contains_bad_language(final_text):
                    print("[AURA] Warning: inappropriate language detected.")

                print("\n" + "=" * 60)
                print(f"RAW TEXT: {wake_text if wake_text else final_text}")
                print(f"CLEANED TEXT: {final_text}")
                print(f"INTENT: {intent}")
                print(f"MOVEMENT: {movement}")
                print("=" * 60)

                await self.callback(final_text, intent, movement)

                print("[AURA] Returning to wake mode.")
                print("[AURA] Waiting for wake word...")
                print("-" * 60)
                await asyncio.sleep(0.05)

            except Exception as e:
                print(f"[STT] Loop error: {e}")
                await asyncio.sleep(0.5)

        self.unload_model()
        print("[STT] Voice loop stopped.")

    def stop(self) -> None:
        self.is_running = False