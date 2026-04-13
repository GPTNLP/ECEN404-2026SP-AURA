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


# ============================================================
# TUNING
# ============================================================
WAKE_AUDIO_THRESHOLD = 0.009
COMMAND_AUDIO_THRESHOLD = 0.0045
END_SILENCE_SECONDS = 1.10
WAKE_CHUNK_SECONDS = 1.00
COMMAND_CHUNK_SECONDS = 0.30
COMMAND_TIMEOUT_SECONDS = 8.0
NEAR_FIELD_BOOST = 1.35
USE_ONLY_LEFT_CHANNEL = False
WAKE_MATCH_MODE = 1

NOISE_FLOOR_SAMPLES = 12
NOISE_FLOOR_MULTIPLIER = 1.35
MIN_DYNAMIC_THRESHOLD = 0.0035

DEFAULT_MODEL_SIZE = "base.en"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_LANGUAGE = "en"
DEFAULT_TASK = "transcribe"

WAKE_COOLDOWN_SECONDS = 1.0

# Optional: manually force a specific mic by index.
MANUAL_INPUT_DEVICE = None

PREFERRED_INPUT_KEYWORDS = [
    "nanomic",
    "usb",
    "microphone",
    "mic",
    "headset",
    "webcam",
]

ENABLE_SECOND_PASS_FOR_WEAK_TEXT = True
SECOND_PASS_MIN_WORDS = 3
RELAX_WEAK_TRANSCRIPT_FILTER = True

MAX_CONSECUTIVE_RECORD_ERRORS = 8
RECALIBRATE_EVERY_SECONDS = 90


# ============================================================
# PHRASES / RULES
# ============================================================
MOVEMENT_PATTERNS = {
    "forward": [
        "move forward",
        "go forward",
        "drive forward",
        "forward",
        "forwards",
        "go straight",
        "straight",
    ],
    "backward": [
        "move backward",
        "go backward",
        "move back",
        "go back",
        "drive backward",
        "backward",
        "back",
        "reverse",
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
                "straight",
            }

            if is_single_word_direction and word_count > 3 and not has_action_word:
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

    movement = detect_last_movement_command(norm)
    if movement is not None:
        return False

    if norm in WAKE_ONLY_PHRASES:
        return True

    words = norm.split()

    weak_two_word_phrases = {
        "speed food",
        "speak food",
        "feed food",
        "big food",
        "good food",
    }

    if norm in weak_two_word_phrases:
        return True

    if RELAX_WEAK_TRANSCRIPT_FILTER:
        if len(words) == 1:
            return True
        return False

    if len(words) == 1:
        return True

    if len(words) <= 2 and "speak" not in words:
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
        input_device: Optional[int] = MANUAL_INPUT_DEVICE,
        device_sample_rate: Optional[int] = None,
        target_sample_rate: int = 16000,
        channels: Optional[int] = None,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        language: str = DEFAULT_LANGUAGE,
        task: str = DEFAULT_TASK,
        log_path: str = "~/SDP/AURA/JetsonLocal/storage/transcriptions.log",
        unload_after_idle_seconds: float = 300.0,
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
        self.last_noise_calibration_ts = 0.0
        self.consecutive_record_errors = 0

        self._resolve_input_device()

    def _resolve_input_device(self) -> None:
        devices = sd.query_devices()

        print("\n[STT] Available input devices:")
        for idx, dev in enumerate(devices):
            max_in = int(dev.get("max_input_channels", 0))
            if max_in > 0:
                print(
                    f"  #{idx}: {dev['name']} | "
                    f"inputs={max_in} | "
                    f"default_sr={int(dev.get('default_samplerate', 0))}"
                )

        if self.input_device is not None:
            info = sd.query_devices(self.input_device, "input")
            if self.device_sample_rate is None:
                self.device_sample_rate = int(info["default_samplerate"])
            if self.channels is None:
                self.channels = max(1, min(2, int(info["max_input_channels"])))
            print(f"[STT] Using configured input device #{self.input_device}: {info['name']}")
            return

        best_index = None
        best_score = -10_000

        for idx, dev in enumerate(devices):
            max_in = int(dev.get("max_input_channels", 0))
            if max_in <= 0:
                continue

            name = str(dev.get("name", "")).lower()
            score = 0

            for keyword in PREFERRED_INPUT_KEYWORDS:
                if keyword in name:
                    score += 10

            if "default" in name:
                score += 4
            if "pulse" in name:
                score += 2
            if "sysdefault" in name:
                score += 2
            if "hdmi" in name:
                score -= 10
            if "monitor" in name:
                score -= 10
            if "output" in name:
                score -= 8
            if "stereo mix" in name:
                score -= 20

            if max_in in (1, 2):
                score += 4

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

    def reinitialize_audio_input(self) -> None:
        print("[STT] Reinitializing audio input...")
        try:
            sd.stop()
        except Exception:
            pass

        try:
            self._resolve_input_device()
        except Exception as e:
            print(f"[STT] Audio reinit failed: {e}")
            raise

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return

        requested_device = self.device
        requested_compute = self.compute_type

        try:
            import torch
            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False

        if requested_device == "cuda" and not cuda_ok:
            print("[STT] CUDA runtime not available, falling back to CPU int8")
            requested_device = "cpu"
            requested_compute = "int8"

        print(
            f"[STT] Loading faster-whisper model "
            f"'{self.model_size}' on device={requested_device} compute_type={requested_compute}..."
        )

        try:
            self.model = WhisperModel(
                self.model_size,
                device=requested_device,
                compute_type=requested_compute,
                cpu_threads=2 if requested_device == "cuda" else 4,
                num_workers=1,
            )
            self.device = requested_device
            self.compute_type = requested_compute
            print("[STT] Model loaded successfully!")
            return

        except Exception as e:
            msg = str(e).lower()
            cuda_related = (
                "cuda support" in msg
                or "compiled with cuda" in msg
                or "cuda" in msg
            )

            if requested_device == "cuda" and cuda_related:
                print("[STT] CUDA-backed CTranslate2 is unavailable. Falling back to CPU int8...")
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=4,
                    num_workers=1,
                )
                self.device = "cpu"
                self.compute_type = "int8"
                print("[STT] Model loaded successfully on CPU fallback!")
                return

            raise

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

    def maybe_recalibrate_noise_floor(self) -> None:
        now = time.time()
        if now - self.last_noise_calibration_ts >= RECALIBRATE_EVERY_SECONDS:
            self.calibrate_noise_floor()

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio is None:
            return np.zeros(1, dtype=np.float32)

        audio = np.asarray(audio, dtype=np.float32)

        if audio.size == 0:
            return np.zeros(1, dtype=np.float32)

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        if audio.ndim == 1:
            mono = audio.astype(np.float32)
        else:
            if USE_ONLY_LEFT_CHANNEL and audio.shape[1] >= 1:
                mono = audio[:, 0].astype(np.float32)
            else:
                mono = np.mean(audio, axis=1).astype(np.float32)

        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

        mono = mono * NEAR_FIELD_BOOST

        peak = np.max(np.abs(mono)) if mono.size else 0.0
        if np.isnan(peak) or np.isinf(peak):
            peak = 0.0

        if 0 < peak < 0.25:
            mono = mono / peak * min(0.75, peak * 2.8)

        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        mono = np.clip(mono, -1.0, 1.0)

        return mono.astype(np.float32)

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if audio is None:
            return np.zeros(1, dtype=np.float32)

        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        if orig_sr == target_sr:
            return audio.astype(np.float32)

        if len(audio) == 0 or orig_sr <= 0 or target_sr <= 0:
            return np.zeros(1, dtype=np.float32)

        duration = len(audio) / orig_sr
        old_times = np.linspace(0, duration, num=len(audio), endpoint=False)
        new_length = max(1, int(duration * target_sr))
        new_times = np.linspace(0, duration, num=new_length, endpoint=False)

        resampled = np.interp(new_times, old_times, audio).astype(np.float32)
        resampled = np.nan_to_num(resampled, nan=0.0, posinf=0.0, neginf=0.0)
        return resampled

    def record_fixed(self, seconds: float) -> np.ndarray:
        if not self.is_running:
            raise RuntimeError("STT recording stopped")

        frames = max(1, int(seconds * self.device_sample_rate))

        try:
            audio = sd.rec(
                frames,
                samplerate=self.device_sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.input_device,
                blocking=True,
            )
        except Exception as e:
            self.consecutive_record_errors += 1
            print(f"[STT] record error ({self.consecutive_record_errors}): {e}")

            if self.consecutive_record_errors >= MAX_CONSECUTIVE_RECORD_ERRORS:
                self.reinitialize_audio_input()
                self.consecutive_record_errors = 0

            raise RuntimeError(f"Audio capture failed: {e}")

        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        if audio.size == 0:
            self.consecutive_record_errors += 1
            raise RuntimeError("Audio capture returned empty buffer")

        self.consecutive_record_errors = 0
        return audio

    def analyze_level(self, audio: np.ndarray) -> Tuple[float, float]:
        mono = self._prepare_audio(audio)

        if mono is None or mono.size == 0:
            return 0.0, 0.0

        abs_mono = np.abs(mono)
        abs_mono = np.nan_to_num(abs_mono, nan=0.0, posinf=0.0, neginf=0.0)

        peak = float(np.max(abs_mono)) if abs_mono.size else 0.0
        mean = float(np.mean(abs_mono)) if abs_mono.size else 0.0

        if np.isnan(peak) or np.isinf(peak):
            peak = 0.0
        if np.isnan(mean) or np.isinf(mean):
            mean = 0.0

        return peak, mean

    def _dynamic_threshold(self, base_threshold: float) -> float:
        dynamic = max(
            base_threshold,
            MIN_DYNAMIC_THRESHOLD,
            self.noise_floor * NOISE_FLOOR_MULTIPLIER,
        )
        return float(dynamic)

    def calibrate_noise_floor(self) -> None:
        if not self.is_running:
            return

        print("[STT] Calibrating noise floor...")
        samples = []

        for _ in range(NOISE_FLOOR_SAMPLES):
            if not self.is_running:
                return

            try:
                audio = self.record_fixed(0.20)
                _, mean = self.analyze_level(audio)
                if not np.isnan(mean) and not np.isinf(mean):
                    samples.append(mean)
            except Exception as e:
                print(f"[STT] Noise calibration sample failed: {e}")

            time.sleep(0.02)

        self.noise_floor = float(np.median(samples)) if samples else 0.0
        if np.isnan(self.noise_floor) or np.isinf(self.noise_floor):
            self.noise_floor = 0.0

        self.last_noise_calibration_ts = time.time()
        print(f"[STT] Noise floor: {self.noise_floor:.6f}")

    def _run_transcribe(
        self,
        audio_16k: np.ndarray,
        beam_size: int,
        best_of: int,
        vad_min_silence_ms: int,
        speech_pad_ms: int,
        condition_on_previous_text: bool,
    ) -> str:
        segments, _ = self.model.transcribe(
            audio_16k,
            task=self.task,
            language=self.language,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": vad_min_silence_ms,
                "speech_pad_ms": speech_pad_ms,
            },
            beam_size=beam_size,
            best_of=best_of,
            temperature=0.0,
            condition_on_previous_text=condition_on_previous_text,
            without_timestamps=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

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

        if audio_16k is None or len(audio_16k) == 0:
            return ""

        audio_16k = np.nan_to_num(audio_16k, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        self.last_audio_activity_ts = time.time()
        self.last_transcribe_ts = self.last_audio_activity_ts

        text = self._run_transcribe(
            audio_16k=audio_16k,
            beam_size=5,
            best_of=5,
            vad_min_silence_ms=350,
            speech_pad_ms=250,
            condition_on_previous_text=False,
        )

        if ENABLE_SECOND_PASS_FOR_WEAK_TEXT:
            word_count = len(normalize_text(text).split())
            if word_count < SECOND_PASS_MIN_WORDS:
                retry_text = self._run_transcribe(
                    audio_16k=audio_16k,
                    beam_size=7,
                    best_of=7,
                    vad_min_silence_ms=500,
                    speech_pad_ms=320,
                    condition_on_previous_text=False,
                )
                if len(normalize_text(retry_text)) > len(normalize_text(text)):
                    text = retry_text

        return text

    def listen_for_wake_word(self, chunk_seconds: float = WAKE_CHUNK_SECONDS):
        if not self.is_running:
            return False, "", "", "stopped"

        audio = self.record_fixed(chunk_seconds)
        peak, mean = self.analyze_level(audio)

        threshold = self._dynamic_threshold(WAKE_AUDIO_THRESHOLD)
        if peak < threshold and mean < (threshold * 0.22):
            return False, "", "", f"too_quiet peak={peak:.4f} mean={mean:.4f} threshold={threshold:.4f}"

        self.last_audio_activity_ts = time.time()

        text = self._transcribe_audio_array(audio)
        if not text:
            return False, "", "", "no_text"

        matched, leftover, reason = wake_score(text)
        print(
            f"[WAKE CHECK] {text} | reason={reason} | "
            f"peak={peak:.4f} | mean={mean:.4f} | thr={threshold:.4f}"
        )
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

        while self.is_running and (time.time() - start_time < timeout_seconds):
            audio = self.record_fixed(chunk_seconds)
            collected.append(audio)

            peak, mean = self.analyze_level(audio)

            if peak >= threshold or mean >= (threshold * 0.22):
                self.last_audio_activity_ts = time.time()
                speech_started = True
                silence_after_speech = 0.0
            else:
                if speech_started:
                    silence_after_speech += chunk_seconds

            if speech_started and silence_after_speech >= end_silence_seconds:
                break

        if not self.is_running:
            return ""

        if not collected:
            return ""

        full_audio = np.concatenate(collected, axis=0)
        full_audio = np.nan_to_num(full_audio, nan=0.0, posinf=0.0, neginf=0.0)

        peak, mean = self.analyze_level(full_audio)

        print(f"[AURA] Peak level: {peak:.6f}")
        print(f"[AURA] Mean level: {mean:.6f}")
        print(f"[AURA] Command threshold: {threshold:.6f}")
        print(f"[AURA] Speech started: {speech_started}")
        print(f"[AURA] Captured seconds: {len(full_audio) / self.device_sample_rate:.2f}")

        if not speech_started:
            return ""

        return self._transcribe_audio_array(full_audio)

    def log_transcript(self, text: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {text}\n")

    async def continuous_stt_loop(self) -> None:
        self.is_running = True
        self._ensure_model_loaded()
        self.calibrate_noise_floor()

        print("[STT] Voice loop active. Say 'Hey AURA' to activate.")
        print("[AURA] Waiting for wake word...")

        while self.is_running:
            try:
                self.maybe_unload_model_for_idle()
                self.maybe_recalibrate_noise_floor()

                woke, wake_text, leftover, reason = await asyncio.to_thread(
                    self.listen_for_wake_word
                )

                if not self.is_running:
                    break

                if not woke:
                    await asyncio.sleep(0.02)
                    continue

                now = time.time()
                if now - self.last_wake_time < WAKE_COOLDOWN_SECONDS:
                    await asyncio.sleep(0.02)
                    continue

                self.last_wake_time = now
                print(f"[AURA] Wake word detected. reason={reason}")

                if leftover:
                    print("[AURA] Immediate speech detected after wake phrase.")
                    final_text = leftover
                else:
                    final_text = await asyncio.to_thread(self.listen_until_done)

                if not self.is_running:
                    break

                final_text = normalize_text(final_text)

                if not final_text:
                    print("[AURA] No speech heard. Returning to wake mode.")
                    print("[AURA] Waiting for wake word...")
                    print("-" * 60)
                    await asyncio.sleep(0.02)
                    continue

                if looks_like_weak_transcript(final_text):
                    print(f"[AURA] Weak transcript detected: {final_text}")
                    if final_text in WAKE_ONLY_PHRASES:
                        print("[AURA] Ignoring wake-only transcript.")
                        print("[AURA] Waiting for wake word...")
                        print("-" * 60)
                        await asyncio.sleep(0.02)
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
                await asyncio.sleep(0.02)

            except RuntimeError as e:
                if "stopped" in str(e).lower():
                    break
                print(f"[STT] Loop runtime error: {e}")
                await asyncio.sleep(0.20)
            except Exception as e:
                print(f"[STT] Loop error: {e}")
                await asyncio.sleep(0.20)

        self.unload_model()
        print("[STT] Voice loop stopped.")

    def stop(self) -> None:
        self.is_running = False
        try:
            sd.stop()
        except Exception:
            pass