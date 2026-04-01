import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import wave
import tempfile
from datetime import datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard


BAD_WORDS = ["badword1", "badword2", "fuck", "shit", "bitch"]


def censor_text(text: str) -> str:
    def repl(m):
        return "█" * len(m.group(0))

    pattern = r"(?i)\b(" + "|".join(map(re.escape, BAD_WORDS)) + r")\b"
    return re.sub(pattern, repl, text)


def contains_bad_language(text: str) -> bool:
    pattern = r"(?i)\b(" + "|".join(map(re.escape, BAD_WORDS)) + r")\b"
    return re.search(pattern, text) is not None


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

        self.is_recording = False
        self.recording = []
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[STT] Audio status: {status}")
        if self.is_recording:
            self.recording.append(indata.copy())

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

    def start_recording(self) -> None:
        if self.is_recording:
            print("[STT] Already recording...")
            return

        print("\n[STT] Recording started... (Press SPACE again to stop)")
        self.is_recording = True
        self.recording = []

        self.stream = sd.InputStream(
            samplerate=self.device_sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.input_device,
            callback=self._callback,
        )
        self.stream.start()

    def stop_recording(self) -> str | None:
        if not self.is_recording:
            print("[STT] Not currently recording...")
            return None

        print("[STT] Recording stopped. Processing...")
        self.is_recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.recording:
            print("[STT] No audio captured.")
            return None

        audio = np.concatenate(self.recording, axis=0)

        peak = float(np.max(np.abs(audio)))
        mean = float(np.mean(np.abs(audio)))
        print(f"[STT] Peak level: {peak:.6f}")
        print(f"[STT] Mean level: {mean:.6f}")

        audio_16k = self._resample_audio(audio, self.device_sample_rate, self.target_sample_rate)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_filename = tmp_file.name

        try:
            self._save_wav(temp_filename, audio_16k, self.target_sample_rate)

            print("[STT] Transcribing with faster-whisper...")
            segments, info = self.model.transcribe(
                temp_filename,
                task=self.task,
                language=self.language,
                vad_filter=True,
                beam_size=5,
            )

            text = " ".join(seg.text.strip() for seg in segments).strip()
            detected_lang = getattr(info, "language", None) or "unknown"

            print(f"[STT] Detected language: {detected_lang.upper()}")
            print(f"[STT] Output: {text}")

            return text

        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def listen_once(self, seconds: int = 6) -> str:
        print(f"[STT] Recording for {seconds} seconds...")
        audio = sd.rec(
            int(seconds * self.device_sample_rate),
            samplerate=self.device_sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.input_device,
        )
        sd.wait()

        peak = float(np.max(np.abs(audio)))
        mean = float(np.mean(np.abs(audio)))
        print(f"[STT] Peak level: {peak:.6f}")
        print(f"[STT] Mean level: {mean:.6f}")

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

            return " ".join(seg.text.strip() for seg in segments).strip()

        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    def log_transcript(self, text: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {text}\n")


def main():
    print("=" * 60)
    print(" JETSON STT TEST")
    print("=" * 60)
    print("\nInstructions:")
    print(" • Press SPACE to start/stop recording")
    print(" • Press ESC to exit the program\n")

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
    )

    print("[STT] Device info:")
    print(sd.query_devices(stt.input_device))
    print("\n[STT] Ready! Press SPACE to start recording...\n")
    print("-" * 60)

    def on_press(key):
        nonlocal stt

        try:
            if key == keyboard.Key.space:
                if not stt.is_recording:
                    stt.start_recording()
                else:
                    text = stt.stop_recording()
                    if text:
                        censored_text = censor_text(text)
                        bad = contains_bad_language(text)

                        print("\n" + "=" * 60)
                        print(" FINAL OUTPUT:")
                        print(f" '{censored_text}'")
                        print("=" * 60)

                        stt.log_transcript(censored_text)

                        if bad:
                            print("[STT] Inappropriate language detected.")

                    print("\n[STT] Ready for next recording (SPACE to start)...")

            elif key == keyboard.Key.esc:
                print("\n[STT] Exiting program...")
                return False

        except AttributeError:
            if key == keyboard.Key.esc:
                print("\n[STT] Exiting program...")
                return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    main()