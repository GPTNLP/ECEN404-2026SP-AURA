import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tempfile
import wave
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


class SpeechToTextService:
    def __init__(
        self,
        input_device=4,
        device_sample_rate=48000,
        target_sample_rate=16000,
        channels=1,
        model_size="base",
        device="cpu",
        compute_type="int8",
    ):
        self.input_device = input_device
        self.device_sample_rate = device_sample_rate
        self.target_sample_rate = target_sample_rate
        self.channels = channels

        print("[STT] Loading Whisper model...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("[STT] Whisper model loaded.")

    def _resample_audio(self, audio, orig_sr, target_sr):
        if orig_sr == target_sr:
            return audio

        audio = audio.flatten()
        duration = len(audio) / orig_sr

        old_times = np.linspace(0, duration, num=len(audio), endpoint=False)
        new_length = int(duration * target_sr)
        new_times = np.linspace(0, duration, num=new_length, endpoint=False)

        resampled = np.interp(new_times, old_times, audio).astype(np.float32)
        return resampled.reshape(-1, 1)

    def _save_wav(self, path, audio, sample_rate):
        audio = np.clip(audio.flatten(), -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

    def record_audio(self, seconds=5):
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

        return audio

    def transcribe_array(self, audio):
        audio_16k = self._resample_audio(
            audio, self.device_sample_rate, self.target_sample_rate
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        try:
            self._save_wav(temp_path, audio_16k, self.target_sample_rate)

            segments, info = self.model.transcribe(
                temp_path,
                language="en",
                task="transcribe",
                vad_filter=True,
                beam_size=5,
            )

            final_text = " ".join(segment.text.strip() for segment in segments).strip()
            return final_text

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def listen_once(self, seconds=5):
        audio = self.record_audio(seconds=seconds)
        text = self.transcribe_array(audio)
        return text