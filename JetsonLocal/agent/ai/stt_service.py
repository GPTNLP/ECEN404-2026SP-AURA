import os
import asyncio
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# Prevent MKL library conflicts on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class STTService:
    def __init__(self, callback):
        self.callback = callback
        self.is_running = False

    async def continuous_stt_loop(self):
        print("[STT] Loading faster-whisper tiny.en model...")
        try:
            # Switched to tiny.en and limited threads to prevent mkl_malloc crash
            model = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=4)
        except Exception as e:
            print(f"[STT] Failed to load Whisper: {e}")
            return
            
        audio_format = pyaudio.paInt16
        channels = 1
        rate = 16000
        chunk = 1024
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
            print("[STT] Microphone active. Listening...")
            self.is_running = True
            
            while self.is_running:
                frames = []
                for _ in range(0, int(rate / chunk * 3)):
                    data = stream.read(chunk, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.int16))
                
                audio_data = np.concatenate(frames).astype(np.float32) / 32768.0
                segments, _ = await asyncio.to_thread(model.transcribe, audio_data, beam_size=5)
                text = " ".join([segment.text for segment in segments]).strip()
                
                if text and len(text) > 5:
                    print(f"[STT] Transcribed: {text}")
                    await self.callback(text)
                    
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[STT] Error: {e}")
            self.is_running = False