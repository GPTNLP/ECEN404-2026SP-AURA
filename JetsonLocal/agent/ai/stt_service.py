import asyncio
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

class STTService:
    def __init__(self, callback):
        self.callback = callback
        self.is_running = False

    async def continuous_stt_loop(self):
        print("[STT] Loading faster-whisper base.en model...")
        model = WhisperModel("base.en", device="cpu", compute_type="int8") # Use "cuda" if CuDNN/TensorRT is available
        
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
                # Buffer 3 seconds of audio
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