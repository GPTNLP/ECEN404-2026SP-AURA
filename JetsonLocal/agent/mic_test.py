import sounddevice as sd
import numpy as np
import queue
import vosk
import json

q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(bytes(indata))

model = vosk.Model("model")  # path to vosk model
rec = vosk.KaldiRecognizer(model, 16000)

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):

    print("🎤 Speak...")

    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            print("🗣️", result.get("text", ""))