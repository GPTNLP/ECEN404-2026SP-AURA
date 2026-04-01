import os
import time
import json
import asyncio
import requests
import serial
import speech_recognition as sr
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Import from your existing files
from JetsonLocal.agent.config import DEFAULT_MODEL, EMBEDDING_MODEL, STORAGE_DIR
from JetsonLocal.agent.lightrag_local import LightRAG, OllamaClient

# Configurations
AZURE_BACKEND_URL = os.getenv("AZURE_BACKEND_URL", "http://localhost:5000")
SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/ttyUSB0")
INPUT_MODE = os.getenv("INPUT_MODE", "voice") # Default to voice for the physical unit
DB_NAME = "jetson_local_db"

app = FastAPI(title="AURA Edge API (Jetson Orin Nano)")

# Serve the frontend static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket Connection Manager for the UI
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

ui_manager = ConnectionManager()
rag_system = None
esp_serial = None

def init_hardware():
    global esp_serial
    try:
        esp_serial = serial.Serial(SERIAL_PORT, 115200, timeout=1)
    except serial.SerialException:
        pass

def init_rag():
    global rag_system
    db_path = os.path.join(STORAGE_DIR, DB_NAME)
    os.makedirs(db_path, exist_ok=True)
    rag_system = LightRAG(working_dir=db_path, llm_model_name=DEFAULT_MODEL, embed_model_name=EMBEDDING_MODEL)

async def parse_intent(user_msg):
    # Same local Ollama intent parsing as before
    client = OllamaClient("http://127.0.0.1:11434", EMBEDDING_MODEL, DEFAULT_MODEL)
    system = "Classify user input as 'MOVEMENT' or 'QUESTION'. Reply with one word."
    prompt = f"Input: '{user_msg}'"
    try:
        res = await client.generate(prompt, system=system, timeout_s=5.0)
        return "MOVEMENT" if "MOVEMENT" in res.upper() else "QUESTION"
    except:
        return "QUESTION"

def listen_mic():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        # Notify UI that we are listening
        asyncio.run(ui_manager.broadcast({"type": "status", "data": "Listening..."}))
        audio = r.listen(source)
        try:
            return r.recognize_google(audio) # Replace with faster-whisper for offline
        except:
            return ""

async def interactive_assistant_loop():
    """Background loop that processes hardware inputs and sends data to the UI screen."""
    await asyncio.sleep(2)
    
    while True:
        await ui_manager.broadcast({"type": "status", "data": "Ready (Awaiting Input)"})
        
        if INPUT_MODE == "keyboard":
            user_msg = await asyncio.to_thread(input, "\nUser: ")
        else:
            user_msg = await asyncio.to_thread(listen_mic)
        
        if not user_msg.strip():
            continue

        # Broadcast the user's input to the screen
        await ui_manager.broadcast({"type": "chat", "sender": "user", "text": user_msg})
        await ui_manager.broadcast({"type": "status", "data": "Processing..."})

        intent = await parse_intent(user_msg)
        
        if intent == "MOVEMENT":
            if esp_serial:
                esp_serial.write(f"MOVE:{user_msg}\n".encode('utf-8'))
            ai_reply = "Movement command routed to ESP rotors."
        else:
            if rag_system:
                res = await rag_system.aquery(user_msg)
                ai_reply = res.get("answer", "No answer found.")
            else:
                ai_reply = "RAG Database offline."
                
        # Broadcast the AI's response to the screen
        await ui_manager.broadcast({"type": "chat", "sender": "ai", "text": ai_reply})

@app.on_event("startup")
async def startup_event():
    init_hardware()
    init_rag()
    asyncio.create_task(interactive_assistant_loop())

@app.get("/")
async def serve_ui():
    """Serves the main frontend UI on the root URL."""
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for the frontend to receive live updates."""
    await ui_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep connection alive
    except WebSocketDisconnect:
        ui_manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)