import sys
from pathlib import Path
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# --- PATH FIX: Add the 'agent' directory to sys.path so sub-folders are recognized ---
AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# Now these imports will resolve correctly
from core.config import STATIC_DIR, DEVICE_ID
from cloud.api_client import ApiClient
from hardware.serial_link import serial_link
from hardware.camera import camera_service
from ai.rag_manager import rag_manager
from ai.intent_parser import parse_intent
from ai.stt_service import STTService

app = FastAPI(title="AURA Edge API (Jetson Orin Nano)")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
api = ApiClient()

class ConnectionManager:
    def __init__(self):
        self.active_connections = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try: await connection.send_json(message)
            except: pass

ui_manager = ConnectionManager()

async def handle_user_message(user_msg: str):
    """Handles STT inputs or Chat inputs, parses intent, and routes accordingly."""
    await ui_manager.broadcast({"type": "chat", "sender": "user", "text": user_msg})
    
    intent = await parse_intent(user_msg)
    
    if intent == "MOVEMENT":
        try:
            ack = serial_link.send_command(user_msg)
            ai_reply = f"Movement executed. {ack}"
        except Exception as e:
            ai_reply = f"Hardware error: {e}"
    else:
        ai_reply = await rag_manager.query(user_msg)

    await ui_manager.broadcast({"type": "chat", "sender": "ai", "text": ai_reply})

# Initialize the STT service with the callback
stt_service = STTService(callback=handle_user_message)

async def command_loop():
    while True:
        try:
            result = await asyncio.to_thread(api.get_next_command, DEVICE_ID)
            command = result.get("command")

            if command:
                cmd_id = command.get("id")
                cmd = (command.get("command") or "").strip().lower()
                payload = command.get("payload") or {}

                # 1. RAG Vectorization
                if cmd == "build_rag":
                    success = await rag_manager.ingest_document(payload.get("url"))
                    status = "completed" if success else "failed"
                    await asyncio.to_thread(api.ack_command, {"command_id": cmd_id, "device_id": DEVICE_ID, "status": status})
                
                # 2. Movement Control
                elif cmd in serial_link.MOVEMENT_COMMANDS:
                    try:
                        serial_link.send_command(cmd, payload.get("value", ""))
                        await asyncio.to_thread(api.ack_command, {"command_id": cmd_id, "device_id": DEVICE_ID, "status": "completed"})
                    except Exception as e:
                        await asyncio.to_thread(api.ack_command, {"command_id": cmd_id, "device_id": DEVICE_ID, "status": "failed", "note": str(e)})
                        
                # 3. Chat Control (from Web UI Simulator)
                elif cmd == "chat_prompt":
                    ai_reply = await rag_manager.query(payload.get("query", ""))
                    await asyncio.to_thread(api.ack_command, {"command_id": cmd_id, "device_id": DEVICE_ID, "status": "completed", "result": {"answer": ai_reply}})

        except Exception as e:
            print(f"[COMMAND] poll failed: {e}")
            
        await asyncio.sleep(0.5)

@app.on_event("startup")
async def startup_event():
    serial_link.connect()
    rag_manager.initialize()
    
    asyncio.create_task(command_loop())
    asyncio.create_task(stt_service.continuous_stt_loop())

@app.get("/")
async def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ui_manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            if raw.strip():
                await handle_user_message(raw.strip())
    except WebSocketDisconnect:
        ui_manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)