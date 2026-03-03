import os
import shutil
import zipfile
import json
import gc
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama

# --- Local Imports ---
from config import DEFAULT_MODEL, SESSIONS_DIR, STORAGE_DIR, CHROMA_DIR, GRAPH_FILE
from database_api import LoadDatabase, ClearMemory
from lightrag_local import LightRAG

app = FastAPI(title="AURA Edge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = None

def initialize_system():
    """Boots up the LightRAG system if both the vector DB and graph file exist."""
    global rag_system
    db = LoadDatabase()
    
    # Ensure both ChromaDB and the NetworkX graph file are present
    if db and os.path.exists(GRAPH_FILE):
        llm = ChatOllama(model=DEFAULT_MODEL, temperature=0.05)
        # Pass the graph file path here
        rag_system = LightRAG(llm=llm, vector_db=db, graph_file_path=GRAPH_FILE)
        print("AURA System Online.")
    else:
        print("Database or Graph file missing. Waiting for Admin sync...")

@app.on_event("startup")
async def startup_event():
    initialize_system()

class ChatRequest(BaseModel):
    query: str

@app.get("/health")
async def health_check():
    return {"status": "online", "rag_ready": rag_system is not None}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="Database missing.")
    try:
        result = rag_system.generate(request.query)
        
        # Log the conversation
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "content": request.query,
            "response": result["answer"],
            "sources": result.get("sources", [])
        }
        with open(os.path.join(SESSIONS_DIR, f"log_{datetime.now().timestamp()}.json"), "w") as f:
            json.dump(log_entry, f)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sync-db")
async def sync_database(file: UploadFile = File(...)):
    global rag_system
    
    # 1. Clear memory before accepting the large zip file
    rag_system = None
    ClearMemory()
    
    os.makedirs(STORAGE_DIR, exist_ok=True)
    zip_path = os.path.join(STORAGE_DIR, "incoming.zip")
    
    # 2. Save incoming zip
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 3. Wipe old ChromaDB (the old graph file gets overwritten automatically on extract)
    if os.path.exists(CHROMA_DIR): 
        shutil.rmtree(CHROMA_DIR)
    
    # 4. Extract new DB and Graph
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(STORAGE_DIR)
    
    os.remove(zip_path)
    
    # 5. Reboot the pipeline
    initialize_system()
    return {"status": "Updated successfully"}

@app.get("/api/logs")
async def get_logs():
    zip_path = os.path.join(STORAGE_DIR, "logs.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        if os.path.exists(SESSIONS_DIR):
            for root, _, files in os.walk(SESSIONS_DIR):
                for file in files:
                    zipf.write(os.path.join(root, file), file)
                    
    return FileResponse(zip_path, filename="logs.zip")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)