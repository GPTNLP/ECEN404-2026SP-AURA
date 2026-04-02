import os
import asyncio
import shutil
import requests
from pypdf import PdfReader
from core.config import STORAGE_DIR, LOCAL_DB_NAME, DEFAULT_MODEL, EMBEDDING_MODEL
from ai.lightrag_local import LightRAG

class RagManager:
    def __init__(self):
        self.rag_system = None
        self.db_path = os.path.join(str(STORAGE_DIR), LOCAL_DB_NAME)
        self.zip_path = os.path.join(str(STORAGE_DIR), f"{LOCAL_DB_NAME}.zip")
        
    def initialize(self):
        os.makedirs(self.db_path, exist_ok=True)
        try:
            # Initializes the Graph and Vector indices
            self.rag_system = LightRAG(
                working_dir=self.db_path,
                llm_model_name=DEFAULT_MODEL,
                embed_model_name=EMBEDDING_MODEL,
            )
            print(f"[RAG] Offline LightRAG initialized at {self.db_path}")
            return True
        except Exception as e:
            print(f"[RAG] Init failed: {e}")
            return False

    def extract_text(self, path: str) -> str:
        try:
            reader = PdfReader(path)
            parts = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(p for p in parts if p.strip())
        except Exception:
            return ""

    async def ingest_document_and_pack(self, pdf_url: str, api_client, device_id: str, db_name: str) -> bool:
        """Downloads PDF, Vectorizes via LightRAG, and sends raw files to cloud."""
        if not self.rag_system: return False
        
        local_pdf = os.path.join(self.db_path, "temp.pdf")
        response = requests.get(pdf_url, stream=True)
        with open(local_pdf, 'wb') as f: shutil.copyfileobj(response.raw, f)
            
        text = self.extract_text(local_pdf)
        if text:
            # 1. Insert into local graph/vector DB
            await asyncio.to_thread(self.rag_system.insert, text)
            
            # 2. Upload raw files to Website Repository
            try:
                await asyncio.to_thread(api_client.upload_vector_db, db_name, self.db_path)
                print("[RAG] DB Synced to Cloud Repository.")
            except Exception as e:
                print(f"[RAG] Built locally, but cloud sync failed (offline?): {e}")
            return True
        return False

    async def load_remote_db(self, zip_url: str, api_client) -> bool:
        """Downloads a pre-built Vector DB from the website and extracts it locally."""
        try:
            # 1. Download ZIP
            await asyncio.to_thread(api_client.download_vector_db, zip_url, self.zip_path)
            
            # 2. Clear existing DB and extract new one
            shutil.rmtree(self.db_path, ignore_errors=True)
            shutil.unpack_archive(self.zip_path, self.db_path)
            
            # 3. Reinitialize LightRAG with new data
            return self.initialize()
        except Exception as e:
            print(f"[RAG] Failed to load remote DB: {e}")
            return False

    async def query(self, prompt: str) -> str:
        if not self.rag_system: return "RAG system is offline."
        try:
            # Utilizes LightRAG's dual-level (local/global) search
            res = await self.rag_system.aquery(prompt)
            return res.get("answer", "No context found in local database.")
        except Exception as e:
            return f"Query failed: {e}"

rag_manager = RagManager()