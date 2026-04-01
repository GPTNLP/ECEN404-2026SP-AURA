import os
import asyncio
import requests
import shutil
from pypdf import PdfReader
from core.config import STORAGE_DIR, LOCAL_DB_NAME, DEFAULT_MODEL, EMBEDDING_MODEL
from ai.lightrag_local import LightRAG

class RagManager:
    def __init__(self):
        self.rag_system = None
        self.db_path = os.path.join(str(STORAGE_DIR), LOCAL_DB_NAME)
        
    def initialize(self):
        os.makedirs(self.db_path, exist_ok=True)
        try:
            self.rag_system = LightRAG(
                working_dir=self.db_path,
                llm_model_name=DEFAULT_MODEL,
                embed_model_name=EMBEDDING_MODEL,
            )
            print(f"[RAG] Initialized successfully at {self.db_path}")
            return True
        except Exception as e:
            print(f"[RAG] Init failed: {e}")
            self.rag_system = None
            return False

    def extract_text_from_pdf(self, path: str) -> str:
        try:
            reader = PdfReader(path)
            parts = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(p for p in parts if p.strip())
        except Exception as e:
            print(f"[RAG] Error reading PDF: {e}")
            return ""

    async def ingest_document(self, url: str) -> bool:
        if not self.rag_system:
            raise RuntimeError("RAG System not initialized.")
            
        local_pdf_path = os.path.join(self.db_path, "temp_doc.pdf")
        
        # 1. Download from Azure
        print(f"[RAG] Downloading document from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_pdf_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
            
        # 2. Extract and Vectorize
        text = self.extract_text_from_pdf(local_pdf_path)
        if text:
            print(f"[RAG] Vectorizing {len(text)} characters...")
            await asyncio.to_thread(self.rag_system.insert, text)
            print("[RAG] Vectorization complete.")
            return True
        return False

    async def query(self, prompt: str) -> str:
        if not self.rag_system:
            return "RAG system is offline."
        try:
            res = await self.rag_system.aquery(prompt)
            return res.get("answer", "No relevant context found.")
        except Exception as e:
            return f"Error querying local LLM: {e}"

rag_manager = RagManager()