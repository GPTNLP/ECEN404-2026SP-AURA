import os
import asyncio
import shutil
from pypdf import PdfReader
from core.config import STORAGE_DIR, LOCAL_DB_NAME, DEFAULT_MODEL, EMBEDDING_MODEL
from ai.lightrag_local import LightRAG


class RagManager:
    def __init__(self):
        self.rag_system: LightRAG = None
        self.db_name = LOCAL_DB_NAME
        self.db_path = os.path.join(str(STORAGE_DIR), LOCAL_DB_NAME)

    def initialize(self) -> bool:
        os.makedirs(self.db_path, exist_ok=True)
        try:
            self.rag_system = LightRAG(
                working_dir=self.db_path,
                llm_model_name=DEFAULT_MODEL,
                embed_model_name=EMBEDDING_MODEL,
            )
            stats = self.rag_system.stats()
            print(f"[RAG] LightRAG initialized at {self.db_path} — {stats['chunk_count']} chunks")
            return True
        except Exception as e:
            print(f"[RAG] Init failed: {e}")
            return False

    def is_ready(self) -> bool:
        return self.rag_system is not None

    def stats(self) -> dict:
        if not self.rag_system:
            return {"chunk_count": 0, "ready": False}
        s = self.rag_system.stats()
        s["ready"] = True
        return s

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def extract_text(self, path: str) -> str:
        try:
            reader = PdfReader(path)
            parts = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(p for p in parts if p.strip())
        except Exception:
            return ""

    async def ingest_pdfs_and_sync(
        self,
        pdf_paths: list,
        api_client,
        db_name: str,
    ) -> dict:
        """
        Ingests a list of local PDF paths into LightRAG, then syncs the
        resulting vector files back to the website as a named vector DB.
        """
        if not self.rag_system:
            self.initialize()
        if not self.rag_system:
            return {"ok": False, "error": "RAG system failed to initialize"}

        inserted = 0
        skipped = 0

        for pdf_path in pdf_paths:
            text = self.extract_text(pdf_path)
            if not text.strip():
                skipped += 1
                continue
            source = os.path.basename(pdf_path)
            await self.rag_system.ainsert(text, meta={"source": source})
            inserted += 1

        self.rag_system.flush()

        # Upload vector files to website repository
        try:
            result = await asyncio.to_thread(
                api_client.upload_vector_db, db_name, self.db_path
            )
            print(f"[RAG] Synced {db_name} to cloud: {result}")
        except Exception as e:
            print(f"[RAG] Cloud sync failed (offline?): {e}")

        return {"ok": True, "inserted": inserted, "skipped": skipped}

    async def load_remote_db(self, db_name: str, api_client) -> bool:
        """
        Downloads a pre-built vector DB from the website and activates it locally.
        This means PDFs do NOT need to be re-processed.
        """
        try:
            dest_dir = os.path.join(str(STORAGE_DIR), db_name)
            os.makedirs(dest_dir, exist_ok=True)

            downloaded = await asyncio.to_thread(
                api_client.download_vector_db, db_name, dest_dir
            )
            print(f"[RAG] Downloaded vector files: {downloaded}")

            if not downloaded:
                return False

            # Point the active db_path at the downloaded DB and reinitialize
            self.db_path = dest_dir
            self.db_name = db_name
            return self.initialize()
        except Exception as e:
            print(f"[RAG] Failed to load remote DB '{db_name}': {e}")
            return False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(self, prompt: str) -> str:
        if not self.rag_system:
            return "RAG system is offline. Please build or load a vector database first."
        try:
            res = await self.rag_system.aquery(prompt)
            return res.get("answer", "No context found in the database.")
        except Exception as e:
            return f"Query failed: {e}"


rag_manager = RagManager()
