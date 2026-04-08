import asyncio
import json
import os
import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader

from ai.lightrag_local import LightRAG
from core.config import DEFAULT_MODEL, EMBEDDING_MODEL, STORAGE_DIR, LOCAL_DB_NAME


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (value or "").strip())
    return cleaned.strip("._") or "default_db"


class RagManager:
    def __init__(self):
        self.root_dir = Path(STORAGE_DIR) / "jetsonlocaldb"
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.rag_system: Optional[LightRAG] = None
        self.active_db_name: Optional[str] = None
        self.active_db_path: Optional[Path] = None

    def get_db_dir(self, db_name: str) -> Path:
        return self.root_dir / _safe_name(db_name)

    def get_temp_pdf_dir(self, db_name: str) -> Path:
        return self.get_db_dir(db_name) / "_temp_pdfs"

    def get_manifest_path(self, db_name: str) -> Path:
        return self.get_db_dir(db_name) / "build_manifest.json"

    def initialize(self) -> bool:
        default_name = LOCAL_DB_NAME or "default_db"
        default_dir = self.get_db_dir(default_name)

        if default_dir.exists():
            try:
                return self.initialize_db(default_name, reset=False)
            except Exception:
                return False

        self.active_db_name = None
        self.active_db_path = None
        self.rag_system = None
        return True

    def initialize_db(self, db_name: str, reset: bool = False) -> bool:
        db_dir = self.get_db_dir(db_name)

        if reset and db_dir.exists():
            shutil.rmtree(db_dir, ignore_errors=True)

        db_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.rag_system = LightRAG(
                working_dir=str(db_dir),
                llm_model_name=DEFAULT_MODEL,
                embed_model_name=EMBEDDING_MODEL,
            )
            self.active_db_name = db_name
            self.active_db_path = db_dir
            print(f"[RAG JOB] local LightRAG ready at {db_dir}")
            return True
        except Exception as e:
            print(f"[RAG JOB] failed to initialize DB '{db_name}': {e}")
            print(traceback.format_exc())
            self.rag_system = None
            self.active_db_name = None
            self.active_db_path = None
            return False

    def stats(self) -> Dict[str, Any]:
        existing_files: List[str] = []
        if self.active_db_path and self.active_db_path.exists():
            for name in [
                "faiss.index",
                "embeddings.npy",
                "meta.json",
                "db.json",
                "entities.json",
                "build_manifest.json",
            ]:
                if (self.active_db_path / name).exists():
                    existing_files.append(name)

        return {
            "active_db_name": self.active_db_name,
            "active_db_path": str(self.active_db_path) if self.active_db_path else None,
            "ready": self.rag_system is not None,
            "files_present": existing_files,
        }

    def extract_text(self, pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            parts = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(part for part in parts if part.strip())
        except Exception as e:
            print(f"[RAG JOB] PDF extract failed for '{pdf_path}': {e}")
            print(traceback.format_exc())
            return ""

    def write_manifest(
        self,
        db_name: str,
        source_paths: List[str],
        processed_files: List[str],
        skipped_files: List[str],
        failed_files: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        manifest = {
            "db_name": db_name,
            "source_count": len(source_paths),
            "processed_count": len(processed_files),
            "skipped_count": len(skipped_files),
            "failed_count": len(failed_files),
            "processed_files": processed_files,
            "skipped_files": skipped_files,
            "failed_files": failed_files,
            "built_at_epoch": time.time(),
            "built_at_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
            "storage_path": str(self.get_db_dir(db_name)),
        }

        manifest_path = self.get_manifest_path(db_name)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    async def build_database_from_document_paths(
        self,
        db_name: str,
        document_paths: List[str],
        api_client,
    ) -> Dict[str, Any]:
        if not document_paths:
            raise RuntimeError("No PDFs were supplied for this build job.")

        db_dir = self.get_db_dir(db_name)
        temp_pdf_dir = self.get_temp_pdf_dir(db_name)

        print(f"[RAG JOB] received {len(document_paths)} PDF(s) for '{db_name}'")
        print(f"[RAG JOB] preparing local DB folder: {db_dir}")

        if not self.initialize_db(db_name, reset=True):
            raise RuntimeError(f"Failed to initialize local DB for '{db_name}'")

        shutil.rmtree(temp_pdf_dir, ignore_errors=True)
        temp_pdf_dir.mkdir(parents=True, exist_ok=True)

        processed_files: List[str] = []
        skipped_files: List[str] = []
        failed_files: List[Dict[str, str]] = []

        try:
            for rel_path in document_paths:
                filename = os.path.basename(rel_path) or "document.pdf"
                local_pdf_path = temp_pdf_dir / filename

                try:
                    print(f"[RAG JOB] received PDF: {filename}")
                    await asyncio.to_thread(
                        api_client.download_document,
                        rel_path,
                        str(local_pdf_path),
                    )

                    if not local_pdf_path.exists():
                        raise RuntimeError("Downloaded file does not exist after download")

                    file_size = local_pdf_path.stat().st_size
                    print(f"[RAG JOB] downloaded '{filename}' ({file_size} bytes)")

                    print(f'[RAG JOB] vectorizing "{filename}"')
                    text = await asyncio.to_thread(self.extract_text, str(local_pdf_path))
                    text_len = len(text.strip())
                    print(f"[RAG JOB] extracted {text_len} chars from '{filename}'")

                    if not text.strip():
                        print(f'[RAG JOB] skipped "{filename}" (no extractable text)')
                        skipped_files.append(filename)
                        continue

                    if self.rag_system is None:
                        raise RuntimeError("LightRAG is not initialized")

                    print(f"[RAG JOB] inserting '{filename}' into LightRAG")
                    await asyncio.to_thread(self.rag_system.insert, text)
                    print(f"[RAG JOB] inserted '{filename}' successfully")

                    processed_files.append(filename)

                except Exception as file_error:
                    err_msg = str(file_error)
                    print(f"[RAG JOB] failed on '{filename}': {err_msg}")
                    print(traceback.format_exc())
                    failed_files.append({
                        "file": filename,
                        "error": err_msg,
                    })
                    continue

            if not processed_files:
                raise RuntimeError(
                    "No PDFs were successfully inserted into LightRAG. "
                    f"Skipped={len(skipped_files)}, Failed={len(failed_files)}"
                )

            manifest = self.write_manifest(
                db_name=db_name,
                source_paths=document_paths,
                processed_files=processed_files,
                skipped_files=skipped_files,
                failed_files=failed_files,
            )

            print("[RAG JOB] deleting temporary PDFs from Jetson")
            shutil.rmtree(temp_pdf_dir, ignore_errors=True)

            print(f"[RAG JOB] local vector DB saved in: {db_dir}")
            print(f"[RAG JOB] sending vector DB copy back to website for '{db_name}'")

            upload_result = await asyncio.to_thread(
                api_client.upload_vector_db,
                db_name,
                str(db_dir),
            )

            print(f"[RAG JOB] vector DB sync complete for '{db_name}'")

            return {
                "ok": True,
                "db_name": db_name,
                "db_dir": str(db_dir),
                "processed_count": len(processed_files),
                "skipped_count": len(skipped_files),
                "failed_count": len(failed_files),
                "processed_files": processed_files,
                "skipped_files": skipped_files,
                "failed_files": failed_files,
                "manifest": manifest,
                "upload_result": upload_result,
            }

        finally:
            if temp_pdf_dir.exists():
                try:
                    print("[RAG JOB] cleaning any remaining temporary PDFs")
                    shutil.rmtree(temp_pdf_dir, ignore_errors=True)
                except Exception:
                    pass

    async def load_remote_db(self, db_name: str, api_client) -> bool:
        db_dir = self.get_db_dir(db_name)
        print(f"[RAG JOB] downloading remote vector DB for '{db_name}' into {db_dir}")

        if db_dir.exists():
            shutil.rmtree(db_dir, ignore_errors=True)
        db_dir.mkdir(parents=True, exist_ok=True)

        try:
            await asyncio.to_thread(api_client.download_vector_db, db_name, str(db_dir))
            return self.initialize_db(db_name, reset=False)
        except Exception as e:
            print(f"[RAG JOB] failed to load remote DB '{db_name}': {e}")
            print(traceback.format_exc())
            return False

    async def query(self, prompt: str) -> str:
        if not self.rag_system:
            return "RAG system is offline."

        try:
            result = await self.rag_system.aquery(prompt)
            if isinstance(result, dict):
                return result.get("answer", "No context found in local database.")
            return str(result)
        except Exception as e:
            return f"Query failed: {e}"


rag_manager = RagManager()