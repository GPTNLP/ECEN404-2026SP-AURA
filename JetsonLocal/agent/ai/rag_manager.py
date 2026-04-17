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


def _clean_pdf_text(text: str) -> str:
    """Remove extraction artifacts common in LaTeX/arXiv PDFs."""
    # Strip arXiv margin stamps (e.g. "arXiv:2410.05779v3 [cs.IR] 15 Jan 2025")
    text = re.sub(r'arXiv:\S+\s*\[[\w.]+\]\s*\d+\s+\w+\s+\d{4}', '', text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

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
        self.build_in_progress: bool = False

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
            chunk_count = len(self.rag_system._rows)
            print(f"[RAG JOB] local LightRAG ready at {db_dir} — {chunk_count} chunk(s) loaded")
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
                "graph.json",
                "entity_list.json",
                "entity_emb.npy",
                "entity_faiss.index",
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
        """
        Extract text from a PDF using the best available method.

        Strategy:
          1. pdfminer.six  — handles complex font encodings in LaTeX/arXiv PDFs;
                             far more reliable than pypdf for academic papers.
          2. pypdf layout  — fallback; extraction_mode="layout" handles multi-column
                             layouts better than the default "plain" mode.

        The root cause of the "1 chunk" bug was that pypdf's plain extractor
        returned only the arXiv margin stamp (~100 chars) from LaTeX-compiled
        PDFs that use Type1/Type3 fonts without full ToUnicode maps.
        pdfminer.six resolves CMap tables correctly and recovers the full text.
        """
        name = os.path.basename(pdf_path)

        # ── Primary: pdfminer.six ─────────────────────────────────────────────
        try:
            from pdfminer.high_level import extract_text as _pm_extract
            text = _clean_pdf_text(_pm_extract(pdf_path) or "")
            if len(text.strip()) > 200:
                print(f"[RAG JOB] pdfminer.six extracted {len(text):,} chars from '{name}'")
                return text
            if text.strip():
                print(
                    f"[RAG JOB] pdfminer.six returned only {len(text.strip())} chars "
                    f"from '{name}' — trying pypdf as well"
                )
                pdfminer_text = text
            else:
                pdfminer_text = ""
        except ImportError:
            print("[RAG JOB] pdfminer.six not installed — falling back to pypdf "
                  "(run: pip install pdfminer.six)")
            pdfminer_text = ""
        except Exception as e:
            print(f"[RAG JOB] pdfminer.six failed for '{name}': {e}")
            pdfminer_text = ""

        # ── Fallback: pypdf with layout mode ──────────────────────────────────
        try:
            reader = PdfReader(pdf_path)
            parts: List[str] = []
            for page in reader.pages:
                try:
                    # layout mode reconstructs reading order in columnar PDFs
                    t = page.extract_text(extraction_mode="layout") or ""
                except Exception:
                    t = page.extract_text() or ""
                if t.strip():
                    parts.append(t)
            pypdf_text = _clean_pdf_text("\n\n".join(parts))
        except Exception as e:
            print(f"[RAG JOB] pypdf failed for '{name}': {e}")
            print(traceback.format_exc())
            pypdf_text = ""

        # Return whichever produced more content
        result = pypdf_text if len(pypdf_text) >= len(pdfminer_text) else pdfminer_text
        if result:
            method = "pypdf" if result is pypdf_text else "pdfminer.six"
            print(f"[RAG JOB] {method} extracted {len(result):,} chars from '{name}'")
        else:
            print(f"[RAG JOB] all extractors returned empty text for '{name}'")
        return result

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

        self.build_in_progress = True
        try:
            if not self.initialize_db(db_name, reset=True):
                raise RuntimeError(f"Failed to initialize local DB for '{db_name}'")

            # Capture the instance this build owns. command_loop may replace
            # self.rag_system concurrently (e.g. a chat query triggers reload),
            # which would corrupt flush() if we kept using self.rag_system.
            _rag = self.rag_system

            shutil.rmtree(temp_pdf_dir, ignore_errors=True)
            temp_pdf_dir.mkdir(parents=True, exist_ok=True)

            processed_files: List[str] = []
            skipped_files: List[str] = []
            failed_files: List[Dict[str, str]] = []

            total_files = len(document_paths)
            try:
                for file_idx, rel_path in enumerate(document_paths):
                    filename = os.path.basename(rel_path) or "document.pdf"
                    local_pdf_path = temp_pdf_dir / filename
                    file_num = file_idx + 1

                    try:
                        print(f"[RAG JOB] ── file {file_num}/{total_files}: '{filename}' ──")
                        print(f"[RAG JOB] downloading '{filename}' from website...")
                        await asyncio.to_thread(
                            api_client.download_document,
                            rel_path,
                            str(local_pdf_path),
                        )

                        if not local_pdf_path.exists():
                            raise RuntimeError("Downloaded file does not exist after download")

                        file_size = local_pdf_path.stat().st_size
                        print(f"[RAG JOB] downloaded '{filename}' ({file_size:,} bytes)")

                        print(f"[RAG JOB] extracting text from '{filename}'...")
                        text = await asyncio.to_thread(self.extract_text, str(local_pdf_path))
                        text_len = len(text.strip())
                        print(f"[RAG JOB] extracted {text_len:,} chars from '{filename}'")

                        if not text.strip():
                            print(f"[RAG JOB] skipped '{filename}' — no extractable text")
                            skipped_files.append(filename)
                            continue

                        chunks_before = len(_rag._rows)
                        print(f"[RAG JOB] vectorizing '{filename}' (this may take several minutes)...")
                        t0 = time.time()
                        await _rag.ainsert(
                            text,
                            meta={
                                "source": rel_path,
                                "filename": filename,
                            },
                        )
                        elapsed = time.time() - t0
                        chunks_added = len(_rag._rows) - chunks_before
                        if chunks_added == 0:
                            print(f"[RAG JOB] WARNING: '{filename}' produced 0 chunks (all filtered or empty text)")
                            failed_files.append({"file": filename, "error": "produced 0 insertable chunks"})
                            continue
                        stats = _rag.stats()
                        print(
                            f"[RAG JOB] '{filename}' done in {elapsed:.1f}s — "
                            f"DB: {stats['chunk_count']} chunks, "
                            f"{stats['entity_count']} entities, "
                            f"{stats['relation_count']} relations"
                        )
                        print(f"[RAG JOB] progress: {file_num}/{total_files} files processed")

                        processed_files.append(filename)

                    except Exception as file_error:
                        err_msg = str(file_error)
                        print(f"[RAG JOB] FAILED on '{filename}': {err_msg}")
                        print(traceback.format_exc())
                        failed_files.append({
                            "file": filename,
                            "error": err_msg,
                        })
                        continue

                if not processed_files:
                    reasons = "; ".join(
                        f"{f['file']}: {f['error']}" for f in failed_files
                    ) or "unknown"
                    raise RuntimeError(
                        "No PDFs were successfully inserted into LightRAG. "
                        f"Skipped={len(skipped_files)}, Failed={len(failed_files)}. "
                        f"Reasons: {reasons}"
                    )

                manifest = self.write_manifest(
                    db_name=db_name,
                    source_paths=document_paths,
                    processed_files=processed_files,
                    skipped_files=skipped_files,
                    failed_files=failed_files,
                )

                print("[RAG JOB] flushing vector DB files to disk")
                await asyncio.to_thread(_rag.flush)

                # Ensure rag_manager points to the fully built instance even if
                # command_loop replaced self.rag_system during the build.
                self.rag_system = _rag
                self.active_db_name = db_name
                self.active_db_path = db_dir

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

        finally:
            self.build_in_progress = False

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