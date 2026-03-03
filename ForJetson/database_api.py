# database_api.py
import os
import json
import shutil
import time
from typing import List, Optional, Dict, Any, Tuple

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from pypdf import PdfReader

from lightrag_local import LightRAG, QueryParam
from security import require_auth, require_ip_allowlist
from aura_db import init_db, doc_set_owner, doc_get_owner, doc_delete_owner, doc_move_owner

router = APIRouter(tags=["database"])
init_db()

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

DOCS_REL = os.getenv("AURA_DOCS_DIR", "storage/documents")
DB_REL = os.getenv("AURA_DB_DIR", "storage/databases")

DOCUMENTS_DIR = DOCS_REL if os.path.isabs(DOCS_REL) else os.path.join(BACKEND_DIR, DOCS_REL)
RAG_ROOT_DIR = DB_REL if os.path.isabs(DB_REL) else os.path.join(BACKEND_DIR, DB_REL)

DEFAULT_LLM = os.getenv("AURA_LLM_MODEL", "llama3.2:3b")
DEFAULT_EMBED = os.getenv("AURA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("AURA_OLLAMA_URL", "http://127.0.0.1:11434")

# Speed defaults (override via env if needed)
DEFAULT_CHAT_MODE = os.getenv("AURA_CHAT_MODE", "vector")  # vector|bm25|hybrid
DEFAULT_TOP_K = int(os.getenv("AURA_TOP_K", "4"))

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(RAG_ROOT_DIR, exist_ok=True)

# ---------------------------
# Auth helpers
# ---------------------------
def _role(payload: Dict[str, Any]) -> str:
    return str(payload.get("role") or "").lower()

def _email(payload: Dict[str, Any]) -> str:
    return str(payload.get("sub") or "").strip().lower()

def require_any_user(request: Request) -> Dict[str, Any]:
    require_ip_allowlist(request)
    return require_auth(request)

def require_admin(request: Request) -> Dict[str, Any]:
    require_ip_allowlist(request)
    p = require_auth(request)
    if _role(p) != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return p

def require_admin_or_ta(request: Request) -> Dict[str, Any]:
    require_ip_allowlist(request)
    p = require_auth(request)
    if _role(p) not in ("admin", "ta"):
        raise HTTPException(status_code=403, detail="Admin or TA only")
    return p

def require_owner_or_admin(request: Request, rel_path: str) -> Dict[str, Any]:
    """
    For file operations: admin can do anything.
    TA can only modify/delete/move files they uploaded.
    Students can't do file ops.
    """
    require_ip_allowlist(request)
    p = require_auth(request)
    r = _role(p)
    if r == "admin":
        return p
    if r != "ta":
        raise HTTPException(status_code=403, detail="Admin or TA only")

    rel_path_norm = (rel_path or "").replace("\\", "/").lstrip("/")
    owner = doc_get_owner(rel_path_norm)
    if not owner:
        raise HTTPException(status_code=403, detail="File has no owner record (admin only)")
    if (owner.get("owner_email") or "").strip().lower() != _email(p):
        raise HTTPException(status_code=403, detail="TA can only modify files they uploaded")
    return p

# ---------------------------
# Path helpers
# ---------------------------
def _safe_join(root: str, rel: str) -> str:
    rel = (rel or "").replace("\\", "/").lstrip("/")
    full = os.path.abspath(os.path.join(root, rel))
    root_abs = os.path.abspath(root)
    if not full.startswith(root_abs):
        raise HTTPException(status_code=400, detail="Invalid path")
    return full

def _db_dir(db_name: str) -> str:
    if not db_name or any(c in db_name for c in r'\/:*?"<>|'):
        raise HTTPException(status_code=400, detail="Invalid database name")
    return os.path.join(RAG_ROOT_DIR, db_name)

def _db_config_path(db_name: str) -> str:
    return os.path.join(_db_dir(db_name), "db.json")

def _db_workdir(db_name: str) -> str:
    return _db_dir(db_name)

def _load_db_config(db_name: str) -> Dict[str, Any]:
    p = _db_config_path(db_name)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Database not found")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_db_config(db_name: str, cfg: Dict[str, Any]):
    os.makedirs(_db_dir(db_name), exist_ok=True)
    with open(_db_config_path(db_name), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def _read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        return "\n\n".join(parts)
    except Exception:
        return ""

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _chunk_text(text: str, max_chars: int = 2400, overlap: int = 250) -> List[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(0, j - overlap)
    return chunks

def _walk_tree(root: str) -> Dict[str, Any]:
    def build(node_path: str) -> Dict[str, Any]:
        name = os.path.basename(node_path) or "documents"
        if os.path.isdir(node_path):
            children = []
            for item in sorted(os.listdir(node_path)):
                children.append(build(os.path.join(node_path, item)))
            return {"name": name, "type": "dir", "children": children}
        return {"name": name, "type": "file"}
    return build(root)

# ---------------------------
# RAG cache (HUGE speed win)
# ---------------------------
_RAG_CACHE: Dict[str, Tuple[LightRAG, float]] = {}  # db_name -> (rag, loaded_at)
_RAG_CACHE_TTL_S = float(os.getenv("AURA_RAG_CACHE_TTL_S", "3600"))  # 1 hour

def _get_rag(db_name: str) -> LightRAG:
    now = time.time()
    hit = _RAG_CACHE.get(db_name)
    if hit:
        rag, ts = hit
        if (now - ts) < _RAG_CACHE_TTL_S:
            return rag

    cfg = _load_db_config(db_name)
    rag = LightRAG(
        working_dir=_db_workdir(db_name),
        llm_model_name=str(cfg.get("llm_model") or DEFAULT_LLM),
        embed_model_name=str(cfg.get("embed_model") or DEFAULT_EMBED),
        ollama_base_url=str(cfg.get("ollama_url") or OLLAMA_URL),
    )
    _RAG_CACHE[db_name] = (rag, now)
    return rag

def _invalidate_rag(db_name: str):
    _RAG_CACHE.pop(db_name, None)

# ---------------------------
# Models
# ---------------------------
class MkdirRequest(BaseModel):
    path: str

class MoveRequest(BaseModel):
    src: str
    dst: str

class CreateDBRequest(BaseModel):
    name: str
    folders: List[str] = []

class BuildDBRequest(BaseModel):
    name: str
    folders: Optional[List[str]] = None
    force: bool = True

class ChatRequest(BaseModel):
    db: str
    query: str
    mode: Optional[str] = None   # vector|bm25|hybrid
    top_k: Optional[int] = None  # overrides default

# ---------------------------
# Endpoints
# ---------------------------
@router.get("/api/documents/tree")
def documents_tree(request: Request):
    require_any_user(request)
    return {"root": "documents", "path": DOCUMENTS_DIR, "tree": _walk_tree(DOCUMENTS_DIR)}

@router.post("/api/documents/mkdir")
def documents_mkdir(req: MkdirRequest, request: Request):
    require_admin_or_ta(request)
    full = _safe_join(DOCUMENTS_DIR, req.path)
    os.makedirs(full, exist_ok=True)
    return {"ok": True, "created": req.path}

@router.post("/api/documents/upload")
async def documents_upload(request: Request, path: str = "", files: List[UploadFile] = File(...)):
    payload = require_admin_or_ta(request)
    dest_dir = _safe_join(DOCUMENTS_DIR, path)
    os.makedirs(dest_dir, exist_ok=True)

    saved = 0
    owner_email = _email(payload)
    owner_role = _role(payload)

    for f in files:
        name = os.path.basename(f.filename or "file")
        out = os.path.join(dest_dir, name)
        with open(out, "wb") as w:
            w.write(await f.read())
        saved += 1

        rel_source = os.path.relpath(out, DOCUMENTS_DIR).replace("\\", "/")
        doc_set_owner(rel_source, owner_email, owner_role)

    return {"ok": True, "saved": saved, "path": path}

@router.delete("/api/documents/delete")
def documents_delete(request: Request, path: str):
    rel_norm = (path or "").replace("\\", "/").lstrip("/")
    full = _safe_join(DOCUMENTS_DIR, path)
    if not os.path.exists(full):
        raise HTTPException(status_code=404, detail="Not found")

    if os.path.isdir(full):
        require_admin(request)
        shutil.rmtree(full)
        return {"ok": True, "deleted": path}

    require_owner_or_admin(request, rel_norm)
    os.remove(full)
    doc_delete_owner(rel_norm)
    return {"ok": True, "deleted": path}

@router.post("/api/documents/move")
def documents_move(req: MoveRequest, request: Request):
    src_rel = (req.src or "").replace("\\", "/").lstrip("/")
    dst_rel = (req.dst or "").replace("\\", "/").lstrip("/")

    src = _safe_join(DOCUMENTS_DIR, req.src)
    dst = _safe_join(DOCUMENTS_DIR, req.dst)
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="Source not found")

    if os.path.isdir(src):
        require_admin(request)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        return {"ok": True, "src": req.src, "dst": req.dst}

    require_owner_or_admin(request, src_rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    doc_move_owner(src_rel, dst_rel)
    return {"ok": True, "src": req.src, "dst": req.dst}

@router.get("/api/databases")
def list_databases(request: Request):
    require_any_user(request)
    out = []
    for name in sorted(os.listdir(RAG_ROOT_DIR)):
        p = os.path.join(RAG_ROOT_DIR, name)
        if not os.path.isdir(p):
            continue
        if os.path.exists(_db_config_path(name)):
            out.append(name)
    return {"databases": out}

@router.post("/api/databases/create")
def create_database(req: CreateDBRequest, request: Request):
    require_admin(request)

    db_dir = _db_dir(req.name)
    os.makedirs(db_dir, exist_ok=True)

    cfg = {
        "name": req.name,
        "folders": req.folders,
        "llm_model": DEFAULT_LLM,
        "embed_model": DEFAULT_EMBED,
        "ollama_url": OLLAMA_URL,
    }
    _save_db_config(req.name, cfg)
    _invalidate_rag(req.name)
    return {"ok": True, "db": req.name, "config": cfg}

@router.get("/api/databases/{db_name}/config")
def get_database_config(db_name: str, request: Request):
    require_any_user(request)
    return _load_db_config(db_name)

@router.get("/api/databases/{db_name}/stats")
def database_stats(db_name: str, request: Request):
    require_any_user(request)
    cfg = _load_db_config(db_name)
    rag = _get_rag(db_name)
    return {"db": db_name, "config": cfg, "stats": rag.stats()}

@router.post("/api/databases/build")
async def build_database(req: BuildDBRequest, request: Request):
    require_admin(request)

    cfg = _load_db_config(req.name)
    folders = req.folders if req.folders is not None else cfg.get("folders", [])

    if not folders:
        raise HTTPException(status_code=400, detail="No folders selected for this database")

    workdir = _db_workdir(req.name)
    os.makedirs(workdir, exist_ok=True)

    rag = _get_rag(req.name)

    if req.force:
        rag.reset()

    all_files: List[str] = []
    for folder in folders:
        base = _safe_join(DOCUMENTS_DIR, folder)
        if not os.path.exists(base) or not os.path.isdir(base):
            raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")

        for root, _, files in os.walk(base):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext in [".pdf", ".txt", ".md"]:
                    all_files.append(os.path.join(root, fn))

    if not all_files:
        raise HTTPException(status_code=400, detail="No indexable files (.pdf/.txt/.md) found")

    inserted_chunks = 0
    skipped_files = 0

    for path in sorted(all_files):
        ext = os.path.splitext(path)[1].lower()
        text = _read_pdf(path) if ext == ".pdf" else _read_text(path)

        if not text.strip():
            skipped_files += 1
            continue

        rel_source = os.path.relpath(path, DOCUMENTS_DIR).replace("\\", "/")
        header = f"[SOURCE FILE: {rel_source}]\n\n"
        chunks = _chunk_text(header + text)

        for c in chunks:
            await rag.ainsert(c, meta={"source": rel_source})
            inserted_chunks += 1

    cfg["folders"] = folders
    _save_db_config(req.name, cfg)

    try:
        rag.flush()
    except Exception:
        pass

    _invalidate_rag(req.name)  # next chat loads fresh store from disk

    return {
        "ok": True,
        "status": "Database built",
        "db": req.name,
        "folders": folders,
        "files_found": len(all_files),
        "skipped_files": skipped_files,
        "inserted_chunks": inserted_chunks,
        "stats": rag.stats(),
    }

@router.post("/api/databases/chat")
async def database_chat(req: ChatRequest, request: Request):
    require_any_user(request)
    try:
        rag = _get_rag(req.db)
        mode = (req.mode or DEFAULT_CHAT_MODE).lower().strip()
        top_k = int(req.top_k or DEFAULT_TOP_K)
        top_k = max(1, min(top_k, 8))  # clamp

        # FAST DEFAULTS:
        # vector mode + low top_k is dramatically faster than hybrid
        param = QueryParam(mode=mode, top_k=top_k)
        return await rag.aquery(req.query, param=param)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))