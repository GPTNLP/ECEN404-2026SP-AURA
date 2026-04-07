import os
import json
import shutil
import time
import math
import re
from typing import List, Optional, Dict, Any, Iterable

from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader

from security import require_auth, require_ip_allowlist
from aura_db import init_db, doc_set_owner, doc_get_owner, doc_delete_owner, doc_move_owner

router = APIRouter(tags=["database"])
init_db()

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# Paths / env
# ---------------------------
DOCS_ABS = (os.getenv("AURA_DOCUMENTS_DIR", "") or "").strip()
DBS_ABS = (os.getenv("AURA_DATABASES_DIR", "") or "").strip()

DOCS_REL_OR_ABS = (os.getenv("AURA_DOCS_DIR", "storage/documents") or "").strip()
DB_REL_OR_ABS = (os.getenv("AURA_DB_DIR", "storage/databases") or "").strip()

if DOCS_ABS:
    DOCUMENTS_DIR = DOCS_ABS
else:
    DOCUMENTS_DIR = (
        DOCS_REL_OR_ABS
        if os.path.isabs(DOCS_REL_OR_ABS)
        else os.path.join(BACKEND_DIR, DOCS_REL_OR_ABS)
    )

if DBS_ABS:
    RAG_ROOT_DIR = DBS_ABS
else:
    RAG_ROOT_DIR = (
        DB_REL_OR_ABS
        if os.path.isabs(DB_REL_OR_ABS)
        else os.path.join(BACKEND_DIR, DB_REL_OR_ABS)
    )

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(RAG_ROOT_DIR, exist_ok=True)

DEFAULT_LLM = os.getenv("AURA_LLM_MODEL", "llama3.2:3b")
DEFAULT_EMBED = os.getenv("AURA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("AURA_OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_TOP_K = int(os.getenv("AURA_TOP_K", "4"))

# Website backend should always be simple-only.
AURA_ENABLE_RAG = False


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


def _db_chunks_path(db_name: str) -> str:
    return os.path.join(_db_dir(db_name), "chunks.jsonl")


def _db_stats_path(db_name: str) -> str:
    return os.path.join(_db_dir(db_name), "stats.json")


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


# ---------------------------
# Reading + chunking
# ---------------------------
def _read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        parts: List[str] = []
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

    chunks: List[str] = []
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
# Simple index helpers
# ---------------------------
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s or "") if t]


def _score_overlap(query_tokens: List[str], text_tokens: List[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    qset = set(query_tokens)
    tset = set(text_tokens)
    inter = len(qset & tset)
    return inter / math.sqrt(max(1.0, float(len(tset))))


def _write_chunks(db_name: str, records: Iterable[Dict[str, Any]]) -> int:
    path = _db_chunks_path(db_name)
    os.makedirs(_db_dir(db_name), exist_ok=True)

    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _read_chunks(db_name: str) -> List[Dict[str, Any]]:
    path = _db_chunks_path(db_name)
    if not os.path.exists(path):
        return []

    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _save_simple_stats(db_name: str, stats: Dict[str, Any]):
    with open(_db_stats_path(db_name), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def _load_simple_stats(db_name: str) -> Dict[str, Any]:
    p = _db_stats_path(db_name)
    if not os.path.exists(p):
        return {"chunk_count": 0, "files_found": 0, "skipped_files": 0, "mode": "simple"}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"chunk_count": 0, "files_found": 0, "skipped_files": 0, "mode": "simple"}


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
    mode: Optional[str] = None
    top_k: Optional[int] = None


# ---------------------------
# Endpoints: Documents
# ---------------------------
@router.get("/api/documents/download")
def download_document(path: str, request: Request):
    require_any_user(request)
    full_path = _safe_join(DOCUMENTS_DIR, path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full_path)


DEVICE_SECRET = os.getenv("DEVICE_SHARED_SECRET", "").strip()
ALLOWED_VECTOR_FILES = {"faiss.index", "embeddings.npy", "meta.json", "db.json", "entities.json"}


def _require_device_secret(x_device_secret: Optional[str]):
    if not DEVICE_SECRET:
        raise HTTPException(status_code=500, detail="DEVICE_SHARED_SECRET not configured on server")
    if (x_device_secret or "").strip() != DEVICE_SECRET:
        raise HTTPException(status_code=401, detail="Invalid device secret")


@router.post("/api/databases/{db_name}/sync_up")
async def sync_db_up(
    db_name: str,
    x_device_secret: Optional[str] = Header(default=None, alias="X-Device-Secret"),
    files: List[UploadFile] = File(...),
):
    _require_device_secret(x_device_secret)

    db_dir = _db_dir(db_name)
    os.makedirs(db_dir, exist_ok=True)

    if not os.path.exists(_db_config_path(db_name)):
        cfg = {
            "name": db_name,
            "folders": [],
            "engine": "simple",
            "synced_from_device": True,
        }
        _save_db_config(db_name, cfg)

    saved = []
    for f in files:
        if f.filename in ALLOWED_VECTOR_FILES:
            out = os.path.join(db_dir, f.filename)
            with open(out, "wb") as w:
                w.write(await f.read())
            saved.append(f.filename)

    return {"ok": True, "db": db_name, "saved": saved}


@router.get("/api/databases/{db_name}/sync_down/{filename}")
def sync_db_down(
    db_name: str,
    filename: str,
    x_device_secret: Optional[str] = Header(default=None, alias="X-Device-Secret"),
):
    _require_device_secret(x_device_secret)

    if filename not in ALLOWED_VECTOR_FILES:
        raise HTTPException(status_code=400, detail="Invalid vector file name")

    full_path = os.path.join(_db_dir(db_name), filename)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Vector file not found")
    return FileResponse(full_path)


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


# ---------------------------
# Endpoints: Databases
# ---------------------------
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
        "engine": "simple",
    }
    _save_db_config(req.name, cfg)

    return {"ok": True, "db": req.name, "config": cfg}


@router.get("/api/databases/{db_name}/config")
def get_database_config(db_name: str, request: Request):
    require_any_user(request)
    return _load_db_config(db_name)


@router.get("/api/databases/{db_name}/stats")
def database_stats(db_name: str, request: Request):
    require_any_user(request)
    cfg = _load_db_config(db_name)
    simple_stats = _load_simple_stats(db_name)

    return {
        "db": db_name,
        "config": cfg,
        "stats": {
            "chunk_count": int(simple_stats.get("chunk_count") or 0),
            "vdb_path": _db_dir(db_name),
            "engine": "simple",
            "files_found": int(simple_stats.get("files_found") or 0),
            "skipped_files": int(simple_stats.get("skipped_files") or 0),
        },
    }


@router.post("/api/databases/build")
async def build_database(req: BuildDBRequest, request: Request):
    require_admin(request)

    cfg = _load_db_config(req.name)
    folders = req.folders if req.folders is not None else cfg.get("folders", [])

    if not folders:
        raise HTTPException(status_code=400, detail="No folders selected for this database")

    db_dir = _db_dir(req.name)
    os.makedirs(db_dir, exist_ok=True)

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

    if req.force:
        try:
            if os.path.exists(_db_chunks_path(req.name)):
                os.remove(_db_chunks_path(req.name))
        except Exception:
            pass
        try:
            if os.path.exists(_db_stats_path(req.name)):
                os.remove(_db_stats_path(req.name))
        except Exception:
            pass

    inserted_chunks = 0
    skipped_files = 0
    records: List[Dict[str, Any]] = []

    for path in sorted(all_files):
        ext = os.path.splitext(path)[1].lower()
        text = _read_pdf(path) if ext == ".pdf" else _read_text(path)

        if not (text or "").strip():
            skipped_files += 1
            continue

        rel_source = os.path.relpath(path, DOCUMENTS_DIR).replace("\\", "/")
        header = f"[SOURCE FILE: {rel_source}]\n\n"
        chunks = _chunk_text(header + text)

        for c in chunks:
            records.append({"source": rel_source, "text": c})
            inserted_chunks += 1

    if inserted_chunks == 0:
        raise HTTPException(status_code=400, detail="No readable text found in indexable files")

    chunk_count = _write_chunks(req.name, records)

    cfg["folders"] = folders
    cfg["engine"] = "simple"
    _save_db_config(req.name, cfg)

    simple_stats = {
        "mode": "simple",
        "chunk_count": chunk_count,
        "files_found": len(all_files),
        "skipped_files": skipped_files,
        "built_ts": int(time.time()),
    }
    _save_simple_stats(req.name, simple_stats)

    return {
        "ok": True,
        "status": "Database built (simple)",
        "db": req.name,
        "folders": folders,
        "files_found": len(all_files),
        "skipped_files": skipped_files,
        "inserted_chunks": chunk_count,
        "stats": {
            "chunk_count": chunk_count,
            "vdb_path": _db_dir(req.name),
            "engine": "simple",
        },
        "engine": "simple",
    }


@router.post("/api/databases/chat")
async def database_chat(req: ChatRequest, request: Request):
    require_any_user(request)

    chunks = _read_chunks(req.db)
    if not chunks:
        raise HTTPException(status_code=400, detail="Database has no built index yet. Click Build first.")

    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query is empty")

    qtok = _tokenize(q)
    scored: List[tuple[float, Dict[str, Any]]] = []

    for r in chunks:
        txt = str(r.get("text") or "")
        score = _score_overlap(qtok, _tokenize(txt))
        if score > 0:
            scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = max(1, min(int(req.top_k or DEFAULT_TOP_K), 8))
    top = scored[:top_k] if scored else []

    sources: List[str] = []
    context_snips: List[str] = []

    for _, r in top:
        src = str(r.get("source") or "")
        if src and src not in sources:
            sources.append(src)

        t = str(r.get("text") or "")
        t = t.replace("\n", " ").strip()
        context_snips.append(t[:500])

    if not top:
        return {
            "answer": "I couldn’t find anything relevant in the indexed documents for that query.",
            "sources": [],
            "engine": "simple",
        }

    answer = "Top matches from your documents:\n\n" + "\n\n".join(
        [f"- {snip}" for snip in context_snips[:3]]
    )

    return {
        "answer": answer,
        "sources": sources,
        "engine": "simple",
    }