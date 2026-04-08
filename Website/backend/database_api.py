import os
import json
import shutil
import time
import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

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

DEVICE_SECRET = os.getenv("DEVICE_SHARED_SECRET", "").strip()
ALLOWED_VECTOR_FILES = {"faiss.index", "embeddings.npy", "meta.json", "db.json", "entities.json"}

BUILD_JOBS_DIR = os.path.join(RAG_ROOT_DIR, "_build_jobs")
os.makedirs(BUILD_JOBS_DIR, exist_ok=True)

ACTIVE_BUILD_STATUSES = {"pending", "claimed", "running"}


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


def _require_device_secret(x_device_secret: Optional[str]):
    if not DEVICE_SECRET:
        raise HTTPException(status_code=500, detail="DEVICE_SHARED_SECRET not configured on server")
    if (x_device_secret or "").strip() != DEVICE_SECRET:
        raise HTTPException(status_code=401, detail="Invalid device secret")


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


def _build_status_path(db_name: str) -> str:
    return os.path.join(_db_dir(db_name), "build_status.json")


def _job_path(job_id: str) -> str:
    return os.path.join(BUILD_JOBS_DIR, f"{job_id}.json")


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


def _save_build_status(db_name: str, payload: Dict[str, Any]) -> None:
    os.makedirs(_db_dir(db_name), exist_ok=True)
    with open(_build_status_path(db_name), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_build_status(db_name: str) -> Dict[str, Any]:
    path = _build_status_path(db_name)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_job(job: Dict[str, Any]) -> None:
    with open(_job_path(job["job_id"]), "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)


def _load_job(job_id: str) -> Dict[str, Any]:
    path = _job_path(job_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Build job not found")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load build job: {e}")


def _collect_pdf_files(folders: List[str]) -> List[str]:
    files: List[str] = []

    for folder in folders:
        base = _safe_join(DOCUMENTS_DIR, folder)
        if not os.path.exists(base):
            continue

        if os.path.isfile(base):
            rel = os.path.relpath(base, DOCUMENTS_DIR).replace("\\", "/")
            if rel.lower().endswith(".pdf"):
                files.append(rel)
            continue

        for root, _, names in os.walk(base):
            for name in names:
                if not name.lower().endswith(".pdf"):
                    continue
                full = os.path.join(root, name)
                rel = os.path.relpath(full, DOCUMENTS_DIR).replace("\\", "/")
                files.append(rel)

    return sorted(set(files))


def _get_active_build_job() -> Optional[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    for name in os.listdir(BUILD_JOBS_DIR):
        if not name.endswith(".json"):
            continue

        path = os.path.join(BUILD_JOBS_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                job = json.load(f)
        except Exception:
            continue

        status = str(job.get("status") or "").lower()
        if status in ACTIVE_BUILD_STATUSES:
            jobs.append(job)

    if not jobs:
        return None

    jobs.sort(key=lambda j: float(j.get("created_at") or 0.0))
    return jobs[0]


def _raise_if_build_in_progress(action_label: str):
    active_job = _get_active_build_job()
    if not active_job:
        return

    active_db = str(active_job.get("db_name") or "unknown")
    active_status = str(active_job.get("status") or "running")
    raise HTTPException(
        status_code=409,
        detail=(
            f'Jetson is currently vectorizing "{active_db}" '
            f'({active_status}). No new database actions can be taken right now.'
        ),
    )


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


class RagBuildJobAckRequest(BaseModel):
    job_id: str
    device_id: str
    status: str
    note: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


# ---------------------------
# Endpoints: Documents
# ---------------------------
@router.get("/api/documents/download")
def download_document(
    path: str,
    request: Request,
    x_device_secret: Optional[str] = Header(default=None, alias="X-Device-Secret"),
):
    if (x_device_secret or "").strip():
        _require_device_secret(x_device_secret)
    else:
        require_any_user(request)

    full_path = _safe_join(DOCUMENTS_DIR, path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full_path)


@router.get("/api/documents/tree")
def documents_tree(request: Request):
    require_any_user(request)

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
    _raise_if_build_in_progress("create")

    db_dir = _db_dir(req.name)
    os.makedirs(db_dir, exist_ok=True)

    cfg = {
        "name": req.name,
        "folders": req.folders,
        "llm_model": DEFAULT_LLM,
        "embed_model": DEFAULT_EMBED,
        "ollama_url": OLLAMA_URL,
        "engine": "jetson_remote",
        "build_host": "jetson",
        "build_trigger": "website_backend",
    }
    _save_db_config(req.name, cfg)

    _save_build_status(
        req.name,
        {
            "db_name": req.name,
            "status": "idle",
            "message": "Database created. Waiting for build.",
            "updated_at": time.time(),
        },
    )

    return {"ok": True, "db": req.name, "config": cfg}


@router.delete("/api/databases/{db_name}")
def delete_database(db_name: str, request: Request):
    require_admin(request)
    _raise_if_build_in_progress("delete")

    db_dir = _db_dir(db_name)
    if not os.path.exists(db_dir):
        raise HTTPException(status_code=404, detail="Database not found")

    shutil.rmtree(db_dir, ignore_errors=True)
    return {"ok": True, "deleted": db_name}


@router.get("/api/databases/{db_name}/config")
def get_database_config(db_name: str, request: Request):
    require_any_user(request)
    return _load_db_config(db_name)


@router.get("/api/databases/{db_name}/stats")
def database_stats(db_name: str, request: Request):
    require_any_user(request)

    cfg = _load_db_config(db_name)
    build_status = _load_build_status(db_name)

    existing_files = []
    db_dir = _db_dir(db_name)
    for name in sorted(ALLOWED_VECTOR_FILES):
        full = os.path.join(db_dir, name)
        if os.path.exists(full):
            existing_files.append(name)

    return {
        "db": db_name,
        "config": cfg,
        "stats": {
            "engine": "jetson_remote",
            "vector_files_present": existing_files,
            "vector_file_count": len(existing_files),
            "vdb_path": db_dir,
        },
        "build": build_status,
    }


@router.post("/api/databases/build")
async def build_database(req: BuildDBRequest, request: Request):
    require_admin(request)
    _raise_if_build_in_progress("build")

    cfg = _load_db_config(req.name)
    folders = req.folders if req.folders is not None else cfg.get("folders", [])

    if not folders:
        raise HTTPException(status_code=400, detail="No folders selected for this database")

    document_paths = _collect_pdf_files(folders)
    if not document_paths:
        raise HTTPException(status_code=400, detail="No PDF files found in selected folders")

    updated_cfg = {
        **cfg,
        "name": req.name,
        "folders": folders,
        "engine": "jetson_remote",
        "build_host": "jetson",
        "build_trigger": "website_backend",
        "last_build_requested_at": time.time(),
    }
    _save_db_config(req.name, updated_cfg)

    now = time.time()
    job_id = uuid.uuid4().hex

    job = {
        "job_id": job_id,
        "db_name": req.name,
        "folders": folders,
        "document_paths": document_paths,
        "file_count": len(document_paths),
        "status": "pending",
        "created_at": now,
        "created_at_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "force": bool(req.force),
    }
    _save_job(job)

    _save_build_status(
        req.name,
        {
            "job_id": job_id,
            "db_name": req.name,
            "status": "pending",
            "file_count": len(document_paths),
            "folders": folders,
            "document_paths": document_paths,
            "created_at": now,
            "created_at_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "message": f"Queued {len(document_paths)} PDF(s) for Jetson build",
        },
    )

    return {
        "ok": True,
        "queued": True,
        "db": req.name,
        "job_id": job_id,
        "file_count": len(document_paths),
        "document_paths": document_paths,
        "message": f'Queued "{req.name}" for Jetson vectorization',
    }


@router.get("/api/databases/build_jobs/next")
def get_next_build_job(
    device_id: str,
    x_device_secret: Optional[str] = Header(default=None, alias="X-Device-Secret"),
):
    _require_device_secret(x_device_secret)

    pending_jobs: List[Dict[str, Any]] = []

    for name in os.listdir(BUILD_JOBS_DIR):
        if not name.endswith(".json"):
            continue

        path = os.path.join(BUILD_JOBS_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                job = json.load(f)
        except Exception:
            continue

        if str(job.get("status") or "").lower() == "pending":
            pending_jobs.append(job)

    pending_jobs.sort(key=lambda j: float(j.get("created_at") or 0.0))

    if not pending_jobs:
        return {"ok": True, "job": None}

    job = pending_jobs[0]
    now = time.time()

    job["status"] = "claimed"
    job["claimed_by_device"] = device_id
    job["claimed_at"] = now
    job["claimed_at_readable"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    _save_job(job)

    _save_build_status(
        job["db_name"],
        {
            **_load_build_status(job["db_name"]),
            "job_id": job["job_id"],
            "db_name": job["db_name"],
            "status": "claimed",
            "file_count": int(job.get("file_count") or 0),
            "folders": job.get("folders") or [],
            "document_paths": job.get("document_paths") or [],
            "claimed_by_device": device_id,
            "claimed_at": now,
            "claimed_at_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "message": f'Jetson "{device_id}" claimed the build job',
        },
    )

    return {"ok": True, "job": job}


@router.post("/api/databases/build_jobs/ack")
def ack_build_job(
    req: RagBuildJobAckRequest,
    x_device_secret: Optional[str] = Header(default=None, alias="X-Device-Secret"),
):
    _require_device_secret(x_device_secret)

    job = _load_job(req.job_id)
    now = time.time()

    job["status"] = req.status
    job["acked_by_device"] = req.device_id
    job["note"] = req.note or ""
    job["extra"] = req.extra or {}
    job["updated_at"] = now
    job["updated_at_readable"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    _save_job(job)

    _save_build_status(
        job["db_name"],
        {
            "job_id": job["job_id"],
            "db_name": job["db_name"],
            "status": req.status,
            "file_count": int(job.get("file_count") or 0),
            "folders": job.get("folders") or [],
            "document_paths": job.get("document_paths") or [],
            "device_id": req.device_id,
            "updated_at": now,
            "updated_at_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "message": req.note or "",
            "extra": req.extra or {},
        },
    )

    return {"ok": True, "job_id": req.job_id, "status": req.status}


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
            "engine": "jetson_remote",
            "build_host": "jetson",
            "build_trigger": "website_backend",
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

    now = time.time()
    _save_build_status(
        db_name,
        {
            **_load_build_status(db_name),
            "db_name": db_name,
            "status": "synced",
            "saved_files": saved,
            "updated_at": now,
            "updated_at_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "message": f"Received synced vector files from Jetson: {', '.join(saved) if saved else 'none'}",
        },
    )

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


@router.post("/api/databases/chat")
async def database_chat(req: ChatRequest, request: Request):
    require_any_user(request)
    raise HTTPException(
        status_code=400,
        detail="Website-side vectorization/chat is disabled. Run document chat from the Jetson-backed flow.",
    )