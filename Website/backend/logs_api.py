import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from security import require_auth

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

router = APIRouter(prefix="/logs", tags=["logs"])

STORAGE_DIR = Path(
    os.getenv("AURA_STORAGE_DIR") or str(Path(__file__).resolve().parent / "storage")
)
LOG_FILE = STORAGE_DIR / "chat_logs.jsonl"
SESSIONS_DIR = STORAGE_DIR / "sessions"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

LOG_INGEST_SECRET = os.getenv("LOG_INGEST_SECRET", "")
DEVICE_SECRET = os.getenv("DEVICE_SHARED_SECRET", "").strip()

print(f"[LOGS] STORAGE_DIR = {STORAGE_DIR}")
print(f"[LOGS] LOG_FILE = {LOG_FILE}")
print(f"[LOGS] SESSIONS_DIR = {SESSIONS_DIR}")


# ============================================================================
# Helpers
# ============================================================================

def require_admin(request: Request) -> Dict[str, Any]:
    payload = require_auth(request)
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return payload


def _append_log(obj: Dict[str, Any]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_logs(limit: int, offset: int) -> List[Dict[str, Any]]:
    if not LOG_FILE.exists():
        return []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines.reverse()  # newest first
    start = max(0, offset)
    end = max(0, offset + limit)

    out: List[Dict[str, Any]] = []
    for line in lines[start:end]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _require_device_secret(x_device_secret: Optional[str]) -> None:
    if not DEVICE_SECRET:
        raise HTTPException(
            status_code=500,
            detail="DEVICE_SHARED_SECRET not configured on server",
        )
    if (x_device_secret or "").strip() != DEVICE_SECRET:
        raise HTTPException(status_code=401, detail="Invalid device secret")


def _sanitize_session_id(session_id: str) -> str:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_.")
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    return safe


def _session_path(session_id: str) -> Path:
    safe = _sanitize_session_id(session_id)
    return SESSIONS_DIR / f"{safe}.json"


def _read_session_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Bad session file: {exc}") from exc


def _write_session_file(data: Dict[str, Any]) -> Dict[str, Any]:
    session_id = data["session_id"]
    path = _session_path(session_id)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return data


def _session_meta(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "session_id": data.get("session_id"),
        "title": data.get("title") or "Untitled chat",
        "owner_email": data.get("owner_email"),
        "owner_role": data.get("owner_role"),
        "device_id": data.get("device_id"),
        "db_name": data.get("db_name"),
        "message_count": data.get("message_count", len(data.get("history", []))),
        "created_ts": data.get("created_ts"),
        "updated_ts": data.get("updated_ts"),
        "source": data.get("source", "website"),
    }


def _list_all_sessions() -> List[Dict[str, Any]]:
    sessions: List[Dict[str, Any]] = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = _read_session_file(p)
            sessions.append(data)
        except Exception:
            continue
    return sessions


def _get_session_or_404(session_id: str) -> Dict[str, Any]:
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return _read_session_file(path)


def _check_session_owner_or_admin(request: Request, session_data: Dict[str, Any]) -> Dict[str, Any]:
    payload = require_auth(request)
    role = payload.get("role")
    email = (payload.get("sub") or "").strip().lower()

    if role == "admin":
        return payload

    owner_email = (session_data.get("owner_email") or "").strip().lower()
    if owner_email != email:
        raise HTTPException(status_code=403, detail="Forbidden")
    return payload


# ============================================================================
# Event log models + routes
# ============================================================================

class LogWrite(BaseModel):
    event: str = "chat"
    prompt: Optional[str] = None
    response_preview: Optional[str] = None
    model: Optional[str] = None
    latency_ms: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


class LogIngest(BaseModel):
    event: str = "chat"
    user_email: Optional[str] = None
    user_role: Optional[str] = None
    prompt: Optional[str] = None
    response_preview: Optional[str] = None
    model: Optional[str] = None
    latency_ms: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


@router.post("/write")
def write_log(data: LogWrite, request: Request):
    payload = require_auth(request)
    now = int(time.time())

    obj = {
        "ts": now,
        "event": data.event,
        "user_email": payload.get("sub"),
        "user_role": payload.get("role"),
        "prompt": data.prompt,
        "response_preview": data.response_preview,
        "model": data.model,
        "latency_ms": data.latency_ms,
        "meta": data.meta or {},
        "source": "frontend_or_api",
    }
    _append_log(obj)
    return {"ok": True}


@router.post("/ingest")
def ingest_log(data: LogIngest, request: Request):
    if not LOG_INGEST_SECRET:
        raise HTTPException(status_code=500, detail="Server missing LOG_INGEST_SECRET")

    secret = request.headers.get("x-log-secret", "")
    if not secret or secret != LOG_INGEST_SECRET:
        raise HTTPException(status_code=403, detail="Bad ingest secret")

    now = int(time.time())
    obj = {
        "ts": now,
        "event": data.event,
        "user_email": data.user_email,
        "user_role": data.user_role,
        "prompt": data.prompt,
        "response_preview": data.response_preview,
        "model": data.model,
        "latency_ms": data.latency_ms,
        "meta": data.meta or {},
        "source": "ml_ingest",
    }
    _append_log(obj)
    return {"ok": True}


@router.get("/mine")
def my_logs(request: Request, limit: int = 200, offset: int = 0):
    payload = require_auth(request)
    me = (payload.get("sub") or "").strip().lower()

    limit = max(1, min(limit, 1000))
    offset = max(0, offset)

    items = _read_logs(limit=5000, offset=0)
    mine = [
        it
        for it in items
        if str(it.get("user_email", "")).strip().lower() == me
    ]
    page = mine[offset: offset + limit]

    return {
        "ok": True,
        "email": me,
        "total": len(mine),
        "limit": limit,
        "offset": offset,
        "items": page,
    }


@router.get("/list")
def list_logs(
    request: Request,
    limit: int = 200,
    offset: int = 0,
    q: str = "",
    role: str = "",
    event: str = "",
):
    require_admin(request)

    limit = max(1, min(limit, 1000))
    offset = max(0, offset)

    items = _read_logs(limit=5000, offset=0)

    q_l = (q or "").strip().lower()
    role_l = (role or "").strip().lower()
    event_l = (event or "").strip().lower()

    def matches(it: Dict[str, Any]) -> bool:
        if event_l and str(it.get("event", "")).lower() != event_l:
            return False
        if role_l and str(it.get("user_role", "")).lower() != role_l:
            return False
        if q_l:
            blob = " ".join(
                [
                    str(it.get("user_email", "")),
                    str(it.get("user_role", "")),
                    str(it.get("event", "")),
                    str(it.get("prompt", "")),
                    str(it.get("response_preview", "")),
                    json.dumps(it.get("meta", {}), ensure_ascii=False),
                ]
            ).lower()
            if q_l not in blob:
                return False
        return True

    filtered = [it for it in items if matches(it)]
    page = filtered[offset: offset + limit]

    return {
        "ok": True,
        "total_scanned": len(items),
        "total_matched": len(filtered),
        "limit": limit,
        "offset": offset,
        "items": page,
    }


# ============================================================================
# Session models
# ============================================================================

class SessionMessage(BaseModel):
    role: str
    content: str
    ts: Optional[int] = None


class SessionCreate(BaseModel):
    title: Optional[str] = None
    db_name: Optional[str] = None
    device_id: Optional[str] = None
    history: List[SessionMessage] = Field(default_factory=list)


class SessionUpdate(BaseModel):
    title: Optional[str] = None
    db_name: Optional[str] = None
    device_id: Optional[str] = None
    history: List[SessionMessage] = Field(default_factory=list)


class SessionIngest(BaseModel):
    session_id: str
    device_id: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
    updated_ts: Optional[int] = None
    title: Optional[str] = None
    db_name: Optional[str] = None
    owner_email: Optional[str] = None
    owner_role: Optional[str] = None


# ============================================================================
# User-owned session routes
# ============================================================================

@router.get("/my-sessions")
def list_my_sessions(request: Request):
    payload = require_auth(request)
    me = (payload.get("sub") or "").strip().lower()

    out: List[Dict[str, Any]] = []
    for data in _list_all_sessions():
        owner_email = (data.get("owner_email") or "").strip().lower()
        if owner_email == me:
            out.append(_session_meta(data))

    return {"ok": True, "sessions": out}


@router.post("/my-sessions/start")
def start_my_session(data: SessionCreate, request: Request):
    payload = require_auth(request)
    now = int(time.time())

    history = [m.model_dump() for m in data.history]
    session_id = str(uuid.uuid4())

    session = {
        "session_id": session_id,
        "title": (data.title or "New chat").strip() or "New chat",
        "owner_email": payload.get("sub"),
        "owner_role": payload.get("role"),
        "device_id": data.device_id,
        "db_name": data.db_name,
        "history": history,
        "message_count": len(history),
        "created_ts": now,
        "updated_ts": now,
        "source": "website",
    }

    _write_session_file(session)
    return {"ok": True, "session": session}


@router.get("/my-sessions/{session_id}")
def get_my_session(session_id: str, request: Request):
    data = _get_session_or_404(session_id)
    _check_session_owner_or_admin(request, data)
    return {"ok": True, "session": data}


@router.post("/my-sessions/{session_id}")
def update_my_session(session_id: str, data: SessionUpdate, request: Request):
    existing = _get_session_or_404(session_id)
    payload = _check_session_owner_or_admin(request, existing)
    now = int(time.time())

    history = [m.model_dump() for m in data.history]

    updated = {
        **existing,
        "session_id": existing["session_id"],
        "title": (data.title or existing.get("title") or "New chat").strip() or "New chat",
        "owner_email": existing.get("owner_email") or payload.get("sub"),
        "owner_role": existing.get("owner_role") or payload.get("role"),
        "device_id": data.device_id if data.device_id is not None else existing.get("device_id"),
        "db_name": data.db_name if data.db_name is not None else existing.get("db_name"),
        "history": history,
        "message_count": len(history),
        "created_ts": existing.get("created_ts") or now,
        "updated_ts": now,
        "source": existing.get("source", "website"),
    }

    _write_session_file(updated)
    return {"ok": True, "session": updated}


# ============================================================================
# Jetson/admin session routes
# ============================================================================

@router.post("/sessions/ingest")
def ingest_session(
    data: SessionIngest,
    x_device_secret: Optional[str] = Header(default=None, alias="X-Device-Secret"),
):
    _require_device_secret(x_device_secret)

    existing: Dict[str, Any] = {}
    path = _session_path(data.session_id)
    if path.exists():
        try:
            existing = _read_session_file(path)
        except Exception:
            existing = {}

    owner_email = data.owner_email or existing.get("owner_email")
    owner_role = data.owner_role or existing.get("owner_role")
    title = (data.title or existing.get("title") or "Jetson session").strip()
    db_name = data.db_name or existing.get("db_name")
    created_ts = existing.get("created_ts") or int(time.time())
    updated_ts = data.updated_ts or int(time.time())

    session = {
        "session_id": data.session_id,
        "title": title,
        "owner_email": owner_email,
        "owner_role": owner_role,
        "device_id": data.device_id or existing.get("device_id"),
        "db_name": db_name,
        "history": data.history,
        "message_count": len(data.history),
        "created_ts": created_ts,
        "updated_ts": updated_ts,
        "source": "jetson",
    }

    _write_session_file(session)
    return {"ok": True, "session_id": data.session_id, "messages": len(data.history)}


@router.get("/sessions/list")
def list_sessions(
    request: Request,
    x_device_secret: Optional[str] = Header(default=None, alias="X-Device-Secret"),
):
    device_authed = DEVICE_SECRET and (x_device_secret or "").strip() == DEVICE_SECRET
    if not device_authed:
        payload = require_auth(request)
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin only")

    sessions = [_session_meta(data) for data in _list_all_sessions()]
    return {"ok": True, "sessions": sessions}


@router.get("/sessions/{session_id}")
def get_session(
    session_id: str,
    request: Request,
    x_device_secret: Optional[str] = Header(default=None, alias="X-Device-Secret"),
):
    device_authed = DEVICE_SECRET and (x_device_secret or "").strip() == DEVICE_SECRET
    if not device_authed:
        payload = require_auth(request)
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin only")

    return _get_session_or_404(session_id)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, request: Request):
    require_admin(request)
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    path.unlink()
    return {"ok": True, "deleted": session_id}