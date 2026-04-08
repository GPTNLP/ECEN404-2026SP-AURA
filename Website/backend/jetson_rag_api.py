import os
from typing import Any, Dict, Optional

import requests
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from security import require_auth, require_ip_allowlist

router = APIRouter(tags=["jetson_rag"])

JETSON_API_BASE = (
    os.getenv("AURA_JETSON_API_BASE", "").strip()
    or os.getenv("JETSON_API_BASE", "").strip()
)

JETSON_TIMEOUT_S = float(os.getenv("AURA_JETSON_PROXY_TIMEOUT_S", "180"))


def require_any_user(request: Request) -> Dict[str, Any]:
    require_ip_allowlist(request)
    return require_auth(request)


def _base() -> str:
    base = (JETSON_API_BASE or "").rstrip("/")
    if not base:
        raise HTTPException(
            status_code=500,
            detail="AURA_JETSON_API_BASE is not configured on the backend",
        )
    return base


def _read_json_or_text(resp: requests.Response) -> Dict[str, Any]:
    try:
        data = resp.json()
        if isinstance(data, dict):
            return data
        return {"data": data}
    except Exception:
        return {"message": resp.text or f"HTTP {resp.status_code}"}


class JetsonChatRequest(BaseModel):
    db_name: str
    query: str
    session_id: Optional[str] = None


class JetsonLoadDbRequest(BaseModel):
    db_name: str


@router.get("/api/jetson/health")
def jetson_health(request: Request):
    require_any_user(request)

    try:
        resp = requests.get(
            f"{_base()}/health",
            timeout=min(JETSON_TIMEOUT_S, 15),
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Jetson health request failed: {e}")

    data = _read_json_or_text(resp)
    if not resp.ok:
        raise HTTPException(
            status_code=502,
            detail=data.get("detail") or data.get("message") or "Jetson health failed",
        )

    return data


@router.post("/api/jetson/load_db")
def jetson_load_db(req: JetsonLoadDbRequest, request: Request):
    require_any_user(request)

    if not req.db_name.strip():
        raise HTTPException(status_code=400, detail="db_name is required")

    try:
        resp = requests.post(
            f"{_base()}/rag/load_db",
            json={"db_name": req.db_name},
            timeout=JETSON_TIMEOUT_S,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Jetson load_db request failed: {e}")

    data = _read_json_or_text(resp)
    if not resp.ok:
        raise HTTPException(
            status_code=502,
            detail=data.get("detail") or data.get("message") or "Jetson load_db failed",
        )

    return data


@router.post("/api/jetson/chat")
def jetson_chat(req: JetsonChatRequest, request: Request):
    user = require_any_user(request)

    db_name = (req.db_name or "").strip()
    query = (req.query or "").strip()

    if not db_name:
        raise HTTPException(status_code=400, detail="db_name is required")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    session_id = (
        req.session_id
        or f"{str(user.get('sub') or 'anon').strip().lower()}::{db_name}"
    )

    # Step 1: make sure the requested DB is active on the Jetson
    try:
        load_resp = requests.post(
            f"{_base()}/rag/load_db",
            json={"db_name": db_name},
            timeout=JETSON_TIMEOUT_S,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Jetson load_db request failed: {e}")

    load_data = _read_json_or_text(load_resp)
    if not load_resp.ok:
        raise HTTPException(
            status_code=502,
            detail=load_data.get("detail") or load_data.get("message") or "Jetson load_db failed",
        )

    # Step 2: ask the Jetson RAG/LLM
    try:
        chat_resp = requests.post(
            f"{_base()}/rag/chat",
            json={
                "query": query,
                "session_id": session_id,
            },
            timeout=JETSON_TIMEOUT_S,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Jetson chat request failed: {e}")

    chat_data = _read_json_or_text(chat_resp)
    if not chat_resp.ok:
        raise HTTPException(
            status_code=502,
            detail=chat_data.get("detail") or chat_data.get("message") or "Jetson chat failed",
        )

    answer = str(chat_data.get("answer") or "").strip()
    if not answer:
        answer = "(No answer returned from Jetson)"

    return {
        "ok": True,
        "answer": answer,
        "session_id": chat_data.get("session_id") or session_id,
        "db_name": db_name,
        "engine": "jetson",
        "load_result": {
            "db_name": load_data.get("db_name") or db_name,
            "stats": load_data.get("stats") or {},
        },
    }