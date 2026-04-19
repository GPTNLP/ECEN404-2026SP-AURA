# backend/security.py
import os
import json
from typing import Dict, Any
from fastapi import Request, HTTPException

from config import ALLOWED_IPS, API_TOKEN, AUTH_ALLOWED_DOMAINS, ADMIN_USERS_PATH
from security_tokens import verify_token
from ta_store import is_ta


# ---------------------------
# Client IP (Azure-friendly)
# ---------------------------
def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def require_ip_allowlist(request: Request):
    if not ALLOWED_IPS:
        return
    ip = get_client_ip(request)
    if ip not in ALLOWED_IPS:
        raise HTTPException(status_code=403, detail="IP not allowed")


# ---------------------------
# Camera token (header OR ?token=)
# ---------------------------
def require_camera_token(request: Request):
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="Server missing API_TOKEN")

    auth = request.headers.get("authorization", "")
    if auth == f"Bearer {API_TOKEN}":
        return

    token_q = request.query_params.get("token", "")
    if token_q == API_TOKEN:
        return

    raise HTTPException(status_code=401, detail="Invalid camera token")


# ---------------------------
# Auth token (cookie preferred, Bearer supported)
# ---------------------------
def _get_auth_token_from_request(request: Request) -> str:
    cookie_name = os.getenv("AUTH_COOKIE_NAME", "aura_token")
    cookie_token = request.cookies.get(cookie_name)
    if cookie_token:
        return cookie_token

    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth.replace("Bearer ", "", 1).strip()

    return ""


def _load_admin_emails() -> set[str]:
    try:
        if not ADMIN_USERS_PATH.exists():
            return set()

        raw = ADMIN_USERS_PATH.read_text(encoding="utf-8").strip()
        if not raw:
            return set()

        data = json.loads(raw)
        admins = data.get("admins", [])
        if not isinstance(admins, list):
            return set()

        out: set[str] = set()
        for item in admins:
            if not isinstance(item, dict):
                continue
            email = (item.get("email") or "").strip().lower()
            if email:
                out.add(email)
        return out
    except Exception:
        return set()


def resolve_current_role(email: str) -> str:
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        return "student"

    if email in _load_admin_emails():
        return "admin"

    try:
        if is_ta(email):
            return "ta"
    except Exception:
        pass

    return "student"


def _normalize_token_role(role: str) -> str:
    role = (role or "").strip().lower()
    if role in {"admin", "ta", "student"}:
        return role
    return "student"


def require_auth(request: Request) -> Dict[str, Any]:
    token = _get_auth_token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token")

    try:
        payload = verify_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    email = (payload.get("sub") or "").strip().lower()
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")

    payload["sub"] = email

    # Keep the role embedded in the signed token stable for the lifetime of that
    # session. Role changes should happen through token revocation + new login,
    # not by silently downgrading users on /auth/me when a backing file is briefly
    # stale, moved, or unreadable.
    payload["role"] = _normalize_token_role(payload.get("role") or "student")
    return payload


def require_role(request: Request, *allowed_roles: str) -> Dict[str, Any]:
    payload = require_auth(request)
    role = (payload.get("role") or "").strip().lower()
    allowed = {r.strip().lower() for r in allowed_roles if r.strip()}

    if allowed and role not in allowed:
        raise HTTPException(status_code=403, detail="Forbidden")

    return payload


# ---------------------------
# Email domain restriction
# ---------------------------
def domain_allowed(email: str) -> bool:
    email = (email or "").strip().lower()
    if "@" not in email:
        return False

    domain = email.split("@", 1)[1].strip().lower()
    allowed = set((AUTH_ALLOWED_DOMAINS or []))
    if not allowed:
        allowed = {"tamu.edu"}

    return domain in allowed
