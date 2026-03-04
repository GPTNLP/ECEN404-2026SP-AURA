# backend/admin_auth_api.py
import os
import time
import json
import random
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, Request, HTTPException, Response
from pydantic import BaseModel

from security import require_ip_allowlist
from security_tokens import mint_app_token
from hash_passwords import verify_password as verify_pbkdf2_password
from otp_store import OTPStore, hash_code

from config import ADMIN_USERS_PATH, ensure_storage_layout  # ✅ persistent path + auto-create

# Load .env ONLY for local dev; Azure uses App Settings (env vars).
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

env = (os.getenv("ENV", "") or "").lower()
if env in ("", "dev", "local") and load_dotenv is not None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)

# SMTP settings
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)

# Admin settings
ADMIN_OTP_TTL_SECONDS = int(os.getenv("ADMIN_OTP_TTL_SECONDS", "300"))  # 5 min
ADMIN_MAX_OTP_ATTEMPTS = int(os.getenv("ADMIN_MAX_OTP_ATTEMPTS", "5"))

# Simple IP rate limit (per window)
ADMIN_LOGIN_RATE_WINDOW = int(os.getenv("ADMIN_LOGIN_RATE_WINDOW", "300"))
ADMIN_LOGIN_RATE_MAX = int(os.getenv("ADMIN_LOGIN_RATE_MAX", "10"))

# Cookie settings
COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "aura_token")
COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax")  # lax/strict/none
COOKIE_DOMAIN = os.getenv("AUTH_COOKIE_DOMAIN", "")         # optional

router = APIRouter(prefix="/auth/admin", tags=["admin-auth"])
otp_store = OTPStore(prefix="adminotp")

class AdminLoginRequest(BaseModel):
    email: str
    password: str

class AdminVerifyRequest(BaseModel):
    email: str
    otp: str

# --- Simple in-memory rate store ---
_RATE: Dict[str, Dict[str, Any]] = {}

def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def _rate_limit_or_429(ip: str):
    now = int(time.time())
    rec = _RATE.get(ip)
    if not rec or now - rec["start"] >= ADMIN_LOGIN_RATE_WINDOW:
        _RATE[ip] = {"start": now, "count": 1}
        return
    rec["count"] += 1
    if rec["count"] > ADMIN_LOGIN_RATE_MAX:
        raise HTTPException(status_code=429, detail="Too many requests")

def _send_otp_email(to_email: str, code: str):
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
        raise HTTPException(status_code=500, detail="SMTP not configured")

    msg = EmailMessage()
    msg["Subject"] = "AURA Admin Login Code"
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg.set_content(
        f"Your AURA admin login code is: {code}\n\n"
        f"This code expires in {max(1, ADMIN_OTP_TTL_SECONDS//60)} minutes.\n"
        f"If you did not request this, ignore this email."
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)

def _load_admins() -> Dict[str, str]:
    """
    Loads admins from persistent storage:
      ADMIN_USERS_PATH (usually /home/site/storage/admin_users.json on Azure)
    Expected format:
      {"admins": [{"email": "...", "password_hash": "..."}]}
    """
    ensure_storage_layout()

    # Auto-create if missing
    if not ADMIN_USERS_PATH.exists():
        ADMIN_USERS_PATH.write_text('{"admins":[]}\n', encoding="utf-8")

    try:
        with open(ADMIN_USERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Admin store unreadable: {e}")

    admins_list = data.get("admins", [])
    if not isinstance(admins_list, list):
        raise HTTPException(status_code=500, detail="Admin store invalid format (admins must be a list)")

    out: Dict[str, str] = {}
    for a in admins_list:
        if not isinstance(a, dict):
            continue
        email = (a.get("email") or "").strip().lower()
        ph = (a.get("password_hash") or "").strip()
        if email and ph:
            out[email] = ph
    return out

def _should_secure_cookie(request: Request) -> bool:
    """
    Secure cookies only work on HTTPS.
    - Local dev (http://127.0.0.1) -> False
    - Azure/real domain (https)     -> True
    """
    env = (os.getenv("ENV", "") or "").lower()
    if env in ("prod", "production"):
        return True
    return request.url.scheme == "https"

@router.post("/login")
async def login(data: AdminLoginRequest, request: Request):
    # Keep if you want; set ALLOWED_IPS empty in config to allow all
    require_ip_allowlist(request)

    ip = _client_ip(request)
    _rate_limit_or_429(ip)

    email = (data.email or "").strip().lower()
    password = (data.password or "").strip()

    invalid = HTTPException(status_code=401, detail="Invalid credentials")

    if not email or "@" not in email or not password:
        raise invalid

    admins = _load_admins()
    stored_hash = admins.get(email)
    if not stored_hash:
        raise invalid

    if not verify_pbkdf2_password(password, stored_hash):
        raise invalid

    code = f"{random.randint(100000, 999999)}"
    otp_store.set(email=email, code=code, ttl_seconds=ADMIN_OTP_TTL_SECONDS)
    _send_otp_email(email, code)

    return {"message": "OTP sent", "otp_expires_in": ADMIN_OTP_TTL_SECONDS}

@router.post("/verify")
async def verify(data: AdminVerifyRequest, request: Request, response: Response):
    require_ip_allowlist(request)

    email = (data.email or "").strip().lower()
    otp = (data.otp or "").strip()

    invalid = HTTPException(status_code=401, detail="Invalid code")

    rec = otp_store.get(email)
    if not rec:
        raise invalid

    if int(rec.get("expires", 0)) and time.time() > int(rec["expires"]):
        otp_store.delete(email)
        raise invalid

    attempts = otp_store.incr_attempts(email)
    if attempts > ADMIN_MAX_OTP_ATTEMPTS:
        otp_store.delete(email)
        raise HTTPException(status_code=429, detail="Too many attempts")

    if hash_code(otp) != rec.get("otp_hash"):
        raise invalid

    otp_store.delete(email)

    result = mint_app_token(email=email, role="admin")
    token = result["token"]

    secure_cookie = _should_secure_cookie(request)

    cookie_kwargs: Dict[str, Any] = {}
    if COOKIE_DOMAIN.strip():
        cookie_kwargs["domain"] = COOKIE_DOMAIN.strip()

    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=secure_cookie,
        samesite=COOKIE_SAMESITE,
        max_age=result["expires_in"],
        **cookie_kwargs,
    )

    # ✅ IMPORTANT: also return token so your frontend session works immediately
    return {"token": token, "user": result["user"], "expires_in": result["expires_in"]}

@router.get("/me")
def me(request: Request):
    require_ip_allowlist(request)
    from security import require_auth
    payload = require_auth(request)
    return {
        "user": {"email": payload.get("sub"), "role": payload.get("role")},
        "exp": payload.get("exp"),
    }

@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}