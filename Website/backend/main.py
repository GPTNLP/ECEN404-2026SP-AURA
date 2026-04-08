# backend/main.py
import os
import importlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ensure_storage_layout

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

ENV = os.getenv("ENV", "").lower()

if ENV in ("", "dev", "local") and load_dotenv is not None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)

docs_url = None if ENV in ("prod", "production") else "/docs"
redoc_url = None if ENV in ("prod", "production") else "/redoc"

app = FastAPI(
    title="AURA Backend",
    version="0.1.0",
    docs_url=docs_url,
    redoc_url=redoc_url,
)

allowed = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if not allowed:
    allowed = ["http://127.0.0.1:5173", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "env": ENV}


def include_router_safely(module_name: str, label: str):
    """
    Imports a module and includes its `router` if present.
    Never blocks app startup if a router fails to import.
    """
    try:
        mod = importlib.import_module(module_name)
        router = getattr(mod, "router", None)
        if router is None:
            raise RuntimeError(f"{module_name} has no attribute 'router'")
        app.include_router(router)
        print(f"Loaded router: {label} ({module_name})")
    except Exception as e:
        print(f"Router not loaded ({label} / {module_name}): {e}")


def m(name: str) -> str:
    if os.getenv("AURA_USE_BACKEND_PACKAGE", "0") == "1":
        return f"backend.{name}"
    return name


include_router_safely(m("auth_me_api"), "auth_me_api")
include_router_safely(m("admin_auth_api"), "admin_auth_api")
include_router_safely(m("student_auth_api"), "student_auth_api")
include_router_safely(m("ta_auth_api"), "ta_auth_api")
include_router_safely(m("ta_admin_api"), "ta_admin_api")
include_router_safely(m("database_api"), "database_api")
include_router_safely(m("logs_api"), "logs_api")
include_router_safely(m("device_api"), "device_api")
include_router_safely(m("device_commands_api"), "device_commands_api")
include_router_safely(m("camera_bridge_api"), "camera_bridge_api")


@app.on_event("startup")
async def _startup():
    ensure_storage_layout()
    print("Backend startup complete")