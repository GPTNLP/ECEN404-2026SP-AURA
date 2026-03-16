# backend/main.py
import os
import importlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ensure_storage_layout  # ✅ ensure persistent dirs/files exist at boot

# Load .env ONLY for local dev; Azure uses App Settings (env vars).
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

ENV = os.getenv("ENV", "").lower()

# ---- Optional local .env loading ----
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

# ---- CORS ----
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
        print(f"✅ Loaded router: {label} ({module_name})")
    except Exception as e:
        print(f"⚠️ Router not loaded ({label} / {module_name}): {e}")

def m(name: str) -> str:
    # If you use backend/ as a package, set AURA_USE_BACKEND_PACKAGE=1 in Azure.
    if os.getenv("AURA_USE_BACKEND_PACKAGE", "0") == "1":
        return f"backend.{name}"
    return name

# ---- Always-safe routers (no ML deps) ----
include_router_safely(m("auth_me_api"), "auth_me_api")
include_router_safely(m("admin_auth_api"), "admin_auth_api")
include_router_safely(m("student_auth_api"), "student_auth_api")
include_router_safely(m("ta_auth_api"), "ta_auth_api")
include_router_safely(m("ta_admin_api"), "ta_admin_api")
include_router_safely(m("database_api"), "database_api")
include_router_safely(m("logs_api"), "logs_api")
include_router_safely(m("device_api"), "device_api")

# ---- Optional / Heavy routers behind feature flags ----
if os.getenv("ENABLE_CAMERA", "0") == "1":
    include_router_safely(m("camera_api"), "camera_api")

if os.getenv("ENABLE_DETECT", "0") == "1":
    include_router_safely(m("detect_api"), "detect_api")

if os.getenv("ENABLE_TTS", "0") == "1":
    include_router_safely(m("tts_api"), "tts_api")

if os.getenv("ENABLE_STT", "0") == "1":
    include_router_safely(m("stt_api"), "stt_api")

@app.on_event("startup")
async def _startup():
    # ✅ Ensure /home/site/storage layout exists (admin store, db dirs, etc.)
    ensure_storage_layout()

    # ---- Ollama warmup OFF by default ----
    if os.getenv("ENABLE_OLLAMA_WARMUP", "0") != "1":
        print("ℹ️ Ollama warmup disabled")
        return

    try:
        from lightrag_local import OllamaClient

        ollama_url = os.getenv("AURA_OLLAMA_URL", "")
        if not ollama_url:
            print("⚠️ Ollama warmup enabled but AURA_OLLAMA_URL is empty; skipping")
            return

        llm = os.getenv("AURA_LLM_MODEL", "llama3.2:3b")
        emb = os.getenv("AURA_EMBED_MODEL", "nomic-embed-text")
        timeout_s = float(os.getenv("AURA_OLLAMA_TIMEOUT_S", "10"))

        client = OllamaClient(base_url=ollama_url, embed_model=emb, llm_model=llm)
        await client.embed("warmup")
        await client.generate(prompt="Say 'ready'.", system="", timeout_s=timeout_s)
        print("✅ Ollama warmup complete")
    except Exception as e:
        print(f"⚠️ Ollama warmup skipped: {e}")