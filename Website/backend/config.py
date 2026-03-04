"""
Configuration file for the AURA project.

IMPORTANT for Azure App Service:
- Only /home is persistent
- Do NOT rely on os.getcwd() for storage paths
"""

import os
from pathlib import Path

def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default

# -------------------------
# Persistent storage root
# -------------------------
# Azure: set AURA_STORAGE_DIR=/home/site/storage in App Settings
# Local: defaults to "<repo>/storage" next to this backend package
DEFAULT_LOCAL_STORAGE = (Path(__file__).resolve().parent / "storage").resolve()
STORAGE_DIR = Path(_env("AURA_STORAGE_DIR", str(DEFAULT_LOCAL_STORAGE))).expanduser().resolve()

# Keep your old names, but make them stable
BASE_DIR = str(Path(__file__).resolve().parent)  # stable path to backend folder (not cwd)

CHROMA_DIR = Path(_env("AURA_CHROMA_DIR", str(STORAGE_DIR / "chroma"))).resolve()
GRAPH_FILE = Path(_env("AURA_GRAPH_FILE", str(STORAGE_DIR / "knowledge_graph.graphml"))).resolve()
SESSIONS_DIR = Path(_env("AURA_SESSIONS_DIR", str(STORAGE_DIR / "sessions"))).resolve()

# If you want staging in persistent storage on Azure, you can also env-override it.
DOCS_STAGING_DIR = Path(_env("AURA_DOCS_STAGING_DIR", str(Path(BASE_DIR).parent / "source_documents"))).resolve()

# Admin store lives in persistent storage
ADMIN_USERS_PATH = Path(_env("ADMIN_USERS_PATH", str(STORAGE_DIR / "admin_users.json"))).resolve()

# Databases/documents dirs (you already set these in Azure; we default them sensibly)
AURA_DATABASES_DIR = Path(_env("AURA_DATABASES_DIR", str(STORAGE_DIR / "databases"))).resolve()
AURA_DOCUMENTS_DIR = Path(_env("AURA_DOCUMENTS_DIR", str(STORAGE_DIR / "documents"))).resolve()
AURA_SQLITE_PATH = Path(_env("AURA_SQLITE_PATH", str(STORAGE_DIR / "aura.sqlite"))).resolve()

def ensure_storage_layout() -> None:
    """Create required folders/files in persistent storage."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    AURA_DATABASES_DIR.mkdir(parents=True, exist_ok=True)
    AURA_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure admin store exists
    if not ADMIN_USERS_PATH.exists():
        # Your code expects {"admins":[...]} shape
        ADMIN_USERS_PATH.write_text('{"admins":[]}\n', encoding="utf-8")

# -------------------------
# AI Settings
# -------------------------
DEFAULT_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
LIGHTRAG_K = 4

# -------------------------
# Auth / Security Settings
# -------------------------
ALLOWED_IPS = {ip.strip() for ip in os.getenv("ALLOWED_IPS", "").split(",") if ip.strip()}
API_TOKEN = os.getenv("API_TOKEN", "")
AUTH_SECRET = os.getenv("AUTH_SECRET", "")

AUTH_ALLOWED_DOMAINS = {
    d.strip().lower()
    for d in (os.getenv("AUTH_ALLOWED_DOMAINS", "tamu.edu")).split(",")
    if d.strip()
}