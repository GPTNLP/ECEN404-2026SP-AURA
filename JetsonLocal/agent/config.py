import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
JETSONLOCAL_DIR = BASE_DIR.parent
STORAGE_DIR = JETSONLOCAL_DIR / "storage"
LOGS_DIR = STORAGE_DIR / "logs"
QUEUE_DIR = STORAGE_DIR / "queue"
STATE_DIR = STORAGE_DIR / "state"
STATIC_DIR = JETSONLOCAL_DIR / "static"

ENV_PATH = JETSONLOCAL_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

API_BASE_URL = os.getenv("AZURE_BACKEND_URL", "").rstrip("/")
DEVICE_ID = os.getenv("DEVICE_ID", "jetson-001").strip()
DEVICE_NAME = os.getenv("DEVICE_NAME", "AURA Jetson").strip()
DEVICE_TYPE = os.getenv("DEVICE_TYPE", "jetson").strip()
DEVICE_SHARED_SECRET = os.getenv("DEVICE_SHARED_SECRET", "").strip()
DEVICE_SOFTWARE_VERSION = os.getenv("DEVICE_SOFTWARE_VERSION", "0.1.0").strip()

OLLAMA_READY_DEFAULT = os.getenv("OLLAMA_READY_DEFAULT", "true").lower() == "true"
VECTOR_DB_READY_DEFAULT = os.getenv("VECTOR_DB_READY_DEFAULT", "true").lower() == "true"
CAMERA_READY_DEFAULT = os.getenv("CAMERA_READY_DEFAULT", "true").lower() == "true"

HEARTBEAT_SECONDS = int(os.getenv("DEVICE_HEARTBEAT_SECONDS", "30"))
STATUS_SECONDS = int(os.getenv("DEVICE_STATUS_SECONDS", "20"))
CONFIG_REFRESH_SECONDS = int(os.getenv("DEVICE_CONFIG_REFRESH_SECONDS", "120"))
OFFLINE_RETRY_SECONDS = int(os.getenv("DEVICE_OFFLINE_RETRY_SECONDS", "10"))

SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/ttyUSB0")
INPUT_MODE = os.getenv("INPUT_MODE", "keyboard").strip().lower()

# local Jetson demo / edge AI config
DEFAULT_MODEL = os.getenv("AURA_LLM_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("AURA_EMBED_MODEL", "nomic-embed-text")
LOCAL_DB_NAME = os.getenv("LOCAL_DB_NAME", "jetson_local_db")

for p in [STORAGE_DIR, LOGS_DIR, QUEUE_DIR, STATE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

AGENT_LOG_FILE = LOGS_DIR / "agent.jsonl"
PENDING_LOGS_FILE = QUEUE_DIR / "pending_logs.jsonl"
PENDING_STATUS_FILE = QUEUE_DIR / "pending_status.jsonl"
RUNTIME_FILE = STATE_DIR / "runtime.json"