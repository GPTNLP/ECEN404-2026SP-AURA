import json
import time
from typing import Any, Dict, Optional

from JetsonLocal.agent.core.config import AGENT_LOG_FILE, DEVICE_ID


def _now() -> int:
    return int(time.time())


def write_local_log(level: str, event: str, message: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    entry = {
        "ts": _now(),
        "device_id": DEVICE_ID,
        "level": (level or "info").lower(),
        "event": event,
        "message": message,
        "meta": meta or {},
    }
    with AGENT_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry