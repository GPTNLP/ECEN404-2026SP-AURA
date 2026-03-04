import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

def _default_path() -> Path:
    # Prefer Azure persistent storage if configured
    p = os.getenv("TA_USERS_PATH", "").strip()
    if p:
        return Path(p)

    # Fallback: repo-local (dev only)
    return Path(__file__).resolve().parent / "storage" / "ta_users.json"

TA_USERS_PATH = _default_path()
TA_USERS_PATH.parent.mkdir(parents=True, exist_ok=True)

def _init_if_missing_or_empty() -> None:
    """
    Ensure the TA store is valid JSON.
    - If missing -> create {"tas":[]}
    - If empty (size 0) -> write {"tas":[]}
    """
    if (not TA_USERS_PATH.exists()) or (TA_USERS_PATH.stat().st_size == 0):
        TA_USERS_PATH.write_text(json.dumps({"tas": []}, indent=2) + "\n", encoding="utf-8")

def _read() -> Dict[str, Any]:
    _init_if_missing_or_empty()
    try:
        return json.loads(TA_USERS_PATH.read_text(encoding="utf-8"))
    except Exception:
        # If corrupted, reset safely (you can also choose to raise)
        TA_USERS_PATH.write_text(json.dumps({"tas": []}, indent=2) + "\n", encoding="utf-8")
        return {"tas": []}

def _write(data: Dict[str, Any]) -> None:
    TA_USERS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

def _norm(email: str) -> str:
    return (email or "").strip().lower()

def list_ta_items() -> List[Dict[str, Any]]:
    """
    Returns:
      [{"email": "...", "added_by": "...", "added_ts": 123}, ...]
    Auto-migrates old format:
      {"tas": ["a@tamu.edu", ...]}
    """
    raw = _read()
    tas = raw.get("tas", [])
    if not isinstance(tas, list):
        tas = []

    # migrate old format (list[str]) -> list[dict]
    if tas and isinstance(tas[0], str):
        migrated = []
        for e in tas:
            e2 = _norm(e)
            if e2 and "@" in e2:
                migrated.append({"email": e2, "added_by": "", "added_ts": 0})
        tas = migrated

    out: List[Dict[str, Any]] = []
    seen = set()
    for it in tas:
        if not isinstance(it, dict):
            continue
        email = _norm(it.get("email", ""))
        if not email or "@" not in email:
            continue
        if email in seen:
            continue
        seen.add(email)
        out.append(
            {
                "email": email,
                "added_by": _norm(it.get("added_by", "")),
                "added_ts": int(it.get("added_ts") or 0),
            }
        )

    out.sort(key=lambda x: x["email"])
    _write({"tas": out})  # persist migration + cleanup
    return out

def list_tas() -> List[str]:
    return [x["email"] for x in list_ta_items()]

def is_ta(email: str) -> bool:
    e = _norm(email)
    return e in set(list_tas())

def add_ta(email: str, added_by: str = "") -> None:
    email = _norm(email)
    if not email or "@" not in email:
        return

    items = list_ta_items()
    if any(x["email"] == email for x in items):
        return

    items.append(
        {
            "email": email,
            "added_by": _norm(added_by),
            "added_ts": int(time.time()),
        }
    )
    items.sort(key=lambda x: x["email"])
    _write({"tas": items})

def remove_ta(email: str) -> None:
    email = _norm(email)
    items = [x for x in list_ta_items() if x["email"] != email]
    _write({"tas": items})