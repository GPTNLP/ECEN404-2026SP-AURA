import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

from config import STORAGE_DIR


def _default_path() -> Path:
    explicit = os.getenv("TA_USERS_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    return (STORAGE_DIR / "ta_users.json").resolve()


TA_USERS_PATH = _default_path()
TA_USERS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _empty_store() -> Dict[str, Any]:
    return {"tas": []}


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _init_if_missing_or_empty() -> None:
    if (not TA_USERS_PATH.exists()) or TA_USERS_PATH.stat().st_size == 0:
        _write_json_atomic(TA_USERS_PATH, _empty_store())


def _read() -> Dict[str, Any]:
    _init_if_missing_or_empty()

    raw = TA_USERS_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return _empty_store()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"TA store is unreadable: {TA_USERS_PATH} ({exc})") from exc

    if not isinstance(data, dict):
        return _empty_store()

    tas = data.get("tas")
    if not isinstance(tas, list):
        data["tas"] = []

    return data


def _write(data: Dict[str, Any]) -> None:
    _write_json_atomic(TA_USERS_PATH, data)


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

    migrated_from_legacy = False
    if tas and isinstance(tas[0], str):
        migrated: List[Dict[str, Any]] = []
        for value in tas:
            email = _norm(value)
            if email and "@" in email:
                migrated.append({"email": email, "added_by": "", "added_ts": 0})
        tas = migrated
        migrated_from_legacy = True

    out: List[Dict[str, Any]] = []
    seen = set()
    for item in tas:
        if not isinstance(item, dict):
            continue

        email = _norm(item.get("email", ""))
        if not email or "@" not in email or email in seen:
            continue

        seen.add(email)
        out.append(
            {
                "email": email,
                "added_by": _norm(item.get("added_by", "")),
                "added_ts": int(item.get("added_ts") or 0),
            }
        )

    out.sort(key=lambda x: x["email"])

    normalized = {"tas": out}
    if migrated_from_legacy or raw.get("tas") != out:
        _write(normalized)

    return out


def list_tas() -> List[str]:
    return [item["email"] for item in list_ta_items()]


def is_ta(email: str) -> bool:
    normalized = _norm(email)
    if not normalized:
        return False
    return normalized in set(list_tas())


def add_ta(email: str, added_by: str = "") -> None:
    email = _norm(email)
    if not email or "@" not in email:
        return

    items = list_ta_items()
    if any(item["email"] == email for item in items):
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
    items = [item for item in list_ta_items() if item["email"] != email]
    _write({"tas": items})
