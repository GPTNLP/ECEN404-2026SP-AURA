import json
from pathlib import Path
from typing import Any, Dict, List

from JetsonLocal.agent.core.config import PENDING_LOGS_FILE, PENDING_STATUS_FILE


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def queue_log(payload: Dict[str, Any]) -> None:
    _append_jsonl(PENDING_LOGS_FILE, payload)


def queue_status(payload: Dict[str, Any]) -> None:
    _append_jsonl(PENDING_STATUS_FILE, payload)


def flush_logs(send_fn) -> int:
    rows = _read_jsonl(PENDING_LOGS_FILE)
    if not rows:
        return 0

    remaining = []
    sent = 0
    for row in rows:
        try:
            send_fn(row)
            sent += 1
        except Exception:
            remaining.append(row)

    _write_jsonl(PENDING_LOGS_FILE, remaining)
    return sent


def flush_statuses(send_fn) -> int:
    rows = _read_jsonl(PENDING_STATUS_FILE)
    if not rows:
        return 0

    remaining = []
    sent = 0
    for row in rows:
        try:
            send_fn(row)
            sent += 1
        except Exception:
            remaining.append(row)

    _write_jsonl(PENDING_STATUS_FILE, remaining)
    return sent