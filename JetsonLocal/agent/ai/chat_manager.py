import json
import os
import time
import asyncio
from core.config import STORAGE_DIR, DEVICE_ID


class ChatSessionManager:
    def __init__(self):
        self.session_dir = os.path.join(str(STORAGE_DIR), "sessions")
        os.makedirs(self.session_dir, exist_ok=True)
        self.active_session_id = "default"
        self.history = []
        self._load_local()

    def set_session(self, session_id: str, remote_history: list = None):
        """Switch to a different chat session (e.g. loaded from website)."""
        self.active_session_id = session_id
        if remote_history is not None:
            self.history = remote_history
            self._save_local()
        else:
            self._load_local()

    def add_message(self, role: str, text: str, api_client=None):
        """Adds a message, saves locally, and attempts background cloud sync."""
        self.history.append({
            "role": role,
            "content": text,
            "ts": int(time.time()),
        })
        self._save_local()

        if api_client is not None:
            # Fire-and-forget sync (won't block UI if offline)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    asyncio.to_thread(
                        api_client.sync_chat_session,
                        self.active_session_id,
                        self._session_payload(),
                    )
                )
            except RuntimeError:
                # No running loop (called from sync context) — skip background sync
                pass

    def get_history(self) -> list:
        return list(self.history)

    def clear(self):
        """Clear the current session history."""
        self.history = []
        self._save_local()

    def list_local_sessions(self) -> list:
        """Returns a list of locally stored session IDs."""
        sessions = []
        for fn in os.listdir(self.session_dir):
            if fn.endswith(".json"):
                sessions.append(fn[:-5])
        return sessions

    def _session_payload(self) -> dict:
        return {
            "session_id": self.active_session_id,
            "device_id": DEVICE_ID,
            "history": self.history,
            "updated_ts": int(time.time()),
        }

    def _get_filepath(self):
        return os.path.join(self.session_dir, f"{self.active_session_id}.json")

    def _save_local(self):
        with open(self._get_filepath(), "w") as f:
            json.dump(self._session_payload(), f, indent=2)

    def _load_local(self):
        path = self._get_filepath()
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                self.history = data.get("history", [])
        else:
            self.history = []


chat_manager = ChatSessionManager()
