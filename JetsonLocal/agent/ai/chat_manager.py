import json
import os
import asyncio
from core.config import STORAGE_DIR

class ChatSessionManager:
    def __init__(self):
        self.session_dir = os.path.join(STORAGE_DIR, "sessions")
        os.makedirs(self.session_dir, exist_ok=True)
        self.active_session_id = "default"
        self.history = []
        self._load_local()

    def set_session(self, session_id: str, remote_history: list = None):
        """Switch to a different chat session (from UI)."""
        self.active_session_id = session_id
        if remote_history is not None:
            self.history = remote_history
            self._save_local()
        else:
            self._load_local()

    def add_message(self, role: str, text: str, api_client, device_id: str):
        """Adds a message, saves locally, and attempts background cloud sync."""
        self.history.append({"role": role, "content": text})
        self._save_local()
        
        # Fire-and-forget sync (won't block UI if offline)
        asyncio.create_task(
            asyncio.to_thread(api_client.sync_chat_log, device_id, self.active_session_id, {"history": self.history})
        )

    def _get_filepath(self):
        return os.path.join(self.session_dir, f"{self.active_session_id}.json")

    def _save_local(self):
        with open(self._get_filepath(), "w") as f:
            json.dump({"session_id": self.active_session_id, "history": self.history}, f)

    def _load_local(self):
        path = self._get_filepath()
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                self.history = data.get("history", [])
        else:
            self.history = []

chat_manager = ChatSessionManager()