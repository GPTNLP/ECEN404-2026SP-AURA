import os
import requests
from typing import Any, Dict

from core.config import API_BASE_URL, DEVICE_ID, DEVICE_SHARED_SECRET


class ApiClient:
    def __init__(self):
        self.base_url = API_BASE_URL.rstrip("/")
        self.timeout = 15
        self.camera_timeout = 4
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Document helpers
    # ------------------------------------------------------------------

    def download_document(self, path: str, dest_path: str):
        r = self.session.get(
            self._url("/api/documents/download"),
            params={"path": path},
            headers=self._headers(),
            stream=True,
            timeout=self.timeout,
        )
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    def list_documents(self) -> Dict[str, Any]:
        """Returns the document tree from the website backend."""
        r = self.session.get(
            self._url("/api/documents/tree"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Vector DB sync
    # ------------------------------------------------------------------

    def upload_vector_db(self, db_name: str, db_dir: str):
        """Uploads individual LightRAG files to the website backend."""
        url = f"{self.base_url}/api/databases/{db_name}/sync_up"
        headers = {"X-Device-Secret": DEVICE_SHARED_SECRET}

        files_to_send = []
        file_handles = []
        allowed_files = ["faiss.index", "embeddings.npy", "meta.json", "db.json", "entities.json"]

        for fn in allowed_files:
            path = os.path.join(db_dir, fn)
            if os.path.exists(path):
                f = open(path, "rb")
                file_handles.append(f)
                files_to_send.append(("files", (fn, f, "application/octet-stream")))

        if not files_to_send:
            return {"ok": False, "error": "No vector files found to upload"}

        try:
            response = self.session.post(url, headers=headers, files=files_to_send, timeout=120.0)
            response.raise_for_status()
            return response.json()
        finally:
            for f in file_handles:
                f.close()

    def download_vector_db(self, db_name: str, dest_dir: str):
        """Downloads individual LightRAG files from the website backend."""
        headers = {"X-Device-Secret": DEVICE_SHARED_SECRET}
        allowed_files = ["faiss.index", "embeddings.npy", "meta.json", "db.json", "entities.json"]
        os.makedirs(dest_dir, exist_ok=True)

        downloaded = []
        for fn in allowed_files:
            url = f"{self.base_url}/api/databases/{db_name}/sync_down/{fn}"
            response = self.session.get(url, headers=headers, stream=True, timeout=120.0)
            if response.status_code == 200:
                with open(os.path.join(dest_dir, fn), "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded.append(fn)
        return downloaded

    def list_databases(self) -> Dict[str, Any]:
        """Returns the list of vector databases available on the website."""
        r = self.session.get(
            self._url("/api/databases"),
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Chat session sync
    # ------------------------------------------------------------------

    def sync_chat_session(self, session_id: str, session_data: dict):
        """Pushes a full conversation session JSON to the website."""
        url = f"{self.base_url}/logs/sessions/ingest"
        headers = {
            "X-Device-Secret": DEVICE_SHARED_SECRET,
            "Content-Type": "application/json",
        }
        try:
            self.session.post(
                url,
                headers=headers,
                json={"session_id": session_id, "device_id": DEVICE_ID, **session_data},
                timeout=8.0,
            )
        except Exception:
            pass  # Fails silently if offline; relies on local JSON

    def get_chat_session(self, session_id: str) -> Dict[str, Any]:
        """Pulls a conversation session from the website."""
        r = self.session.get(
            self._url(f"/logs/sessions/{session_id}"),
            headers={"X-Device-Secret": DEVICE_SHARED_SECRET},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def list_chat_sessions(self) -> Dict[str, Any]:
        """Lists all chat sessions stored on the website."""
        r = self.session.get(
            self._url("/logs/sessions/list"),
            headers={"X-Device-Secret": DEVICE_SHARED_SECRET},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def upload_camera_frame(self, device_id: str, mode: str, jpeg_bytes: bytes) -> Dict[str, Any]:
        r = self.session.post(
            self._url("/device/camera/frame"),
            params={"device_id": device_id, "mode": mode},
            data=jpeg_bytes,
            headers={
                "Content-Type": "image/jpeg",
                "X-Device-Secret": DEVICE_SHARED_SECRET,
                "Connection": "keep-alive",
            },
            timeout=self.camera_timeout,
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Device telemetry
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Device-Secret": DEVICE_SHARED_SECRET,
        }

    def _url(self, path: str) -> str:
        if not self.base_url:
            raise RuntimeError("AZURE_BACKEND_URL is not set")
        return f"{self.base_url}{path}"

    def health(self) -> Dict[str, Any]:
        r = self.session.get(self._url("/health"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def register(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self.session.post(
            self._url("/device/register"),
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def heartbeat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self.session.post(
            self._url("/device/heartbeat"),
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def status(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self.session.post(
            self._url("/device/status"),
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def log(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self.session.post(
            self._url("/device/logs"),
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def get_config(self, device_id: str) -> Dict[str, Any]:
        r = self.session.get(
            self._url("/device/config"),
            params={"device_id": device_id},
            headers={"X-Device-Secret": DEVICE_SHARED_SECRET},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def get_next_command(self, device_id: str) -> Dict[str, Any]:
        r = self.session.get(
            self._url("/device/command/next"),
            params={"device_id": device_id},
            headers={"X-Device-Secret": DEVICE_SHARED_SECRET},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def ack_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self.session.post(
            self._url("/device/command/ack"),
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
