import os
from typing import Any, Dict, Optional

import requests

from core.config import API_BASE_URL, DEVICE_SHARED_SECRET


class ApiClient:
    def __init__(self):
        self.base_url = API_BASE_URL.rstrip("/")
        self.secret = DEVICE_SHARED_SECRET
        self.timeout = 15
        self.camera_timeout = 4
        self.session = requests.Session()

    def _headers(self, content_type: Optional[str] = "application/json") -> Dict[str, str]:
        headers: Dict[str, str] = {
            "X-Device-Secret": self.secret,
        }
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _url(self, path: str) -> str:
        if not self.base_url:
            raise RuntimeError("AZURE_BACKEND_URL / API_BASE_URL is not set")
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
            headers={"X-Device-Secret": self.secret},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def get_next_command(self, device_id: str) -> Dict[str, Any]:
        r = self.session.get(
            self._url("/device/command/next"),
            params={"device_id": device_id},
            headers={"X-Device-Secret": self.secret},
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

    def download_document(self, path: str, dest_path: str) -> None:
        r = self.session.get(
            self._url("/api/documents/download"),
            params={"path": path},
            headers={"X-Device-Secret": self.secret},
            stream=True,
            timeout=120.0,
        )
        r.raise_for_status()

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def upload_vector_db(self, db_name: str, db_dir: str) -> Dict[str, Any]:
        url = self._url(f"/api/databases/{db_name}/sync_up")

        allowed_files = [
            "faiss.index",
            "embeddings.npy",
            "meta.json",
            "db.json",
            "entities.json",
            "graph.json",
            "entity_list.json",
            "entity_emb.npy",
            "entity_faiss.index",
        ]

        files_to_send = []
        file_handles = []

        try:
            for filename in allowed_files:
                full_path = os.path.join(db_dir, filename)
                if not os.path.exists(full_path):
                    continue

                handle = open(full_path, "rb")
                file_handles.append(handle)
                files_to_send.append(
                    ("files", (filename, handle, "application/octet-stream"))
                )

            if not files_to_send:
                raise RuntimeError(f"No vector DB files found to upload in: {db_dir}")

            response = self.session.post(
                url,
                headers={"X-Device-Secret": self.secret},
                files=files_to_send,
                timeout=300.0,
            )
            response.raise_for_status()
            return response.json()
        finally:
            for handle in file_handles:
                try:
                    handle.close()
                except Exception:
                    pass

    def download_vector_db(self, db_name: str, dest_dir: str) -> None:
        allowed_files = [
            "faiss.index",
            "embeddings.npy",
            "meta.json",
            "db.json",
            "entities.json",
            "graph.json",
            "entity_list.json",
            "entity_emb.npy",
            "entity_faiss.index",
        ]

        os.makedirs(dest_dir, exist_ok=True)

        for filename in allowed_files:
            url = self._url(f"/api/databases/{db_name}/sync_down/{filename}")
            response = self.session.get(
                url,
                headers={"X-Device-Secret": self.secret},
                stream=True,
                timeout=120.0,
            )

            if response.status_code == 404:
                continue

            response.raise_for_status()

            out_path = os.path.join(dest_dir, filename)
            with open(out_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    def get_next_rag_build_job(self, device_id: str) -> Dict[str, Any]:
        r = self.session.get(
            self._url("/api/databases/build_jobs/next"),
            params={"device_id": device_id},
            headers={"X-Device-Secret": self.secret},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()

    def ack_rag_build_job(
        self,
        job_id: str,
        device_id: str,
        status: str,
        note: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "job_id": job_id,
            "device_id": device_id,
            "status": status,
            "note": note,
            "extra": extra or {},
        }

        r = self.session.post(
            self._url("/api/databases/build_jobs/ack"),
            json=payload,
            headers=self._headers(),
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()

    def sync_chat_log(self, device_id: str, log_data: dict) -> Dict[str, Any]:
        """
        Sync full chat session history to the backend session store.
        This is best-effort, but now targets the correct session endpoint.
        """
        payload = {
            "session_id": (log_data or {}).get("session_id"),
            "device_id": device_id,
            "title": (log_data or {}).get("title"),
            "db_name": (log_data or {}).get("db_name"),
            "history": (log_data or {}).get("history", []),
        }

        r = self.session.post(
            self._url("/logs/sessions/ingest"),
            json=payload,
            headers=self._headers(),
            timeout=5.0,
        )
        r.raise_for_status()
        return r.json()

    def upload_camera_frame(self, device_id: str, mode: str, jpeg_bytes: bytes) -> Dict[str, Any]:
        r = self.session.post(
            self._url("/device/camera/frame"),
            params={"device_id": device_id, "mode": mode},
            data=jpeg_bytes,
            headers={
                "Content-Type": "image/jpeg",
                "X-Device-Secret": self.secret,
                "Connection": "keep-alive",
            },
            timeout=self.camera_timeout,
        )
        r.raise_for_status()
        return r.json()

    def get_chat_session(self, session_id: str) -> Dict[str, Any]:
        r = self.session.get(
            self._url(f"/logs/sessions/{session_id}"),
            headers={"X-Device-Secret": self.secret},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()