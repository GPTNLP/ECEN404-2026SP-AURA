import requests
from typing import Any, Dict
import os

from core.config import API_BASE_URL, DEVICE_SHARED_SECRET


class ApiClient:
    def __init__(self):
        self.base_url = API_BASE_URL.rstrip("/")
        self.timeout = 15
        self.camera_timeout = 4
        self.session = requests.Session()

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

    def upload_vector_db(self, db_name: str, working_dir: str):
        url = self._url(f"/api/databases/{db_name}/sync_up")
        files = []
        for fn in ["faiss.index", "embeddings.npy", "meta.json"]:
            fp = os.path.join(working_dir, fn)
            if os.path.exists(fp):
                files.append(("files", (fn, open(fp, "rb"))))

        if files:
            r = self.session.post(
                url,
                files=files,
                headers={"X-Device-Secret": DEVICE_SHARED_SECRET},
                timeout=self.timeout,
            )
            r.raise_for_status()
            for _, (_, f) in files:
                f.close()

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