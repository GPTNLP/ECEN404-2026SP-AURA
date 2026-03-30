import requests
from typing import Any, Dict

from config import API_BASE_URL, DEVICE_SHARED_SECRET


class ApiClient:
    def __init__(self):
        self.base_url = API_BASE_URL.rstrip("/")
        self.timeout = 15

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
        r = requests.get(self._url("/health"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def register(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            self._url("/device/register"),
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def heartbeat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            self._url("/device/heartbeat"),
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def status(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            self._url("/device/status"),
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def log(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            self._url("/device/logs"),
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def get_config(self, device_id: str) -> Dict[str, Any]:
        r = requests.get(
            self._url("/device/config"),
            params={"device_id": device_id},
            headers={"X-Device-Secret": DEVICE_SHARED_SECRET},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def get_next_command(self, device_id: str) -> Dict[str, Any]:
        r = requests.get(
            self._url("/device/command/next"),
            params={"device_id": device_id},
            headers={"X-Device-Secret": DEVICE_SHARED_SECRET},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def ack_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            self._url("/device/command/ack"),
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()