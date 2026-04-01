from typing import Dict, Any

from JetsonLocal.agent.core.config import (
    DEVICE_ID,
    DEVICE_SOFTWARE_VERSION,
    OLLAMA_READY_DEFAULT,
    VECTOR_DB_READY_DEFAULT,
)
from device_info import collect_device_info
from JetsonLocal.agent.hardware.camera import get_camera_status


def build_heartbeat_payload() -> Dict[str, Any]:
    info = collect_device_info()
    cam = get_camera_status()

    return {
        "device_id": DEVICE_ID,
        "software_version": DEVICE_SOFTWARE_VERSION,
        "uptime_seconds": info["uptime_seconds"],
        "hostname": info["hostname"],
        "local_ip": info["local_ip"],
        "ollama_ready": OLLAMA_READY_DEFAULT,
        "vector_db_ready": VECTOR_DB_READY_DEFAULT,
        "camera_ready": cam["camera_ready"],
    }