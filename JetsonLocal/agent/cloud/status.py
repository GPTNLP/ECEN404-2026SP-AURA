from typing import Dict, Any

from core.config import (
    DEVICE_ID,
    OLLAMA_READY_DEFAULT,
    VECTOR_DB_READY_DEFAULT,
    LOCAL_DB_NAME,
)
from device_info import collect_device_info
from battery import read_battery_status
from hardware.camera import get_camera_status


def build_status_payload() -> Dict[str, Any]:
    info = collect_device_info()
    batt = read_battery_status()
    cam = get_camera_status()

    gpu_percent = info.get("gpu_percent")

    return {
        "device_id": DEVICE_ID,
        "battery_percent": batt["battery_percent"],
        "battery_voltage": batt["battery_voltage"],
        "charging": batt["charging"],
        "cpu_percent": info["cpu_percent"],
        "gpu_percent": gpu_percent,
        "ram_percent": info["ram_percent"],
        "disk_percent": info["disk_percent"],
        "temperature_c": info["temperature_c"],
        "camera_ready": cam["camera_ready"],
        "mic_ready": True,
        "speaker_ready": True,
        "ollama_ready": OLLAMA_READY_DEFAULT,
        "vector_db_ready": VECTOR_DB_READY_DEFAULT,
        "current_mode": "idle",
        "current_task": "waiting",
        "extra": {
            "hostname": info["hostname"],
            "local_ip": info["local_ip"],
            "uptime_seconds": info["uptime_seconds"],
            "gpu_percent": gpu_percent,
            "db_name": LOCAL_DB_NAME,
        },
    }