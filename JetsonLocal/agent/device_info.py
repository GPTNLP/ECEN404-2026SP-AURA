import socket
import time
import shutil
from pathlib import Path
from typing import Any

try:
    import psutil
except Exception:
    psutil = None

try:
    from jtop import jtop
except Exception:
    jtop = None


START_TIME = time.time()

_THERMAL_PATHS = (
    Path("/sys/class/thermal/thermal_zone0/temp"),
    Path("/sys/devices/virtual/thermal/thermal_zone0/temp"),
)

_GPU_SYSFS_PATHS = (
    Path("/sys/class/devfreq/17000000.ga10b/load"),
    Path("/sys/class/devfreq/gpu/load"),
    Path("/sys/devices/gpu.0/load"),
)

# psutil.cpu_percent(interval=None) returns % since the previous call.
# This primer establishes a baseline so the first real call isn't 0.0.
if psutil:
    try:
        psutil.cpu_percent(interval=None)
    except Exception:
        pass


def get_hostname() -> str:
    return socket.gethostname()


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def get_uptime_seconds() -> int:
    return int(time.time() - START_TIME)


def get_temperature_c() -> float | None:
    for path in _THERMAL_PATHS:
        try:
            value = float(path.read_text(encoding="utf-8").strip())
            if value > 1000:
                value /= 1000.0
            return round(value, 1)
        except Exception:
            pass
    return None


def get_cpu_percent() -> float | None:
    if psutil:
        try:
            return round(psutil.cpu_percent(interval=None), 1)
        except Exception:
            return None
    return None


def _read_gpu_from_jtop() -> float | None:
    if jtop is None:
        return None
    try:
        with jtop() as jetson:
            if not jetson.ok():
                return None
            stats = jetson.stats
            for key in ("GPU", "GPU1", "GR3D_FREQ"):
                if key in stats:
                    try:
                        value = float(stats[key])
                        if value > 100:
                            value /= 10.0
                        return round(value, 1)
                    except (ValueError, TypeError):
                        pass
    except Exception:
        return None
    return None


def _read_gpu_from_sysfs() -> float | None:
    for path in _GPU_SYSFS_PATHS:
        try:
            if not path.exists():
                continue
            value = float(path.read_text(encoding="utf-8").strip())
            # Jetson often reports 0..1000
            if value > 100:
                value /= 10.0
            return round(value, 1)
        except Exception:
            pass
    return None


def get_gpu_percent() -> float | None:
    v = _read_gpu_from_jtop()
    return v if v is not None else _read_gpu_from_sysfs()


def get_ram_percent() -> float | None:
    if psutil:
        try:
            return round(psutil.virtual_memory().percent, 1)
        except Exception:
            return None
    return None


def get_disk_percent() -> float | None:
    try:
        total, used, _ = shutil.disk_usage("/")
        if total <= 0:
            return None
        return round((used / total) * 100.0, 1)
    except Exception:
        return None


def collect_device_info() -> dict[str, Any]:
    return {
        "hostname": get_hostname(),
        "local_ip": get_local_ip(),
        "uptime_seconds": get_uptime_seconds(),
        "cpu_percent": get_cpu_percent(),
        "gpu_percent": get_gpu_percent(),
        "ram_percent": get_ram_percent(),
        "disk_percent": get_disk_percent(),
        "temperature_c": get_temperature_c(),
    }
