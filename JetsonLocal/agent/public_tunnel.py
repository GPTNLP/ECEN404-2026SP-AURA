import os
import re
import shutil
import subprocess
import threading
from typing import Optional

_PUBLIC_URL: Optional[str] = None
_TUNNEL_PROC: Optional[subprocess.Popen] = None

_CF_URL_RE = re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com")
_NGROK_URL_RE = re.compile(r"https://[a-zA-Z0-9.-]+\.ngrok(?:-free)?\.(?:app|io)")


def get_public_url() -> str:
    return _PUBLIC_URL or ""


def _set_public_url(url: str):
    global _PUBLIC_URL
    _PUBLIC_URL = (url or "").strip()
    if _PUBLIC_URL:
        print(f"[TUNNEL] public url: {_PUBLIC_URL}")


def _consume_stdout(proc: subprocess.Popen, provider: str):
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue

            print(f"[TUNNEL:{provider}] {line}")

            if provider == "cloudflared":
                m = _CF_URL_RE.search(line)
                if m:
                    _set_public_url(m.group(0))
            elif provider == "ngrok":
                m = _NGROK_URL_RE.search(line)
                if m:
                    _set_public_url(m.group(0))
    except Exception as e:
        print(f"[TUNNEL] stdout reader failed: {e}")


def start_public_tunnel(local_port: int = 8000) -> str:
    """
    Starts a testing tunnel if enabled by env.
    Supported:
      AURA_ENABLE_PUBLIC_TUNNEL=1
      AURA_PUBLIC_TUNNEL_PROVIDER=cloudflared | ngrok
    """
    global _TUNNEL_PROC

    enabled = os.getenv("AURA_ENABLE_PUBLIC_TUNNEL", "0").strip() == "1"
    if not enabled:
        print("[TUNNEL] disabled")
        return ""

    provider = os.getenv("AURA_PUBLIC_TUNNEL_PROVIDER", "cloudflared").strip().lower()
    target = f"http://127.0.0.1:{int(local_port)}"

    if provider == "cloudflared":
        exe = shutil.which("cloudflared")
        if not exe:
            print("[TUNNEL] cloudflared not found in PATH")
            return ""

        cmd = [exe, "tunnel", "--url", target, "--no-autoupdate"]

    elif provider == "ngrok":
        exe = shutil.which("ngrok")
        if not exe:
            print("[TUNNEL] ngrok not found in PATH")
            return ""

        # ngrok must already have authtoken configured
        cmd = [exe, "http", str(int(local_port)), "--log", "stdout"]

    else:
        print(f"[TUNNEL] unsupported provider: {provider}")
        return ""

    try:
        _TUNNEL_PROC = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        t = threading.Thread(target=_consume_stdout, args=(_TUNNEL_PROC, provider), daemon=True)
        t.start()
        print(f"[TUNNEL] started {provider} for {target}")
    except Exception as e:
        print(f"[TUNNEL] failed to start {provider}: {e}")
        return ""

    return ""


def stop_public_tunnel():
    global _TUNNEL_PROC
    try:
        if _TUNNEL_PROC and _TUNNEL_PROC.poll() is None:
            _TUNNEL_PROC.terminate()
            print("[TUNNEL] stopped")
    except Exception as e:
        print(f"[TUNNEL] stop failed: {e}")
    finally:
        _TUNNEL_PROC = None