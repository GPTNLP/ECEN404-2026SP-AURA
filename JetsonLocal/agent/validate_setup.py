#Run after running setup_orin.sh

import os
import sys
import requests
import serial
import speech_recognition as sr
from jtop import jtop

def print_status(component, status, error=""):
    color = "\033[92m" if status else "\033[91m"
    end = "\033[0m"
    print(f"[{color}{'PASS' if status else 'FAIL'}{end}] {component} {error}")

def print_warn(component, msg=""):
    print(f"[\033[93mWARN\033[0m] {component} {msg}")

def run_diagnostics():
    print("=== AURA Edge Diagnostic Suite ===")

    # 1. Check Ollama Local Server
    try:
        res = requests.get("http://127.0.0.1:11434/", timeout=2)
        print_status("Ollama Server", res.status_code == 200)
    except Exception as e:
        print_status("Ollama Server", False, f"({e})")

    # 2. Check Ollama GPU usage via /api/ps
    #    /api/ps lists running models with size_vram (bytes on GPU).
    #    If size_vram == 0 the model is CPU-only; if it equals size the model
    #    is fully GPU-resident.  A partial value means a CPU/GPU split.
    try:
        ps = requests.get("http://127.0.0.1:11434/api/ps", timeout=3).json()
        models = ps.get("models") or []
        if not models:
            print_warn("Ollama GPU Check", "— no model currently loaded (send a prompt first, then re-run)")
        else:
            for m in models:
                name      = m.get("name", "?")
                size_total = m.get("size", 0)
                size_vram  = m.get("size_vram", 0)
                if size_total > 0:
                    pct = round(100 * size_vram / size_total)
                    fully_gpu = size_vram >= size_total * 0.99
                    print_status(
                        f"GPU resident: {name}",
                        fully_gpu,
                        f"({pct}% in VRAM — {'fully GPU ✓' if fully_gpu else 'CPU/GPU split — check OLLAMA_NUM_GPU'})"
                    )
                else:
                    print_warn(f"Ollama model {name}", "— size info not available")
    except Exception as e:
        print_warn("Ollama GPU Check", f"— could not query /api/ps ({e})")

    # 3. Check Ollama performance environment variables
    #    These live in the Ollama systemd service, not in this process's env.
    #    We read the unit file directly to verify they are set.
    OLLAMA_SVC = "/etc/systemd/system/ollama.service"
    perf_vars = {
        "OLLAMA_NUM_GPU":        ("999",  "forces all model layers onto Jetson GPU"),
        "OLLAMA_FLASH_ATTENTION":("1",    "reduces VRAM per call; critical for GPU stability"),
        "OLLAMA_KV_CACHE_TYPE":  ("q8_0", "halves KV-cache VRAM; prevents CPU fallback"),
    }
    if os.path.exists(OLLAMA_SVC):
        with open(OLLAMA_SVC) as f:
            svc_text = f.read()
        for var, (expected, hint) in perf_vars.items():
            present = var in svc_text
            correct = f'"{var}={expected}"' in svc_text or f"{var}={expected}" in svc_text
            if correct:
                print_status(f"Ollama env: {var}={expected}", True)
            elif present:
                print_warn(f"Ollama env: {var}", f"— present but value differs from recommended {expected} ({hint})")
            else:
                print_warn(f"Ollama env: {var}", f"— MISSING. Add to {OLLAMA_SVC} ({hint}). Re-run setup_orin.sh to fix.")
    else:
        print_warn("Ollama service file", f"— {OLLAMA_SVC} not found; cannot verify env vars")

    # 4. Check ESP Serial Port
    serial_port = os.getenv("SERIAL_PORT", "/dev/ttyUSB0")
    try:
        s = serial.Serial(serial_port)
        s.close()
        print_status(f"Serial Connection ({serial_port})", True)
    except Exception as e:
        print_status(f"Serial Connection ({serial_port})", False, "- Is the ESP plugged in? Check permissions.")

    # 5. Check jtop (Hardware Telemetry)
    try:
        with jtop() as jetson:
            print_status("Jetson Stats (jtop)", jetson.ok())
    except Exception as e:
        print_status("Jetson Stats (jtop)", False, "- Try running 'sudo systemctl restart jtop'")

    # 6. Check Audio Input
    try:
        mics = sr.Microphone.list_microphone_names()
        if len(mics) > 0:
            print_status("USB Microphone Detected", True)
        else:
            print_status("USB Microphone Detected", False, "- No audio devices found.")
    except Exception as e:
        print_status("Audio Subsystem", False, f"({e})")

if __name__ == "__main__":
    run_diagnostics()
