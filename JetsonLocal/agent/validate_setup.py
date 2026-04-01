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

def run_diagnostics():
    print("=== AURA Edge Diagnostic Suite ===")
    
    # 1. Check Ollama Local Server
    try:
        res = requests.get("http://127.0.0.1:11434/", timeout=2)
        print_status("Ollama Server", res.status_code == 200)
    except Exception as e:
        print_status("Ollama Server", False, f"({e})")

    # 2. Check ESP Serial Port
    serial_port = os.getenv("SERIAL_PORT", "/dev/ttyUSB0")
    try:
        s = serial.Serial(serial_port)
        s.close()
        print_status(f"Serial Connection ({serial_port})", True)
    except Exception as e:
        print_status(f"Serial Connection ({serial_port})", False, "- Is the ESP plugged in? Check permissions.")

    # 3. Check jtop (Hardware Telemetry)
    try:
        with jtop() as jetson:
            print_status("Jetson Stats (jtop)", jetson.ok())
    except Exception as e:
        print_status("Jetson Stats (jtop)", False, "- Try running 'sudo systemctl restart jtop'")

    # 4. Check Audio Input
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