#!/usr/bin/env python3
import io
import json
import queue
import re
import subprocess
import threading
import tkinter as tk
from collections import Counter
from datetime import datetime
from tkinter import font as tkfont
from urllib import error, parse, request

try:
    from PIL import Image, ImageTk
except Exception as exc:
    raise RuntimeError(
        "nano_main.py now needs Pillow for the camera touchscreen UI. "
        "Install it with: pip install pillow"
    ) from exc


SERVICE_NAME = "aura-agent.service"
API_BASE = "http://127.0.0.1:8000"
MAX_LOG_LINES = 220
POLL_MS = 150
FRAME_MS = 90

VISION_MODES = {
    "detection": {
        "title": "Component Detection",
        "subtitle": "Runs the component model and shows detected parts below the camera.",
        "button": "Detection",
    },
    "colorcode": {
        "title": "Color Code",
        "subtitle": "Runs the color-code model and shows band / object labels below the camera.",
        "button": "Color Code",
    },
    "face": {
        "title": "Face Detection",
        "subtitle": "Runs the face model and shows face detections below the camera.",
        "button": "Face Detection",
    },
}

STATUS_STYLES = {
    "BOOTING": {"fg": "#7dd3fc", "sub": "Starting services"},
    "READY": {"fg": "#a7f3d0", "sub": "Ready"},
    "LISTENING": {"fg": "#67e8f9", "sub": "Listening for wake word"},
    "THINKING": {"fg": "#fcd34d", "sub": "Thinking"},
    "SPEAKING": {"fg": "#c4b5fd", "sub": "Speaking"},
    "VISION": {"fg": "#f9a8d4", "sub": "Vision mode active"},
    "VECTORIZING": {"fg": "#f9a8d4", "sub": "Vectorizing PDFs"},
    "COMMAND": {"fg": "#fdba74", "sub": "Running command"},
    "ERROR": {"fg": "#fca5a5", "sub": "Check live console"},
    "OFFLINE": {"fg": "#94a3b8", "sub": "Waiting for agent"},
}


class AuraConsoleApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AURA")
        self.root.configure(bg="#05070a")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.on_close)
        self.root.bind("q", self.on_close)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.running = True
        self.reader_process = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.ui_mode = "home"
        self.active_vision_mode = None
        self.voice_should_resume = False
        self.current_frame_image = None

        self.state = "BOOTING"
        self.status_text = tk.StringVar(value="BOOTING")
        self.sub_text = tk.StringVar(value=STATUS_STYLES["BOOTING"]["sub"])
        self.banner_text = tk.StringVar(value="AURA")
        self.clock_text = tk.StringVar(value="")
        self.vision_title_text = tk.StringVar(value="")
        self.vision_status_text = tk.StringVar(value="Waiting for camera...")
        self.vision_meta_text = tk.StringVar(value="")
        self.detection_text = tk.StringVar(value="No detections yet.")

        self._build_ui()
        self._start_reader()
        self._tick_clock()
        self._poll_logs()
        self._poll_vision()

    def _build_ui(self):
        sw = self.root.winfo_screenwidth()

        title_font = tkfont.Font(family="Courier", size=max(22, int(sw * 0.030)), weight="bold")
        status_font = tkfont.Font(family="Courier", size=max(28, int(sw * 0.050)), weight="bold")
        sub_font = tkfont.Font(family="Courier", size=max(11, int(sw * 0.016)))
        section_font = tkfont.Font(family="Courier", size=max(17, int(sw * 0.024)), weight="bold")
        log_font = tkfont.Font(family="Courier", size=max(9, int(sw * 0.013)))
        button_font = tkfont.Font(family="Courier", size=max(14, int(sw * 0.018)), weight="bold")
        vision_title_font = tkfont.Font(family="Courier", size=max(18, int(sw * 0.024)), weight="bold")
        vision_info_font = tkfont.Font(family="Courier", size=max(12, int(sw * 0.015)))

        outer = tk.Frame(self.root, bg="#05070a")
        outer.pack(fill="both", expand=True, padx=12, pady=12)

        self.header = tk.Frame(
            outer,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        self.header.pack(fill="x", pady=(0, 10))

        self.header_label = tk.Label(
            self.header,
            textvariable=self.banner_text,
            fg="#14f195",
            bg="#0b0f14",
            font=title_font,
            anchor="center",
            padx=10,
            pady=10,
        )
        self.header_label.pack(fill="x")

        self.content = tk.Frame(outer, bg="#05070a")
        self.content.pack(fill="both", expand=True)

        self.home_frame = tk.Frame(self.content, bg="#05070a")
        self.vision_frame = tk.Frame(self.content, bg="#05070a")

        self._build_home_ui(self.home_frame, status_font, sub_font, section_font, log_font, button_font)
        self._build_vision_ui(self.vision_frame, button_font, vision_title_font, vision_info_font)

        self._show_home()

    def _build_home_ui(self, parent, status_font, sub_font, section_font, log_font, button_font):
        self.status_card = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        self.status_card.pack(fill="x", pady=(0, 10))

        self.status_label = tk.Label(
            self.status_card,
            textvariable=self.status_text,
            fg=STATUS_STYLES["BOOTING"]["fg"],
            bg="#0b0f14",
            font=status_font,
            pady=12,
        )
        self.status_label.pack(fill="x")

        self.sub_label = tk.Label(
            self.status_card,
            textvariable=self.sub_text,
            fg="#94a3b8",
            bg="#0b0f14",
            font=sub_font,
            pady=6,
        )
        self.sub_label.pack(fill="x", pady=(0, 12))

        meta_row = tk.Frame(parent, bg="#05070a")
        meta_row.pack(fill="x", pady=(0, 10))

        self.clock_card = self._make_meta_card(meta_row, "TIME", self.clock_text, sub_font)
        self.clock_card.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.source_card = self._make_meta_card(meta_row, "SOURCE", tk.StringVar(value=SERVICE_NAME), sub_font)
        self.source_card.pack(side="left", fill="x", expand=True, padx=5)

        self.mode_card = self._make_meta_card(meta_row, "UI", tk.StringVar(value="HOME"), sub_font)
        self.mode_card.pack(side="left", fill="x", expand=True, padx=(5, 0))
        self.mode_value_label = self.mode_card.winfo_children()[1]

        vision_card = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        vision_card.pack(fill="x", pady=(0, 10))

        tk.Label(
            vision_card,
            text="VISION MODES",
            fg="#14f195",
            bg="#0b0f14",
            font=section_font,
            anchor="w",
            padx=14,
            pady=12,
        ).pack(fill="x")

        buttons_wrap = tk.Frame(vision_card, bg="#0b0f14")
        buttons_wrap.pack(fill="x", padx=12, pady=(0, 12))

        for idx, mode in enumerate(("face", "detection", "colorcode")):
            cfg = VISION_MODES[mode]
            btn = tk.Button(
                buttons_wrap,
                text=cfg["button"],
                command=lambda m=mode: self.enter_vision_mode(m),
                font=button_font,
                bg="#111827",
                fg="#ecfeff",
                activebackground="#1d4ed8",
                activeforeground="#ffffff",
                relief="flat",
                bd=0,
                cursor="hand2",
                padx=20,
                pady=18,
            )
            btn.grid(row=0, column=idx, sticky="nsew", padx=6)
            buttons_wrap.grid_columnconfigure(idx, weight=1)

        logs_card = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        logs_card.pack(fill="both", expand=True)

        tk.Label(
            logs_card,
            text="LIVE CONSOLE",
            fg="#14f195",
            bg="#0b0f14",
            font=section_font,
            anchor="w",
            padx=14,
            pady=10,
        ).pack(fill="x")

        self.log_text = tk.Text(
            logs_card,
            bg="#05070a",
            fg="#9effc7",
            insertbackground="#9effc7",
            relief="flat",
            wrap="word",
            font=log_font,
            padx=12,
            pady=12,
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True)

    def _build_vision_ui(self, parent, button_font, title_font, info_font):
        topbar = tk.Frame(parent, bg="#05070a")
        topbar.pack(fill="x", pady=(0, 10))

        self.back_button = tk.Button(
            topbar,
            text="← Back",
            command=self.leave_vision_mode,
            font=button_font,
            bg="#111827",
            fg="#ecfeff",
            activebackground="#374151",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=18,
            pady=12,
        )
        self.back_button.pack(side="left")

        self.vision_title_label = tk.Label(
            topbar,
            textvariable=self.vision_title_text,
            fg="#14f195",
            bg="#05070a",
            font=title_font,
            anchor="center",
        )
        self.vision_title_label.pack(side="left", fill="x", expand=True, padx=16)

        mode_row = tk.Frame(parent, bg="#05070a")
        mode_row.pack(fill="x", pady=(0, 10))

        self.mode_buttons = {}
        for idx, mode in enumerate(("face", "detection", "colorcode")):
            cfg = VISION_MODES[mode]
            btn = tk.Button(
                mode_row,
                text=cfg["button"],
                command=lambda m=mode: self.switch_vision_mode(m),
                font=button_font,
                bg="#111827",
                fg="#ecfeff",
                activebackground="#1d4ed8",
                activeforeground="#ffffff",
                relief="flat",
                bd=0,
                cursor="hand2",
                padx=20,
                pady=14,
            )
            btn.grid(row=0, column=idx, sticky="nsew", padx=6)
            mode_row.grid_columnconfigure(idx, weight=1)
            self.mode_buttons[mode] = btn

        camera_card = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        camera_card.pack(fill="both", expand=True)

        self.camera_label = tk.Label(
            camera_card,
            text="Starting camera...",
            fg="#cbd5e1",
            bg="#020617",
            font=info_font,
            anchor="center",
            justify="center",
        )
        self.camera_label.pack(fill="both", expand=True, padx=12, pady=12)

        info_card = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        info_card.pack(fill="x", pady=(10, 0))

        tk.Label(
            info_card,
            textvariable=self.vision_status_text,
            fg="#e2e8f0",
            bg="#0b0f14",
            font=info_font,
            anchor="w",
            justify="left",
            padx=14,
            pady=4,
        ).pack(fill="x", pady=(12, 4))

        tk.Label(
            info_card,
            textvariable=self.vision_meta_text,
            fg="#94a3b8",
            bg="#0b0f14",
            font=info_font,
            anchor="w",
            justify="left",
            padx=14,
            pady=3,
        ).pack(fill="x", pady=(0, 6))

        tk.Label(
            info_card,
            textvariable=self.detection_text,
            fg="#f8fafc",
            bg="#0b0f14",
            font=info_font,
            anchor="w",
            justify="left",
            wraplength=max(600, int(self.root.winfo_screenwidth() * 0.9)),
            padx=14,
            pady=6,
        ).pack(fill="x", pady=(0, 12))

    def _make_meta_card(self, parent, title, variable, info_font):
        frame = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        tk.Label(
            frame,
            text=title,
            fg="#14f195",
            bg="#0b0f14",
            font=info_font,
            anchor="w",
            padx=10,
            pady=4,
        ).pack(fill="x", pady=(8, 2))
        tk.Label(
            frame,
            textvariable=variable,
            fg="#e2e8f0",
            bg="#0b0f14",
            font=info_font,
            anchor="w",
            padx=10,
            pady=4,
        ).pack(fill="x", pady=(0, 8))
        return frame

    def _show_home(self):
        self.ui_mode = "home"
        self.vision_frame.pack_forget()
        self.home_frame.pack(fill="both", expand=True)
        self.mode_value_label.configure(text="HOME")

    def _show_vision(self):
        self.ui_mode = "vision"
        self.home_frame.pack_forget()
        self.vision_frame.pack(fill="both", expand=True)
        self.mode_value_label.configure(text="VISION")

    def _start_reader(self):
        thread = threading.Thread(target=self._reader_worker, daemon=True)
        thread.start()

    def _reader_worker(self):
        cmd = ["journalctl", "-u", SERVICE_NAME, "-f", "-n", "150", "--no-pager", "-o", "cat"]
        try:
            self.reader_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.log_queue.put(f"[UI ERROR] Failed to start journal reader: {exc}")
            return

        if self.reader_process.stdout is None:
            self.log_queue.put("[UI ERROR] journalctl stdout unavailable")
            return

        for raw_line in self.reader_process.stdout:
            if not self.running:
                break
            line = raw_line.rstrip("\n")
            if line.strip():
                self.log_queue.put(line)

    def _tick_clock(self):
        self.clock_text.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if self.running:
            self.root.after(500, self._tick_clock)

    def _poll_logs(self):
        processed = 0
        while processed < 60:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)
            self._update_state_from_line(line)
            processed += 1

        if self.running:
            self.root.after(POLL_MS, self._poll_logs)

    def _append_log(self, line: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        display_line = f"[{timestamp}] {line}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", display_line)
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            self.log_text.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_status(self, status: str, substatus: str):
        style = STATUS_STYLES.get(status, STATUS_STYLES["READY"])
        self.status_text.set(status)
        self.sub_text.set(substatus or style["sub"])
        self.status_label.configure(fg=style["fg"])

    def _clean_event(self, line: str) -> str:
        line = re.sub(r"\s+", " ", line).strip()
        return line[:150]

    def _update_state_from_line(self, line: str):
        lower = line.lower()
        clean = self._clean_event(line)

        if "[startup]" in lower or "telemetry agent running" in lower:
            self._set_status("READY", "AURA services are up")
            return
        if "[voice]" in lower and "listening" in lower:
            self._set_status("LISTENING", "Wake word detection active")
            return
        if "[voice]" in lower and "answered" in lower:
            self._set_status("SPEAKING", clean)
            return
        if "[chat]" in lower and ("running rag query" in lower or "received command" in lower):
            self._set_status("THINKING", clean)
            return
        if "[jetson db]" in lower and ("loading" in lower or "loaded" in lower or "vector" in lower or "build" in lower):
            self._set_status("VECTORIZING", clean)
            return
        if "[command]" in lower:
            self._set_status("COMMAND", clean)
            return
        if "[status] ok" in lower:
            if self.ui_mode != "vision":
                self._set_status("READY", "Systems nominal")
            return
        if "error" in lower or "failed" in lower or "traceback" in lower:
            self._set_status("ERROR", clean)
            return

    def _http_json(self, method: str, path: str, timeout: float = 2.0):
        req = request.Request(f"{API_BASE}{path}", method=method.upper())
        req.add_header("Accept", "application/json")
        with request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        return json.loads(data) if data else {}

    def _http_bytes(self, path: str, timeout: float = 2.0) -> bytes:
        req = request.Request(f"{API_BASE}{path}", method="GET")
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def _set_mode_button_styles(self):
        for mode, btn in self.mode_buttons.items():
            active = mode == self.active_vision_mode
            btn.configure(
                bg="#1d4ed8" if active else "#111827",
                fg="#ffffff" if active else "#ecfeff",
            )

    def enter_vision_mode(self, mode: str):
        self._show_vision()
        self.switch_vision_mode(mode)

    def switch_vision_mode(self, mode: str):
        if mode not in VISION_MODES:
            return

        try:
            voice_status = self._http_json("GET", "/voice/status")
            self.voice_should_resume = bool(voice_status.get("running"))
        except Exception:
            self.voice_should_resume = False

        try:
            self._http_json("POST", "/voice/stop", timeout=3.0)
        except Exception as exc:
            self.vision_status_text.set(f"Voice stop warning: {exc}")

        try:
            query = parse.urlencode({"mode": mode})
            self._http_json("POST", f"/camera/activate?{query}", timeout=4.0)
        except Exception as exc:
            self.vision_status_text.set(f"Camera activation failed: {exc}")
            self.detection_text.set("Check the AURA agent logs below for details.")
            return

        self.active_vision_mode = mode
        cfg = VISION_MODES[mode]
        self.vision_title_text.set(cfg["title"])
        self.vision_status_text.set(cfg["subtitle"])
        self.detection_text.set("Waiting for detections...")
        self._set_mode_button_styles()
        self._set_status("VISION", f"{cfg['button']} active")

    def leave_vision_mode(self):
        try:
            self._http_json("POST", "/camera/deactivate", timeout=3.0)
        except Exception:
            pass

        if self.voice_should_resume:
            try:
                self._http_json("POST", "/voice/start", timeout=3.0)
            except Exception:
                pass

        self.active_vision_mode = None
        self.voice_should_resume = False
        self.camera_label.configure(image="", text="Camera stopped.")
        self.current_frame_image = None
        self.vision_title_text.set("")
        self.vision_status_text.set("Vision mode closed.")
        self.vision_meta_text.set("")
        self.detection_text.set("No detections yet.")
        self._set_mode_button_styles()
        self._show_home()
        self._set_status("READY", "Returned to home screen")

    def _poll_vision(self):
        if self.running and self.ui_mode == "vision" and self.active_vision_mode:
            self._refresh_vision_frame()
            self._refresh_detections()
        if self.running:
            self.root.after(FRAME_MS, self._poll_vision)

    def _refresh_vision_frame(self):
        if not self.active_vision_mode:
            return
        try:
            frame_bytes = self._http_bytes(
                f"/camera/frame.jpg?ts={int(datetime.now().timestamp() * 1000)}",
                timeout=2.0,
            )
            image = Image.open(io.BytesIO(frame_bytes))

            max_w = max(640, int(self.root.winfo_screenwidth() * 0.94))
            max_h = max(360, int(self.root.winfo_screenheight() * 0.56))
            image.thumbnail((max_w, max_h))

            photo = ImageTk.PhotoImage(image)
            self.current_frame_image = photo
            self.camera_label.configure(image=photo, text="")
        except error.HTTPError as exc:
            self.camera_label.configure(image="", text=f"Camera HTTP error: {exc.code}")
            self.current_frame_image = None
        except Exception as exc:
            self.camera_label.configure(image="", text=f"Waiting for camera...\n{exc}")
            self.current_frame_image = None

    def _refresh_detections(self):
        if not self.active_vision_mode:
            return

        try:
            status = self._http_json("GET", "/camera/status")
            detections = self._http_json("GET", "/camera/detections")
        except Exception as exc:
            self.vision_meta_text.set(f"Status unavailable: {exc}")
            return

        model_loaded = status.get("models_loaded", {}).get(self.active_vision_mode)
        count = int(status.get("detection_count") or 0)
        fps = status.get("actual_fps") or status.get("fps") or 0
        resolution = status.get("actual_resolution") or {}
        w = resolution.get("width") or status.get("resolution", {}).get("width") or "?"
        h = resolution.get("height") or status.get("resolution", {}).get("height") or "?"
        backend = status.get("capture_backend") or "unknown"
        last_error = status.get("last_error")

        self.vision_meta_text.set(
            f"Mode: {self.active_vision_mode}   |   Detections: {count}   |   "
            f"Resolution: {w}x{h}   |   FPS: {fps}   |   Backend: {backend}"
        )

        items = detections.get("detections", []) or []
        if not model_loaded:
            path_map = status.get("model_paths", {})
            model_path = path_map.get(self.active_vision_mode, "model file missing")
            self.detection_text.set(
                f"Model for {self.active_vision_mode} is not loaded. Expected path: {model_path}."
            )
            if last_error:
                self.vision_status_text.set(f"Vision warning: {last_error}")
            return

        if not items:
            self.detection_text.set("No detections in current frame.")
            if last_error:
                self.vision_status_text.set(f"Vision warning: {last_error}")
            else:
                self.vision_status_text.set(VISION_MODES[self.active_vision_mode]["subtitle"])
            return

        counts = Counter(item.get("label", "unknown") for item in items)
        lines = []
        for idx, item in enumerate(items[:8], start=1):
            label = item.get("label", "unknown")
            conf = item.get("confidence")
            bbox = item.get("bbox") or []
            bbox_text = f" bbox={bbox}" if bbox else ""
            conf_text = f" ({conf:.2f})" if isinstance(conf, (float, int)) else ""
            lines.append(f"{idx}. {label}{conf_text}{bbox_text}")

        summary = ", ".join(f"{label}: {qty}" for label, qty in counts.most_common())
        self.detection_text.set("Summary: " + summary + "\n\n" + "\n".join(lines))
        self.vision_status_text.set(f"{len(items)} detection(s) in current frame.")

    def on_close(self, event=None):
        self.running = False
        try:
            self.leave_vision_mode()
        except Exception:
            pass
        try:
            if self.reader_process and self.reader_process.poll() is None:
                self.reader_process.terminate()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    AuraConsoleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()