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
FRAME_MS = 70
DETECTION_EVERY_N_POLLS = 2
CAMERA_ERROR_THRESHOLD = 4

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
    "LISTENING": {"fg": "#67e8f9", "sub": "Listening"},
    "THINKING": {"fg": "#fcd34d", "sub": "Thinking"},
    "SPEAKING": {"fg": "#c4b5fd", "sub": "Speaking"},
    "VISION": {"fg": "#f9a8d4", "sub": "Vision mode active"},
    "VECTORIZING": {"fg": "#f9a8d4", "sub": "Vectorizing PDFs"},
    "COMMAND": {"fg": "#fdba74", "sub": "Running command"},
    "ERROR": {"fg": "#fca5a5", "sub": "Check live feed"},
    "OFFLINE": {"fg": "#94a3b8", "sub": "Waiting for agent"},
}

BTN_BG = "#111827"
BTN_FG = "#ecfeff"
BTN_ACTIVE_BG = "#0f172a"
BTN_ACTIVE_FG = "#ffffff"
BTN_GREEN = "#14f195"
BTN_GREEN_ACTIVE = "#22c55e"


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
        self.view_mode = "console"
        self.active_vision_mode = None
        self.voice_should_resume = False
        self.current_frame_image = None

        self.state = "BOOTING"
        self.status_text = tk.StringVar(value="BOOTING")
        self.sub_text = tk.StringVar(value=STATUS_STYLES["BOOTING"]["sub"])
        self.banner_text = tk.StringVar(value="AURA")
        self.vision_title_text = tk.StringVar(value="")
        self.vision_status_text = tk.StringVar(value="Waiting for camera...")
        self.vision_meta_text = tk.StringVar(value="")
        self.detection_text = tk.StringVar(value="No detections yet.")
        self.voice_button_text = tk.StringVar(value="Tap Mic")
        self.voice_status_text = tk.StringVar(
            value="Press the mic, speak once, and AURA will respond."
        )
        self.voice_result_text = tk.StringVar(
            value="Last voice result will appear here."
        )
        self.voice_busy = False

        self._llm_thinking = False
        self._llm_history = []

        self._vision_poll_counter = 0
        self._camera_fail_count = 0

        self._build_ui()
        self._start_reader()
        self._poll_logs()
        self._poll_vision()

    # =========================
    # UI
    # =========================

    def _build_ui(self):
        sw = self.root.winfo_screenwidth()

        title_font = tkfont.Font(
            family="Courier", size=max(22, int(sw * 0.030)), weight="bold"
        )
        sub_font = tkfont.Font(
            family="Courier", size=max(11, int(sw * 0.016))
        )
        section_font = tkfont.Font(
            family="Courier", size=max(17, int(sw * 0.024)), weight="bold"
        )
        log_font = tkfont.Font(
            family="Courier", size=max(10, int(sw * 0.015)), weight="bold"
        )
        button_font = tkfont.Font(
            family="Courier", size=max(14, int(sw * 0.018)), weight="bold"
        )
        vision_title_font = tkfont.Font(
            family="Courier", size=max(18, int(sw * 0.024)), weight="bold"
        )
        vision_info_font = tkfont.Font(
            family="Courier", size=max(12, int(sw * 0.015))
        )

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

        self._build_home_ui(
            self.home_frame, sub_font, section_font, log_font, button_font
        )
        self._build_vision_ui(
            self.vision_frame, button_font, vision_title_font, vision_info_font
        )

        self._show_home()

    def _is_scrolled_near_bottom(self, widget, threshold: float = 0.04) -> bool:
        try:
            _top, bottom = widget.yview()
            return bottom >= (1.0 - threshold)
        except Exception:
            return True
    
    def _outline_button(
        self,
        parent,
        text,
        command,
        font,
        *,
        fg=BTN_FG,
        highlight=BTN_GREEN,
        padx=20,
        pady=12,
        fill=False,
    ):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=font,
            bg=BTN_BG,
            fg=fg,
            activebackground=BTN_ACTIVE_BG,
            activeforeground=BTN_ACTIVE_FG,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=padx,
            pady=pady,
            highlightthickness=1,
            highlightbackground=highlight,
            highlightcolor=highlight,
        )
        if fill:
            btn.pack(fill="x", padx=12, pady=12)
        return btn

    def _build_home_ui(self, parent, sub_font, section_font, log_font, button_font):
        mode_card = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        mode_card.pack(fill="x", pady=(0, 10))

        btn_row = tk.Frame(mode_card, bg="#0b0f14")
        btn_row.pack(side="left", padx=12, pady=10)

        self.console_btn = self._outline_button(
            btn_row,
            "CONSOLE",
            lambda: self._switch_view("console"),
            button_font,
            fg="#ffffff",
            highlight=BTN_GREEN,
        )
        self.console_btn.pack(side="left", padx=(0, 8))

        self.llm_btn = self._outline_button(
            btn_row,
            "LLM CHAT",
            lambda: self._switch_view("llm"),
            button_font,
            fg=BTN_FG,
            highlight="#94a3b8",
        )
        self.llm_btn.pack(side="left")

        self.status_mini_label = tk.Label(
            mode_card,
            textvariable=self.status_text,
            fg=STATUS_STYLES["BOOTING"]["fg"],
            bg="#0b0f14",
            font=sub_font,
            anchor="e",
            padx=16,
        )
        self.status_mini_label.pack(side="right", fill="y")

        camera_card = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        camera_card.pack(fill="x", pady=(0, 10))

        self.camera_home_button = self._outline_button(
            camera_card,
            "CAMERA",
            lambda: self.enter_vision_mode("face"),
            button_font,
            fg=BTN_GREEN,
            highlight=BTN_GREEN,
            pady=18,
            fill=True,
        )

        voice_card = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        voice_card.pack(fill="x", pady=(0, 10))

        voice_top = tk.Frame(voice_card, bg="#0b0f14")
        voice_top.pack(fill="x", padx=12, pady=(12, 8))

        tk.Label(
            voice_top,
            text="VOICE",
            fg="#14f195",
            bg="#0b0f14",
            font=section_font,
            anchor="w",
        ).pack(side="left")

        self.voice_button = self._outline_button(
            voice_top,
            "Tap Mic",
            self._run_voice_button,
            button_font,
            fg=BTN_GREEN,
            highlight=BTN_GREEN,
        )
        self.voice_button.pack(side="right")

        info_font = tkfont.Font(
            family="Courier",
            size=max(10, int(self.root.winfo_screenwidth() * 0.012)),
        )

        tk.Label(
            voice_card,
            textvariable=self.voice_status_text,
            fg="#cbd5e1",
            bg="#0b0f14",
            font=info_font,
            anchor="w",
            justify="left",
            padx=14,
            pady=2,
        ).pack(fill="x")

        tk.Label(
            voice_card,
            textvariable=self.voice_result_text,
            fg="#94a3b8",
            bg="#0b0f14",
            font=info_font,
            anchor="w",
            justify="left",
            wraplength=max(600, int(self.root.winfo_screenwidth() * 0.88)),
            padx=14,
            pady=0,
        ).pack(fill="x", pady=(0, 12))

        self.content_stack = tk.Frame(parent, bg="#05070a")
        self.content_stack.pack(fill="both", expand=True)

        self.console_panel = tk.Frame(self.content_stack, bg="#05070a")
        self.llm_panel = tk.Frame(self.content_stack, bg="#05070a")

        self._build_console_panel(self.console_panel, section_font, log_font)
        self._build_llm_panel(self.llm_panel, section_font, log_font, button_font)

        self._switch_view("console")

    def _build_console_panel(self, parent, section_font, log_font):
        logs_card = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        logs_card.pack(fill="both", expand=True)

        tk.Label(
            logs_card,
            text="LIVE FEED",
            fg="#14f195",
            bg="#0b0f14",
            font=section_font,
            anchor="w",
            padx=14,
            pady=10,
        ).pack(fill="x")

        live_font = tkfont.Font(
            family="Courier",
            size=max(12, int(self.root.winfo_screenwidth() * 0.016)),
            weight="bold",
        )

        self.log_text = tk.Text(
            logs_card,
            bg="#05070a",
            fg="#9effc7",
            insertbackground="#9effc7",
            relief="flat",
            wrap="word",
            font=live_font,
            padx=14,
            pady=14,
            state="disabled",
            spacing1=3,
            spacing2=3,
            spacing3=3,
        )
        self.log_text.pack(fill="both", expand=True)

    def _build_llm_panel(self, parent, section_font, log_font, button_font):
        chat_card = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        chat_card.pack(fill="both", expand=True)

        tk.Label(
            chat_card,
            text="LLM CHAT",
            fg="#14f195",
            bg="#0b0f14",
            font=section_font,
            anchor="w",
            padx=14,
            pady=10,
        ).pack(fill="x")

        self.llm_chat_text = tk.Text(
            chat_card,
            bg="#05070a",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief="flat",
            wrap="word",
            font=log_font,
            padx=12,
            pady=12,
            state="disabled",
        )
        self.llm_chat_text.pack(fill="both", expand=True)

        bold_log = tkfont.Font(
            family="Courier",
            size=max(9, int(log_font.cget("size"))),
            weight="bold",
        )

        self.llm_chat_text.tag_configure("user_label", foreground="#14f195", font=bold_log)
        self.llm_chat_text.tag_configure("user_text", foreground="#ffffff")
        self.llm_chat_text.tag_configure("aura_label", foreground="#fcd34d", font=bold_log)
        self.llm_chat_text.tag_configure("aura_text", foreground="#cbd5e1")
        self.llm_chat_text.tag_configure("thinking", foreground="#64748b")
        self.llm_chat_text.tag_configure("error_label", foreground="#fca5a5", font=bold_log)
        self.llm_chat_text.tag_configure("error_text", foreground="#fca5a5")

        input_frame = tk.Frame(chat_card, bg="#0b0f14", pady=8, padx=8)
        input_frame.pack(fill="x")

        self.llm_entry = tk.Entry(
            input_frame,
            bg="#111827",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief="flat",
            font=button_font,
        )
        self.llm_entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=10)
        self.llm_entry.bind("<Return>", lambda _e: self._llm_submit())

        self.llm_send_btn = tk.Button(
            input_frame,
            text="Ask",
            command=self._llm_submit,
            font=button_font,
            bg="#111827",
            fg="#ecfeff",
            activebackground="#0f172a",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=20,
            pady=10,
            highlightthickness=1,
            highlightbackground="#94a3b8",
            highlightcolor="#14f195",
        )
        self.llm_send_btn.pack(side="right")

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
            activebackground="#0f172a",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=18,
            pady=12,
            highlightthickness=1,
            highlightbackground="#14f195",
            highlightcolor="#14f195",
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
                activebackground="#0f172a",
                activeforeground="#ffffff",
                relief="flat",
                bd=0,
                cursor="hand2",
                padx=20,
                pady=14,
                highlightthickness=1,
                highlightbackground="#14f195",
                highlightcolor="#14f195",
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

    # =========================
    # View switching
    # =========================

    def _show_home(self):
        self.ui_mode = "home"
        self.vision_frame.pack_forget()
        self.home_frame.pack(fill="both", expand=True)

    def _show_vision(self):
        self.ui_mode = "vision"
        self.home_frame.pack_forget()
        self.vision_frame.pack(fill="both", expand=True)

    def _switch_view(self, mode: str):
        self.view_mode = mode
        if mode == "llm":
            self.console_panel.pack_forget()
            self.llm_panel.pack(fill="both", expand=True)
            self.console_btn.configure(bg="#111827", fg="#ecfeff", highlightbackground="#94a3b8")
            self.llm_btn.configure(bg="#111827", fg="#ffffff", highlightbackground="#14f195")
            self.llm_entry.focus_set()
        else:
            self.llm_panel.pack_forget()
            self.console_panel.pack(fill="both", expand=True)
            self.console_btn.configure(bg="#111827", fg="#ffffff", highlightbackground="#14f195")
            self.llm_btn.configure(bg="#111827", fg="#ecfeff", highlightbackground="#94a3b8")

    # =========================
    # LLM chat
    # =========================

    def _llm_submit(self):
        query = self.llm_entry.get().strip()
        if not query or self._llm_thinking:
            return

        self.llm_entry.delete(0, "end")
        self._llm_thinking = True
        self.llm_send_btn.configure(state="disabled", bg="#111827")
        self.llm_entry.configure(state="disabled")

        self._llm_history.append(("user", query))
        self._llm_redraw()

        def _call():
            try:
                result = self._http_json_post("/rag/chat", {"query": query}, timeout=120.0)
                answer = result.get("answer") or "(no response from model)"
                self.root.after(0, lambda: self._llm_got_response(answer, None))
            except Exception as exc:
                self.root.after(0, lambda: self._llm_got_response(None, str(exc)))

        threading.Thread(target=_call, daemon=True).start()

    def _llm_got_response(self, answer, error):
        self._llm_thinking = False
        if error:
            self._llm_history.append(("error", error))
        else:
            self._llm_history.append(("assistant", answer))
        self._llm_redraw()
        self.llm_send_btn.configure(state="normal", bg="#111827")
        self.llm_entry.configure(state="normal")
        self.llm_entry.focus_set()

    def _llm_redraw(self):
        should_follow = self._is_scrolled_near_bottom(self.llm_chat_text)
        self.llm_chat_text.configure(state="normal")
        self.llm_chat_text.delete("1.0", "end")

        for role, text in self._llm_history:
            if role == "user":
                self.llm_chat_text.insert("end", "\nYOU:  ", "user_label")
                self.llm_chat_text.insert("end", text + "\n", "user_text")
            elif role == "error":
                self.llm_chat_text.insert("end", "\nERROR: ", "error_label")
                self.llm_chat_text.insert("end", text + "\n", "error_text")
            else:
                self.llm_chat_text.insert("end", "\nAURA: ", "aura_label")
                self.llm_chat_text.insert("end", text + "\n", "aura_text")

        if self._llm_thinking:
            self.llm_chat_text.insert("end", "\nAURA: thinking...\n", "thinking")

        if should_follow:
            self.llm_chat_text.see("end")
        self.llm_chat_text.configure(state="disabled")

    # =========================
    # Voice
    # =========================

    def _set_voice_busy(self, busy: bool, status: str = ""):
        self.voice_busy = busy
        self.voice_button_text.set("Listening..." if busy else "Tap Mic")
        if hasattr(self, "voice_button"):
            self.voice_button.configure(text=self.voice_button_text.get())
        self.voice_status_text.set(
            status or ("Listening for your question..." if busy else "Press the mic, speak once, and AURA will respond.")
        )

    def _run_voice_button(self):
        if self.voice_busy:
            return

        self._set_voice_busy(True, "Listening for your question...")
        self.voice_result_text.set(
            "Speak normally. AURA will process after about 1 second of silence."
        )

        def _call():
            try:
                result = self._http_json_post("/voice/listen_once", {}, timeout=180.0)
                self.root.after(0, lambda: self._voice_request_done(result, None))
            except Exception as exc:
                self.root.after(0, lambda: self._voice_request_done(None, str(exc)))

        threading.Thread(target=_call, daemon=True).start()

    def _voice_request_done(self, result, error):
        self._set_voice_busy(False)

        if error:
            self.voice_status_text.set("Voice request failed.")
            self.voice_result_text.set(error)
            return

        transcript = (result or {}).get("transcript", "").strip()
        response_text = (result or {}).get("response_text", "").strip()
        action = (result or {}).get("action", "").strip() or "unknown"

        if transcript:
            self.voice_status_text.set(f"Done ({action}).")
            pieces = [f'Heard: "{transcript}"']
            if response_text:
                pieces.append(f'Result: "{response_text}"')
            self.voice_result_text.set("   ".join(pieces))
        else:
            self.voice_status_text.set("No speech detected.")
            self.voice_result_text.set(
                response_text or "Try again and speak a little closer to the mic."
            )

    # =========================
    # Live feed formatting
    # =========================

    def _format_live_line(self, line: str):
        line = (line or "").strip()
        if not line:
            return None

        if line.startswith("[UI_STATE]"):
            return None
        if line.startswith("=") or line.startswith("-" * 10):
            return None
        if line.startswith("RAW TEXT:"):
            return None

        if line.startswith("CLEANED TEXT:"):
            return "Heard: " + line.split(":", 1)[1].strip()

        if line.startswith("INTENT:"):
            value = line.split(":", 1)[1].strip().upper()
            if value == "LLM":
                return "Mode: CHAT"
            if value == "MOVEMENT":
                return "Mode: MOVE"
            return "Mode: " + value

        if line.startswith("MOVEMENT:"):
            value = line.split(":", 1)[1].strip()
            if value and value != "None":
                return "Move: " + value.upper()
            return None

        if "[VOICE] question received:" in line:
            return "Heard: " + line.split(":", 1)[1].strip()

        if "[VOICE] speaking:" in line:
            return "Speaking..."

        if "[VOICE] answered:" in line:
            return "Done."

        if "[VOICE] button heard:" in line:
            return "Heard: " + line.split(":", 1)[1].strip()

        if "[VOICE] button capture loading model" in line:
            return "Loading speech model..."

        if "[AURA] Listening for command..." in line:
            return "Listening..."

        if "[AURA] No speech heard." in line:
            return "No speech heard."

        if "[AURA] Returning to wake mode." in line or "[AURA] Waiting for wake word..." in line:
            return None

        if "[STARTUP] RAG build worker ready" in line:
            return "RAG ready"
        if "[STARTUP] telemetry agent running" in line:
            return "Telemetry online"
        if "[STARTUP] local device id=" in line:
            return "Device ready"
        if "[STARTUP] idle while website activates raw mode" in line:
            return "Camera idle"
        if "[STARTUP] LLM warmup skipped" in line:
            return "LLM warmup skipped"
        if "[CAMERA] ready" in line:
            return "Camera ready"
        if "[SERIAL] Connect failed:" in line:
            return "ESP32 not connected"

        if "[UI]" in line or "[UI ERROR]" in line:
            return line

        # RAG vectorization progress — [RAG JOB] from rag_manager, [LightRAG] from lightrag_local
        if "[RAG JOB]" in line:
            text = line.split("[RAG JOB]", 1)[-1].strip()
            return f"[RAG] {text}"[:120]

        if "[LightRAG]" in line:
            text = line.split("[LightRAG]", 1)[-1].strip()
            return f"[RAG] {text}"[:120]

        if "error" in line.lower() or "failed" in line.lower():
            return line[:120]

        if (
            "[STARTUP]" in line
            or "[RAG]" in line
            or "[TTS]" in line
            or "[CAMERA]" in line
            or "[COMMAND]" in line
            or "[CHAT]" in line
            or "[VOICE]" in line
        ):
            return line[:120]

        return None
    # =========================
    # Journal reader
    # =========================

    def _start_reader(self):
        thread = threading.Thread(target=self._reader_worker, daemon=True)
        thread.start()

    def _reader_worker(self):
        commands = [
            ["journalctl", "-u", SERVICE_NAME, "-f", "-n", "150", "--no-pager", "-o", "cat"],
            ["journalctl", "-f", "-n", "150", "--no-pager", "-o", "cat"],
        ]

        last_error = None

        for cmd in commands:
            try:
                self.log_queue.put(f"[UI] Log source: {' '.join(cmd)}")
                self.reader_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                if self.reader_process.stdout is None:
                    raise RuntimeError("journalctl stdout unavailable")

                for raw_line in self.reader_process.stdout:
                    if not self.running:
                        break
                    line = raw_line.rstrip("\n")
                    if line.strip():
                        self.log_queue.put(line)

                return

            except Exception as exc:
                last_error = exc
                self.log_queue.put(f"[UI ERROR] Log source failed: {exc}")

        if last_error is not None:
            self.log_queue.put(
                "[UI ERROR] Could not attach to journalctl. "
                "If you launched agent/main.py manually, the UI will not see those logs here."
            )
            

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
        formatted = self._format_live_line(line)
        if not formatted:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        display_line = f"[{timestamp}] {formatted}\n"
        should_follow = self._is_scrolled_near_bottom(self.log_text)

        self.log_text.configure(state="normal")
        self.log_text.insert("end", display_line)

        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            self.log_text.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")

        if should_follow:
            self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # =========================
    # Status
    # =========================

    def _set_status(self, status: str, substatus: str):
        style = STATUS_STYLES.get(status, STATUS_STYLES["READY"])
        self.status_text.set(status)
        self.sub_text.set(substatus or style["sub"])
        if hasattr(self, "status_mini_label"):
            self.status_mini_label.configure(fg=style["fg"])

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
            self._set_status("LISTENING", "Voice active")
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
        if "[rag job]" in lower or "[lightrag]" in lower:
            if "sync complete" in lower or "insert done" in lower or "vector db sync" in lower:
                self._set_status("READY", "RAG build complete")
            else:
                self._set_status("VECTORIZING", clean[:80])
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

    # =========================
    # HTTP helpers
    # =========================

    def _http_json(self, method: str, path: str, timeout: float = 2.0):
        req = request.Request(f"{API_BASE}{path}", method=method.upper())
        req.add_header("Accept", "application/json")
        with request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        return json.loads(data) if data else {}

    def _http_json_post(self, path: str, data: dict, timeout: float = 90.0) -> dict:
        body = json.dumps(data).encode("utf-8")
        req = request.Request(f"{API_BASE}{path}", data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw else {}

    def _http_bytes(self, path: str, timeout: float = 1.6) -> bytes:
        req = request.Request(f"{API_BASE}{path}", method="GET")
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    # =========================
    # Vision mode helpers
    # =========================

    def _set_mode_button_styles(self):
        for mode, btn in self.mode_buttons.items():
            active = mode == self.active_vision_mode
            btn.configure(
                bg="#111827",
                fg="#ffffff" if active else "#ecfeff",
                activebackground="#0f172a",
                activeforeground="#ffffff",
                highlightbackground="#14f195" if active else "#94a3b8",
                highlightcolor="#14f195",
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
            self.detection_text.set("Check live feed for details.")
            return

        self.active_vision_mode = mode
        self._camera_fail_count = 0
        self._vision_poll_counter = 0

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
        self._camera_fail_count = 0
        self.current_frame_image = None
        self.camera_label.configure(image="", text="Camera stopped.")
        self.vision_title_text.set("")
        self.vision_status_text.set("Vision mode closed.")
        self.vision_meta_text.set("")
        self.detection_text.set("No detections yet.")
        self._set_mode_button_styles()
        self._show_home()
        self._set_status("READY", "Returned to home screen")

    # =========================
    # Vision polling
    # =========================

    def _poll_vision(self):
        if self.running and self.ui_mode == "vision" and self.active_vision_mode:
            self._vision_poll_counter += 1
            self._refresh_vision_frame()

            if self._vision_poll_counter % DETECTION_EVERY_N_POLLS == 0:
                self._refresh_detections()

        if self.running:
            self.root.after(FRAME_MS, self._poll_vision)

    def _refresh_vision_frame(self):
        if not self.active_vision_mode:
            return

        try:
            frame_bytes = self._http_bytes(
                f"/camera/frame.jpg?ts={int(datetime.now().timestamp() * 1000)}",
                timeout=1.6,
            )
            image = Image.open(io.BytesIO(frame_bytes))

            max_w = max(640, int(self.root.winfo_screenwidth() * 0.94))
            max_h = max(360, int(self.root.winfo_screenheight() * 0.56))
            image.thumbnail((max_w, max_h))

            photo = ImageTk.PhotoImage(image)
            self.current_frame_image = photo
            self.camera_label.configure(image=photo, text="")
            self._camera_fail_count = 0

        except error.HTTPError as exc:
            self._camera_fail_count += 1
            if self._camera_fail_count >= CAMERA_ERROR_THRESHOLD:
                self.camera_label.configure(
                    image="",
                    text=f"Camera HTTP error: {exc.code}"
                )
                self.current_frame_image = None

        except Exception as exc:
            self._camera_fail_count += 1
            if self._camera_fail_count >= CAMERA_ERROR_THRESHOLD:
                self.camera_label.configure(
                    image="",
                    text=f"Waiting for camera...\n{exc}"
                )
                self.current_frame_image = None
            else:
                # keep last good frame instead of blanking immediately
                self.vision_status_text.set("Camera reconnecting...")

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

    # =========================
    # Shutdown
    # =========================

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
        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


def main():
    root = tk.Tk()
    AuraConsoleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()