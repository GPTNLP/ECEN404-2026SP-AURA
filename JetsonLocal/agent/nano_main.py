#!/usr/bin/env python3
import io
import json
import queue
import re
import subprocess
import os
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
MAX_RAW_LOG_LINES = 600
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
        self._reboot_armed_until = None

        self.ui_mode = "home"
        self.view_mode = "home"
        self.previous_ui_mode = "home"
        self.previous_view_mode = "home"
        self.active_vision_mode = None
        self.voice_should_resume = False
        self.current_frame_image = None

        self.status_text = tk.StringVar(value="BOOTING")
        self.sub_text = tk.StringVar(value=STATUS_STYLES["BOOTING"]["sub"])
        self.banner_text = tk.StringVar(value="AURA")
        self.vision_title_text = tk.StringVar(value="")
        self.vision_status_text = tk.StringVar(value="Waiting for camera...")
        self.vision_meta_text = tk.StringVar(value="")
        self.detection_text = tk.StringVar(value="No detections yet.")
        self.voice_status_text = tk.StringVar(
            value="Press the mic, speak once, and AURA will respond."
        )
        self.voice_result_text = tk.StringVar(
            value="Last voice result will appear here."
        )
        self.voice_busy = False

        self._settings_volume_var = tk.DoubleVar(value=50.0)
        self._settings_brightness_var = tk.DoubleVar(value=100.0)
        self._settings_volume_text = tk.StringVar(value="50%")
        self._settings_brightness_text = tk.StringVar(value="100%")
        self._settings_status_text = tk.StringVar(
            value="Adjust volume and brightness for the touchscreen."
        )
        self._pending_volume_job = None
        self._pending_brightness_job = None

        self._llm_thinking = False
        self._llm_history = []
        self._llm_input_buffer = ""
        self._llm_input_var = tk.StringVar(value="")
        self._osk_shift = False
        self._osk_caps = False
        self._osk_mode = "alpha"

        self._vision_poll_counter = 0
        self._camera_fail_count = 0

        self._rag_dataset_var = tk.StringVar(value="None")
        self._rag_dataset_loaded = False

        self._build_ui()
        self.root.bind("<KeyPress>", self._handle_root_keypress, add="+")
        self.root.after(300, self._best_effort_disable_system_keyboard)
        self._start_reader()
        self._poll_logs()
        self._poll_rag_dataset()
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

        header_inner = tk.Frame(self.header, bg="#0b0f14", height=82)
        header_inner.pack(fill="x")
        header_inner.pack_propagate(False)

        icon_font = tkfont.Font(
            family="Courier", size=max(18, int(sw * 0.0215)), weight="bold"
        )
        settings_icon_font = tkfont.Font(
            family="Courier", size=max(20, int(sw * 0.024)), weight="bold"
        )

        header_actions = tk.Frame(header_inner, bg="#0b0f14")
        header_actions.pack(side="right", padx=(0, 10), pady=10)

        button_common = {
            "bg": "#0b0f14",
            "fg": "#14f195",
            "activebackground": "#052e1c",
            "activeforeground": "#eafff3",
            "relief": "flat",
            "bd": 0,
            "cursor": "hand2",
            "width": 3,
            "height": 1,
            "padx": 8,
            "pady": 10,
            "highlightthickness": 1,
            "highlightbackground": "#14f195",
            "highlightcolor": "#14f195",
        }

        self.settings_button = tk.Button(
            header_actions,
            text="⚙",
            command=self.open_settings,
            font=settings_icon_font,
            **button_common,
        )
        self.settings_button.pack(side="right")

        self.reboot_button = tk.Button(
            header_actions,
            text="↻",
            command=self._tap_reboot,
            font=icon_font,
            **button_common,
        )
        self.reboot_button.pack(side="right", padx=(0, 8))

        self.header_label = tk.Label(
            header_inner,
            textvariable=self.banner_text,
            fg="#14f195",
            bg="#0b0f14",
            font=title_font,
            anchor="center",
        )
        self.header_label.place(relx=0.5, rely=0.5, anchor="center")
        header_actions.lift()

        self.content = tk.Frame(outer, bg="#05070a")
        self.content.pack(fill="both", expand=True)

        self.home_frame = tk.Frame(self.content, bg="#05070a")
        self.vision_frame = tk.Frame(self.content, bg="#05070a")
        self.settings_frame = tk.Frame(self.content, bg="#05070a")

        self._build_home_ui(
            self.home_frame, sub_font, section_font, log_font, button_font
        )
        self._build_vision_ui(
            self.vision_frame, button_font, vision_title_font, vision_info_font
        )
        self._build_settings_ui(
            self.settings_frame, section_font, button_font
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
        bg=BTN_BG,
        active_bg=BTN_ACTIVE_BG,
        active_fg=BTN_ACTIVE_FG,
        padx=20,
        pady=12,
        fill=False,
    ):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=font,
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=active_fg,
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

    def _prevent_system_keyboard_focus(self, event=None):
        self._best_effort_disable_system_keyboard()
        try:
            self.root.after(1, self.root.focus_set)
        except Exception:
            pass
        return "break"

    def _passive_text_click(self, event=None):
        self._best_effort_disable_system_keyboard()
        try:
            self.root.focus_set()
        except Exception:
            pass
        return "break"

    def _make_text_passive(self, widget):
        try:
            widget.configure(takefocus=0, cursor="arrow")
        except Exception:
            pass
        for sequence in ("<FocusIn>", "<Button-1>", "<Double-Button-1>", "<Triple-Button-1>", "<B1-Motion>"):
            try:
                widget.bind(sequence, self._passive_text_click)
            except Exception:
                pass

    def _llm_input_click(self, event=None):
        self._best_effort_disable_system_keyboard()
        try:
            self.root.focus_set()
        except Exception:
            pass
        return "break"

    def _refresh_llm_input_display(self):
        cursor = "▌" if (self.view_mode == "llm" and self.ui_mode == "home" and not self._llm_thinking) else ""
        self._llm_input_var.set(f"{self._llm_input_buffer}{cursor}")

    def _llm_set_input_text(self, text: str):
        self._llm_input_buffer = text or ""
        self._refresh_llm_input_display()

    def _llm_append_input_text(self, text: str):
        if not text:
            return
        self._llm_input_buffer += text
        self._refresh_llm_input_display()

    def _llm_backspace(self):
        if self._llm_input_buffer:
            self._llm_input_buffer = self._llm_input_buffer[:-1]
            self._refresh_llm_input_display()

    def _handle_root_keypress(self, event):
        if self.ui_mode != "home" or self.view_mode != "llm":
            return
        if self._llm_thinking:
            return "break"

        self._best_effort_disable_system_keyboard()

        state = getattr(event, "state", 0) or 0
        ctrl_pressed = bool(state & 0x4)
        keysym = event.keysym or ""
        char = event.char or ""

        if keysym in {"Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Super_L", "Super_R"}:
            return "break"

        if ctrl_pressed:
            low = keysym.lower()
            if low == "v":
                try:
                    clip = self.root.clipboard_get()
                except Exception:
                    clip = ""
                if clip:
                    self._llm_append_input_text(clip)
                return "break"
            if low == "u":
                self._llm_set_input_text("")
                return "break"
            return "break"

        if keysym in {"BackSpace"}:
            self._llm_backspace()
            return "break"
        if keysym in {"Return", "KP_Enter"}:
            self._llm_submit()
            return "break"
        if keysym == "Tab":
            self._llm_append_input_text("    ")
            return "break"
        if keysym == "space" or char == " ":
            self._llm_append_input_text(" ")
            return "break"
        if len(char) == 1 and char.isprintable():
            self._llm_append_input_text(char)
            return "break"

        return "break"

    def _build_home_ui(self, parent, sub_font, section_font, log_font, button_font):
        mode_card = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        mode_card.pack(fill="x", pady=(0, 10))

        tabs_wrap = tk.Frame(mode_card, bg="#0b0f14")
        tabs_wrap.pack(side="left", fill="x", expand=True, padx=12, pady=10)

        self.home_btn = self._outline_button(
            tabs_wrap,
            "HOME",
            lambda: self._switch_view("home"),
            button_font,
            fg="#9effc7",
            highlight=BTN_GREEN,
            padx=10,
            pady=14,
        )
        self.live_btn = self._outline_button(
            tabs_wrap,
            "LIVE",
            lambda: self._switch_view("live"),
            button_font,
            fg="#9effc7",
            highlight=BTN_GREEN,
            padx=10,
            pady=14,
        )
        self.llm_btn = self._outline_button(
            tabs_wrap,
            "LLM CHAT",
            lambda: self._switch_view("llm"),
            button_font,
            fg="#9effc7",
            highlight=BTN_GREEN,
            padx=10,
            pady=14,
        )

        for idx, btn in enumerate((self.home_btn, self.live_btn, self.llm_btn)):
            btn.grid(row=0, column=idx, sticky="nsew", padx=5)
            tabs_wrap.grid_columnconfigure(idx, weight=1)

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

        self.home_dashboard = tk.Frame(parent, bg="#05070a")
        self.live_panel = tk.Frame(parent, bg="#05070a")
        self.llm_panel = tk.Frame(parent, bg="#05070a")

        self.camera_card = tk.Frame(
            self.home_dashboard,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        self.camera_card.pack(fill="x", pady=(0, 10))

        self.camera_home_button = self._outline_button(
            self.camera_card,
            "CAMERA",
            lambda: self.enter_vision_mode("face"),
            button_font,
            fg=BTN_GREEN,
            highlight=BTN_GREEN,
            pady=18,
            fill=True,
        )

        self.voice_card = tk.Frame(
            self.home_dashboard,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        self.voice_card.pack(fill="x", pady=(0, 10))

        voice_top = tk.Frame(self.voice_card, bg="#0b0f14")
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
            self.voice_card,
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
            self.voice_card,
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

        self.home_console_panel = tk.Frame(self.home_dashboard, bg="#05070a")
        self.home_console_panel.pack(fill="both", expand=True)
        self._build_console_panel(self.home_console_panel, section_font, title="LIVE FEED", raw=False)

        self._build_live_panel(self.live_panel, section_font)
        self._build_llm_panel(self.llm_panel, section_font, button_font)

        self._switch_view("home")

    def _build_console_panel(self, parent, section_font, *, title="LIVE FEED", raw=False):
        logs_card = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        logs_card.pack(fill="both", expand=True)

        tk.Label(
            logs_card,
            text=title,
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

        log_body = tk.Frame(logs_card, bg="#0b0f14")
        log_body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        scroll = tk.Scrollbar(
            log_body,
            orient="vertical",
            width=26,
            troughcolor="#05070a",
            activebackground="#14f195",
            bg="#111827",
            highlightthickness=0,
            relief="flat",
        )
        scroll.pack(side="right", fill="y")

        text_widget = tk.Text(
            log_body,
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
            yscrollcommand=scroll.set,
        )
        text_widget.pack(side="left", fill="both", expand=True)
        scroll.config(command=text_widget.yview)
        self._make_text_passive(text_widget)

        if raw:
            self.raw_log_text = text_widget
        else:
            self.log_text = text_widget

    def _build_live_panel(self, parent, section_font):
        self._build_console_panel(parent, section_font, title="LIVE TERMINAL", raw=True)


    def _build_llm_panel(self, parent, section_font, button_font):
        chat_card = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        chat_card.pack(fill="both", expand=True)

        title_row = tk.Frame(chat_card, bg="#0b0f14")
        title_row.pack(fill="x")

        tk.Label(
            title_row,
            text="LLM CHAT",
            fg="#14f195",
            bg="#0b0f14",
            font=section_font,
            anchor="w",
            padx=14,
            pady=10,
        ).pack(side="left")

        ds_info_font = tkfont.Font(
            family="Courier",
            size=max(11, int(self.root.winfo_screenwidth() * 0.014)),
            weight="bold",
        )

        dataset_bubble = tk.Frame(
            title_row,
            bg="#052e1c",
            highlightbackground="#14f195",
            highlightthickness=1,
            padx=12,
            pady=6,
        )
        dataset_bubble.pack(side="right", padx=12, pady=6)

        tk.Label(
            dataset_bubble,
            text="DATASET:",
            fg="#bbf7d0",
            bg="#052e1c",
            font=ds_info_font,
            anchor="w",
        ).pack(side="left")

        self._dataset_label = tk.Label(
            dataset_bubble,
            textvariable=self._rag_dataset_var,
            fg="#14f195",
            bg="#052e1c",
            font=ds_info_font,
            anchor="w",
            padx=6,
        )
        self._dataset_label.pack(side="left")

        chat_history_wrap = tk.Frame(chat_card, bg="#0b0f14")
        chat_history_wrap.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        self.llm_chat_scroll = tk.Scrollbar(
            chat_history_wrap,
            orient="vertical",
            width=26,
            troughcolor="#05070a",
            activebackground="#14f195",
            bg="#111827",
            highlightthickness=0,
            relief="flat",
        )
        self.llm_chat_scroll.pack(side="right", fill="y")

        llm_font = tkfont.Font(
            family="Courier",
            size=max(11, int(self.root.winfo_screenwidth() * 0.015)),
            weight="bold",
        )

        self.llm_chat_text = tk.Text(
            chat_history_wrap,
            bg="#05070a",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief="flat",
            wrap="word",
            font=llm_font,
            padx=12,
            pady=12,
            height=6,
            state="disabled",
            yscrollcommand=self.llm_chat_scroll.set,
        )
        self.llm_chat_text.pack(side="left", fill="both", expand=True)
        self.llm_chat_scroll.config(command=self.llm_chat_text.yview)
        self._make_text_passive(self.llm_chat_text)

        bold_log = tkfont.Font(
            family="Courier",
            size=max(10, int(llm_font.cget("size"))),
            weight="bold",
        )

        self.llm_chat_text.tag_configure("user_label", foreground="#14f195", font=bold_log)
        self.llm_chat_text.tag_configure("user_text", foreground="#ffffff")
        self.llm_chat_text.tag_configure("aura_label", foreground="#86efac", font=bold_log)
        self.llm_chat_text.tag_configure("aura_text", foreground="#d1fae5")
        self.llm_chat_text.tag_configure("thinking", foreground="#64748b")
        self.llm_chat_text.tag_configure("error_label", foreground="#fca5a5", font=bold_log)
        self.llm_chat_text.tag_configure("error_text", foreground="#fca5a5")

        composer_wrap = tk.Frame(
            chat_card,
            bg="#07130b",
            highlightbackground="#14f195",
            highlightthickness=2,
            bd=0,
            padx=4,
            pady=4,
        )
        composer_wrap.pack(fill="x", padx=8, pady=(0, 4))

        entry_shell = tk.Frame(
            composer_wrap,
            bg="#0b1a11",
            highlightthickness=0,
            bd=0,
        )
        entry_shell.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.llm_entry = tk.Label(
            entry_shell,
            textvariable=self._llm_input_var,
            bg="#0b1a11",
            fg="#ffffff",
            relief="flat",
            font=button_font,
            bd=0,
            anchor="w",
            justify="left",
            padx=10,
            pady=12,
        )
        self.llm_entry.pack(fill="x", expand=True)
        for sequence in ("<Button-1>", "<ButtonRelease-1>", "<B1-Motion>", "<FocusIn>"):
            self.llm_entry.bind(sequence, self._llm_input_click)

        self.llm_send_btn = tk.Button(
            composer_wrap,
            text="Ask",
            command=self._llm_submit,
            font=button_font,
            bg="#14f195",
            fg="#04130b",
            activebackground="#22c55e",
            activeforeground="#04130b",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=20,
            pady=10,
            highlightthickness=1,
            highlightbackground="#14f195",
            highlightcolor="#14f195",
        )
        self.llm_send_btn.pack(side="right")

        self._refresh_llm_input_display()

        self._build_osk(chat_card)

    def _build_osk(self, parent):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self._osk_key_font = tkfont.Font(
            family="Courier",
            size=max(7, int(sw * 0.0080)),
            weight="bold",
        )
        self._osk_keyboard_bg = "#06150d"
        self._osk_key_bg = "#0d2618"
        self._osk_key_active = "#14532d"
        self._osk_key_fg = "#d1fae5"
        self._osk_key_border = "#14f195"
        self._osk_key_height = 1

        self._osk_outer = tk.Frame(
            parent,
            bg=self._osk_keyboard_bg,
            pady=1,
            padx=3,
            highlightbackground=self._osk_key_border,
            highlightthickness=1,
        )
        self._osk_outer.pack(fill="x", side="bottom", padx=8, pady=(0, 8))
        self._render_osk()

    def _render_osk(self):
        if not hasattr(self, "_osk_outer"):
            return

        for child in self._osk_outer.winfo_children():
            child.destroy()

        shift = self._osk_shift
        caps = self._osk_caps
        upper = caps ^ shift

        if self._osk_mode == "alpha":
            rows = [
                [("1",1),("2",1),("3",1),("4",1),("5",1),("6",1),("7",1),("8",1),("9",1),("0",1)],
                [(c,1) for c in (list("QWERTYUIOP") if upper else list("qwertyuiop"))],
                [("CAPS",1.6)] + [(c,1) for c in (list("ASDFGHJKL") if upper else list("asdfghjkl"))] + [("⌫",1.6)],
                [("SHIFT",1.9)] + [(c,1) for c in (list("ZXCVBNM") if upper else list("zxcvbnm"))] + [("123",1.9)],
                [("SYM",1.4),(",",1),("SPACE",4.4),(".",1),("/",1),("↩",1.8)],
            ]
        elif self._osk_mode == "num":
            rows = [
                [("1",1),("2",1),("3",1),("4",1),("5",1),("6",1),("7",1),("8",1),("9",1),("0",1)],
                [("-",1),("/",1),(":",1),(";",1),("(",1),(")",1),("$",1),("&",1),("@",1),('"',1)],
                [(".",1),(",",1),("?",1),("!",1),("'",1),("#",1),("%",1),("^",1),("*",1),("⌫",1.5)],
                [("ABC",1.8),("+",1),("=",1),("_",1),("\\",1),("|",1),("~",1),("<",1),(">",1),("SYM",1.8)],
                [("TAB",1.4),("[",1),("SPACE",4.2),("]",1),("↩",1.8)],
            ]
        else:
            rows = [
                [("`",1),("{",1),("}",1),("[",1),("]",1),("(",1),(")",1),("<",1),(">",1),("⌫",1.5)],
                [("+",1),("-",1),("*",1),("/",1),("=",1),("_",1),("|",1),("~",1),("^",1),("%",1)],
                [("#",1),("@",1),("$",1),("&",1),("!",1),("?",1),(":",1),(";",1),('"',1),("'",1)],
                [("ABC",1.8),(".",1),(",",1),("\\",1),("€",1),("£",1),("¥",1),("•",1),("…",1),("123",1.8)],
                [("TAB",1.4),("CLR",1.4),("SPACE",3.8),("↩",1.8)],
            ]

        for row in rows:
            rf = tk.Frame(self._osk_outer, bg=self._osk_keyboard_bg)
            rf.pack(fill="x", pady=0)
            total_weight = sum(weight for _, weight in row)
            col = 0
            for label, weight in row:
                self._make_osk_key(rf, label, col, weight, total_weight)
                col += 1

    def _make_osk_key(self, parent_frame, label, column, weight, total_weight):
        accent = label == "↩"
        toggle_on = (label == "SHIFT" and self._osk_shift) or (label == "CAPS" and self._osk_caps)

        bg = "#14f195" if accent else ("#14532d" if toggle_on else self._osk_key_bg)
        fg = "#04130b" if accent else self._osk_key_fg
        active_bg = "#22c55e" if accent else self._osk_key_active

        parent_frame.grid_columnconfigure(column, weight=int(weight * 100))

        btn = tk.Button(
            parent_frame,
            text=label,
            command=lambda k=label: self._osk_key(k),
            font=self._osk_key_font,
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground="#04130b" if accent else "#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=0,
            pady=2,
            height=self._osk_key_height,
            highlightthickness=1,
            highlightbackground=self._osk_key_border,
            highlightcolor=self._osk_key_border,
        )
        btn.grid(row=0, column=column, sticky="nsew", padx=2, pady=0)

    def _osk_key(self, key: str):
        self._best_effort_disable_system_keyboard()
        try:
            self.root.focus_set()
        except Exception:
            pass

        if self._llm_thinking:
            return

        if key == "⌫":
            self._llm_backspace()
            return
        if key == "SHIFT":
            self._osk_shift = not self._osk_shift
            self._render_osk()
            return
        if key == "CAPS":
            self._osk_caps = not self._osk_caps
            self._render_osk()
            return
        if key == "123":
            self._osk_mode = "num"
            self._osk_shift = False
            self._render_osk()
            return
        if key == "SYM":
            self._osk_mode = "sym"
            self._osk_shift = False
            self._render_osk()
            return
        if key == "ABC":
            self._osk_mode = "alpha"
            self._osk_shift = False
            self._render_osk()
            return
        if key == "CLR":
            self._llm_set_input_text("")
            return
        if key == "SPACE":
            self._llm_append_input_text(" ")
            if self._osk_mode == "alpha" and self._osk_shift and not self._osk_caps:
                self._osk_shift = False
                self._render_osk()
            return
        if key == "TAB":
            self._llm_append_input_text("    ")
            return
        if key == "↩":
            self._llm_submit()
            return

        self._llm_append_input_text(key)
        if self._osk_mode == "alpha" and self._osk_shift and not self._osk_caps:
            self._osk_shift = False
            self._render_osk()

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
            pady=2,
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


    def _build_settings_ui(self, parent, section_font, button_font):
        topbar = tk.Frame(parent, bg="#05070a", height=64)
        topbar.pack(fill="x", pady=(0, 10))
        topbar.pack_propagate(False)

        self.settings_back_button = tk.Button(
            topbar,
            text="← Back",
            command=self.close_settings,
            font=button_font,
            bg="#111827",
            fg="#ecfeff",
            activebackground="#0f172a",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=16,
            pady=10,
            highlightthickness=1,
            highlightbackground="#14f195",
            highlightcolor="#14f195",
        )
        self.settings_back_button.pack(side="left", padx=(0, 6), pady=8)

        settings_title_font = tkfont.Font(
            family="Courier",
            size=max(17, int(self.root.winfo_screenwidth() * 0.024)),
            weight="bold",
        )
        tk.Label(
            topbar,
            text="SETTINGS",
            fg="#14f195",
            bg="#05070a",
            font=settings_title_font,
            anchor="center",
        ).place(relx=0.5, rely=0.5, anchor="center")

        value_font = tkfont.Font(
            family="Courier",
            size=max(13, int(self.root.winfo_screenwidth() * 0.016)),
            weight="bold",
        )
        slider_font = tkfont.Font(
            family="Courier",
            size=max(10, int(self.root.winfo_screenwidth() * 0.013)),
            weight="bold",
        )

        settings_card = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        settings_card.pack(fill="both", expand=True)

        tk.Label(
            settings_card,
            text="DISPLAY & AUDIO",
            fg="#14f195",
            bg="#0b0f14",
            font=section_font,
            anchor="w",
            padx=14,
            pady=12,
        ).pack(fill="x")

        volume_row = tk.Frame(settings_card, bg="#0b0f14")
        volume_row.pack(fill="x", padx=14, pady=(8, 8))

        tk.Label(
            volume_row,
            text="Volume",
            fg="#e2e8f0",
            bg="#0b0f14",
            font=value_font,
            anchor="w",
        ).pack(side="left")

        tk.Label(
            volume_row,
            textvariable=self._settings_volume_text,
            fg="#14f195",
            bg="#052e1c",
            font=value_font,
            anchor="e",
            padx=12,
            pady=4,
            highlightbackground="#14f195",
            highlightthickness=1,
        ).pack(side="right")

        self.volume_scale = tk.Scale(
            settings_card,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self._settings_volume_var,
            command=self._on_volume_change,
            showvalue=False,
            resolution=1,
            troughcolor="#163a28",
            activebackground="#14f195",
            bg="#0b0f14",
            fg="#14f195",
            font=slider_font,
            highlightthickness=0,
            bd=0,
            relief="flat",
            sliderlength=38,
            width=22,
            length=max(520, int(self.root.winfo_screenwidth() * 0.74)),
        )
        self.volume_scale.pack(fill="x", padx=18, pady=(0, 18))

        brightness_row = tk.Frame(settings_card, bg="#0b0f14")
        brightness_row.pack(fill="x", padx=14, pady=(10, 8))

        tk.Label(
            brightness_row,
            text="Brightness",
            fg="#e2e8f0",
            bg="#0b0f14",
            font=value_font,
            anchor="w",
        ).pack(side="left")

        tk.Label(
            brightness_row,
            textvariable=self._settings_brightness_text,
            fg="#14f195",
            bg="#052e1c",
            font=value_font,
            anchor="e",
            padx=12,
            pady=4,
            highlightbackground="#14f195",
            highlightthickness=1,
        ).pack(side="right")

        self.brightness_scale = tk.Scale(
            settings_card,
            from_=10,
            to=100,
            orient="horizontal",
            variable=self._settings_brightness_var,
            command=self._on_brightness_change,
            showvalue=False,
            resolution=1,
            troughcolor="#163a28",
            activebackground="#14f195",
            bg="#0b0f14",
            fg="#14f195",
            font=slider_font,
            highlightthickness=0,
            bd=0,
            relief="flat",
            sliderlength=38,
            width=22,
            length=max(520, int(self.root.winfo_screenwidth() * 0.74)),
        )
        self.brightness_scale.pack(fill="x", padx=18, pady=(0, 12))

    # =========================
    # View switching / actions
    # =========================

    def _show_home(self):
        self.ui_mode = "home"
        self.vision_frame.pack_forget()
        self.settings_frame.pack_forget()
        self.home_frame.pack(fill="both", expand=True)

    def _show_vision(self):
        self.ui_mode = "vision"
        self.settings_frame.pack_forget()
        self.home_frame.pack_forget()
        self.vision_frame.pack(fill="both", expand=True)

    def _show_settings(self):
        self.ui_mode = "settings"
        self.home_frame.pack_forget()
        self.vision_frame.pack_forget()
        self.settings_frame.pack(fill="both", expand=True)

    def _switch_view(self, mode: str):
        self.view_mode = mode
        self.home_dashboard.pack_forget()
        self.live_panel.pack_forget()
        self.llm_panel.pack_forget()

        inactive = {"bg": "#0b0f14", "fg": "#9effc7", "highlightbackground": "#14f195"}
        active = {"bg": "#052e1c", "fg": "#eafff3", "highlightbackground": "#22c55e"}

        self.home_btn.configure(**inactive)
        self.live_btn.configure(**inactive)
        self.llm_btn.configure(**inactive)

        if mode == "llm":
            self.llm_panel.pack(fill="both", expand=True)
            self.llm_btn.configure(**active)
            self.root.focus_set()
            self._refresh_llm_input_display()
            self.root.after(80, self._best_effort_disable_system_keyboard)
        elif mode == "live":
            self.live_panel.pack(fill="both", expand=True)
            self.live_btn.configure(**active)
        else:
            self.home_dashboard.pack(fill="both", expand=True)
            self.home_btn.configure(**active)

    def _refresh_ui_data(self):
        self._poll_rag_dataset()
        self._poll_vision()
        self._best_effort_disable_system_keyboard()

    def open_settings(self):
        self.previous_ui_mode = self.ui_mode
        self.previous_view_mode = self.view_mode
        self._load_settings_state()
        self._show_settings()
        self._set_status("READY", "Settings open")

    def close_settings(self):
        if self.previous_ui_mode == "vision" and self.active_vision_mode:
            self._show_vision()
            self._set_status("VISION", f"{VISION_MODES[self.active_vision_mode]['button']} active")
            return

        self._show_home()
        self._switch_view(self.previous_view_mode or "home")
        self._set_status("READY", "Returned from settings")

    def _tap_reboot(self):
        now_ms = int(datetime.now().timestamp() * 1000)
        if self._reboot_armed_until and now_ms <= self._reboot_armed_until:
            self._reboot_now()
            return

        self._reboot_armed_until = now_ms + 4000
        self.reboot_button.configure(text="↻!")

        def _clear_reboot_arm():
            current_ms = int(datetime.now().timestamp() * 1000)
            if self._reboot_armed_until and current_ms > self._reboot_armed_until:
                self._reboot_armed_until = None
                self.reboot_button.configure(text="↻")

        self.root.after(4100, _clear_reboot_arm)

    def _reboot_now(self):
        self.reboot_button.configure(text="...")
        def _worker():
            for cmd in (["systemctl", "reboot"], ["sudo", "systemctl", "reboot"], ["reboot"]):
                try:
                    subprocess.Popen(cmd)
                    return
                except Exception:
                    continue
        threading.Thread(target=_worker, daemon=True).start()

    def _best_effort_disable_system_keyboard(self):
        commands = [
            ["gsettings", "set", "org.gnome.desktop.a11y.applications", "screen-keyboard-enabled", "false"],
            ["gsettings", "set", "org.gnome.desktop.a11y.keyboard", "enable", "false"],
            ["pkill", "-f", "onboard"],
            ["pkill", "-f", "maliit"],
            ["pkill", "-f", "caribou"],
        ]
        for cmd in commands:
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1)
            except Exception:
                pass


    def _load_settings_state(self):
        volume = self._get_system_volume()
        if volume is not None:
            self._settings_volume_var.set(volume)
            self._settings_volume_text.set(f"{int(volume)}%")
        else:
            self._settings_volume_text.set(f"{int(self._settings_volume_var.get())}%")

        brightness = self._get_system_brightness()
        if brightness is not None:
            self._settings_brightness_var.set(brightness)
            self._settings_brightness_text.set(f"{int(brightness)}%")
        else:
            self._settings_brightness_text.set(f"{int(self._settings_brightness_var.get())}%")

        self._settings_status_text.set("Adjust volume and brightness for the touchscreen.")

    def _on_volume_change(self, value):
        try:
            percent = max(0, min(100, int(float(value))))
        except Exception:
            percent = int(self._settings_volume_var.get())
        self._settings_volume_text.set(f"{percent}%")

        if self._pending_volume_job is not None:
            try:
                self.root.after_cancel(self._pending_volume_job)
            except Exception:
                pass
        self._pending_volume_job = self.root.after(120, lambda p=percent: self._apply_volume(p))

    def _on_brightness_change(self, value):
        try:
            percent = max(10, min(100, int(float(value))))
        except Exception:
            percent = int(self._settings_brightness_var.get())
        self._settings_brightness_text.set(f"{percent}%")

        if self._pending_brightness_job is not None:
            try:
                self.root.after_cancel(self._pending_brightness_job)
            except Exception:
                pass
        self._pending_brightness_job = self.root.after(120, lambda p=percent: self._apply_brightness(p))

    def _apply_volume(self, percent: int):
        self._pending_volume_job = None
        if self._best_effort_set_volume(percent):
            self._settings_status_text.set(f"Volume set to {percent}%.")
        else:
            self._settings_status_text.set("Could not change volume on this device.")

    def _apply_brightness(self, percent: int):
        self._pending_brightness_job = None
        if self._best_effort_set_brightness(percent):
            self._settings_status_text.set(f"Brightness set to {percent}%.")
        else:
            self._settings_status_text.set("Could not change brightness on this device.")

    def _get_system_volume(self):
        commands = [
            ["amixer", "get", "Master"],
            ["amixer", "-D", "pulse", "get", "Master"],
            ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
        ]
        for cmd in commands:
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=2,
                )
                match = re.search(r"(\d{1,3})%", proc.stdout or "")
                if match:
                    return max(0, min(100, int(match.group(1))))
            except Exception:
                continue
        return None

    def _best_effort_set_volume(self, percent: int) -> bool:
        commands = [
            ["amixer", "sset", "Master", f"{percent}%"],
            ["amixer", "-D", "pulse", "sset", "Master", f"{percent}%"],
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"],
        ]
        for cmd in commands:
            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                    check=False,
                )
                return True
            except Exception:
                continue
        return False

    def _get_system_brightness(self):
        try:
            backlight_root = "/sys/class/backlight"
            if os.path.isdir(backlight_root):
                for name in os.listdir(backlight_root):
                    base = os.path.join(backlight_root, name)
                    try:
                        with open(os.path.join(base, "brightness"), "r", encoding="utf-8") as f:
                            cur = int(f.read().strip())
                        with open(os.path.join(base, "max_brightness"), "r", encoding="utf-8") as f:
                            max_val = int(f.read().strip())
                        if max_val > 0:
                            return max(1, min(100, int(round((cur / max_val) * 100))))
                    except Exception:
                        continue
        except Exception:
            pass
        return None

    def _best_effort_set_brightness(self, percent: int) -> bool:
        percent = max(10, min(100, int(percent)))

        try:
            backlight_root = "/sys/class/backlight"
            if os.path.isdir(backlight_root):
                for name in os.listdir(backlight_root):
                    base = os.path.join(backlight_root, name)
                    try:
                        with open(os.path.join(base, "max_brightness"), "r", encoding="utf-8") as f:
                            max_val = int(f.read().strip())
                        target = max(1, int(round((percent / 100.0) * max_val)))
                        with open(os.path.join(base, "brightness"), "w", encoding="utf-8") as f:
                            f.write(str(target))
                        return True
                    except Exception:
                        continue
        except Exception:
            pass

        try:
            proc = subprocess.run(
                ["xrandr", "--current"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
            for line in (proc.stdout or "").splitlines():
                if " connected" in line:
                    output_name = line.split()[0]
                    brightness_value = max(0.10, min(1.0, percent / 100.0))
                    subprocess.run(
                        ["xrandr", "--output", output_name, "--brightness", f"{brightness_value:.2f}"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=2,
                        check=False,
                    )
                    return True
        except Exception:
            pass

        return False

    # =========================
    # LLM chat
    # =========================

    def _llm_submit(self):
        self._best_effort_disable_system_keyboard()
        query = self._llm_input_buffer.strip()
        if not query or self._llm_thinking:
            return

        self._llm_set_input_text("")
        self._llm_thinking = True
        self.llm_send_btn.configure(state="disabled", bg="#0d2618")
        self._refresh_llm_input_display()

        self._llm_history.append(("user", query))
        # Reserve an assistant slot that we'll fill progressively via streaming
        self._llm_history.append(("assistant", ""))
        self._llm_redraw()

        def _call():
            try:
                # SSE streaming: update the panel sentence-by-sentence so the
                # touchscreen sees tokens appear rather than waiting for the full answer.
                req = request.Request(
                    f"{API_BASE}/rag/chat?stream=true",
                    data=json.dumps({"query": query}).encode(),
                    method="POST",
                )
                req.add_header("Content-Type", "application/json")
                req.add_header("Accept", "text/event-stream")

                accumulated = ""
                with request.urlopen(req, timeout=120.0) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        payload_str = line[5:].strip()
                        try:
                            obj = json.loads(payload_str)
                        except Exception:
                            continue
                        if obj.get("done"):
                            break
                        chunk = obj.get("text", "")
                        if chunk:
                            accumulated += chunk
                            _acc = accumulated
                            self.root.after(0, lambda a=_acc: self._llm_stream_chunk(a))

                self.root.after(0, lambda: self._llm_got_response(accumulated or "(no response)", None))
            except Exception as exc:
                self.root.after(0, lambda: self._llm_got_response(None, str(exc)))

        threading.Thread(target=_call, daemon=True).start()

    def _llm_stream_chunk(self, accumulated: str):
        """Replace the last assistant entry with the growing streamed answer."""
        if self._llm_history and self._llm_history[-1][0] == "assistant":
            self._llm_history[-1] = ("assistant", accumulated)
            self._llm_redraw()

    def _llm_got_response(self, answer, err):
        self._llm_thinking = False
        if err:
            if self._llm_history and self._llm_history[-1][0] == "assistant":
                self._llm_history[-1] = ("error", err)
            else:
                self._llm_history.append(("error", err))
        else:
            if self._llm_history and self._llm_history[-1][0] == "assistant":
                self._llm_history[-1] = ("assistant", answer)
            else:
                self._llm_history.append(("assistant", answer))
        self._llm_redraw()
        self.llm_send_btn.configure(state="normal", bg="#14f195")
        self.root.focus_set()
        self._refresh_llm_input_display()
        self.root.after(40, self._best_effort_disable_system_keyboard)

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
    # RAG dataset polling
    # =========================

    def _poll_rag_dataset(self):
        def _fetch():
            try:
                result = self._http_json("GET", "/rag/stats", timeout=2.0)
                db_name = result.get("active_db_name") or None
                ready = bool(result.get("ready"))
                self.root.after(0, lambda: self._update_dataset_label(db_name, ready))
            except Exception:
                self.root.after(0, lambda: self._update_dataset_label(None, False))

        threading.Thread(target=_fetch, daemon=True).start()
        if self.running:
            self.root.after(3000, self._poll_rag_dataset)

    def _update_dataset_label(self, db_name: str, ready: bool):
        if db_name and ready:
            self._rag_dataset_var.set(db_name)
            self._rag_dataset_loaded = True
            if hasattr(self, "_dataset_label"):
                self._dataset_label.configure(fg="#14f195")
        else:
            self._rag_dataset_var.set("None")
            self._rag_dataset_loaded = False
            if hasattr(self, "_dataset_label"):
                self._dataset_label.configure(fg="#86efac")

    # =========================
    # Voice
    # =========================

    def _set_voice_busy(self, busy: bool, status: str = ""):
        self.voice_busy = busy
        if hasattr(self, "voice_button"):
            self.voice_button.configure(text="Listening..." if busy else "Tap Mic")
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

    def _voice_request_done(self, result, err):
        self._set_voice_busy(False)

        if err:
            self.voice_status_text.set("Voice request failed.")
            self.voice_result_text.set(err)
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
        threading.Thread(target=self._reader_worker, daemon=True).start()

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
            self._append_raw_log(line)
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

    def _append_raw_log(self, line: str):
        if not hasattr(self, "raw_log_text"):
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        should_follow = self._is_scrolled_near_bottom(self.raw_log_text)
        self.raw_log_text.configure(state="normal")
        self.raw_log_text.insert("end", f"[{timestamp}] {line.rstrip()}\n")

        line_count = int(self.raw_log_text.index("end-1c").split(".")[0])
        if line_count > MAX_RAW_LOG_LINES:
            self.raw_log_text.delete("1.0", f"{line_count - MAX_RAW_LOG_LINES}.0")

        if should_follow:
            self.raw_log_text.see("end")
        self.raw_log_text.configure(state="disabled")

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
        return re.sub(r"\s+", " ", line).strip()[:150]

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
        if hasattr(self, "camera_label"):
            self.camera_label.configure(image="", text="Camera stopped.")
        self.vision_title_text.set("")
        self.vision_status_text.set("Vision mode closed.")
        self.vision_meta_text.set("")
        self.detection_text.set("No detections yet.")
        if hasattr(self, "mode_buttons"):
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
                self.camera_label.configure(image="", text=f"Camera HTTP error: {exc.code}")
                self.current_frame_image = None

        except Exception as exc:
            self._camera_fail_count += 1
            if self._camera_fail_count >= CAMERA_ERROR_THRESHOLD:
                self.camera_label.configure(image="", text=f"Waiting for camera...\n{exc}")
                self.current_frame_image = None
            else:
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
