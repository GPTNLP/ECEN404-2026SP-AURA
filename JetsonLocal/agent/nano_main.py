#!/usr/bin/env python3
import queue
import re
import subprocess
import threading
import tkinter as tk
from tkinter import font as tkfont


SERVICE_NAME = "aura-agent.service"
MAX_LOG_LINES = 220

STATUS_STYLES = {
    "BOOTING": {"fg": "#7dd3fc", "sub": "Starting services"},
    "READY": {"fg": "#a7f3d0", "sub": "Ready"},
    "LISTENING": {"fg": "#67e8f9", "sub": "Listening for wake word"},
    "THINKING": {"fg": "#fcd34d", "sub": "Thinking"},
    "SPEAKING": {"fg": "#c4b5fd", "sub": "Speaking"},
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

        self.state = "BOOTING"

        self.status_text = tk.StringVar(value="BOOTING")
        self.sub_text = tk.StringVar(value=STATUS_STYLES["BOOTING"]["sub"])
        self.banner_text = tk.StringVar(value="AURA")

        self._build_ui()
        self._start_reader()
        self._poll_logs()

    def _build_ui(self):
        sw = self.root.winfo_screenwidth()

        title_font = tkfont.Font(family="Courier", size=max(22, int(sw * 0.030)), weight="bold")
        status_font = tkfont.Font(family="Courier", size=max(28, int(sw * 0.050)), weight="bold")
        sub_font = tkfont.Font(family="Courier", size=max(11, int(sw * 0.016)))
        section_font = tkfont.Font(family="Courier", size=max(17, int(sw * 0.024)), weight="bold")
        log_font = tkfont.Font(family="Courier", size=max(9, int(sw * 0.013)))

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

        self.status_card = tk.Frame(
            outer,
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
            anchor="center",
            pady=12,
        )
        self.status_label.pack(fill="x")

        self.sub_label = tk.Label(
            self.status_card,
            textvariable=self.sub_text,
            fg="#b8c4cf",
            bg="#0b0f14",
            font=sub_font,
            justify="center",
            wraplength=max(220, sw - 80),
            padx=14,
            pady=(0, 12),
        )
        self.sub_label.pack(fill="x")

        self.console_card = tk.Frame(
            outer,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        self.console_card.pack(fill="both", expand=True)

        tk.Label(
            self.console_card,
            text="LIVE CONSOLE",
            fg="#14f195",
            bg="#0b0f14",
            font=section_font,
            anchor="w",
            padx=12,
            pady=10,
        ).pack(fill="x")

        self.log_text = tk.Text(
            self.console_card,
            bg="#05070a",
            fg="#a7f3d0",
            insertbackground="#a7f3d0",
            relief="flat",
            wrap="word",
            font=log_font,
            padx=12,
            pady=12,
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True)

        self._apply_state_style("BOOTING")

    def _start_reader(self):
        threading.Thread(target=self._reader_worker, daemon=True).start()

    def _reader_worker(self):
        cmd = [
            "journalctl",
            "-u",
            SERVICE_NAME,
            "-f",
            "-n",
            "150",
            "--no-pager",
            "-o",
            "cat",
        ]
        try:
            self.reader_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            self.log_queue.put(f"[UI ERROR] Failed to start journal reader: {e}")
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

    def _poll_logs(self):
        processed = 0
        while processed < 50:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break

            self._handle_line(line)
            processed += 1

        if self.running:
            self.root.after(100, self._poll_logs)

    def _handle_line(self, line: str):
        self._append_log(line)

        explicit_state, explicit_detail = self._extract_explicit_state(line)
        if explicit_state:
            self._set_state(explicit_state, explicit_detail or self._friendly_subtext(explicit_state))
            return

        lower = line.lower()
        compact = self._compact(line)

        if "[wake check]" in lower or "[voice] listening" in lower:
            self._set_state("LISTENING", "Listening for wake word")
            return

        if "[voice] question received" in lower or "[chat] running rag query" in lower:
            self._set_state("THINKING", compact)
            return

        if "[voice] speaking" in lower or "[voice] answered" in lower or "[tts]" in lower:
            self._set_state("SPEAKING", compact)
            return

        if "[jetson db]" in lower or "[rag job]" in lower:
            if "failed" in lower:
                self._set_state("ERROR", compact)
            elif "completed" in lower or "loaded" in lower or "deleted" in lower or "ready" in lower:
                self._set_state("READY", compact)
            else:
                self._set_state("VECTORIZING", compact)
            return

        if "[command]" in lower:
            self._set_state("COMMAND", compact)
            return

        if "[status] ok" in lower and self.state in {"BOOTING", "OFFLINE"}:
            self._set_state("READY", "Ready")
            return

        if "traceback" in lower or "failed" in lower or "error" in lower:
            self._set_state("ERROR", compact)

    def _extract_explicit_state(self, line: str):
        match = re.search(r"\[UI_STATE\]\s+([A-Z_]+)(?:\s+\|\s+(.*))?$", line)
        if not match:
            return None, None
        state = match.group(1).strip().upper()
        detail = (match.group(2) or "").strip()
        return state, detail

    def _friendly_subtext(self, state: str) -> str:
        return STATUS_STYLES.get(state, STATUS_STYLES["READY"])["sub"]

    def _compact(self, line: str) -> str:
        line = re.sub(r"\s+", " ", line).strip()
        return line[:110]

    def _set_state(self, state: str, subtitle: str):
        if not subtitle:
            subtitle = self._friendly_subtext(state)

        self.state = state
        self.status_text.set(state)
        self.sub_text.set(subtitle)
        self._apply_state_style(state)

    def _apply_state_style(self, state: str):
        style = STATUS_STYLES.get(state, STATUS_STYLES["READY"])
        self.status_label.configure(fg=style["fg"])
        self.status_card.configure(highlightbackground=style["fg"])
        self.header.configure(highlightbackground=style["fg"])

    def _append_log(self, line: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")

        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            self.log_text.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")

        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def on_close(self, event=None):
        self.running = False
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