#!/usr/bin/env python3
import re
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import font as tkfont
from datetime import datetime


SERVICE_NAME = "aura-agent.service"
MAX_LOG_LINES = 250


class AuraConsoleApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AURA Console")
        self.root.configure(bg="#05070a")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.on_escape)
        self.root.bind("q", self.on_escape)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.reader_thread = None
        self.reader_process = None
        self.running = True

        self.status_text = tk.StringVar(value="BOOTING")
        self.substatus_text = tk.StringVar(value="Waiting for AURA agent logs...")
        self.clock_text = tk.StringVar(value="")
        self.mode_text = tk.StringVar(value="MODE: INIT")
        self.agent_text = tk.StringVar(value=f"SOURCE: {SERVICE_NAME}")
        self.last_event_text = tk.StringVar(value="LAST EVENT: none")

        self._build_ui()
        self._start_reader()
        self._tick_clock()
        self._poll_logs()

    def _build_ui(self):
        screen_w = self.root.winfo_screenwidth()

        title_font = tkfont.Font(
            family="Courier",
            size=max(22, int(screen_w * 0.022)),
            weight="bold",
        )
        status_font = tkfont.Font(
            family="Courier",
            size=max(36, int(screen_w * 0.04)),
            weight="bold",
        )
        sub_font = tkfont.Font(
            family="Courier",
            size=max(14, int(screen_w * 0.015)),
        )
        meta_font = tkfont.Font(
            family="Courier",
            size=max(12, int(screen_w * 0.012)),
        )
        log_font = tkfont.Font(
            family="Courier",
            size=max(12, int(screen_w * 0.012)),
        )

        outer = tk.Frame(self.root, bg="#05070a")
        outer.pack(fill="both", expand=True, padx=18, pady=18)

        header = tk.Frame(
            outer,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        header.pack(fill="x", pady=(0, 14))

        tk.Label(
            header,
            text="AURA // AUTONOMOUS UNIVERSITY ROBOT ASSISTANT",
            fg="#14f195",
            bg="#0b0f14",
            font=title_font,
            anchor="w",
            padx=18,
            pady=12,
        ).pack(fill="x")

        mid = tk.Frame(outer, bg="#05070a")
        mid.pack(fill="both", expand=False)

        status_card = tk.Frame(
            mid,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        status_card.pack(fill="x", pady=(0, 14))

        tk.Label(
            status_card,
            textvariable=self.status_text,
            fg="#eafff5",
            bg="#0b0f14",
            font=status_font,
        ).pack(fill="x", pady=(16, 6))

        tk.Label(
            status_card,
            textvariable=self.substatus_text,
            fg="#8bb8a7",
            bg="#0b0f14",
            font=sub_font,
        ).pack(fill="x", pady=(0, 14))

        meta = tk.Frame(mid, bg="#05070a")
        meta.pack(fill="x", pady=(0, 14))

        self._meta_box(meta, "STATE", self.mode_text, meta_font).pack(
            side="left", fill="x", expand=True, padx=(0, 7)
        )
        self._meta_box(meta, "TIME", self.clock_text, meta_font).pack(
            side="left", fill="x", expand=True, padx=7
        )
        self._meta_box(meta, "AGENT", self.agent_text, meta_font).pack(
            side="left", fill="x", expand=True, padx=(7, 0)
        )

        event_card = tk.Frame(
            outer,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )
        event_card.pack(fill="x", pady=(0, 14))

        tk.Label(
            event_card,
            textvariable=self.last_event_text,
            fg="#d8fff0",
            bg="#0b0f14",
            font=sub_font,
            anchor="w",
            justify="left",
            padx=16,
            pady=10,
        ).pack(fill="x")

        logs_card = tk.Frame(
            outer,
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
            font=title_font,
            anchor="w",
            padx=16,
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
            padx=16,
            pady=16,
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True)

    def _meta_box(self, parent, title: str, var: tk.StringVar, font_obj):
        box = tk.Frame(
            parent,
            bg="#0b0f14",
            highlightbackground="#14f195",
            highlightthickness=1,
        )

        tk.Label(
            box,
            text=title,
            fg="#14f195",
            bg="#0b0f14",
            font=font_obj,
            anchor="w",
            padx=12,
            pady=10,
        ).pack(fill="x")

        tk.Label(
            box,
            textvariable=var,
            fg="#eafff5",
            bg="#0b0f14",
            font=font_obj,
            anchor="w",
            padx=12,
            pady=10,
        ).pack(fill="x")

        return box

    def _start_reader(self):
        self.reader_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.reader_thread.start()

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

        try:
            if self.reader_process.stdout is None:
                self.log_queue.put("[UI ERROR] journalctl stdout unavailable")
                return

            for raw_line in self.reader_process.stdout:
                if not self.running:
                    break

                line = raw_line.rstrip("\n")
                if not line.strip():
                    continue

                self.log_queue.put(line)

        except Exception as e:
            self.log_queue.put(f"[UI ERROR] Reader crashed: {e}")

    def _tick_clock(self):
        self.clock_text.set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if self.running:
            self.root.after(500, self._tick_clock)

    def _poll_logs(self):
        processed = 0

        while processed < 50:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break

            self._append_log(line)
            self._update_state_from_line(line)
            processed += 1

        if self.running:
            self.root.after(100, self._poll_logs)

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

    def _set_status(self, status: str, substatus: str, mode: str, last_event: str):
        self.status_text.set(status)
        self.substatus_text.set(substatus)
        self.mode_text.set(f"MODE: {mode}")
        self.last_event_text.set(f"LAST EVENT: {last_event}")

    def _clean_event(self, line: str) -> str:
        line = re.sub(r"\s+", " ", line).strip()
        return line[:140]

    def _update_state_from_line(self, line: str):
        lower = line.lower()
        clean = self._clean_event(line)

        if "[startup]" in lower or "telemetry agent running" in lower:
            self._set_status("ONLINE", "AURA services are up", "READY", clean)
            return

        if "[wake check]" in lower or "wake" in lower:
            self._set_status("LISTENING", "Wake word detection active", "VOICE", clean)
            return

        if "listening" in lower and "[voice" in lower:
            self._set_status("LISTENING", "Microphone input in progress", "VOICE", clean)
            return

        if "[voice]" in lower and "answered" in lower:
            self._set_status("SPEAKING", "AURA is replying out loud", "VOICE", clean)
            return

        if "[voice]" in lower or "[tts]" in lower or "tts" in lower:
            self._set_status("VOICE", "Voice pipeline active", "VOICE", clean)
            return

        if "[chat]" in lower and (
            "running rag query" in lower or "received command" in lower
        ):
            self._set_status("THINKING", "Running local RAG/LLM query", "LLM", clean)
            return

        if "[chat]" in lower and (
            "raw answer" in lower or "ack sent successfully" in lower
        ):
            self._set_status("READY", "Response completed", "LLM", clean)
            return

        if "[jetson db]" in lower and (
            "loading" in lower
            or "loaded" in lower
            or "vector" in lower
            or "build" in lower
            or "chunk" in lower
            or "delete" in lower
            or "database" in lower
        ):
            if "loaded" in lower or "deleted" in lower:
                self._set_status("READY", "Database operation completed", "DATABASE", clean)
            else:
                self._set_status("VECTORIZING", "Working on local database files", "DATABASE", clean)
            return

        if "[command]" in lower:
            self._set_status("COMMAND", "Processing control command", "CONTROL", clean)
            return

        if "[status] ok" in lower:
            self._set_status("READY", "Systems nominal", "READY", clean)
            return

        if "error" in lower or "failed" in lower or "traceback" in lower:
            self._set_status("ERROR", "Check live console for details", "FAULT", clean)
            return

    def on_escape(self, event=None):
        self.on_close()

    def on_close(self):
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