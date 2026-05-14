from __future__ import annotations

import base64
import threading
from io import BytesIO
from typing import Callable

from PIL import Image as PIL_Image, ImageTk
import tkinter as tk
from tkinter import ttk

from dotenv import load_dotenv

from app import (
    Callbacks,
    create_client,
    create_dispatcher,
    must_get_env,
    run_task,
)
from tools.screen import get_screenshot


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LLM Desktop Agent")
        self.root.geometry("1200x900")

        self._stop_event = threading.Event()
        self._agent_thread: threading.Thread | None = None
        self._run_id = 0  # generation counter to reject stale _on_done calls

        self._pending_logs: list[tuple[str, str]] = []
        self._pending_screenshot: str | None = None
        self._lock = threading.Lock()

        self._photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._flush_updates()
        self._refresh_screenshot()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(0, weight=1)

        # Screenshot
        self._scr_frame = ttk.Frame(self.root)
        self._scr_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 2))
        self._scr_frame.grid_rowconfigure(0, weight=1)
        self._scr_frame.grid_columnconfigure(0, weight=1)

        self._scr_label = ttk.Label(self._scr_frame, background="#1e1e1e")
        self._scr_label.grid(row=0, column=0, sticky="nsew")

        # Bottom: log + controls
        bottom = ttk.Frame(self.root)
        bottom.grid(row=1, column=0, sticky="ew", padx=4, pady=(2, 4))
        bottom.grid_rowconfigure(0, weight=1)
        bottom.grid_columnconfigure(0, weight=1)

        self._log = tk.Text(bottom, height=8, wrap=tk.WORD, state=tk.DISABLED, font=("SF Mono", 11))
        self._log.grid(row=0, column=0, sticky="nsew")

        self._log.tag_configure("reasoning", foreground="gray")
        self._log.tag_configure("content", foreground="black")
        self._log.tag_configure("tool_call", foreground="#0066cc")
        self._log.tag_configure("tool_result", foreground="#008800")
        self._log.tag_configure("error", foreground="#cc0000")
        self._log.tag_configure("info", foreground="#888888")

        scrollbar = ttk.Scrollbar(bottom, command=self._log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._log.configure(yscrollcommand=scrollbar.set)

        # Control bar
        control = ttk.Frame(bottom)
        control.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        control.grid_columnconfigure(0, weight=1)

        self._task_entry = ttk.Entry(control, font=("SF Mono", 12))
        self._task_entry.grid(row=0, column=0, sticky="ew")
        self._task_entry.bind("<Return>", lambda _e: self._run())

        self._run_btn = ttk.Button(control, text="Run", command=self._run)
        self._run_btn.grid(row=0, column=1, padx=(4, 0))

        self._stop_btn = ttk.Button(control, text="Stop", command=self._stop, state=tk.DISABLED)
        self._stop_btn.grid(row=0, column=2, padx=(4, 0))

    # ------------------------------------------------------------------
    # Thread-safe UI helpers
    # ------------------------------------------------------------------

    def _flush_updates(self) -> None:
        with self._lock:
            logs = self._pending_logs[:]
            self._pending_logs.clear()
            screenshot = self._pending_screenshot
            self._pending_screenshot = None

        if logs:
            self._log.configure(state=tk.NORMAL)
            for tag, text in logs:
                self._log.insert(tk.END, text, tag)
            self._log.see(tk.END)
            self._log.configure(state=tk.DISABLED)

        if screenshot is not None:
            self._display_screenshot(screenshot)

        self.root.after(50, self._flush_updates)

    def _push_log(self, tag: str, text: str) -> None:
        with self._lock:
            self._pending_logs.append((tag, text))

    def _push_screenshot(self, data_url: str) -> None:
        with self._lock:
            self._pending_screenshot = data_url

    def _refresh_screenshot(self) -> None:
        self._display_screenshot(get_screenshot(format="JPEG", max_size=1024))

    def _display_screenshot(self, data_url: str) -> None:
        try:
            _header, b64 = data_url.split(",", 1)
            img = PIL_Image.open(BytesIO(base64.b64decode(b64)))

            max_w = self._scr_frame.winfo_width()
            if max_w > 10:
                img.thumbnail((max_w, 99999), PIL_Image.Resampling.LANCZOS)

            self._photo = ImageTk.PhotoImage(img)
            self._scr_label.configure(image=self._photo)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        task = self._task_entry.get().strip()
        if not task:
            return
        # Interrupt any running task
        if self._agent_thread is not None and self._agent_thread.is_alive():
            self._stop_event.set()
            self._agent_thread.join(timeout=2)
        self._run_id += 1
        self._task_entry.delete(0, tk.END)
        self._stop_event.clear()
        self._run_btn.configure(state=tk.DISABLED)
        self._stop_btn.configure(state=tk.NORMAL)

        self._push_log("info", f"> {task}\n")
        self._refresh_screenshot()

        self._agent_thread = threading.Thread(target=self._run_agent, args=(task, self._run_id), daemon=True)
        self._agent_thread.start()

    def _stop(self) -> None:
        self._stop_event.set()
        self._push_log("info", "\n[stopped]\n")
        # Don't call _on_done() — the agent thread will call it when it
        # detects the stop event and exits.

    def _on_done(self, run_id: int) -> None:
        if run_id != self._run_id:
            return  # stale callback from an interrupted task
        self._run_btn.configure(state=tk.NORMAL)
        self._stop_btn.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Agent (runs on background thread)
    # ------------------------------------------------------------------

    def _run_agent(self, task: str, run_id: int) -> None:
        try:
            load_dotenv()
            client = create_client()
            model = must_get_env("OPENAI_MODEL")
            dispatcher = create_dispatcher()
            tools = list(dispatcher.tool_schemas().values())
            cb = _GuiCallbacks(self)
            run_task(client, model, tools, dispatcher, task, cb)
        except Exception as e:
            self._push_log("error", f"\nFatal: {e}\n")
        finally:
            self.root.after(0, lambda: self._on_done(run_id))


class _GuiCallbacks:
    """Bridge between the agent thread and the GUI. All methods are called
    from the agent thread; they marshal updates through the App's pending
    queues so tkinter widgets are only touched on the main thread."""

    def __init__(self, app: App) -> None:
        self._app = app

    def on_extra_content(self, data: str) -> None:
        self._app._push_log("reasoning", f"\n[extra_content: {data}]")

    def on_reasoning(self, token: str) -> None:
        self._app._push_log("reasoning", token)

    def on_content(self, token: str) -> None:
        self._app._push_log("content", token)

    def on_tool_call(self, name: str, kwargs_str: str) -> None:
        self._app._push_log("tool_call", f"\n[{name}({kwargs_str})]")

    def on_tool_result(self, output: str) -> None:
        self._app._push_log("tool_result", f"  -> {output}")

    def on_tool_error(self, error: str) -> None:
        self._app._push_log("error", f"\n  ! {error}")

    def on_screenshot(self, data_url: str) -> None:
        self._app._push_screenshot(data_url)

    def is_stopped(self) -> bool:
        return self._app._stop_event.is_set()


def main() -> None:
    load_dotenv()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
