from __future__ import annotations

import base64
import os
import threading
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image as PIL_Image, ImageTk
import tkinter as tk
from tkinter import ttk

from app import (
    SYSTEM_PROMPT,
    Callbacks,
    create_client,
    create_dispatcher,
    run_task,
)
from tools.js_runtime import JSRuntimeTool
from tools.keyboard import KeyboardTool
from tools.mouse import MouseTool
from tools.pikafish import PikaFishTool
from tools.screen import ScreenTool, get_screenshot

_ENV_PATH = Path(__file__).resolve().parent / ".env"


# ------------------------------------------------------------------
# .env persistence
# ------------------------------------------------------------------

def _read_env() -> dict[str, str]:
    """Return all KEY=VALUE pairs from .env (stripped of quotes)."""
    result: dict[str, str] = {}
    if not _ENV_PATH.exists():
        return result
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        result[k.strip()] = v.strip().strip("\"'")
    return result



# ------------------------------------------------------------------
# GUI
# ------------------------------------------------------------------

class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LLM Desktop Agent")
        self.root.geometry("1000x600")

        self._stop_event = threading.Event()
        self._agent_thread: threading.Thread | None = None
        self._run_id = 0

        self._pending_logs: list[tuple[str, str]] = []
        self._pending_screenshot: str | None = None
        self._lock = threading.Lock()

        self._photo: ImageTk.PhotoImage | None = None
        self._env = _read_env()

        # Create tool instances once — they hold state (JS runtime,
        # chess engine process, etc.)
        self._tool_instances = {
            "mouse": MouseTool(),
            "keyboard": KeyboardTool(),
            "pikafish": PikaFishTool(),
            "js_runtime": JSRuntimeTool(),
            "screen": ScreenTool(),
        }

        self._tool_vars: dict[str, tk.BooleanVar] = {}
        self._build_ui()
        self._flush_updates()
        self._refresh_screenshot()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Red.TButton", foreground="red")
        style.map("Red.TButton", foreground=[("disabled", "gray")])

        # Vertical split: screenshot area (top) | log area (bottom)
        vpane = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        vpane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ----- Top pane: left | screenshot | right -----
        self._hpane = hpane = ttk.PanedWindow(vpane, orient=tk.HORIZONTAL)

        # Left panel — full
        self._left_full = ttk.Frame(hpane, width=220)
        self._build_left_panel(self._left_full)
        hpane.add(self._left_full, weight=0)

        # Left panel — collapsed (button only)
        self._left_collapsed = ttk.Frame(hpane, width=36)
        ttk.Button(self._left_collapsed, text="≡", command=self._toggle_left_panel, width=2).pack(anchor="nw", padx=(4, 0), pady=(4, 0))

        scr = ttk.Frame(hpane)
        self._build_screenshot(scr)
        hpane.add(scr, weight=1)

        # Right panel — full
        self._right_full = ttk.Frame(hpane, width=280)
        self._build_right_panel(self._right_full)
        hpane.add(self._right_full, weight=0)

        # Right panel — collapsed (button only)
        self._right_collapsed = ttk.Frame(hpane, width=36)
        ttk.Button(self._right_collapsed, text="≡", command=self._toggle_right_panel, width=2).pack(anchor="ne", padx=(0, 4), pady=(4, 0))

        self._left_visible = True
        self._right_visible = True

        vpane.add(hpane, weight=1)

        # ----- Bottom pane: log + controls -----
        bottom = ttk.Frame(vpane)
        vpane.add(bottom, weight=1)
        bottom.grid_rowconfigure(0, weight=1)
        bottom.grid_columnconfigure(0, weight=1)

        self._log = tk.Text(bottom, height=8, wrap=tk.WORD, state=tk.DISABLED,
                            font=("SF Mono", 11))
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

        self._task_entry = tk.Text(control, height=3, font=("SF Mono", 12), wrap=tk.WORD)
        self._task_entry.grid(row=0, column=0, sticky="ew")
        self._task_entry.bind("<Return>", lambda _e: self._run() or "break")
        self._task_entry.bind("<Shift-Return>", lambda _e: self._task_entry.insert(tk.INSERT, "\n") or "break")

        self._run_btn = ttk.Button(control, text="Run", command=self._run)
        self._run_btn.grid(row=0, column=1, padx=(4, 0))

        self._stop_btn = ttk.Button(control, text="Stop", command=self._stop, state=tk.DISABLED, style="Red.TButton")
        self._stop_btn.grid(row=0, column=2, padx=(4, 0))

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        # Collapse button — always visible at top
        ttk.Button(parent, text="≡", command=self._toggle_left_panel, width=2).pack(anchor="nw", padx=(4, 0), pady=(4, 0))

        # Content — collapsible (includes title + fields)
        self._left_content = frame = ttk.Frame(parent, padding=4)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.grid_columnconfigure(0, weight=0)

        ttk.Label(frame, text="Environment", font=("SF Mono", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        # BASE_URL
        ttk.Label(frame, text="BASE_URL").grid(row=1, column=0, sticky="e", padx=(0, 4))
        self._url_var = tk.StringVar(value=self._env.get("OPENAI_BASE_URL", ""))
        url_entry = ttk.Entry(frame, textvariable=self._url_var, width=28)
        url_entry.grid(row=1, column=1, sticky="ew", pady=(0, 2))
        url_entry.bind("<FocusOut>", lambda _e: self._save_env_var("OPENAI_BASE_URL", self._url_var.get()))

        # API_KEY (masked)
        ttk.Label(frame, text="API_KEY").grid(row=2, column=0, sticky="e", padx=(0, 4))
        self._key_var = tk.StringVar(value=self._env.get("OPENAI_API_KEY", ""))
        key_entry = ttk.Entry(frame, textvariable=self._key_var, width=28, show="*")
        key_entry.grid(row=2, column=1, sticky="ew", pady=(0, 2))
        key_entry.bind("<FocusOut>", lambda _e: self._save_env_var("OPENAI_API_KEY", self._key_var.get()))

        # MODEL
        ttk.Label(frame, text="MODEL").grid(row=3, column=0, sticky="e", padx=(0, 4))
        self._model_var = tk.StringVar(value=self._env.get("OPENAI_MODEL", ""))
        model_entry = ttk.Entry(frame, textvariable=self._model_var, width=28)
        model_entry.grid(row=3, column=1, sticky="ew", pady=(0, 2))
        model_entry.bind("<FocusOut>", lambda _e: self._save_env_var("OPENAI_MODEL", self._model_var.get()))

        # Clear History button
        ttk.Button(frame, text="Clear History", command=self._clear_history, style="Red.TButton").grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(12, 0))

    def _build_screenshot(self, parent: ttk.Frame) -> None:
        self._scr_frame = ttk.Frame(parent)
        self._scr_frame.pack(fill=tk.BOTH, expand=True, padx=4)
        self._scr_frame.grid_rowconfigure(0, weight=1)
        self._scr_frame.grid_columnconfigure(0, weight=1)

        self._scr_label = ttk.Label(self._scr_frame, background="#1e1e1e")
        self._scr_label.grid(row=0, column=0, sticky="nsew")

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        # Collapse button — always visible at top
        ttk.Button(parent, text="≡", command=self._toggle_right_panel, width=2).pack(anchor="ne", padx=(0, 4), pady=(4, 0))

        # Content — collapsible (includes title + fields)
        self._right_content = frame = ttk.Frame(parent, padding=4)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ttk.Label(frame, text="System Prompt", font=("SF Mono", 11, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 4))

        self._sys_text = tk.Text(frame, width=34, height=14, wrap=tk.WORD,
                                 font=("SF Mono", 10))
        self._sys_text.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        self._sys_text.insert("1.0", SYSTEM_PROMPT)

        # Tools
        ttk.Label(frame, text="Tools", font=("SF Mono", 11, "bold")).grid(
            row=2, column=0, sticky="w", pady=(0, 4))

        tool_names = [
            ("mouse", "Mouse"),
            ("keyboard", "Keyboard"),
            ("pikafish", "Pikafish"),
            ("js_runtime", "JS Runtime"),
            ("screen", "Screen"),
        ]
        for key, label in tool_names:
            var = tk.BooleanVar(value=True)
            self._tool_vars[key] = var
            ttk.Checkbutton(frame, text=label, variable=var).grid(
                row=3 + list(self._tool_vars).index(key), column=0, sticky="w")

    # ------------------------------------------------------------------
    # Sidebar toggle
    # ------------------------------------------------------------------

    def _toggle_left_panel(self) -> None:
        if self._left_visible:
            self._hpane.forget(self._left_full)
            self._hpane.insert(0, self._left_collapsed, weight=0)
            self._left_visible = False
        else:
            self._hpane.forget(self._left_collapsed)
            self._hpane.insert(0, self._left_full, weight=0)
            self._left_visible = True

    def _toggle_right_panel(self) -> None:
        if self._right_visible:
            self._hpane.forget(self._right_full)
            self._hpane.add(self._right_collapsed, weight=0)
            self._right_visible = False
        else:
            self._hpane.forget(self._right_collapsed)
            self._hpane.add(self._right_full, weight=0)
            self._right_visible = True

    # ------------------------------------------------------------------
    # Env persistence
    # ------------------------------------------------------------------

    def _save_env_var(self, key: str, value: str) -> None:
        if value:
            os.environ[key] = value

    # ------------------------------------------------------------------
    # Clear history
    # ------------------------------------------------------------------

    def _clear_history(self) -> None:
        self._log.configure(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.configure(state=tk.DISABLED)

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
        task = self._task_entry.get("1.0", "end-1c").strip()
        if not task:
            return
        if self._agent_thread is not None and self._agent_thread.is_alive():
            self._stop_event.set()
            self._agent_thread.join(timeout=2)
        self._run_id += 1
        self._task_entry.delete("1.0", tk.END)
        self._stop_event.clear()
        self._task_entry.configure(state=tk.DISABLED)
        self._run_btn.configure(state=tk.DISABLED)
        self._stop_btn.configure(state=tk.NORMAL)

        self._push_log("info", f"> {task}\n")
        self._refresh_screenshot()

        self._agent_thread = threading.Thread(
            target=self._run_agent, args=(task, self._run_id), daemon=True)
        self._agent_thread.start()

    def _stop(self) -> None:
        self._stop_event.set()
        self._push_log("info", "\n[stopped]\n")
        self._task_entry.configure(state=tk.NORMAL)
        self._run_btn.configure(state=tk.NORMAL)

    def _on_done(self, run_id: int) -> None:
        if run_id != self._run_id:
            return
        self._task_entry.configure(state=tk.NORMAL)
        self._run_btn.configure(state=tk.NORMAL)
        self._stop_btn.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Agent (runs on background thread)
    # ------------------------------------------------------------------

    def _run_agent(self, task: str, run_id: int) -> None:
        try:
            load_dotenv()
            client = create_client()
            model = os.environ.get("OPENAI_MODEL", "")
            enabled = {k for k, v in self._tool_vars.items() if v.get()}
            dispatcher = create_dispatcher(enabled, instances=self._tool_instances)
            tools = list(dispatcher.tool_schemas().values())
            system_prompt = self._sys_text.get("1.0", tk.END).strip()
            cb = _GuiCallbacks(self)
            run_task(client, model, tools, dispatcher, task, cb, system_prompt=system_prompt)
        except Exception as e:
            self._push_log("error", f"\nFatal: {e}\n")
        finally:
            self.root.after(0, lambda: self._on_done(run_id))


class _GuiCallbacks:
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
