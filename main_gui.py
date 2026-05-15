from __future__ import annotations

import base64
import json
import os
import threading
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image as PIL_Image

from PySide6.QtCore import (
    QEvent,
    QThread,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QPixmap,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from llm_client import (
    LLMClient,
    PROMPT_GENERATE_META,
    PROMPT_TIER1_EXPLICIT,
    PROMPT_TIER2_GUIDED,
    PROMPT_TIER3_CONCISE,
    PROMPT_TIER4_MINIMAL,
    SYSTEM_PROMPT,
    ToolList, discover_tools,
)
from tools.screen import get_screenshot

_ENV_PATH = Path(__file__).resolve().parent / ".env"

_PROMPTS: dict[str, str] = {
    "tier1_explicit": PROMPT_TIER1_EXPLICIT,
    "tier2_guided": PROMPT_TIER2_GUIDED,
    "tier3_concise": PROMPT_TIER3_CONCISE,
    "tier4_minimal": PROMPT_TIER4_MINIMAL,
}

_TOOL_LABELS: dict[str, str] = {
    "mouse": "Mouse",
    "keyboard": "Keyboard",
    "pikafish": "Pikafish",
    "js_runtime": "JS Runtime",
    "screen": "Screen",
}


# ------------------------------------------------------------------
# .env read-only (never writes)
# ------------------------------------------------------------------

def _read_env() -> dict[str, str]:
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
# Log formatter — converts tag/text to QTextCharFormat
# ------------------------------------------------------------------

_TAG_COLORS: dict[str, QColor] = {
    "reasoning":   QColor("#888888"),
    "tool_call":   QColor("#3d8fd9"),
    "tool_result": QColor("#4a9a4a"),
    "error":       QColor("#d04040"),
    "info":        QColor("#999999"),
}


def _format_for_tag(tag: str, text: str) -> tuple[str, QTextCharFormat]:
    fmt = QTextCharFormat()
    if tag in _TAG_COLORS:
        fmt.setForeground(_TAG_COLORS[tag])
    return text, fmt


# ------------------------------------------------------------------
# Agent worker thread
# ------------------------------------------------------------------

class AgentThread(QThread):
    log_signal = Signal(str, str)        # tag, text
    screenshot_signal = Signal(str)      # data_url
    finished_signal = Signal(int)        # run_id

    def __init__(
        self,
        app: LLMClient,
        run_id: int,
        user_message: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._app = app
        self._run_id = run_id
        self._user_message = user_message
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            cb = _ThreadCallbacks(self)
            self._app.append_user_message_and_generate(self._user_message, cb)
            self.finished_signal.emit(self._run_id)
        except Exception as e:
            self.log_signal.emit("error", f"\nFatal: {e}\n")
            self.finished_signal.emit(self._run_id)


class _ThreadCallbacks:
    def __init__(self, thread: AgentThread) -> None:
        self._t = thread

    def on_extra_content(self, data: str) -> None:
        self._t.log_signal.emit("reasoning", f"\n[extra_content: {data}]")

    def on_reasoning(self, token: str) -> None:
        self._t.log_signal.emit("reasoning", token)

    def on_content(self, token: str) -> None:
        self._t.log_signal.emit("content", token)

    def on_tool_call(self, name: str, kwargs_str: str) -> None:
        self._t.log_signal.emit("tool_call", f"\n[{name}({kwargs_str})]")

    def on_tool_result(self, output: str) -> None:
        self._t.log_signal.emit("tool_result", f"  -> {output}")

    def on_tool_error(self, error: str) -> None:
        self._t.log_signal.emit("error", f"\n  ! {error}")

    def on_screenshot(self, data_url: str) -> None:
        self._t.screenshot_signal.emit(data_url)

    def is_stopped(self) -> bool:
        return self._t._stop_event.is_set()


# ------------------------------------------------------------------
# Generate prompt worker thread
# ------------------------------------------------------------------

class GenerateThread(QThread):
    log_signal = Signal(str, str)
    result_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, app: LLMClient, meta_prompt: str, system_prompt: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._app = app
        self._meta_prompt = meta_prompt
        self._system_prompt = system_prompt
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            cb = _ThreadCallbacks(self)
            self.log_signal.emit("info", "\n[auto-generate]\n")
            result = self._app.generate_prompt(self._meta_prompt, self._system_prompt, cb)
            self.result_signal.emit(result)
            self.log_signal.emit("info", "\n")
        except Exception as e:
            self.error_signal.emit(str(e))


# ------------------------------------------------------------------
# Main window
# ------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LLM Desktop Agent")
        self.resize(1200, 750)

        self._env = _read_env()
        tools_dir = Path(__file__).resolve().parent / "tools"
        self._all_tools = discover_tools(tools_dir)
        self._app = LLMClient(
            base_url=self._env.get("OPENAI_BASE_URL", ""),
            api_key=self._env.get("OPENAI_API_KEY", ""),
            model=self._env.get("OPENAI_MODEL", ""),
            tool=ToolList(*self._all_tools.values()),
            temperature=float(self._env.get("OPENAI_TEMPERATURE", "0.7")),
            top_p=float(self._env.get("OPENAI_TOP_P", "1.0")),
            max_tokens=int(self._env.get("OPENAI_MAX_TOKENS", "4096")),
        )
        self._app.append_system_message(SYSTEM_PROMPT)
        self._tool_checkboxes: dict[str, QCheckBox] = {}

        self._agent_thread: AgentThread | None = None
        self._gen_thread: GenerateThread | None = None
        self._run_id = 0

        self._build_ui()
        self._apply_style()
        self._refresh_screenshot()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Central widget: screenshot (top) | log + controls (bottom)
        central_splitter = QSplitter(Qt.Orientation.Vertical)
        self.setCentralWidget(central_splitter)

        # Screenshot
        self._scr_label = QLabel()
        self._scr_label.setObjectName("scrLabel")
        self._scr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scr_label.setMinimumSize(200, 200)
        central_splitter.addWidget(self._scr_label)

        # Bottom: log + controls
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)))
        bottom_layout.addWidget(self._log)

        # Control bar
        control = QWidget()
        control_layout = QHBoxLayout(control)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(6)

        self._task_entry = QLineEdit()
        self._task_entry.setPlaceholderText("Enter task...")
        self._task_entry.returnPressed.connect(self._run_stop)
        control_layout.addWidget(self._task_entry)

        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self._run_stop)
        control_layout.addWidget(self._run_btn)

        bottom_layout.addWidget(control)
        central_splitter.addWidget(bottom)
        central_splitter.setStretchFactor(0, 2)
        central_splitter.setStretchFactor(1, 1)

        # Left dock — Environment
        self._left_dock = QDockWidget("Environment")
        self._left_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self._left_dock.setWidget(self._build_left_panel())
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._left_dock)

        # Right dock — System Prompt
        self._right_dock = QDockWidget("System Prompt")
        self._right_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self._right_dock.setWidget(self._build_right_panel())
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._right_dock)

        # View menu — toggles docks back after closing
        menu = self.menuBar().addMenu("View")
        menu.addAction(self._left_dock.toggleViewAction())
        menu.addAction(self._right_dock.toggleViewAction())

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # URL
        layout.addWidget(QLabel("BASE_URL"))
        self._url_edit = QLineEdit(self._env.get("OPENAI_BASE_URL", ""))
        self._url_edit.setPlaceholderText("https://api.openai.com/v1")
        layout.addWidget(self._url_edit)

        # Key
        layout.addWidget(QLabel("API_KEY"))
        self._key_edit = QLineEdit(self._env.get("OPENAI_API_KEY", ""))
        self._key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self._key_edit)

        # Model
        layout.addWidget(QLabel("MODEL"))
        self._model_edit = QLineEdit(self._env.get("OPENAI_MODEL", ""))
        layout.addWidget(self._model_edit)

        # Save env to os.environ
        for edit, key in [
            (self._url_edit, "OPENAI_BASE_URL"),
            (self._key_edit, "OPENAI_API_KEY"),
            (self._model_edit, "OPENAI_MODEL"),
        ]:
            edit.editingFinished.connect(
                lambda e=edit, k=key: self._save_env_var(k, e.text()))

        # Sampling
        label = QLabel("Sampling")
        label.setStyleSheet("font-weight: bold; margin-top: 12px;")
        layout.addWidget(label)

        layout.addWidget(QLabel("TEMPERATURE"))
        self._temp_edit = QLineEdit(self._env.get("OPENAI_TEMPERATURE", "0.7"))
        layout.addWidget(self._temp_edit)

        layout.addWidget(QLabel("TOP_P"))
        self._top_p_edit = QLineEdit(self._env.get("OPENAI_TOP_P", "1.0"))
        layout.addWidget(self._top_p_edit)

        layout.addWidget(QLabel("MAX_TOKENS"))
        self._max_tok_edit = QLineEdit(self._env.get("OPENAI_MAX_TOKENS", "4096"))
        layout.addWidget(self._max_tok_edit)

        layout.addStretch()

        clear = QPushButton("Clear History")
        clear.setStyleSheet("color: #e05555;")
        clear.clicked.connect(self._clear_history)
        layout.addWidget(clear)

        return w

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # System prompt editor
        layout.addWidget(QLabel("System Prompt"))
        self._sys_edit = QTextEdit()
        self._sys_edit.setFont(QFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)))
        self._sys_edit.setPlainText(SYSTEM_PROMPT)
        self._sys_edit.setMinimumHeight(150)
        layout.addWidget(self._sys_edit, stretch=1)

        # Preset selector
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(4)
        preset_layout.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("custom")
        for name in _PROMPTS:
            self._preset_combo.addItem(name)
        self._preset_combo.currentTextChanged.connect(self._load_preset)
        preset_layout.addWidget(self._preset_combo)
        layout.addLayout(preset_layout)

        # Auto-generate
        self._gen_btn = QPushButton("Auto Generate Prompt")
        self._gen_btn.clicked.connect(self._auto_generate_prompt)
        layout.addWidget(self._gen_btn)

        # Tools
        layout.addWidget(QLabel("Tools"))
        for key in self._all_tools:
            label = _TOOL_LABELS.get(key, key.replace("_", " ").title())
            cb = QCheckBox(label)
            cb.setChecked(True)
            self._tool_checkboxes[key] = cb
            layout.addWidget(cb)

        return w

    # ------------------------------------------------------------------
    # Styling (dark / light follows system)
    # ------------------------------------------------------------------

    _DARK = """
        QMainWindow { background-color: #1a1a1a; }
        QMenuBar { background-color: #252525; color: #cccccc; border-bottom: 1px solid #333333; }
        QMenuBar::item:selected { background-color: #3a3a3a; }
        QDockWidget { color: #cccccc; }
        QDockWidget::title { background-color: #252525; padding: 6px 8px; border-bottom: 1px solid #333333; }
        QLabel { color: #aaaaaa; font-size: 12px; }
        QLineEdit, QTextEdit, QPlainTextEdit, QComboBox { background-color: #2a2a2a; color: #cccccc; border: 1px solid #3a3a3a; border-radius: 4px; padding: 4px 6px; font-size: 12px; }
        QLineEdit:focus, QTextEdit:focus, QComboBox:focus { border-color: #5ea2e8; }
        QPushButton { background-color: #3a3a3a; color: #cccccc; border: 1px solid #4a4a4a; border-radius: 4px; padding: 6px 14px; font-size: 12px; }
        QPushButton:hover { background-color: #4a4a4a; }
        QPushButton:pressed { background-color: #2a2a2a; }
        QPushButton:disabled { color: #666666; }
        QCheckBox { color: #cccccc; font-size: 12px; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #4a4a4a; border-radius: 3px; background-color: #2a2a2a; }
        QCheckBox::indicator:checked { background-color: #5ea2e8; border-color: #5ea2e8; }
        QComboBox::drop-down { border: none; padding-right: 6px; }
        QComboBox QAbstractItemView { background-color: #2a2a2a; color: #cccccc; selection-background-color: #3a3a3a; border: 1px solid #3a3a3a; }
        QScrollBar:vertical { background-color: #1a1a1a; width: 10px; border: none; }
        QScrollBar::handle:vertical { background-color: #3a3a3a; border-radius: 5px; min-height: 20px; }
        QScrollBar::handle:vertical:hover { background-color: #4a4a4a; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QSplitter::handle { background-color: #333333; height: 2px; }
        QLabel#scrLabel { background-color: #1a1a1a; border: none; }
    """

    _LIGHT = """
        QMainWindow { background-color: #f5f5f5; }
        QMenuBar { background-color: #e8e8e8; color: #333333; border-bottom: 1px solid #d0d0d0; }
        QMenuBar::item:selected { background-color: #d0d0d0; }
        QDockWidget { color: #333333; }
        QDockWidget::title { background-color: #e8e8e8; padding: 6px 8px; border-bottom: 1px solid #d0d0d0; }
        QLabel { color: #555555; font-size: 12px; }
        QLineEdit, QTextEdit, QPlainTextEdit, QComboBox { background-color: #ffffff; color: #333333; border: 1px solid #c0c0c0; border-radius: 4px; padding: 4px 6px; font-size: 12px; }
        QLineEdit:focus, QTextEdit:focus, QComboBox:focus { border-color: #3d8fd9; }
        QPushButton { background-color: #e0e0e0; color: #333333; border: 1px solid #c0c0c0; border-radius: 4px; padding: 6px 14px; font-size: 12px; }
        QPushButton:hover { background-color: #d0d0d0; }
        QPushButton:pressed { background-color: #c0c0c0; }
        QPushButton:disabled { color: #999999; }
        QCheckBox { color: #333333; font-size: 12px; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #b0b0b0; border-radius: 3px; background-color: #ffffff; }
        QCheckBox::indicator:checked { background-color: #3d8fd9; border-color: #3d8fd9; }
        QComboBox::drop-down { border: none; padding-right: 6px; }
        QComboBox QAbstractItemView { background-color: #ffffff; color: #333333; selection-background-color: #d0d0d0; border: 1px solid #c0c0c0; }
        QScrollBar:vertical { background-color: #f0f0f0; width: 10px; border: none; }
        QScrollBar::handle:vertical { background-color: #c0c0c0; border-radius: 5px; min-height: 20px; }
        QScrollBar::handle:vertical:hover { background-color: #a0a0a0; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QSplitter::handle { background-color: #d0d0d0; height: 2px; }
        QLabel#scrLabel { background-color: #e8e8e8; border: none; }
    """

    _applying_style = False

    def _apply_style(self) -> None:
        if MainWindow._applying_style:
            return
        MainWindow._applying_style = True
        try:
            if QApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark:
                self.setStyleSheet(self._DARK)
            else:
                self.setStyleSheet(self._LIGHT)
        finally:
            MainWindow._applying_style = False

    def changeEvent(self, event: object) -> None:
        if event.type() == QEvent.Type.StyleChange:
            self._apply_style()
        super().changeEvent(event)

    # ------------------------------------------------------------------
    # Env
    # ------------------------------------------------------------------

    def _save_env_var(self, key: str, value: str) -> None:
        if value:
            os.environ[key] = value

    # ------------------------------------------------------------------
    # Log / screenshot
    # ------------------------------------------------------------------

    def _append_log(self, tag: str, text: str) -> None:
        text, fmt = _format_for_tag(tag, text)
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text, fmt)
        self._log.setTextCursor(cursor)
        self._log.ensureCursorVisible()

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_last_screenshot"):
            self._display_screenshot(self._last_screenshot)

    def _display_screenshot(self, data_url: str) -> None:
        self._last_screenshot = data_url
        try:
            _header, b64 = data_url.split(",", 1)
            img = QImage.fromData(base64.b64decode(b64))
            pixmap = QPixmap.fromImage(img)
            scaled = pixmap.scaled(
                self._scr_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._scr_label.setPixmap(scaled)
        except Exception:
            pass

    def _refresh_screenshot(self) -> None:
        self._display_screenshot(get_screenshot(format="JPEG", max_size=1024))

    def _clear_history(self) -> None:
        self._log.clear()
        self._app.clear_history()
        self._app.append_system_message(self._sys_edit.toPlainText().strip())

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def _load_preset(self, name: str) -> None:
        if name == "custom":
            return
        text = _PROMPTS.get(name)
        if text is not None:
            self._sys_edit.setPlainText(text)
            self._app.clear_history()
            self._app.append_system_message(text)

    def _auto_generate_prompt(self) -> None:
        current = self._sys_edit.toPlainText().strip()
        if not current:
            return
        meta_prompt = PROMPT_GENERATE_META.format(current=current)
        self._gen_btn.setEnabled(False)
        self._gen_thread = GenerateThread(self._app, meta_prompt, current)
        self._gen_thread.log_signal.connect(self._append_log)
        self._gen_thread.result_signal.connect(self._on_generate_result)
        self._gen_thread.error_signal.connect(
            lambda e: self._append_log("error", f"\nGenerate failed: {e}\n"))
        self._gen_thread.finished.connect(lambda: self._gen_btn.setEnabled(True))
        self._gen_thread.start()

    def _on_generate_result(self, text: str) -> None:
        self._sys_edit.setPlainText(text)
        self._app.clear_history()
        self._app.append_system_message(text)

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------

    def _run_stop(self) -> None:
        if self._agent_thread is not None and self._agent_thread.isRunning():
            self._agent_thread.stop()
            self._append_log("info", "\n[stopped]\n")
            return

        task = self._task_entry.text().strip()
        if not task:
            return
        self._run_id += 1
        self._task_entry.clear()
        self._task_entry.setEnabled(False)
        self._run_btn.setText("Stop")

        self._append_log("info", f"> {task}\n")
        self._refresh_screenshot()

        # Sync state from UI into app
        self._app.set_model(model=self._model_edit.text().strip() or self._app.model)

        checked = [self._all_tools[k] for k, cb in self._tool_checkboxes.items() if cb.isChecked()]
        self._app.set_tool(ToolList(*checked) if checked else None)

        self._agent_thread = AgentThread(
            self._app, self._run_id, task)
        self._agent_thread.log_signal.connect(self._append_log)
        self._agent_thread.screenshot_signal.connect(self._display_screenshot)
        self._agent_thread.finished_signal.connect(self._on_done)
        self._agent_thread.start()

    def _on_done(self, run_id: int) -> None:
        if run_id != self._run_id:
            return
        self._task_entry.setEnabled(True)
        self._run_btn.setText("Run")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    app = QApplication([])
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
