#!/usr/bin/env python3
"""LLM GUI Client — thin entry point.

Usage:
    uv run python run_app.py
"""

from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication
from llm_client_app.app import MainWindow


def main() -> None:
    load_dotenv()
    app = QApplication([])
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
