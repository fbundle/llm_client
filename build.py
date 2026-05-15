"""Build a standalone .app (macOS) or .exe (Windows) with PyInstaller.

Usage:
  python build.py          # detect platform, build
  python build.py --mac    # force macOS build
  python build.py --win    # force Windows build (cross-compile not supported — run on Windows)

Requires: pyinstaller (pip install pyinstaller)
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "LLM Agent"
ICON_MAC = ROOT / "assets" / "icon.icns"
ICON_WIN = ROOT / "assets" / "icon.ico"


def build(icon: Path | None, windowed: bool) -> None:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", NAME,
        "--add-data", f"{ROOT / 'vendor'}:vendor",
    ]
    if windowed:
        cmd.append("--windowed")
    if icon and icon.exists():
        cmd += ["--icon", str(icon)]

    cmd.append(str(ROOT / "main_gui.py"))
    print(f"[build] {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    p = argparse.ArgumentParser(description="Build standalone app with PyInstaller")
    p.add_argument("--mac", action="store_true")
    p.add_argument("--win", action="store_true")
    args = p.parse_args()

    system = platform.system()

    if args.mac or (system == "Darwin" and not args.win):
        print("[build] targeting macOS .app bundle")
        build(icon=ICON_MAC, windowed=True)

    elif args.win or system == "Windows":
        print("[build] targeting Windows .exe")
        build(icon=ICON_WIN, windowed=True)

    else:
        print(f"[build] unknown platform: {system}")


if __name__ == "__main__":
    main()
