#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN="$ROOT/bin"
NAME="LLM Agent"

mkdir -p "$BIN"

case "$(uname -s)" in
    Darwin)
        echo "[build] targeting macOS .app bundle"
        pyinstaller \
            --onefile \
            --windowed \
            --name "$NAME" \
            --icon "$ROOT/assets/icon.icns" \
            --add-data "$ROOT/vendor:vendor" \
            --distpath "$BIN" \
            "$ROOT/llm_client_window/window.py"
        ;;
    Linux|*)
        echo "[build] targeting Linux binary"
        pyinstaller \
            --onefile \
            --windowed \
            --name "$NAME" \
            --add-data "$ROOT/vendor:vendor" \
            --distpath "$BIN" \
            "$ROOT/llm_client_window/window.py"
        ;;
esac

echo "[build] done → $BIN"
