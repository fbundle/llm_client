from __future__ import annotations

import json
import os
import selectors
import subprocess
from pathlib import Path

_BINARY = str(Path(__file__).resolve().parent.parent / "vendor" / "Pikafish" / "src" / "pikafish")


class PikaFish:
    """UCI xiangqi engine client using the Pikafish binary."""

    def __init__(self) -> None:
        self._p: subprocess.Popen[bytes] | None = None

    def _ensure_started(self) -> None:
        if self._p is not None:
            return
        self._p = subprocess.Popen(
            [_BINARY],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._send("uci")
        self._read_until(("uciok",))
        self._send("setoption name UCI_ShowWDL value true")
        self._send("isready")
        self._read_until(("readyok",))

    def go(self, fen: str, depth: int) -> tuple[str, str]:
        """Analyze *fen* to *depth* and return ``(move, ponder)``."""
        self._ensure_started()
        assert self._p is not None
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        out = self._read_until(("bestmove",))
        line = out.splitlines()[-1]
        parts = line.split()
        move = parts[1]
        ponder = parts[3] if len(parts) > 3 and parts[2] == "ponder" else ""
        return move, ponder

    def pikafish_go(self, fen: str, depth: int) -> str:
        """Analyze *fen* to *depth* and return a result string."""
        move, ponder = self.go(fen, depth)
        if ponder:
            return f"bestmove {move} ponder {ponder}"
        return move

    def call(self, name: str, args: str) -> str:
        if name == "pikafish_go":
            try:
                kwargs = json.loads(args)
                return self.pikafish_go(**kwargs)
            except Exception as e:
                return str(e)
        else:
            return "tool name not found"

    def finish(self) -> None:
        """Gracefully shut down the engine."""
        if self._p is None:
            return
        self._send("quit")
        self._p.stdin.close()
        self._p.wait(timeout=2)
        self._p = None

    def openai_tools(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "pikafish_go",
                    "description": "Analyze a xiangqi (Chinese chess) position given its FEN and return the best move.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fen": {
                                "type": "string",
                                "description": "FEN string of the xiangqi position.",
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Search depth (higher = stronger but slower).",
                            },
                        },
                        "required": ["fen", "depth"],
                    },
                },
            },
        ]

    def _send(self, line: str) -> None:
        assert self._p is not None and self._p.stdin is not None
        self._p.stdin.write((line.rstrip("\n") + "\n").encode())
        self._p.stdin.flush()

    def _read_until(self, end_markers: tuple[str, ...]) -> str:
        assert self._p is not None and self._p.stdout is not None and self._p.stderr is not None

        out_buf = bytearray()
        err_buf = bytearray()
        markers = tuple(m.encode() for m in end_markers)
        sel = selectors.DefaultSelector()
        sel.register(self._p.stdout, selectors.EVENT_READ)
        sel.register(self._p.stderr, selectors.EVENT_READ)

        try:
            while True:
                events = sel.select(timeout=10.0)
                if not events:
                    code = self._p.poll()
                    if code is not None:
                        raise RuntimeError(f"process terminated (exit_code={code})")
                    continue

                for key, _mask in events:
                    chunk = os.read(key.fileobj.fileno(), 4096)
                    if chunk == b"":
                        code = self._p.poll()
                        raise RuntimeError(f"process terminated (exit_code={code})")

                    if key.fileobj is self._p.stdout:
                        out_buf.extend(chunk)
                    else:
                        err_buf.extend(chunk)

                    for marker in markers:
                        match_end: int | None = None
                        out_end: int

                        if out_buf.startswith(marker):
                            out_end = 0
                            nl = out_buf.find(b"\n")
                            match_end = (nl + 1) if nl != -1 else len(out_buf)
                        else:
                            needle = b"\n" + marker
                            pos = out_buf.find(needle)
                            if pos != -1:
                                out_end = pos + 1
                                nl = out_buf.find(b"\n", out_end)
                                match_end = (nl + 1) if nl != -1 else len(out_buf)

                        if match_end is not None:
                            out = bytes(out_buf[:match_end])
                            del out_buf[:match_end]
                            return out.decode(errors="replace").rstrip("\n")
        finally:
            sel.close()
