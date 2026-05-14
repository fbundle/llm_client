from __future__ import annotations

import os
import selectors
import subprocess
from pathlib import Path

from tools.tool import ChatCompletionFunctionToolParam, Tool, ToolOutput

_BINARY = str(Path(__file__).resolve().parent.parent / "vendor" / "Pikafish" / "src" / "pikafish")


class PikaFish:
    """UCI xiangqi engine client using the Pikafish binary."""

    def __init__(self) -> None:
        self._p: subprocess.Popen[bytes] | None = None

    def _ensure_started(self) -> None:
        if self._p is not None and self._p.poll() is None:
            return
        if self._p is not None:
            self._p = None
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

    def finish(self) -> None:
        if self._p is None:
            return
        self._send("quit")
        self._p.stdin.close()
        self._p.wait(timeout=2)
        self._p = None

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


class PikaFishTool(Tool):
    def __init__(self) -> None:
        self.pikafish = PikaFish()

    def submit_board(self, fen: str, depth: int) -> ToolOutput:
        try:
            move, ponder = self.pikafish.go(fen, depth)
        except Exception as e:
            return ToolOutput(state_change=False, output="", error=str(e))
        if ponder:
            return ToolOutput(state_change=False, output=f"bestmove {move} ponder {ponder}", error="")
        return ToolOutput(state_change=False, output=move, error="")

    def dispatch(self, name: str, kwargs: dict[str, object]) -> ToolOutput:
        if name != "submit_board":
            return ToolOutput(state_change=False, output="", error=f"unknown tool: {name}")
        try:
            return self.submit_board(str(kwargs["fen"]), int(kwargs["depth"]))
        except Exception as e:
            return ToolOutput(state_change=False, output="", error=str(e))

    def tool_schemas(self) -> dict[str, ChatCompletionFunctionToolParam]:
        return {
            "submit_board": {
                "type": "function",
                "function": {
                    "name": "submit_board",
                    "description": (
                        "Submit a xiangqi position in FEN notation to get the engine's best move. "
                        "FEN format: ranks separated by '/', then space, then 'w' or 'b' for side to move. "
                        "Ranks are listed from black's side (rank 9) down to red's side (rank 0). "
                        "Pieces: K/k=king, A/a=advisor, B/b=elephant, N/n=knight, R/r=rook, C/c=cannon, P/p=pawn. "
                        "Uppercase=red, lowercase=black. Digits for consecutive empty squares.\n"
                        "Example starting position: rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fen": {
                                "type": "string",
                                "description": "Xiangqi FEN string, e.g. 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w'",
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
        }
