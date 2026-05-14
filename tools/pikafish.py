from __future__ import annotations

import json
import os
import selectors
import subprocess
from pathlib import Path

from tools.tool import Tool

_BINARY = str(Path(__file__).resolve().parent.parent / "vendor" / "Pikafish" / "src" / "pikafish")


class PikaFish:
    """UCI xiangqi engine client using the Pikafish binary."""

    def __init__(self) -> None:
        self._p: subprocess.Popen[bytes] | None = None

    def _ensure_started(self) -> None:
        if self._p is not None and self._p.poll() is None:
            return
        if self._p is not None:
            self._p = None  # process died, restart
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
        """Gracefully shut down the engine."""
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


PIECES = {"K", "A", "B", "N", "R", "C", "P", "k", "a", "b", "n", "r", "c", "p"}


def grid_to_fen(grid: list[list[str]], side_to_move: str) -> str:
    """Convert a 10x9 piece grid to xiangqi FEN.

    grid[0] = red's back rank (bottom of board).
    UPPERCASE = red, lowercase = black, "" = empty.
    """
    fen_ranks = []
    for rank in reversed(grid):  # FEN starts from black's side (top)
        empty = 0
        fen_rank = ""
        for piece in rank:
            if piece == "":
                empty += 1
            else:
                if empty:
                    fen_rank += str(empty)
                    empty = 0
                fen_rank += piece
        if empty:
            fen_rank += str(empty)
        fen_ranks.append(fen_rank)
    return "/".join(fen_ranks) + " " + side_to_move


class PikaFishTool(Tool):
    def __init__(self, *args, **kwargs):
        self.pikafish = PikaFish(*args, **kwargs)

    def submit_board(self, grid: list[list[str]], side_to_move: str, depth: int) -> tuple[str, bool]:
        """Convert *grid* to FEN, analyze with Pikafish, return best move and whether to re-screenshot."""
        fen = grid_to_fen(grid, side_to_move)
        move, ponder = self.pikafish.go(fen, depth)
        if ponder:
            return f"bestmove {move} ponder {ponder}", False
        return move, False

    def call(self, name: str, args: str) -> tuple[str, bool]:
        if name == "submit_board":
            try:
                kwargs = json.loads(args)
                grid = [[str(c) for c in row] for row in kwargs["grid"]]
                side = str(kwargs["side_to_move"])
                depth = int(kwargs["depth"])
                return self.submit_board(grid, side, depth)
            except Exception as e:
                return str(e), False
        else:
            return "tool name not found", False

    def openai_tools(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_board",
                    "description": (
                        "Submit the xiangqi board position as a 10x9 grid of piece letters and get the best move. "
                        "grid[0] is red's back rank (bottom), grid[9] is black's back rank (top). "
                        "UPPERCASE = red, lowercase = black. "
                        "Pieces: K=king, A=advisor, B=elephant, N=knight, R=rook, C=cannon, P=pawn. "
                        'Use "" for empty squares.'
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "grid": {
                                "type": "array",
                                "description": "10x9 grid of piece letters. 10 rows (rank 0 to 9 from red's bottom), each row has 9 columns (file a to i from red's left). Each cell is a piece letter (K/A/B/N/R/C/P uppercase=red, lowercase=black) or empty string.",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 9,
                                    "maxItems": 9,
                                },
                                "minItems": 10,
                                "maxItems": 10,
                            },
                            "side_to_move": {
                                "type": "string",
                                "enum": ["w", "b"],
                                "description": '"w" if red to move, "b" if black to move.',
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Search depth (higher = stronger but slower).",
                            },
                        },
                        "required": ["grid", "side_to_move", "depth"],
                    },
                },
            },
        ]
