"""PrefixDict — a bounded dict keyed by token lists with longest-prefix lookup.

Keys are token sequences stored as ``"_".join(tokens)`` strings for efficient
prefix matching.
"""

import time
from typing import TypeVar

V = TypeVar("V")


def _to_key(tokens: list[int]) -> str:
    return "_".join(str(t) for t in tokens)


def _from_key(key: str) -> list[int]:
    return [int(s) for s in key.split("_")]


class PrefixDict[V]:
    """Two operations: push (store) and pop (longest-prefix lookup).

    Capacity-limited — when full, push evicts the oldest entry.
    """

    def __init__(self, capacity: int = 8):
        self._capacity = capacity
        self._data: dict[str, tuple[V, float]] = {}

    def push(self, tokens: list[int], value: V) -> None:
        if len(self._data) >= self._capacity:
            oldest = min(self._data.keys(), key=lambda k: self._data[k][1])
            del self._data[oldest]
        self._data[_to_key(tokens)] = (value, time.monotonic())

    def pop(self, tokens: list[int]) -> V | None:
        """Return the value for the longest key that is a prefix of *tokens*."""
        target = _to_key(tokens)
        best_key = ""
        for key_str in self._data:
            if len(key_str) > len(best_key):
                if target.startswith(key_str + "_") or target == key_str:
                    best_key = key_str
        if best_key:
            return self._data[best_key][0]
        return None

    def __len__(self) -> int:
        return len(self._data)
