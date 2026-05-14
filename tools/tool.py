from __future__ import annotations

from typing import Any, Protocol


class Tool(Protocol):
    def call(self, name: str, args: str) -> tuple[str, bool]: ...
    def openai_tools(self) -> list[dict[str, object]]: ...


class Dispatcher:
    def __init__(self, *tools: Tool) -> None:
        self._tools = list(tools)
        self._lookup: dict[str, Tool] = {}
        for t in tools:
            for func in t.openai_tools():
                name = func["function"]["name"]
                self._lookup[name] = t

    def call(self, name: str, args: str) -> tuple[str, bool]:
        tool = self._lookup.get(name)
        if tool is None:
            return f"unknown tool: {name}", False
        return tool.call(name, args)

    def openai_tools(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for t in self._tools:
            result.extend(t.openai_tools())
        return result
