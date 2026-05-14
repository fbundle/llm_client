from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

@dataclass
class ToolOutput:
    state_change: bool
    output: str
    error: str


class Tool(Protocol):
    def call(self, name: str, args: str) -> ToolOutput: ...
    def openai_tools(self) -> list[dict[str, object]]: ...


class Dispatcher(Tool):
    def __init__(self, *tools: Tool) -> None:
        self._tools = list(tools)
        self._lookup: dict[str, Tool] = {}
        for t in tools:
            for func in t.openai_tools():
                name = func["function"]["name"]
                self._lookup[name] = t

    def call(self, name: str, args: str) -> ToolOutput:
        tool = self._lookup.get(name)
        if tool is None:
            return ToolOutput(state_change=False, output="", error=f"unknown tool: {name}")
        return tool.call(name, args)

    def openai_tools(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for t in self._tools:
            result.extend(t.openai_tools())
        return result
