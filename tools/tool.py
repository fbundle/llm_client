from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Protocol

from openai.types.chat import ChatCompletionFunctionToolParam

@dataclass
class ToolOutput:
    state_change: bool
    output: str
    error: str


class Tool(Protocol):
    def dispatch(self, name: str, kwargs: dict[str, object]) -> ToolOutput: ...
    def tool_schemas(self) -> dict[str, ChatCompletionFunctionToolParam]: ...


class ToolList(Tool):
    def __init__(self, *tools: Tool) -> None:
        self._tools = list(tools)
        self._lookup: dict[str, Tool] = {}
        for t in tools:
            for name in t.tool_schemas():
                self._lookup[name] = t

    def dispatch(self, name: str, kwargs: dict[str, object]) -> ToolOutput:
        tool = self._lookup.get(name)
        if tool is None:
            return ToolOutput(state_change=False, output="", error=f"unknown tool: {name}")
        return tool.dispatch(name, kwargs)

    def tool_schemas(self) -> dict[str, ChatCompletionFunctionToolParam]:
        result: dict[str, ChatCompletionFunctionToolParam] = {}
        for t in self._tools:
            result.update(t.tool_schemas())
        return result


class NameMapping(Tool):
    def __init__(self, tool: Tool, name_map: dict[str, str]) -> None:
        self._tool = tool
        self._to_original = {new: old for old, new in name_map.items()}
        self._to_new = name_map

    def dispatch(self, name: str, kwargs: dict[str, object]) -> ToolOutput:
        original = self._to_original.get(name, name)
        return self._tool.dispatch(original, kwargs)

    def tool_schemas(self) -> dict[str, ChatCompletionFunctionToolParam]:
        schemas = copy.deepcopy(self._tool.tool_schemas())
        for old_name, new_name in self._to_new.items():
            if old_name in schemas:
                schemas[new_name] = schemas.pop(old_name)
                schemas[new_name]["function"]["name"] = new_name
        return schemas
