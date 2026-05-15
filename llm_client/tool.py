from __future__ import annotations

import copy
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from openai.types.chat import ChatCompletionFunctionToolParam

@dataclass
class ToolOutput:
    state_change: bool
    output: str
    error: str
    output_image: str = ""


class Tool(Protocol):
    def dispatch(self, name: str, kwargs: dict[str, object]) -> ToolOutput: ...
    def tool_schemas(self) -> dict[str, ChatCompletionFunctionToolParam]: ...


class ToolList(Tool):
    def __init__(self, *tools: Tool) -> None:
        self._tools = list(tools)
        self._lookup: dict[str, Tool] = {}
        for t in tools:
            for name in t.tool_schemas():
                if name in self._lookup:
                    raise ValueError(f"duplicate tool name: {name}")
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


def discover_tools(tools_dir: Path) -> dict[str, Tool]:
    """Discover and instantiate all Tool classes in *tools_dir*.

    Each ``.py`` file (excluding ``_*`` and ``tool.py``) is loaded, and any
    class with ``dispatch`` + ``tool_schemas`` methods (that isn't ToolList
    or NameMapping) is instantiated and keyed by module stem.
    """
    result: dict[str, Tool] = {}
    for f in sorted(tools_dir.glob("*.py")):
        if f.name.startswith("_") or f.name == "tool.py":
            continue
        spec = importlib.util.spec_from_file_location(f"llm_client_tools.{f.stem}", f)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for _name, obj in inspect.getmembers(mod, inspect.isclass):
            if hasattr(obj, "dispatch") and hasattr(obj, "tool_schemas"):
                if obj not in (ToolList, NameMapping):
                    result[f.stem] = obj()
                    break
    return result
