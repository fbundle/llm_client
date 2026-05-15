import asyncio
import json
import os
from pathlib import Path
from threading import Lock

from llm_client.tool import ChatCompletionFunctionToolParam, Tool, ToolOutput


def _load_config(config: dict | str | Path) -> dict:
    if isinstance(config, (str, Path)):
        with open(config) as f:
            return json.load(f)
    return config


class MCPTool(Tool):
    def __init__(self, config: dict | str | Path | None = None) -> None:
        self._lock = Lock()
        self._sessions: dict[str, object] = {}       # server_name -> ClientSession
        self._transports: dict[str, object] = {}     # server_name -> transport
        self._tool_map: dict[str, str] = {}          # tool_name -> server_name
        self._schemas: dict[str, ChatCompletionFunctionToolParam] = {}

        if config is not None:
            self._connect(_load_config(config))

    def _connect(self, config: dict) -> None:
        try:
            asyncio.get_running_loop()
            # already in async context — caller must handle this
            raise RuntimeError("MCPTool.connect must be called from a sync context")
        except RuntimeError:
            pass

        servers = config.get("mcpServers", {})
        asyncio.run(self._connect_all(servers))
        # warm up tool discovery
        asyncio.run(self._discover_tools())

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            pass
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    async def _connect_one_stdio(self, name: str, cfg: dict) -> None:
        from mcp.client.stdio import stdio_client

        env = os.environ.copy()
        env.update(cfg.get("env", {}))
        transport = await stdio_client(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=env,
        )
        self._transports[name] = transport

    async def _connect_one_sse(self, name: str, cfg: dict) -> None:
        from mcp.client.sse import sse_client

        transport = await sse_client(cfg["url"])
        self._transports[name] = transport

    async def _connect_all(self, servers: dict) -> None:
        from mcp import ClientSession

        for name, cfg in servers.items():
            if "url" in cfg:
                await self._connect_one_sse(name, cfg)
            elif "command" in cfg:
                await self._connect_one_stdio(name, cfg)
            else:
                continue

            read, write = self._transports[name]
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()
            self._sessions[name] = session

    async def _discover_tools(self) -> None:
        self._tool_map.clear()
        self._schemas.clear()
        for name, session in self._sessions.items():
            result = await session.list_tools()
            for tool in result.tools:
                full_name = f"mcp_{name}_{tool.name}"
                self._tool_map[full_name] = name
                self._schemas[full_name] = {
                    "type": "function",
                    "function": {
                        "name": full_name,
                        "description": tool.description or f"MCP tool: {tool.name}",
                        "parameters": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
                    },
                }

    # -- Tool protocol ---------------------------------------------------

    def dispatch(self, name: str, kwargs: dict[str, object]) -> ToolOutput:
        server_name = self._tool_map.get(name)
        if server_name is None:
            return ToolOutput(state_change=False, output="", error=f"unknown tool: {name}")

        session = self._sessions.get(server_name)
        if session is None:
            return ToolOutput(state_change=False, output="", error=f"MCP server not connected: {server_name}")

        # strip mcp_{server}_ prefix to get original tool name
        tool_name = name[len(f"mcp_{server_name}_"):]

        try:
            loop = self._get_or_create_loop()
            result = loop.run_until_complete(session.call_tool(tool_name, kwargs))
            content = getattr(result, "content", [])
            text = "\n".join(c.text for c in content if hasattr(c, "text"))
            return ToolOutput(state_change=False, output=text, error="")
        except Exception as e:
            return ToolOutput(state_change=False, output="", error=str(e))

    def tool_schemas(self) -> dict[str, ChatCompletionFunctionToolParam]:
        return dict(self._schemas)

    # -- lifecycle -------------------------------------------------------

    async def close(self) -> None:
        for session in self._sessions.values():
            await session.__aexit__(None, None, None)
        self._sessions.clear()
        self._transports.clear()
        self._tool_map.clear()
        self._schemas.clear()

    def __del__(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(self.close())
            except Exception:
                pass
