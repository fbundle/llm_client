from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL

from llm_client.tool import Tool, ToolOutput


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _make_client(base_url: str, api_key: str) -> OpenAI | None:
    if not base_url or not api_key:
        return None
    return OpenAI(base_url=base_url, api_key=api_key)


def safe_json_kwargs(kwargs_str: str) -> str:
    try:
        json.loads(kwargs_str)
        return kwargs_str
    except json.JSONDecodeError as e:
        if e.msg == "Extra data":
            return kwargs_str[: e.pos]
    return kwargs_str


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str = ""
    name: str = ""
    kwargs_str: str = ""
    extra_content: dict[str, object] | None = None


class Callbacks(Protocol):
    def on_extra_content(self, data: str) -> None: ...
    def on_reasoning(self, token: str) -> None: ...
    def on_content(self, token: str) -> None: ...
    def on_tool_call(self, name: str, kwargs_str: str) -> None: ...
    def on_tool_result(self, output: str) -> None: ...
    def on_tool_error(self, error: str) -> None: ...
    def on_screenshot(self, data_url: str) -> None: ...
    def is_stopped(self) -> bool: ...


class DefaultCallbacks:
    def on_extra_content(self, data: str) -> None:
        print(f"\033[2m[extra_content: {data}]\033[0m")

    def on_reasoning(self, token: str) -> None:
        print(f"\033[2m{token}\033[0m", end="", flush=True)

    def on_content(self, token: str) -> None:
        print(token, end="", flush=True)

    def on_tool_call(self, name: str, kwargs_str: str) -> None:
        print(f"[*] tool call: {name}({kwargs_str})")

    def on_tool_result(self, output: str) -> None:
        print(f"[*] tool output: {output}")

    def on_tool_error(self, error: str) -> None:
        print(f"[!] tool error: {error}")

    def on_screenshot(self, data_url: str) -> None:
        pass

    def is_stopped(self) -> bool:
        return False


# ------------------------------------------------------------------
# XML fallback parser
# ------------------------------------------------------------------

_XML_TC_RE = re.compile(
    r"<tool_call>\s*"
    r"<function=([^>]+)>\s*"
    r"(.*?)"
    r"</function>\s*"
    r"</tool_call>",
    re.DOTALL,
)
_XML_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _parse_xml_tool_calls(content: str) -> tuple[list[ToolCall], str]:
    tool_calls: list[ToolCall] = []
    cleaned = content
    for m in _XML_TC_RE.finditer(content):
        name = m.group(1).strip()
        params_block = m.group(2)
        kwargs: dict[str, object] = {}
        for pm in _XML_PARAM_RE.finditer(params_block):
            key = pm.group(1).strip()
            val = pm.group(2).strip()
            try:
                kwargs[key] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                kwargs[key] = val
        tc = ToolCall(
            id=f"xml_{uuid.uuid4().hex[:8]}",
            name=name,
            kwargs_str=json.dumps(kwargs),
        )
        tool_calls.append(tc)
        cleaned = cleaned.replace(m.group(0), "")
    return tool_calls, cleaned.strip()


def _make_tool_call_param(tc: ToolCall) -> ChatCompletionMessageFunctionToolCallParam:
    p: ChatCompletionMessageFunctionToolCallParam = {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.name,
            "arguments": safe_json_kwargs(tc.kwargs_str),
        },
    }
    if tc.extra_content is not None:
        p["extra_content"] = tc.extra_content  # type: ignore[typeddict-unknown-key]
    return p


# ------------------------------------------------------------------
# Streaming & tool execution (internal helpers)
# ------------------------------------------------------------------

def _stream_response(
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    tools: list[ChatCompletionFunctionToolParam],
    max_tokens: int,
    temperature: float,
    top_p: float,
    cb: Callbacks,
) -> tuple[str, list[ToolCall]]:
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )

    content_buf = ""
    reasoning_buf = ""
    tool_call_buf: dict[int, ToolCall] = {}

    for chunk in stream:
        if cb.is_stopped():
            break

        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue

        if reasoning := getattr(delta, "reasoning_content", None):
            cb.on_reasoning(reasoning)
            reasoning_buf += reasoning

        if delta.content:
            cb.on_content(delta.content)
            content_buf += delta.content

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_call_buf:
                    tool_call_buf[idx] = ToolCall()
                tc = tool_call_buf[idx]
                if tc_delta.id:
                    tc.id = tc_delta.id
                if ec := getattr(tc_delta, "extra_content", None):
                    tc.extra_content = ec.model_dump() if hasattr(ec, "model_dump") else dict(ec)
                    cb.on_extra_content(json.dumps(tc.extra_content))
                if tc_delta.function:
                    if tc_delta.function.name:
                        tc.name = tc_delta.function.name
                    if tc_delta.function.arguments:
                        tc.kwargs_str += tc_delta.function.arguments

    tool_calls = sorted(tool_call_buf.values(), key=lambda t: t.id)

    if reasoning_buf:
        xml_calls, reasoning_buf = _parse_xml_tool_calls(reasoning_buf)
        if xml_calls:
            for tc in xml_calls:
                cb.on_tool_call(tc.name, tc.kwargs_str)
            tool_calls += xml_calls

    if content_buf:
        xml_calls, content_buf = _parse_xml_tool_calls(content_buf)
        if xml_calls:
            for tc in xml_calls:
                cb.on_tool_call(tc.name, tc.kwargs_str)
            tool_calls += xml_calls

    tool_calls = sorted(tool_calls, key=lambda t: t.id)
    return content_buf, reasoning_buf, tool_calls


def _execute_tools(
    tool_calls: list[ToolCall],
    tool: Tool,
    cb: Callbacks,
) -> list[ChatCompletionMessageParam]:
    results: list[ChatCompletionMessageParam] = []
    for tc in tool_calls:
        cb.on_tool_call(tc.name, tc.kwargs_str)
        try:
            kwargs = json.loads(tc.kwargs_str)
        except json.JSONDecodeError as e:
            out = ToolOutput(state_change=False, output="", error=f"invalid JSON: {e}")
        else:
            try:
                out = tool.dispatch(tc.name, kwargs)
            except Exception as e:
                out = ToolOutput(state_change=False, output="", error=f"dispatch error: {e}")
        cb.on_tool_result(out.output)
        results.append(ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=tc.id,
            content=out.output,
        ))
        if out.output_image:
            cb.on_screenshot(out.output_image)
            results.append(ChatCompletionUserMessageParam(
                role="user",
                content=[
                    ChatCompletionContentPartTextParam(type="text", text="Latest screenshot."),
                    ChatCompletionContentPartImageParam(type="image_url", image_url=ImageURL(url=out.output_image, detail="low")),
                ],
            ))
        if out.error:
            cb.on_tool_error(out.error)
            results.append(ChatCompletionUserMessageParam(
                role="user",
                content=[ChatCompletionContentPartTextParam(type="text", text=f"Error: {out.error}")],
            ))
    return results


# ------------------------------------------------------------------
# LLMClient — state machine that owns messages, tools, model, sampling
# ------------------------------------------------------------------

class LLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        tool: Tool | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 4096,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self._client = _make_client(base_url, api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.tool = tool
        self.messages: list[ChatCompletionMessageParam] = []

    # -- mutators ---------------------------------------------------------

    def set_model(
        self,
        *,
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        if model:
            self.model = model
        if base_url:
            self.base_url = base_url
        if api_key:
            self.api_key = api_key
        if base_url or api_key:
            self._client = _make_client(self.base_url, self.api_key)
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if max_tokens is not None:
            self.max_tokens = max_tokens

    def set_tool(self, tool: Tool | None) -> None:
        self.tool = tool

    # -- actions ---------------------------------------------------------

    def append_system_message(self, system_prompt: str) -> None:
        self.messages.append({"role": "system", "content": system_prompt})

    def append_user_message_and_generate(self, user_message: str, cb: Callbacks | None = None) -> None:
        if cb is None:
            cb = DefaultCallbacks()
        msg_idx = len(self.messages)
        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_message}],
        })

        if self._client is None:
            cb.on_tool_error("Missing credentials — set model, base_url, and api_key first")
            return

        tools = list(self.tool.tool_schemas().values()) if self.tool is not None else []

        while not cb.is_stopped():
            try:
                content, reasoning, tool_calls = _stream_response(
                    self._client, self.model, self.messages, tools,
                    self.max_tokens, self.temperature, self.top_p, cb,
                )
            except Exception as e:
                cb.on_tool_error(f"API error: {e}")
                del self.messages[msg_idx:]  # roll back entire turn on error
                break

            if not tool_calls or self.tool is None:
                msg: dict = {"role": "assistant", "content": content or None}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                self.messages.append(ChatCompletionAssistantMessageParam(**msg))
                break

            tool_results = _execute_tools(tool_calls, self.tool, cb)
            msg = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [_make_tool_call_param(tc) for tc in tool_calls],
            }
            if reasoning:
                msg["reasoning_content"] = reasoning
            self.messages = self.messages + [ChatCompletionAssistantMessageParam(**msg), *tool_results]

    def generate_prompt(self, meta_prompt: str, system_prompt: str, cb: Callbacks | None = None) -> str:
        if cb is None:
            cb = DefaultCallbacks()
        if self._client is None:
            cb.on_tool_error("Missing credentials — set model, base_url, and api_key first")
            return ""
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": meta_prompt},
            ],
            stream=True,
        )
        buf = ""
        for chunk in stream:
            if cb.is_stopped():
                break
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            if reasoning := getattr(delta, "reasoning_content", None):
                cb.on_reasoning(reasoning)
            if delta.content:
                buf += delta.content
                cb.on_content(delta.content)
        return buf

    def clear_history(self) -> None:
        self.messages = []
