from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    Function as ToolCallFunction,
)

from tools.js_runtime import JSRuntimeTool
from tools.keyboard import KeyboardTool
from tools.mouse import MouseTool
from tools.pikafish import PikaFishTool
from tools.screen import ScreenTool
from tools.tool import ToolList, ToolOutput


def must_get_env(key: str) -> str:
    val = os.environ.get(key)
    assert val is not None, f"missing env var: {key}"
    return val


@dataclass
class ToolCall:
    id: str = ""
    name: str = ""
    kwargs_str: str = ""
    # Gemini-specific: extra_content.google.thought_signature must be
    # echoed back verbatim on every function call, or Gemini returns 400.
    # We capture the entire extra_content blob and pass it through as-is.
    extra_content: dict[str, object] | None = None


def safe_json_kwargs(kwargs_str: str) -> str:
    try:
        json.loads(kwargs_str)
        return kwargs_str
    except json.JSONDecodeError as e:
        if e.msg == "Extra data":
            return kwargs_str[: e.pos]
    return kwargs_str


class Callbacks(Protocol):
    def on_extra_content(self, data: str) -> None: ...
    def on_reasoning(self, token: str) -> None: ...
    def on_content(self, token: str) -> None: ...
    def on_tool_call(self, name: str, kwargs_str: str) -> None: ...
    def on_tool_result(self, output: str) -> None: ...
    def on_tool_error(self, error: str) -> None: ...
    def on_screenshot(self, data_url: str) -> None: ...
    def is_stopped(self) -> bool: ...


SYSTEM_PROMPT = (
    "You control a macOS computer. Call take_screenshot first to see the screen.\n"
    "Coordinates: (0,0)=top-left, (1,1)=bottom-right (fractions, NOT pixels).\n"
    "The cursor icon is drawn on screenshots — it won't change shape.\n"
    "\n"
    "Mouse:\n"
    "  mouse_move   — move cursor to (x, y)\n"
    "  mouse_click  — click at (x, y) if given, or click in-place\n"
    "  mouse_drag   — drag from current position to (x, y)\n"
    "\n"
    "Keyboard:\n"
    "  key_type     — type a string character-by-character\n"
    "  key_press    — press a single key (enter, tab, escape, f1, etc.)\n"
    "  key_hotkey is NOT available — use menu navigation with mouse and clicks instead.\n"
    "\n"
    "Screen:\n"
    "  take_screenshot  — capture a fresh screenshot\n"
    "  If you are unsure what is on screen, take a screenshot — do NOT guess.\n"
    "\n"
    "Other:\n"
    "  js_eval      — evaluate JavaScript code, returns the last expression\n"
    "  submit_board — submit a xiangqi FEN string, returns best move\n"
    "\n"
    "One tool call per function. Stop calling tools when the task is done."
)


def create_client() -> OpenAI:
    return OpenAI(
        base_url=must_get_env("OPENAI_BASE_URL"),
        api_key=must_get_env("OPENAI_API_KEY"),
    )


def create_dispatcher(
    enabled: set[str] | None = None,
    instances: dict[str, MouseTool | KeyboardTool | PikaFishTool | JSRuntimeTool | ScreenTool] | None = None,
) -> ToolList:
    if instances is not None:
        names = enabled if enabled is not None else set(instances)
        return ToolList(*(t for name, t in instances.items() if name in names))
    # CLI path: create fresh instances each time
    all_tools = {
        "mouse": MouseTool(),
        "keyboard": KeyboardTool(),
        "pikafish": PikaFishTool(),
        "js_runtime": JSRuntimeTool(),
        "screen": ScreenTool(),
    }
    names = enabled if enabled is not None else set(all_tools)
    return ToolList(*(t for name, t in all_tools.items() if name in names))


def stream_response(
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    tools: list[ChatCompletionFunctionToolParam],
    cb: Callbacks,
) -> tuple[str, list[ToolCall]]:
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=True,
    )

    content_buf = ""
    tool_call_buf: dict[int, ToolCall] = {}

    for chunk in stream:
        if cb.is_stopped():
            break

        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue

        if reasoning := getattr(delta, "reasoning_content", None):
            cb.on_reasoning(reasoning)

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
                # Gemini-specific: capture the entire extra_content blob
                # and echo it back verbatim. Not standard OpenAI API.
                if ec := getattr(tc_delta, "extra_content", None):
                    tc.extra_content = ec.model_dump() if hasattr(ec, "model_dump") else dict(ec)
                    cb.on_extra_content(json.dumps(tc.extra_content))
                if tc_delta.function:
                    if tc_delta.function.name:
                        tc.name = tc_delta.function.name
                    if tc_delta.function.arguments:
                        tc.kwargs_str += tc_delta.function.arguments

    tool_calls = sorted(tool_call_buf.values(), key=lambda t: t.id)
    return content_buf, tool_calls


def execute_tools(
    tool_calls: list[ToolCall],
    dispatcher: ToolList,
    cb: Callbacks,
) -> tuple[list[ChatCompletionMessageParam], bool]:
    results: list[ChatCompletionMessageParam] = []
    any_state_change = False
    for tc in tool_calls:
        cb.on_tool_call(tc.name, tc.kwargs_str)
        try:
            kwargs = json.loads(tc.kwargs_str)
        except json.JSONDecodeError as e:
            out = ToolOutput(state_change=False, output="", error=f"invalid JSON: {e}")
        else:
            out = dispatcher.dispatch(tc.name, kwargs)
        cb.on_tool_result(out.output)
        if out.state_change:
            any_state_change = True
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
    return results, any_state_change


def run_task(
    client: OpenAI,
    model: str,
    tools: list[ChatCompletionFunctionToolParam],
    dispatcher: ToolList,
    task: str,
    cb: Callbacks,
    system_prompt: str = SYSTEM_PROMPT,
) -> None:
    system: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": system_prompt,
    }
    messages: list[ChatCompletionMessageParam] = [
        system,
        ChatCompletionUserMessageParam(
            role="user",
            content=[ChatCompletionContentPartTextParam(type="text", text=task)],
        ),
    ]

    while not cb.is_stopped():
        try:
            content, tool_calls = stream_response(client, model, messages, tools, cb)
        except Exception as e:
            cb.on_tool_error(f"API error: {e}")
            break

        if not tool_calls:
            break

        tool_results, _ = execute_tools(tool_calls, dispatcher, cb)
        messages += [
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=content or None,
                tool_calls=[
                    _make_tool_call_param(tc) for tc in tool_calls
                ],
            ),
            *tool_results,
        ]


def _make_tool_call_param(tc: ToolCall) -> ChatCompletionMessageFunctionToolCallParam:
    p: ChatCompletionMessageFunctionToolCallParam = {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.name,
            "arguments": safe_json_kwargs(tc.kwargs_str),
        },
    }
    # Gemini-specific: echo back the entire extra_content blob verbatim.
    # Not standard OpenAI API. Without this Gemini returns 400.
    if tc.extra_content is not None:
        p["extra_content"] = tc.extra_content  # type: ignore[typeddict-unknown-key]
    return p
