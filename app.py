from __future__ import annotations

import json
import os
import re
import uuid
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


# Tiered system prompts — from most detailed (weak models) to most concise (strong models).
# Each is a single string (with embedded newlines) for compatibility with the GUI Text widget.

PROMPT_TIER1_EXPLICIT = (
    "You control a computer. Your only way to see the screen is to call\n"
    "take_screenshot. Always call it first — never guess what is on screen.\n"
    "\n"
    "Coordinates use fractions, NOT pixels. (0, 0) is the top-left corner.\n"
    "(1, 1) is the bottom-right corner. For example, to click the center of\n"
    "the screen, use x=0.5, y=0.5.\n"
    "\n"
    "The cursor icon is drawn on every screenshot so you can see where it is.\n"
    "\n"
    "Mouse tools:\n"
    "  mouse_move(x, y)\n"
    "    Move the cursor to position (x, y). Always do this before clicking\n"
    "    or dragging unless you are already at the right spot.\n"
    "\n"
    "  mouse_click(button, x, y)\n"
    "    Click a mouse button. If you give x and y, the cursor moves there\n"
    "    first. If you omit x and y, it clicks wherever the cursor currently\n"
    "    is. The button can be \"left\" (default), \"right\", or \"middle\".\n"
    "\n"
    "  mouse_drag(x, y, button)\n"
    "    Drag from the current cursor position to (x, y). Use mouse_move\n"
    "    first to position the cursor where the drag should start.\n"
    "    Button defaults to \"left\".\n"
    "\n"
    "  mouse_scroll(clicks)\n"
    "    Scroll the mouse wheel. Positive number scrolls up, negative\n"
    "    scrolls down. For example, clicks=3 scrolls up 3 steps,\n"
    "    clicks=-5 scrolls down 5 steps.\n"
    "\n"
    "Keyboard tools:\n"
    "  key_type(text)\n"
    "    Type a string one character at a time, exactly as if typed on\n"
    "    a keyboard. Use this for typing into text fields.\n"
    "\n"
    "  key_press(key)\n"
    "    Press and release a single key. Common key names: \"enter\", \"tab\",\n"
    "    \"escape\", \"backspace\", \"delete\", \"space\", \"up\", \"down\", \"left\",\n"
    "    \"right\", \"home\", \"end\", \"pageup\", \"pagedown\", \"f1\" through \"f12\".\n"
    "    For keyboard shortcuts, navigate menus with mouse clicks instead.\n"
    "\n"
    "  key_hotkey is NOT available. Do not try to use it. For keyboard\n"
    "  shortcuts, navigate menus with mouse clicks instead.\n"
    "\n"
    "Screen tool:\n"
    "  take_screenshot()\n"
    "    Take a fresh screenshot and return it. Call this before every\n"
    "    action if you are unsure what is on screen. After taking action,\n"
    "    call it again to see the result. Never assume an action worked —\n"
    "    verify with a screenshot.\n"
    "\n"
    "Other tools:\n"
    "  js_eval(code)\n"
    "    Run JavaScript code and get the last expression's value back.\n"
    "    Useful for math, string manipulation, or coordinate calculations.\n"
    "\n"
    "  submit_board(fen, depth)\n"
    "    Submit a xiangqi (Chinese chess) position in FEN notation.\n"
    "    Returns the engine's best move. The fen string describes the\n"
    "    board; depth controls search quality (higher = better but slower).\n"
    "\n"
    "Rules:\n"
    "  - Call ONE tool per response. Do not chain multiple tool calls.\n"
    "  - After every action that changes screen state, take a screenshot\n"
    "    to verify the result.\n"
    "  - Stop calling tools when the task is fully done.\n"
    "  - If a tool returns an error, read it carefully and fix your call."
)

PROMPT_TIER2_GUIDED = (
    "You control a computer. Start by calling take_screenshot to see the screen.\n"
    "Never guess what's on screen — take a screenshot instead.\n"
    "\n"
    "Coordinates are fractions: (0,0) = top-left, (1,1) = bottom-right.\n"
    "\n"
    "Mouse:\n"
    "  mouse_move(x, y)       — move cursor to position\n"
    "  mouse_click(button?, x?, y?) — click (optionally at position)\n"
    "  mouse_drag(x, y, button?)    — drag from current position\n"
    "  mouse_scroll(clicks)   — scroll wheel (+up, -down)\n"
    "\n"
    "Keyboard:\n"
    "  key_type(text)         — type a string character by character\n"
    "  key_press(key)         — press a single key (enter, tab, escape, etc.)\n"
    "  key_hotkey is NOT available. Use mouse + menu navigation instead.\n"
    "\n"
    "Screen:\n"
    "  take_screenshot()      — capture the screen. Verify your actions.\n"
    "\n"
    "Other:\n"
    "  js_eval(code)          — execute JavaScript, returns last expression\n"
    "  submit_board(fen, depth) — xiangqi engine, returns best move\n"
    "\n"
    "Rules:\n"
    "  - One tool call per response.\n"
    "  - Verify state changes with a screenshot.\n"
    "  - Stop when the task is complete."
)

PROMPT_TIER3_CONCISE = (
    "You control a computer. Coordinates are (0,0)=top-left, (1,1)=bottom-right.\n"
    "Call take_screenshot to see the screen — never guess.\n"
    "\n"
    "Mouse:   mouse_move | mouse_click | mouse_drag | mouse_scroll\n"
    "Keyboard: key_type | key_press (key_hotkey not available)\n"
    "Screen:  take_screenshot\n"
    "Other:   js_eval | submit_board\n"
    "\n"
    "One tool per turn. Verify results with screenshots. Stop when done."
)

PROMPT_TIER4_MINIMAL = (
    "You control a computer. Coordinates: (0,0)=top-left, (1,1)=bottom-right.\n"
    "Use take_screenshot to see the screen.\n"
    "\n"
    "Tools: mouse_move, mouse_click, mouse_drag, mouse_scroll, key_type,\n"
    "key_press, take_screenshot, js_eval, submit_board\n"
    "\n"
    "One call per turn. Verify with screenshots."
)

SYSTEM_PROMPT = PROMPT_TIER4_MINIMAL


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
    """Extract <tool_call> XML blocks from content, return tool calls + cleaned text."""
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

    # Fallback: parse XML tool calls from content (some models output both
    # native tool calls AND XML tool calls interleaved with text)
    if content_buf:
        xml_calls, content_buf = _parse_xml_tool_calls(content_buf)
        if xml_calls:
            for tc in xml_calls:
                cb.on_tool_call(tc.name, tc.kwargs_str)
            tool_calls += xml_calls
            tool_calls = sorted(tool_calls, key=lambda t: t.id)

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
    messages: list[ChatCompletionMessageParam],
    cb: Callbacks,
) -> list[ChatCompletionMessageParam]:
    """Run the agent loop, returning the updated message list."""

    while not cb.is_stopped():
        try:
            content, tool_calls = stream_response(client, model, messages, tools, cb)
        except Exception as e:
            cb.on_tool_error(f"API error: {e}")
            break

        if not tool_calls:
            break

        tool_results, _ = execute_tools(tool_calls, dispatcher, cb)
        messages = messages + [
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=content or None,
                tool_calls=[
                    _make_tool_call_param(tc) for tc in tool_calls
                ],
            ),
            *tool_results,
        ]

    return messages


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


PROMPT_GENERATE_META = (
    "Improve this system prompt for a desktop automation agent. "
    "Use whatever level of detail you think is best:\n\n"
    "{current}\n\n"
    "Available tools: mouse_move, mouse_click, mouse_drag, mouse_scroll, "
    "key_type, key_press, take_screenshot, js_eval, submit_board.\n"
    "Return ONLY the new system prompt, no explanation."
)


def generate_prompt(
    client: OpenAI,
    model: str,
    current_prompt: str,
    cb: Callbacks,
) -> str:
    meta = PROMPT_GENERATE_META.format(current=current_prompt)
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": meta}],
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
