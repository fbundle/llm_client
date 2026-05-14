from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
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
from tools.screen import get_screenshot
from tools.tool import ToolList, ToolOutput


def must_get_env(key: str) -> str:
    val = os.environ.get(key)
    assert val is not None, f"missing env var: {key}"
    return val


@dataclass
class _ToolCall:
    id: str = ""
    name: str = ""
    kwargs_str: str = ""


def _safe_json_kwargs(kwargs_str: str) -> str:
    """Take only the first valid JSON object from a possibly concatenated string."""
    try:
        json.loads(kwargs_str)
        return kwargs_str
    except json.JSONDecodeError as e:
        if e.msg == "Extra data":
            return kwargs_str[: e.pos]
    return kwargs_str


def _stream_response(
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    tools: list[ChatCompletionFunctionToolParam],
) -> tuple[str, list[_ToolCall]]:
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=True,
    )

    content_buf = ""
    tool_call_buf: dict[int, _ToolCall] = {}

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue

        if reasoning := getattr(delta, "reasoning_content", None):
            print(f"\033[2m{reasoning}\033[0m", end="", flush=True)

        if delta.content:
            print(delta.content, end="", flush=True)
            content_buf += delta.content

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_call_buf:
                    tool_call_buf[idx] = _ToolCall()
                tc = tool_call_buf[idx]
                if tc_delta.id:
                    tc.id = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tc.name = tc_delta.function.name
                    if tc_delta.function.arguments:
                        tc.kwargs_str += tc_delta.function.arguments

    tool_calls = sorted(tool_call_buf.values(), key=lambda t: t.id)
    return content_buf, tool_calls


def _execute_tools(
    tool_calls: list[_ToolCall],
    dispatcher: ToolList,
) -> tuple[list[ChatCompletionMessageParam], bool]:
    results: list[ChatCompletionMessageParam] = []
    any_state_change = False
    for tc in tool_calls:
        print(f"[*] tool call: {tc.name}({tc.kwargs_str})")
        try:
            kwargs = json.loads(tc.kwargs_str)
        except json.JSONDecodeError as e:
            out = ToolOutput(state_change=False, output="", error=f"invalid JSON: {e}")
        else:
            out = dispatcher.dispatch(tc.name, kwargs)
        print(f"[*] tool output: {out.output}")
        if out.state_change:
            any_state_change = True
        results.append(ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=tc.id,
            content=out.output,
        ))
        if out.error:
            print(f"[!] tool error: {out.error}")
            results.append(ChatCompletionUserMessageParam(
                role="user",
                content=[ChatCompletionContentPartTextParam(type="text", text=f"Error: {out.error}")],
            ))
    return results, any_state_change




def run_task(
    client: OpenAI,
    model: str,
    tools: list[ChatCompletionFunctionToolParam],
    system: ChatCompletionSystemMessageParam,
    dispatcher: ToolList,
    task: str,
) -> None:
    print("[*] taking screenshot...")
    screenshot = get_screenshot(format="JPEG", temp_file="tmp/screenshot.jpg", max_size=1024)
    print("[*] sending to model...")

    messages: list[ChatCompletionMessageParam] = [system, ChatCompletionUserMessageParam(
        role="user",
        content=[
            ChatCompletionContentPartTextParam(type="text", text=task),
            ChatCompletionContentPartImageParam(type="image_url", image_url=ImageURL(url=screenshot, detail="low")),
        ],
    )]

    while True:
        try:
            content, tool_calls = _stream_response(client, model, messages, tools)
        except Exception as e:
            print(f"\n[!] API error: {e}")
            break

        if not tool_calls:
            print()
            break

        print()
        tool_results, state_changed = _execute_tools(tool_calls, dispatcher)
        messages += [
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=content or None,
                tool_calls=[
                    ChatCompletionMessageFunctionToolCallParam(
                        id=tc.id,
                        type="function",
                        function=ToolCallFunction(name=tc.name, arguments=_safe_json_kwargs(tc.kwargs_str)),
                    )
                    for tc in tool_calls
                ],
            ),
            *tool_results,
        ]

        if state_changed:
            time.sleep(0.5)
            print("[*] taking screenshot...")
            screenshot = get_screenshot(format="JPEG", temp_file="tmp/screenshot.jpg", max_size=1024)
            messages += [ChatCompletionUserMessageParam(
                role="user",
                content=[
                    ChatCompletionContentPartTextParam(type="text", text="Latest screenshot. Continue the task."),
                    ChatCompletionContentPartImageParam(type="image_url", image_url=ImageURL(url=screenshot, detail="low")),
                ],
            )]


def main() -> None:
    load_dotenv()

    client = OpenAI(
        base_url=must_get_env("OPENAI_BASE_URL"),
        api_key=must_get_env("OPENAI_API_KEY"),
    )
    model = must_get_env("OPENAI_MODEL")

    mouse = MouseTool()
    keyboard = KeyboardTool()
    pikafish = PikaFishTool()
    js = JSRuntimeTool()
    dispatcher = ToolList(mouse, keyboard, pikafish, js)
    tools: list[ChatCompletionFunctionToolParam] = list(dispatcher.tool_schemas().values())

    system: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": (
            "You control a computer. Each message includes a screenshot.\n"
            "Coordinates: (0,0)=top-left, (1,1)=bottom-right.\n"
            "The cursor icon shows mouse position.\n"
            "\n"
            "- mouse_move: move cursor to a position\n"
            "- mouse_click: provide x,y to move-and-click, or omit them to click in place\n"
            "- key_type: type text into focused field\n"
            "- key_press: press a single key (enter, tab, escape, etc.)\n"
            "- key_hotkey: press combo like ctrl+t\n"
            "\n"
            "Stop calling tools when the task is done."
        ),
    }

    while True:
        task = input("Task: ").strip()
        if not task:
            continue
        run_task(client, model, tools, system, dispatcher, task)


if __name__ == "__main__":
    main()
