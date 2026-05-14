from __future__ import annotations

import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from tools.js_runtime import JSRuntimeTool
from tools.keyboard import KeyboardTool
from tools.mouse import MouseTool
from tools.pikafish import PikaFishTool
from tools.screen import get_screenshot
from tools.tool import ToolList


def must_get_env(key: str) -> str:
    val = os.environ.get(key)
    assert val is not None, f"missing env var: {key}"
    return val


def _safe_json_args(args: str) -> str:
    """Take only the first valid JSON object from a possibly concatenated string."""
    try:
        json.loads(args)
        return args
    except json.JSONDecodeError as e:
        if e.msg == "Extra data":
            return args[: e.pos]
    return args


def _stream_response(
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    tools: list[ChatCompletionFunctionToolParam],
) -> tuple[str, list[dict[str, object]]]:
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=True,
    )

    content_buf = ""
    tool_call_buf: dict[int, dict[str, object]] = {}

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue

        if getattr(delta, "reasoning_content", None):
            print(f"\033[2m{delta.reasoning_content}\033[0m", end="", flush=True) # type: ignore

        if delta.content:
            print(delta.content, end="", flush=True)
            content_buf += delta.content

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_call_buf:
                    tool_call_buf[idx] = {"id": tc_delta.id or "", "name": "", "args": ""}
                entry = tool_call_buf[idx]
                if tc_delta.id:
                    entry["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        entry["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        entry["args"] += tc_delta.function.arguments # type: ignore

    tool_calls = sorted(tool_call_buf.values(), key=lambda t: str(t.get("id", "")))
    return content_buf, tool_calls


def _execute_tools(
    tool_calls: list[dict[str, object]],
    dispatcher: ToolList,
) -> tuple[list[ChatCompletionMessageParam], bool]:
    results: list[ChatCompletionMessageParam] = []
    any_state_change = False
    for tc in tool_calls:
        name = str(tc["name"])
        args = str(tc["args"])
        print(f"[*] tool call: {name}({args})")
        out = dispatcher.dispatch(name, args)
        print(f"[*] tool output: {out.output}")
        if out.state_change:
            any_state_change = True
        results.append(ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=str(tc["id"]),
            content=out.output,
        ))
        if out.error:
            print(f"[!] tool error: {out.error}")
            results.append(ChatCompletionUserMessageParam(
                role="user",
                content=[{"type": "text", "text": f"Error: {out.error}"}],
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

    messages: list[ChatCompletionMessageParam] = [system, {
        "role": "user",
        "content": [
            {"type": "text", "text": task},
            {"type": "image_url", "image_url": {"url": screenshot, "detail": "low"}},
        ],
    }]

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
            {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": _safe_json_args(str(tc["args"]))},
                    }
                    for tc in tool_calls
                ],
            },
            *tool_results,
        ]

        if state_changed:
            time.sleep(0.5)
            print("[*] taking screenshot...")
            screenshot = get_screenshot(format="JPEG", temp_file="tmp/screenshot.jpg", max_size=1024)
            messages += [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Latest screenshot. Continue the task."},
                    {"type": "image_url", "image_url": {"url": screenshot, "detail": "low"}},
                ],
            }]


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
