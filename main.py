from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from tools.keyboard import KeyboardTool
from tools.mouse import MouseTool
from tools.screen import get_screenshot
from tools.tool import Dispatcher


def must_get_env(key: str) -> str:
    val = os.environ.get(key)
    assert val is not None, f"missing env var: {key}"
    return val


def _stream_response(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Stream a chat completion, return (content, tool_calls)."""
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
            print(f"\033[2m{delta.reasoning_content}\033[0m", end="", flush=True)

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
                        entry["args"] += tc_delta.function.arguments

    tool_calls = sorted(tool_call_buf.values(), key=lambda t: str(t.get("id", "")))
    return content_buf, tool_calls


def _execute_tools(
    tool_calls: list[dict[str, Any]],
    dispatcher: Dispatcher,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for tc in tool_calls:
        name = str(tc["name"])
        args = str(tc["args"])
        print(f"[*] tool call: {name}({args})")
        out = dispatcher.call(name, args)
        print(f"[*] tool output: {out.output}")
        results.append({
            "role": "tool",
            "tool_call_id": str(tc["id"]),
            "content": out.output,
        })
        if out.error:
            print(f"[!] tool error: {out.error}")
            results.append({
                "role": "user",
                "content": [{"type": "text", "text": f"Error: {out.error}"}],
            })
    return results


def _strip_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of messages with all image content removed from user messages."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            text_only = [p for p in msg["content"] if p.get("type") != "image_url"] or [{"type": "text", "text": ""}]
            out.append({"role": "user", "content": text_only})
        else:
            out.append(msg)
    return out


def run_task(
    client: OpenAI,
    model: str,
    tools: list[dict[str, Any]],
    system: dict[str, Any],
    dispatcher: Dispatcher,
    task: str,
) -> None:
    """Run a single task, keeping text history but only the latest screenshot."""
    print("[*] taking screenshot...")
    screenshot = get_screenshot(format="JPEG", temp_file="tmp/screenshot.jpg", max_size=1024)
    print("[*] sending to model...")

    messages: list[dict[str, Any]] = [system, {
        "role": "user",
        "content": [
            {"type": "text", "text": task},
            {"type": "image_url", "image_url": {"url": screenshot, "detail": "low"}},
        ],
    }]

    while True:
        content, tool_calls = _stream_response(client, model, messages, tools)

        if not tool_calls:
            print()
            break

        print()
        messages += [
            {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["args"]},
                    }
                    for tc in tool_calls
                ],
            },
            *_execute_tools(tool_calls, dispatcher),
        ]

        messages = _strip_images(messages)

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
    dispatcher = Dispatcher(mouse, keyboard)
    tools = dispatcher.openai_tools()

    system: dict[str, Any] = {
        "role": "system",
        "content": (
            "You control a computer. Each message includes a screenshot.\n"
            "Coordinates: (0,0)=top-left, (1,1)=bottom-right.\n"
            "The cursor icon shows mouse position.\n"
            "\n"
            "Pick ONE action per turn:\n"
            "- mouse_move: move cursor to target\n"
            "- mouse_click: click where cursor already is\n"
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
