from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools.mouse import MouseTool
from tools.pikafish import PikaFishTool
from tools.screen import get_screenshot


def must_get_env(key: str) -> str:
    val = os.environ.get(key)
    assert val is not None, f"missing env var: {key}"
    return val


def _dispatch(name: str, args: str, *tool_instances: object) -> tuple[str, bool]:
    for t in tool_instances:
        result, take_screenshot = t.call(name, args)
        if result != "tool name not found":
            return result, take_screenshot
    return f"unknown tool: {name}", False


def main() -> None:
    load_dotenv()

    client = OpenAI(
        base_url=must_get_env("LMS_BASE_URL"),
        api_key=must_get_env("LMS_API_KEY"),
    )
    model = must_get_env("LMS_MODEL")

    mouse = MouseTool()
    pikafish = PikaFishTool()
    tools = mouse.openai_tools() + pikafish.openai_tools()

    system: dict[str, object] = {
        "role": "system",
        "content": (
            "You are a xiangqi (Chinese chess) bot. Each turn:\n"
            "1. Look at the screenshot of the board.\n"
            "2. Convert the position to a 10x9 grid. UPPERCASE = red, lowercase = black.\n"
            "   Pieces: K=king, A=advisor, B=elephant, N=knight, R=rook, C=cannon, P=pawn.\n"
            "   grid[0] is red's back rank (bottom of the board in the image).\n"
            "   grid[9] is black's back rank (top of the board in the image).\n"
            '   Each row has 9 columns (a-i from red\'s left). Use "" for empty squares.\n'
            "3. Call submit_board with the grid and side_to_move to get the best move.\n"
            "4. Execute the move in two steps per square: first mouse_move to the square, then mouse_click to click.\n"
            "   Do this for the source square, then for the destination square.\n"
            "   The screenshot shows a green circle at the current cursor position.\n"
            "   After each mouse_move, check that the green circle is on the correct square before calling mouse_click.\n"
            "   If it's off, call mouse_move again with adjusted coordinates — do NOT click until the circle is correct.\n"
            "   The move notation uses standard xiangqi algebraic: columns a-i from red's left, ranks 0-9 from red's bottom.\n"
            "   Column a = left edge of board, column i = right edge. Rank 0 = bottom edge, rank 9 = top edge.\n"
            "   Visually estimate the board boundaries in the image, then interpolate each square's position.\n"
            "   DO NOT guess — compute the fractional position from the column and rank indices.\n"
            "Coordinates are [0, 1] relative to the image: (0,0) top-left, (1,1) bottom-right."
        ),
    }

    while True:
        input("Press Enter to take a turn...")

        print("[*] taking screenshot...")
        screenshot = get_screenshot(format="JPEG", temp_file="tmp/screenshot.jpg")
        print("[*] sending to model...")

        messages: list[dict[str, object]] = [system, {
            "role": "user",
            "content": [
                {"type": "text", "text": "Your turn. Find the best move and click it."},
                {"type": "image_url", "image_url": {"url": screenshot, "detail": "low"}},
            ],
        }]

        while True:
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

            if not tool_call_buf:
                print()
                break

            print()
            tool_calls = sorted(tool_call_buf.values(), key=lambda t: str(t.get("id", "")))

            messages.append({
                "role": "assistant",
                "content": content_buf or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["args"]},
                    }
                    for tc in tool_calls
                ],
            })

            need_screenshot = False
            for tc in tool_calls:
                name = str(tc["name"])
                args = str(tc["args"])
                print(f"[*] tool call: {name}({args})")
                result, take_screenshot = _dispatch(name, args, mouse, pikafish)
                print(f"[*] tool result: {result}")
                if take_screenshot:
                    need_screenshot = True

                messages.append({
                    "role": "tool",
                    "tool_call_id": str(tc["id"]),
                    "content": result,
                })

            if need_screenshot:
                print("[*] taking screenshot...")
                screenshot = get_screenshot(format="JPEG", temp_file="tmp/screenshot.jpg", max_size=1024)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Board after move. Verify and continue if needed."},
                        {"type": "image_url", "image_url": {"url": screenshot, "detail": "low"}},
                    ],
                })


if __name__ == "__main__":
    main()
