from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from llm_client import LLMClient, Callbacks, SYSTEM_PROMPT, ToolList, discover_tools


def _must_get_env(key: str) -> str:
    val = os.environ.get(key)
    assert val is not None, f"missing env var: {key}"
    return val


class CliCallbacks(Callbacks):
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


def main() -> None:
    load_dotenv()

    tools_dir = Path(__file__).resolve().parent / "tools"
    all_tools = discover_tools(tools_dir)
    tool = ToolList(*all_tools.values())

    app = LLMClient(
        base_url=_must_get_env("OPENAI_BASE_URL"),
        api_key=_must_get_env("OPENAI_API_KEY"),
        model=_must_get_env("OPENAI_MODEL"),
        tool=tool,
    )
    app.append_system_message(SYSTEM_PROMPT)
    cb = CliCallbacks()

    while True:
        task = input("Task: ").strip()
        if not task:
            continue
        print()
        app.append_user_message_and_generate(task, cb)
        print()


if __name__ == "__main__":
    main()
