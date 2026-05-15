from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from llm_client import LLMClient, SYSTEM_PROMPT, ToolList, discover_tools


def _must_get_env(key: str) -> str:
    val = os.environ.get(key)
    assert val is not None, f"missing env var: {key}"
    return val


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

    while True:
        task = input("Task: ").strip()
        if not task:
            continue
        print()
        app.append_user_message_and_generate(task)
        print()


if __name__ == "__main__":
    main()
