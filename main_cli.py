from __future__ import annotations

from dotenv import load_dotenv

from app import (
    Callbacks,
    create_client,
    create_dispatcher,
    must_get_env,
    run_task,
)


class CliCallbacks:
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

    client = create_client()
    model = must_get_env("OPENAI_MODEL")
    dispatcher = create_dispatcher()
    tools = list(dispatcher.tool_schemas().values())
    cb = CliCallbacks()

    while True:
        task = input("Task: ").strip()
        if not task:
            continue
        run_task(client, model, tools, dispatcher, task, cb)


if __name__ == "__main__":
    main()
