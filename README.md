# LLM Desktop Agent

An LLM-powered desktop automation agent. The model sees your screen via screenshots, controls your mouse and keyboard, and executes tasks autonomously.

## Quick start

```bash
# Install with GUI + all tools
pip install llm-gui-client[gui]

# Or clone and run from source
git clone https://github.com/fbundle/llm_client.git
cd llm_client
uv sync --extra gui
```

Create a `.env` file (gitignored):

```
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

Launch:

```bash
python llm_client_window/window.py

# Or via module:
python -m llm_client_window
```

## Packages

| Package | Contents | PyPI deps |
|---|---|---|
| `llm_client` | LLM client, streaming, prompt tiers, tool protocol | `openai` |
| `llm_client_tools` | mouse, keyboard, screen, JS runtime, xiangqi engine | `pyautogui`, `pillow`, `mini-racer` |
| `llm_client_window` | PySide6 GUI | `pyside6`, `python-dotenv` |

Install what you need:

```bash
pip install llm-gui-client              # core only
pip install llm-gui-client[tools]       # + tool implementations
pip install llm-gui-client[gui]         # + GUI + tools (everything)
```

## Features

- **Screenshot-driven**: model sees the screen at (0,0)-(1,1) fractional coordinates
- **Streaming output**: reasoning and content tokens stream in real-time via callbacks
- **GUI with sidebars**: collapsible environment config (left) and system prompt editor (right)
- **Tiered prompts**: 4 built-in prompt levels from explicit (weak models) to minimal (strong models)
- **Auto-generate prompts**: have the model improve the current system prompt
- **Run/Stop toggle**: single button toggles between running and stopping
- **Tool toggles**: enable/disable individual tools per run
- **Gemini-compatible**: handles `extra_content` / `thought_signature`
- **Stateful tools**: JS runtime and chess engine persist across turns
- **Threaded agent**: clean stop/interrupt with threaded callbacks

## Prompt tiers

| Variable | Level | For |
|---|---|---|
| `PROMPT_TIER1_EXPLICIT` | Very detailed | Weak models |
| `PROMPT_TIER2_GUIDED` | Moderately detailed | Lower-mid models |
| `PROMPT_TIER3_CONCISE` | Brief | Upper-mid models |
| `PROMPT_TIER4_MINIMAL` | Minimal (default) | Strong models |

## Build standalone .app

```bash
./build_app
```

Outputs to `bin/`. Requires `uv`.

## Example (CLI)

```python
from llm_client import LLMClient, SYSTEM_PROMPT, ToolList, discover_tools

app = LLMClient(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o",
    tool=ToolList(*discover_tools(Path("llm_client_tools")).values()),
)
app.append_system_message(SYSTEM_PROMPT)
app.append_user_message_and_generate("Open Safari and go to wikipedia.org")
```
