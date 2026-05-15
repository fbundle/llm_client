# LLM Desktop Agent

An LLM-powered desktop automation agent. The model sees your screen via screenshots, controls your mouse and keyboard, and executes tasks autonomously.

## Quick start

```bash
# Install dependencies
uv sync

# Configure (edit .env — this file is gitignored)
OPENAI_BASE_URL=https://api.openai.com/v1   # or any OpenAI-compatible endpoint
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# Launch GUI
python main_gui.py

# Or CLI (non-interactive)
python main_cli.py "open Safari and go to wikipedia.org"
```

## Project structure

```
app.py              Core agent loop, streaming, tool orchestration, prompt tiers
main_gui.py         tkinter GUI with collapsible sidebars, log, preset prompts
main_cli.py         Thin CLI entry point
tools/
  mouse.py          mouse_move, mouse_click, mouse_drag, mouse_scroll
  keyboard.py       key_press, key_type
  screen.py         take_screenshot (with cursor overlay)
  js_runtime.py     js_eval (JavaScript evaluation via MiniRacer)
  pikafish.py       submit_board (xiangqi/Chinese chess engine)
  tool.py           Tool/ToolOutput/ToolList base classes
check_secrets.py    Scans repo for accidentally committed secrets
```

## Features

- **Screenshot-driven**: model sees the screen at (0,0)-(1,1) fractional coordinates
- **Streaming output**: reasoning and content tokens stream in real-time
- **GUI with sidebars**: collapsible environment config (left) and system prompt editor (right)
- **Tiered prompts**: 4 built-in prompt levels from explicit (weak models) to minimal (strong models) — editable in-app
- **Auto-generate prompts**: click to have the model improve the current system prompt
- **Resizable layout**: draggable sashes between panels and log area
- **Gemini-compatible**: handles `extra_content` / `thought_signature` for Gemini OpenAI-compatible endpoints
- **Stateful tools**: JS runtime and chess engine persist across turns
- **Stop/interrupt**: threaded agent loop with clean shutdown

## Prompt tiers (in app.py)

| Variable | Level | For |
|---|---|---|
| `PROMPT_TIER1_EXPLICIT` | Very detailed | Weak models |
| `PROMPT_TIER2_GUIDED` | Moderately detailed | Lower-mid models |
| `PROMPT_TIER3_CONCISE` | Brief | Upper-mid models |
| `PROMPT_TIER4_MINIMAL` | Minimal (default) | Strong models |

## Sampling config (optional env vars)

```
OPENAI_TEMPERATURE=0.7
OPENAI_TOP_P=1.0
OPENAI_MAX_TOKENS=4096
```

## Security scanning

```bash
python check_secrets.py          # scan working tree
python check_secrets.py --full   # scan full git history
python check_secrets.py --json   # JSON output
```
