# Playwright Browser Tool

## Setup

Install Playwright and Chromium:

    # from PyPI
    pip install llm-gui-client[tools]
    playwright install chromium

    # local dev
    uv sync --extra tools
    uv run playwright install chromium

## Start Chrome with remote debugging

Quit any existing Chrome first (Cmd+Q on macOS).

On macOS:

    open -a "Google Chrome" --args --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug-profile

On Linux:

    google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug-profile

## Tools

| Tool | Description |
|---|---|
| `browser_navigate(url)` | Navigate to a URL |
| `browser_click(selector)` | Click an element by CSS selector |
| `browser_type(selector, text)` | Type text into an input |
| `browser_screenshot` | Screenshot the viewport |
| `browser_content` | Get visible text content |
| `browser_evaluate(js)` | Execute JavaScript |
| `browser_scroll(direction)` | Scroll up or down |
| `browser_press_key(key)` | Press a keyboard key |
