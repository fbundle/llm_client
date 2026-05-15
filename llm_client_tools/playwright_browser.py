from __future__ import annotations

import base64
from contextlib import contextmanager
from typing import Iterator, Literal

from playwright.sync_api import Page, sync_playwright

from llm_client.tool import ChatCompletionFunctionToolParam, Tool, ToolOutput


class PlaywrightBrowser:
    """Stateless browser automation — connects, acts, disconnects per call."""

    def __init__(self, cdp_url: str = "http://localhost:9222") -> None:
        self._cdp_url = cdp_url

    @contextmanager
    def _page(self) -> Iterator[Page]:
        pw = sync_playwright().start()
        try:
            browser = pw.chromium.connect_over_cdp(self._cdp_url)
            ctx = browser.contexts[0]
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            yield page
            browser.close()
        finally:
            pw.stop()

    def navigate(self, url: str) -> str:
        with self._page() as page:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return f"navigated to {url} — title: {page.title()}"

    def click(self, selector: str) -> str:
        with self._page() as page:
            page.click(selector, timeout=10000)
            return f"clicked {selector}"

    def type_text(self, selector: str, text: str) -> str:
        with self._page() as page:
            page.fill(selector, text, timeout=10000)
            return f"typed into {selector}"

    def screenshot(self) -> bytes:
        with self._page() as page:
            return page.screenshot(type="jpeg", quality=80, full_page=False)

    def content(self) -> str:
        with self._page() as page:
            return page.inner_text("body")

    def evaluate(self, js: str) -> str:
        with self._page() as page:
            return str(page.evaluate(js))

    def scroll(self, direction: Literal["up", "down"]) -> str:
        with self._page() as page:
            page.evaluate(f"window.scrollBy(0, {'-window.innerHeight' if direction == 'up' else 'window.innerHeight'})")
            return f"scrolled {direction}"

    def press_key(self, key: str) -> str:
        with self._page() as page:
            page.keyboard.press(key)
            return f"pressed {key}"


class PlaywrightBrowserTool(Tool):
    def __init__(self, cdp_url: str = "http://localhost:9222") -> None:
        self._browser = PlaywrightBrowser(cdp_url)

    def dispatch(self, name: str, kwargs: dict[str, object]) -> ToolOutput:
        try:
            if name == "browser_navigate":
                return ToolOutput(state_change=True, output=self._browser.navigate(str(kwargs["url"])), error="")
            elif name == "browser_click":
                return ToolOutput(state_change=True, output=self._browser.click(str(kwargs["selector"])), error="")
            elif name == "browser_type":
                return ToolOutput(state_change=True, output=self._browser.type_text(str(kwargs["selector"]), str(kwargs["text"])), error="")
            elif name == "browser_screenshot":
                data = self._browser.screenshot()
                b64 = base64.b64encode(data).decode()
                return ToolOutput(state_change=False, output="screenshot taken", error="", output_image=f"data:image/jpeg;base64,{b64}")
            elif name == "browser_content":
                return ToolOutput(state_change=False, output=self._browser.content(), error="")
            elif name == "browser_evaluate":
                return ToolOutput(state_change=False, output=self._browser.evaluate(str(kwargs["js"])), error="")
            elif name == "browser_scroll":
                return ToolOutput(state_change=True, output=self._browser.scroll(str(kwargs["direction"])), error="")
            elif name == "browser_press_key":
                return ToolOutput(state_change=True, output=self._browser.press_key(str(kwargs["key"])), error="")
            return ToolOutput(state_change=False, output="", error=f"unknown tool: {name}")
        except Exception as e:
            return ToolOutput(state_change=False, output="", error=str(e))

    def tool_schemas(self) -> dict[str, ChatCompletionFunctionToolParam]:
        return {
            "browser_navigate": {
                "type": "function",
                "function": {
                    "name": "browser_navigate",
                    "description": "Navigate the browser to a URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The URL to navigate to."},
                        },
                        "required": ["url"],
                    },
                },
            },
            "browser_click": {
                "type": "function",
                "function": {
                    "name": "browser_click",
                    "description": "Click an element on the page via CSS selector.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector of the element to click."},
                        },
                        "required": ["selector"],
                    },
                },
            },
            "browser_type": {
                "type": "function",
                "function": {
                    "name": "browser_type",
                    "description": "Type text into an input element.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector of the input element."},
                            "text": {"type": "string", "description": "Text to type."},
                        },
                        "required": ["selector", "text"],
                    },
                },
            },
            "browser_screenshot": {
                "type": "function",
                "function": {
                    "name": "browser_screenshot",
                    "description": "Take a screenshot of the current browser viewport.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            "browser_content": {
                "type": "function",
                "function": {
                    "name": "browser_content",
                    "description": "Get the visible text content of the current page.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            "browser_evaluate": {
                "type": "function",
                "function": {
                    "name": "browser_evaluate",
                    "description": "Execute JavaScript in the browser and return the result.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "js": {"type": "string", "description": "JavaScript code to execute."},
                        },
                        "required": ["js"],
                    },
                },
            },
            "browser_scroll": {
                "type": "function",
                "function": {
                    "name": "browser_scroll",
                    "description": "Scroll the page up or down.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": ["up", "down"],
                                "description": "Direction to scroll.",
                            },
                        },
                        "required": ["direction"],
                    },
                },
            },
            "browser_press_key": {
                "type": "function",
                "function": {
                    "name": "browser_press_key",
                    "description": "Press a keyboard key (e.g. Enter, Tab, ArrowDown).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "Key name to press."},
                        },
                        "required": ["key"],
                    },
                },
            },
        }
