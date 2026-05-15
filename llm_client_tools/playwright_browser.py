from __future__ import annotations

import base64
from contextlib import contextmanager
from typing import Iterator

from playwright.sync_api import Page, sync_playwright

from llm_client.tool import ChatCompletionFunctionToolParam, Tool, ToolOutput


def _slice(text: str, beg: int, end: int) -> str:
    if end == -1:
        return text[beg:]
    return text[beg:end]


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

    # -- navigation --------------------------------------------------------

    def navigate(self, url: str, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return _slice(f"navigated to {url} — title: {page.title()}", beg, end)

    def go_back(self, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            page.go_back(wait_until="domcontentloaded", timeout=10000)
            return _slice(f"went back — {page.url}", beg, end)

    def go_forward(self, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            page.go_forward(wait_until="domcontentloaded", timeout=10000)
            return _slice(f"went forward — {page.url}", beg, end)

    def reload(self, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            page.reload(wait_until="domcontentloaded", timeout=30000)
            return _slice(f"reloaded — {page.url}", beg, end)

    # -- mouse -------------------------------------------------------------

    def click(self, selector: str, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            page.click(selector, timeout=10000)
            return _slice(f"clicked {selector}", beg, end)

    def dblclick(self, selector: str, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            page.dblclick(selector, timeout=10000)
            return _slice(f"double-clicked {selector}", beg, end)

    def hover(self, selector: str, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            page.hover(selector, timeout=10000)
            return _slice(f"hovered {selector}", beg, end)

    # -- input -------------------------------------------------------------

    def type_text(self, selector: str, text: str, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            page.fill(selector, text, timeout=10000)
            return _slice(f"typed into {selector}", beg, end)

    def press_key(self, key: str, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            page.keyboard.press(key)
            return _slice(f"pressed {key}", beg, end)

    def select_option(self, selector: str, value: str, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            page.select_option(selector, value, timeout=10000)
            return _slice(f"selected '{value}' in {selector}", beg, end)

    def check(self, selector: str, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            page.check(selector, timeout=10000)
            return _slice(f"checked {selector}", beg, end)

    def uncheck(self, selector: str, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            page.uncheck(selector, timeout=10000)
            return _slice(f"unchecked {selector}", beg, end)

    # -- read --------------------------------------------------------------

    def screenshot(self, beg: int = 0, end: int = -1) -> bytes:
        with self._page() as page:
            return page.screenshot(type="jpeg", quality=80, full_page=False)

    def content(self, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            return _slice(page.inner_text("body"), beg, end)

    def html(self, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            return _slice(page.content(), beg, end)

    def url(self, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            return _slice(page.url, beg, end)

    def title(self, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            return _slice(page.title(), beg, end)

    def evaluate(self, js: str, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            return _slice(str(page.evaluate(js)), beg, end)

    # -- scroll ------------------------------------------------------------

    def scroll(self, direction: str, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            page.evaluate(f"window.scrollBy(0, {'-window.innerHeight' if direction == 'up' else 'window.innerHeight'})")
            return _slice(f"scrolled {direction}", beg, end)

    # -- wait --------------------------------------------------------------

    def wait(self, selector: str = "", ms: int = 0, beg: int = 0, end: int = -1) -> str:
        with self._page() as page:
            import time
            if ms > 0:
                time.sleep(ms / 1000)
                return _slice(f"waited {ms}ms", beg, end)
            if selector:
                page.wait_for_selector(selector, timeout=15000)
                return _slice(f"element appeared: {selector}", beg, end)
            return _slice("nothing to wait for", beg, end)


# ------------------------------------------------------------------
# Tool dispatch
# ------------------------------------------------------------------

_DISPATCH: dict[str, str] = {
    "browser_navigate":      "navigate",
    "browser_go_back":       "go_back",
    "browser_go_forward":    "go_forward",
    "browser_reload":        "reload",
    "browser_click":         "click",
    "browser_dblclick":      "dblclick",
    "browser_hover":         "hover",
    "browser_type":          "type_text",
    "browser_press_key":     "press_key",
    "browser_select_option": "select_option",
    "browser_check":         "check",
    "browser_uncheck":       "uncheck",
    "browser_screenshot":    "screenshot",
    "browser_content":       "content",
    "browser_html":          "html",
    "browser_get_url":       "url",
    "browser_get_title":     "title",
    "browser_evaluate":      "evaluate",
    "browser_scroll":        "scroll",
    "browser_wait":          "wait",
}

_BEG_END = {
    "beg": {"type": "integer", "description": "Start index for partial text extraction (0-based, default 0)."},
    "end": {"type": "integer", "description": "End index for partial text extraction (-1 = to end, default -1)."},
}

_SCHEMAS: dict[str, ChatCompletionFunctionToolParam] = {
    "browser_navigate": {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "Navigate the browser to a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to navigate to."},
                    **_BEG_END,
                },
                "required": ["url"],
            },
        },
    },
    "browser_go_back": {
        "type": "function",
        "function": {
            "name": "browser_go_back",
            "description": "Go back to the previous page.",
            "parameters": {"type": "object", "properties": {**_BEG_END}},
        },
    },
    "browser_go_forward": {
        "type": "function",
        "function": {
            "name": "browser_go_forward",
            "description": "Go forward to the next page.",
            "parameters": {"type": "object", "properties": {**_BEG_END}},
        },
    },
    "browser_reload": {
        "type": "function",
        "function": {
            "name": "browser_reload",
            "description": "Reload the current page.",
            "parameters": {"type": "object", "properties": {**_BEG_END}},
        },
    },
    "browser_click": {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Click an element via CSS selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the element to click."},
                    **_BEG_END,
                },
                "required": ["selector"],
            },
        },
    },
    "browser_dblclick": {
        "type": "function",
        "function": {
            "name": "browser_dblclick",
            "description": "Double-click an element via CSS selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the element to double-click."},
                    **_BEG_END,
                },
                "required": ["selector"],
            },
        },
    },
    "browser_hover": {
        "type": "function",
        "function": {
            "name": "browser_hover",
            "description": "Hover the mouse over an element via CSS selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the element to hover over."},
                    **_BEG_END,
                },
                "required": ["selector"],
            },
        },
    },
    "browser_type": {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": "Type text into an input element (clears existing content first).",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the input element."},
                    "text": {"type": "string", "description": "Text to type."},
                    **_BEG_END,
                },
                "required": ["selector", "text"],
            },
        },
    },
    "browser_press_key": {
        "type": "function",
        "function": {
            "name": "browser_press_key",
            "description": "Press a keyboard key (e.g. Enter, Tab, ArrowDown, Escape).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Key name to press."},
                    **_BEG_END,
                },
                "required": ["key"],
            },
        },
    },
    "browser_select_option": {
        "type": "function",
        "function": {
            "name": "browser_select_option",
            "description": "Select an option in a <select> dropdown by value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the <select> element."},
                    "value": {"type": "string", "description": "Value attribute of the option to select."},
                    **_BEG_END,
                },
                "required": ["selector", "value"],
            },
        },
    },
    "browser_check": {
        "type": "function",
        "function": {
            "name": "browser_check",
            "description": "Check a checkbox or radio button.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the checkbox/radio."},
                    **_BEG_END,
                },
                "required": ["selector"],
            },
        },
    },
    "browser_uncheck": {
        "type": "function",
        "function": {
            "name": "browser_uncheck",
            "description": "Uncheck a checkbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of the checkbox."},
                    **_BEG_END,
                },
                "required": ["selector"],
            },
        },
    },
    "browser_screenshot": {
        "type": "function",
        "function": {
            "name": "browser_screenshot",
            "description": "Take a screenshot of the current browser viewport.",
            "parameters": {"type": "object", "properties": {**_BEG_END}},
        },
    },
    "browser_content": {
        "type": "function",
        "function": {
            "name": "browser_content",
            "description": "Get the visible text content of the current page.",
            "parameters": {"type": "object", "properties": {**_BEG_END}},
        },
    },
    "browser_html": {
        "type": "function",
        "function": {
            "name": "browser_html",
            "description": "Get the full HTML source of the current page.",
            "parameters": {"type": "object", "properties": {**_BEG_END}},
        },
    },
    "browser_get_url": {
        "type": "function",
        "function": {
            "name": "browser_get_url",
            "description": "Get the current page URL.",
            "parameters": {"type": "object", "properties": {**_BEG_END}},
        },
    },
    "browser_get_title": {
        "type": "function",
        "function": {
            "name": "browser_get_title",
            "description": "Get the current page title.",
            "parameters": {"type": "object", "properties": {**_BEG_END}},
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
                    **_BEG_END,
                },
                "required": ["js"],
            },
        },
    },
    "browser_scroll": {
        "type": "function",
        "function": {
            "name": "browser_scroll",
            "description": "Scroll the page up or down by one viewport height.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Direction to scroll.",
                    },
                    **_BEG_END,
                },
                "required": ["direction"],
            },
        },
    },
    "browser_wait": {
        "type": "function",
        "function": {
            "name": "browser_wait",
            "description": "Wait for an element to appear or a delay in milliseconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector to wait for."},
                    "ms": {"type": "integer", "description": "Milliseconds to sleep."},
                    **_BEG_END,
                },
            },
        },
    },
}


class PlaywrightBrowserTool(Tool):
    def __init__(self, cdp_url: str = "http://localhost:9222") -> None:
        self._browser = PlaywrightBrowser(cdp_url)

    def dispatch(self, name: str, kwargs: dict[str, object]) -> ToolOutput:
        method_name = _DISPATCH.get(name)
        if method_name is None:
            return ToolOutput(state_change=False, output="", error=f"unknown tool: {name}")

        beg = int(kwargs.get("beg", 0))
        end = int(kwargs.get("end", -1))

        try:
            method = getattr(self._browser, method_name)

            sig: dict[str, object] = {}
            if name == "browser_navigate":
                sig["url"] = kwargs["url"]
            elif name in ("browser_click", "browser_dblclick", "browser_hover"):
                sig["selector"] = kwargs["selector"]
            elif name == "browser_type":
                sig["selector"] = kwargs["selector"]
                sig["text"] = kwargs["text"]
            elif name == "browser_press_key":
                sig["key"] = kwargs["key"]
            elif name == "browser_select_option":
                sig["selector"] = kwargs["selector"]
                sig["value"] = kwargs["value"]
            elif name in ("browser_check", "browser_uncheck"):
                sig["selector"] = kwargs["selector"]
            elif name == "browser_scroll":
                sig["direction"] = kwargs["direction"]
            elif name == "browser_evaluate":
                sig["js"] = kwargs["js"]
            elif name == "browser_wait":
                sig["selector"] = kwargs.get("selector", "")
                sig["ms"] = kwargs.get("ms", 0)
            elif name == "browser_screenshot":
                sig["beg"] = kwargs.get("beg", 0)
                sig["end"] = kwargs.get("end", -1)

            if name != "browser_screenshot":
                sig["beg"] = beg
                sig["end"] = end

            out = method(**sig)
            if name in ("browser_screenshot",):
                b64 = base64.b64encode(out).decode()
                return ToolOutput(state_change=False, output="screenshot taken", error="", output_image=f"data:image/jpeg;base64,{b64}")
            return ToolOutput(state_change=True, output=str(out), error="")
        except Exception as e:
            return ToolOutput(state_change=False, output="", error=str(e))

    def tool_schemas(self) -> dict[str, ChatCompletionFunctionToolParam]:
        return dict(_SCHEMAS)
