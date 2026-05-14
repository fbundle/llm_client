
from typing import Literal

import pyautogui

from tools.tool import ChatCompletionFunctionToolParam, Tool, ToolOutput

class MouseTool(Tool):
    def mouse_move(self, x: float, y: float) -> ToolOutput:
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return ToolOutput(state_change=False, output="", error=f"x and y must be in [0, 1], got x={x}, y={y}")
        sw, sh = pyautogui.size()
        pyautogui.moveTo(x * sw, y * sh)
        return ToolOutput(state_change=True, output="mouse_move ok", error="")

    def mouse_click(
        self,
        button: Literal["left", "right", "middle"] = "left",
        x: float | None = None,
        y: float | None = None,
    ) -> ToolOutput:
        if x is not None and y is not None:
            if not (0 <= x <= 1 and 0 <= y <= 1):
                return ToolOutput(state_change=False, output="", error=f"x and y must be in [0, 1], got x={x}, y={y}")
            sw, sh = pyautogui.size()
            pyautogui.moveTo(x * sw, y * sh)
        pyautogui.click(button=button)
        return ToolOutput(state_change=True, output="mouse_click ok", error="")

    def mouse_drag(
        self,
        x: float,
        y: float,
        button: Literal["left", "right", "middle"] = "left",
    ) -> ToolOutput:
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return ToolOutput(state_change=False, output="", error=f"x and y must be in [0, 1], got x={x}, y={y}")
        sw, sh = pyautogui.size()
        pyautogui.drag(x * sw, y * sh, button=button)
        return ToolOutput(state_change=True, output="mouse_drag ok", error="")

    def mouse_scroll(self, clicks: int) -> ToolOutput:
        pyautogui.scroll(clicks)
        return ToolOutput(state_change=True, output=f"mouse_scroll {clicks} ok", error="")

    def dispatch(self, name: str, kwargs: dict[str, object]) -> ToolOutput:
        if name == "mouse_move":
            try:
                return self.mouse_move(float(kwargs["x"]), float(kwargs["y"]))
            except Exception as e:
                return ToolOutput(state_change=False, output="", error=str(e))
        elif name == "mouse_click":
            try:
                button = str(kwargs.get("button", "left"))
                x = float(kwargs["x"]) if "x" in kwargs else None
                y = float(kwargs["y"]) if "y" in kwargs else None
                return self.mouse_click(button=button, x=x, y=y)
            except Exception as e:
                return ToolOutput(state_change=False, output="", error=str(e))
        elif name == "mouse_drag":
            try:
                button = str(kwargs.get("button", "left"))
                return self.mouse_drag(float(kwargs["x"]), float(kwargs["y"]), button=button)
            except Exception as e:
                return ToolOutput(state_change=False, output="", error=str(e))
        elif name == "mouse_scroll":
            try:
                return self.mouse_scroll(int(kwargs["clicks"]))
            except Exception as e:
                return ToolOutput(state_change=False, output="", error=str(e))
        else:
            return ToolOutput(state_change=False, output="", error=f"unknown tool: {name}")

    def tool_schemas(self) -> dict[str, ChatCompletionFunctionToolParam]:
        return {
            "mouse_move": {
                "type": "function",
                "function": {
                    "name": "mouse_move",
                    "description": "Move cursor to (x, y) as fractions of screen size (0.0–1.0).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "Fraction of screen width (0.0=left, 1.0=right).",
                            },
                            "y": {
                                "type": "number",
                                "description": "Fraction of screen height (0.0=top, 1.0=bottom).",
                            },
                        },
                        "required": ["x", "y"],
                    },
                },
            },
            "mouse_click": {
                "type": "function",
                "function": {
                    "name": "mouse_click",
                    "description": "Click mouse. If (x, y) given, moves there first. If omitted, clicks at current position.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "button": {
                                "type": "string",
                                "enum": ["left", "right", "middle"],
                                "description": "Mouse button (default: left).",
                            },
                            "x": {
                                "type": "number",
                                "description": "Fraction of screen width (0.0–1.0).",
                            },
                            "y": {
                                "type": "number",
                                "description": "Fraction of screen height (0.0–1.0).",
                            },
                        },
                    },
                },
            },
            "mouse_drag": {
                "type": "function",
                "function": {
                    "name": "mouse_drag",
                    "description": "Drag from current position to (x, y). Use mouse_move first to set the starting point.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "Fraction of screen width (0.0–1.0) — drag destination.",
                            },
                            "y": {
                                "type": "number",
                                "description": "Fraction of screen height (0.0–1.0) — drag destination.",
                            },
                            "button": {
                                "type": "string",
                                "enum": ["left", "right", "middle"],
                                "description": "Button held during drag (default: left).",
                            },
                        },
                        "required": ["x", "y"],
                    },
                },
            },
            "mouse_scroll": {
                "type": "function",
                "function": {
                    "name": "mouse_scroll",
                    "description": "Scroll the mouse wheel. Positive clicks scroll up, negative scroll down.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "clicks": {
                                "type": "integer",
                                "description": "Number of scroll clicks. Positive = up, negative = down.",
                            },
                        },
                        "required": ["clicks"],
                    },
                },
            },
        }
