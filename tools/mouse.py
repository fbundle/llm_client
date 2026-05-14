
import json
from typing import Literal

import pyautogui

from tools.tool import Tool, ToolOutput

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

    def call(self, name: str, args: str) -> ToolOutput:
        try:
            kwargs = json.loads(args)
        except Exception as e:
            return ToolOutput(state_change=False, output="", error=str(e))

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
        else:
            return ToolOutput(state_change=False, output="", error=f"unknown tool: {name}")

    def openai_tools(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": "mouse_move",
                    "description": (
                        "Move cursor to a position. x and y MUST be between 0.0 and 1.0 — "
                        "they are FRACTIONS of the screen, NOT pixel values. "
                        "0.0=left/top edge, 0.5=center, 1.0=right/bottom edge. "
                        "Example: to click a button at the center of the screen, use x=0.5 y=0.5."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "Fraction of screen width (0.0 to 1.0). NOT pixels.",
                            },
                            "y": {
                                "type": "number",
                                "description": "Fraction of screen height (0.0 to 1.0). NOT pixels.",
                            },
                        },
                        "required": ["x", "y"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mouse_click",
                    "description": "Click at a position. If x and y are given, moves there first then clicks. If omitted, clicks at current cursor position.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "button": {
                                "type": "string",
                                "enum": ["left", "right", "middle"],
                                "description": "Mouse button. Default is left.",
                            },
                            "x": {
                                "type": "number",
                                "description": "Fraction of screen width (0.0 to 1.0). NOT pixels.",
                            },
                            "y": {
                                "type": "number",
                                "description": "Fraction of screen height (0.0 to 1.0). NOT pixels.",
                            },
                        },
                    },
                },
            },
        ]
