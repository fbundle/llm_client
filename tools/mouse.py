
import json
from typing import Literal

import pyautogui

from tools.tool import Tool

class MouseTool(Tool):
    def mouse_move(self, x: float, y: float) -> tuple[str, bool]:
        """Move cursor to (*x*, *y*) where both are in [0, 1] relative to the screen."""
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return f"error: x and y must be in [0, 1], got x={x}, y={y}", False
        sw, sh = pyautogui.size()
        pyautogui.moveTo(x * sw, y * sh)
        return "mouse_move ok", True

    def mouse_click(self, button: Literal["left", "right", "middle"] = "left") -> tuple[str, bool]:
        """Click at the current cursor position."""
        pyautogui.click(button=button)
        return "mouse_click ok", True

    def call(self, name: str, args: str) -> tuple[str, bool]:
        try:
            kwargs = json.loads(args)
        except Exception as e:
            return str(e), False

        if name == "mouse_move":
            try:
                return self.mouse_move(float(kwargs["x"]), float(kwargs["y"]))
            except Exception as e:
                return str(e), False
        elif name == "mouse_click":
            try:
                button = str(kwargs.get("button", "left"))
                return self.mouse_click(button)
            except Exception as e:
                return str(e), False
        else:
            return "tool name not found", False

    def openai_tools(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": "mouse_move",
                    "description": (
                        "Move the cursor to a fractional position relative to the screen. "
                        "x and y are floats between 0.0 and 1.0. "
                        "(0.0, 0.0) is top-left, (0.5, 0.5) is center, (1.0, 1.0) is bottom-right. "
                        "Use this to position the cursor over a board square before clicking."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "Fractional horizontal position. Must be between 0.0 and 1.0.",
                            },
                            "y": {
                                "type": "number",
                                "description": "Fractional vertical position. Must be between 0.0 and 1.0.",
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
                    "description": "Click at the current cursor position.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "button": {
                                "type": "string",
                                "enum": ["left", "right", "middle"],
                                "description": "Mouse button to press. Default is left.",
                            },
                        },
                    },
                },
            },
        ]
