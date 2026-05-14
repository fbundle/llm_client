

from typing import Literal

import pyautogui

def mouse_click(
    x: float,
    y: float,
    button: Literal["left", "right", "middle"] = "left",
) -> None:
    """Click at (*x*, *y*) where both are in [0, 1] relative to the screen."""
    sw, sh = pyautogui.size()
    pyautogui.click(x * sw, y * sh, button=button)

openai_tools = {
    "type": "function",
    "function": {
        "name": "mouse_click",
        "description": (
            "Click on the screen at a position relative to the provided image. "
            "(0, 0) is top-left corner, (1, 1) is bottom-right corner."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "Horizontal position from 0 (left) to 1 (right).",
                },
                "y": {
                    "type": "number",
                    "description": "Vertical position from 0 (top) to 1 (bottom).",
                },
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                    "description": "Mouse button to press.",
                },
            },
            "required": ["x", "y"],
        },
    },
}
