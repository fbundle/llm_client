import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Literal

import pyautogui
from PIL import Image as PIL_Image

from tools.tool import ChatCompletionFunctionToolParam, Tool, ToolOutput

_CURSOR_PATH = Path(__file__).resolve().parent / "cursor.png"
_CURSOR_IMG = PIL_Image.open(_CURSOR_PATH)


def _draw_cursor(
    canvas: PIL_Image.Image,
    px: int,
    py: int,
    content_width: int,
    content_height: int,
) -> None:
    csize = max(8, min(content_width, content_height) // 214)
    cursor = _CURSOR_IMG.resize(
        (csize, int(csize * _CURSOR_IMG.height / _CURSOR_IMG.width)),
        PIL_Image.Resampling.LANCZOS,
    )
    canvas.paste(cursor, (px, py), cursor)


def get_screenshot(
    format: Literal["PNG", "JPEG"] = "JPEG",
    temp_file: str | None = None,
    max_size: int = 1024,
) -> str:
    output_template = "data:image/{format_lowercase};base64,{data}"

    im = pyautogui.screenshot()
    orig_width, orig_height = im.size
    sw, sh = pyautogui.size()
    scale_x = orig_width / sw
    scale_y = orig_height / sh

    if max(orig_width, orig_height) > max_size:
        im.thumbnail(size=(max_size, max_size), resample=PIL_Image.Resampling.LANCZOS)
    if format == "JPEG":
        im = im.convert("RGB")

    new_width, new_height = im.size
    px = int(pyautogui.position()[0] * scale_x * (new_width / orig_width))
    py = int(pyautogui.position()[1] * scale_y * (new_height / orig_height))
    _draw_cursor(im, px, py, new_width, new_height)

    buffer_io = BytesIO()
    im.save(buffer_io, format=format, quality=80)
    buffer = buffer_io.getvalue()

    if temp_file is not None:
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(buffer)

    buffer_b64 = base64.b64encode(buffer)
    return output_template.format(
        format_lowercase=format.lower(),
        data=buffer_b64.decode("utf-8"),
    )


class ScreenTool(Tool):
    def take_screenshot(self) -> ToolOutput:
        image = get_screenshot(format="JPEG", temp_file="tmp/screenshot.jpg", max_size=1024)
        return ToolOutput(state_change=False, output="screenshot taken", error="", output_image=image)

    def dispatch(self, name: str, kwargs: dict[str, object]) -> ToolOutput:
        if name == "take_screenshot":
            return self.take_screenshot()
        return ToolOutput(state_change=False, output="", error=f"unknown tool: {name}")

    def tool_schemas(self) -> dict[str, ChatCompletionFunctionToolParam]:
        return {
            "take_screenshot": {
                "type": "function",
                "function": {
                    "name": "take_screenshot",
                    "description": "Capture a fresh screenshot of the screen. Call this first to see what's on screen.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        }
