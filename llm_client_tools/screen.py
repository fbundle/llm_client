import base64
from io import BytesIO
from pathlib import Path
from typing import Literal

import pyautogui
from PIL import Image as PIL_Image

from llm_client.tool import ChatCompletionFunctionToolParam, Tool, ToolOutput

_CURSOR_PATH = Path(__file__).resolve().parent.parent / "assets" / "cursor.png"
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


def get_screenshot(max_size: int = 1024) -> PIL_Image.Image:
    """Capture a screenshot, resize if needed, and draw the cursor overlay."""
    im = pyautogui.screenshot()
    orig_width, orig_height = im.size

    if max(orig_width, orig_height) > max_size:
        im.thumbnail(size=(max_size, max_size), resample=PIL_Image.Resampling.LANCZOS)

    new_width, new_height = im.size
    scale_x = orig_width / pyautogui.size()[0]
    scale_y = orig_height / pyautogui.size()[1]

    mx, my = pyautogui.position()
    px = int(mx * scale_x * (new_width / orig_width))
    py = int(my * scale_y * (new_height / orig_height))
    _draw_cursor(im, px, py, new_width, new_height)

    return im


def crop_screenshot(
    im: PIL_Image.Image,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> PIL_Image.Image:
    """Crop *im* to the fractional region defined by (x1,y1)-(x2,y2) in [0,1] range."""
    w, h = pyautogui.size()
    im_w, im_h = im.size
    scale_x = im_w / w
    scale_y = im_h / h

    x1_px = max(0, int(x1 * w * scale_x))
    y1_px = max(0, int(y1 * h * scale_y))
    x2_px = min(im_w, int(x2 * w * scale_x))
    y2_px = min(im_h, int(y2 * h * scale_y))

    return im.crop((x1_px, y1_px, x2_px, y2_px))


def encode_base64(im: PIL_Image.Image, format: Literal["PNG", "JPEG"] = "JPEG") -> str:
    """Encode *im* to a base64 data URL string."""
    if format == "JPEG":
        im = im.convert("RGB")
    buffer_io = BytesIO()
    im.save(buffer_io, format=format, quality=80)
    buffer = buffer_io.getvalue()
    buffer_b64 = base64.b64encode(buffer)
    return f"data:image/{format.lower()};base64,{buffer_b64.decode('utf-8')}"


class ScreenTool(Tool):
    def take_screenshot(
        self,
        x1: float | None = None,
        y1: float | None = None,
        x2: float | None = None,
        y2: float | None = None,
    ) -> ToolOutput:
        im = get_screenshot()

        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            im = crop_screenshot(im, x1, y1, x2, y2)

        image = encode_base64(im)
        return ToolOutput(state_change=False, output="screenshot taken", error="", output_image=image)

    def dispatch(self, name: str, kwargs: dict[str, object]) -> ToolOutput:
        if name != "take_screenshot":
            return ToolOutput(state_change=False, output="", error=f"unknown tool: {name}")
        try:
            return self.take_screenshot(
                x1=kwargs.get("x1"),
                y1=kwargs.get("y1"),
                x2=kwargs.get("x2"),
                y2=kwargs.get("y2"),
            )
        except Exception as e:
            return ToolOutput(state_change=False, output="", error=str(e))

    def tool_schemas(self) -> dict[str, ChatCompletionFunctionToolParam]:
        return {
            "take_screenshot": {
                "type": "function",
                "function": {
                    "name": "take_screenshot",
                    "description": "Capture a fresh screenshot of the screen. Call this first to see what's on screen. Optionally crop to a region with x1,y1,x2,y2 in fractional coordinates (0.0-1.0).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x1": {
                                "type": "number",
                                "description": "Top-left X of crop region (0.0 = leftmost, 1.0 = rightmost).",
                            },
                            "y1": {
                                "type": "number",
                                "description": "Top-left Y of crop region (0.0 = topmost, 1.0 = bottommost).",
                            },
                            "x2": {
                                "type": "number",
                                "description": "Bottom-right X of crop region (0.0 = leftmost, 1.0 = rightmost).",
                            },
                            "y2": {
                                "type": "number",
                                "description": "Bottom-right Y of crop region (0.0 = topmost, 1.0 = bottommost).",
                            },
                        },
                    },
                },
            },
        }
