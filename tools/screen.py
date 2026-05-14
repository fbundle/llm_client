import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Literal

import pyautogui
from PIL import Image as PIL_Image
from PIL import ImageDraw, ImageFont

_CURSOR_PATH = Path(__file__).resolve().parent / "cursor.png"
_CURSOR_IMG = PIL_Image.open(_CURSOR_PATH)

_RULER_SIZE = 36  # px margin for rulers
_DRAW_RULERS = False


def _draw_rulers(draw: ImageDraw.Draw, width: int, height: int) -> None:
    """Draw coordinate rulers on the top and left margins (0.0–1.0)."""
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except OSError:
        font = ImageFont.load_default()

    # Top ruler (x-axis)
    draw.rectangle([(0, 0), (width - 1, _RULER_SIZE - 1)], fill=(40, 40, 40))
    for i in range(11):  # 0.0 to 1.0
        x = int(_RULER_SIZE + i * (width - _RULER_SIZE) / 10)
        # Major tick
        draw.line([(x, 0), (x, _RULER_SIZE)], fill=(180, 180, 180), width=1)
        label = f"{i / 10:.1f}"
        bbox = font.getbbox(label)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw // 2, 2), label, fill=(220, 220, 220), font=font)
        # Sub-ticks (every 0.05)
        for j in (1, 2):
            xi = int(_RULER_SIZE + (i + j * 0.5 / 10) * (width - _RULER_SIZE) / 10)
            draw.line([(xi, _RULER_SIZE - 10), (xi, _RULER_SIZE)], fill=(120, 120, 120), width=1)

    # Left ruler (y-axis)
    draw.rectangle([(0, 0), (_RULER_SIZE - 1, height - 1)], fill=(40, 40, 40))
    for i in range(11):  # 0.0 to 1.0
        y = int(_RULER_SIZE + i * (height - _RULER_SIZE) / 10)
        # Major tick
        draw.line([(0, y), (_RULER_SIZE, y)], fill=(180, 180, 180), width=1)
        label = f"{i / 10:.1f}"
        bbox = font.getbbox(label)
        th = bbox[3] - bbox[1]
        draw.text((2, y - th // 2), label, fill=(220, 220, 220), font=font)
        # Sub-ticks (every 0.05)
        for j in (1, 2):
            yi = int(_RULER_SIZE + (i + j * 0.5 / 10) * (height - _RULER_SIZE) / 10)
            draw.line([(_RULER_SIZE - 10, yi), (_RULER_SIZE, yi)], fill=(120, 120, 120), width=1)

    # Corner box
    draw.rectangle([(0, 0), (_RULER_SIZE - 1, _RULER_SIZE - 1)], fill=(50, 50, 50))


def _draw_cursor(
    canvas: PIL_Image.Image,
    px: int,
    py: int,
    content_width: int,
    content_height: int,
) -> None:
    """Paste the cursor image onto the canvas at (px, py)."""
    csize = max(20, min(content_width, content_height) // 25)
    cursor = _CURSOR_IMG.resize(
        (csize, int(csize * _CURSOR_IMG.height / _CURSOR_IMG.width)),
        PIL_Image.Resampling.LANCZOS,
    )
    canvas.paste(cursor, (px, py), cursor)


def _annotate_screenshot(
    im: PIL_Image.Image,
    orig_width: int,
    orig_height: int,
    scale_x: float,
    scale_y: float,
) -> PIL_Image.Image:
    """Build a new image with rulers and cursor overlay from a raw screenshot."""
    mx, my = pyautogui.position()
    new_width, new_height = im.size

    if not _DRAW_RULERS:
        px = int(mx * scale_x * (new_width / orig_width))
        py = int(my * scale_y * (new_height / orig_height))
        _draw_cursor(im, px, py, new_width, new_height)
        return im

    canvas = PIL_Image.new("RGB", (new_width + _RULER_SIZE, new_height + _RULER_SIZE), (30, 30, 30))
    canvas.paste(im, (_RULER_SIZE, _RULER_SIZE))

    px = int(_RULER_SIZE + mx * scale_x * (new_width / orig_width))
    py = int(_RULER_SIZE + my * scale_y * (new_height / orig_height))
    _draw_cursor(canvas, px, py, new_width, new_height)

    _draw_rulers(ImageDraw.Draw(canvas), canvas.width, canvas.height)
    return canvas


def get_screenshot(
    format: Literal["PNG", "JPEG"] = "JPEG",
    temp_file: str | None = None,
    max_size: int = 1920,
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

    canvas = _annotate_screenshot(im, orig_width, orig_height, scale_x, scale_y)

    buffer_io = BytesIO()
    canvas.save(buffer_io, format=format, quality=80)
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
