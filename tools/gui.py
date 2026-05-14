import base64
from io import BytesIO
import os
from typing import Literal
from PIL import Image as PIL_Image
import pyautogui

def get_screenshot(
    format: Literal["PNG", "JPEG"] = "JPEG",
    temp_file: str | None = None,
    max_size: int = 1920,
) -> str:
    output_template = "data:image/{format_lowercase};base64,{data}"

    # capture the screen
    im: PIL_Image.Image = pyautogui.screenshot()

    # process screenshot
    width, height = im.size
    if max(width, height) > max_size:
        im.thumbnail(size=(max_size, max_size), resample=PIL_Image.Resampling.LANCZOS)
    if format == "JPEG":
        im = im.convert("RGB")

    # make buffer
    buffer_io: BytesIO = BytesIO()
    im.save(buffer_io, format=format, quality=80)
    buffer: bytes = buffer_io.getvalue()


    # write to temp
    if temp_file is not None:
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(buffer)

    # encode to base64
    buffer_b64: bytes = base64.b64encode(buffer)

    return output_template.format(
        format_lowercase=format.lower(),
        data=buffer_b64.decode("utf-8"),
    )