import os
import base64
from typing import Literal
import pyautogui
from PIL import Image as PIL_Image
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv

def must_get_env(key: str, default: str | None = None) -> str:
    val: str | None = os.environ.get(key, default=default)
    assert isinstance(val, str)
    return val

load_dotenv()

model = must_get_env("LMS_MODEL")

client = OpenAI(
    base_url=must_get_env("LMS_BASE_URL"),
    api_key=must_get_env("LMS_API_KEY"),
)

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


screenshot = get_screenshot(format="JPEG", temp_file="tmp/screenshot.jpg")


# 2. Call OpenAI with Vision
response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "You are a robot vision system. Look at the screen and describe the current state."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is happening on the screen right now?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": screenshot,
                        "detail": "low",
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
