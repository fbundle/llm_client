import json

import pyautogui

from tools.tool import Tool


class KeyboardTool(Tool):
    def key_press(self, key: str) -> tuple[str, bool]:
        """Press and release a single key."""
        pyautogui.press(key)
        return f"key_press {key} ok", True

    def key_type(self, text: str) -> tuple[str, bool]:
        """Type a string character by character."""
        pyautogui.typewrite(text)
        return f"key_type {len(text)} chars ok", True

    def key_hotkey(self, keys: list[str]) -> tuple[str, bool]:
        """Press a key combination (e.g. ctrl+c)."""
        pyautogui.hotkey(*keys)
        return f"key_hotkey {'+'.join(keys)} ok", True

    def call(self, name: str, args: str) -> tuple[str, bool]:
        try:
            kwargs = json.loads(args)
        except Exception as e:
            return str(e), False

        if name == "key_press":
            try:
                return self.key_press(str(kwargs["key"]))
            except Exception as e:
                return str(e), False
        elif name == "key_type":
            try:
                return self.key_type(str(kwargs["text"]))
            except Exception as e:
                return str(e), False
        elif name == "key_hotkey":
            try:
                keys = [str(k) for k in kwargs["keys"]]
                return self.key_hotkey(keys)
            except Exception as e:
                return str(e), False
        else:
            return "tool name not found", False

    def openai_tools(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": "key_press",
                    "description": "Press and release a single keyboard key.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Key name, e.g. 'enter', 'space', 'tab', 'a', '1', 'f1'.",
                            },
                        },
                        "required": ["key"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "key_type",
                    "description": "Type a string character by character.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to type.",
                            },
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "key_hotkey",
                    "description": "Press a key combination like ctrl+c or alt+tab.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keys": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Keys to press together, e.g. ['ctrl', 'c'].",
                            },
                        },
                        "required": ["keys"],
                    },
                },
            },
        ]
