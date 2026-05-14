import pyautogui

from tools.tool import ChatCompletionFunctionToolParam, Tool, ToolOutput


class KeyboardTool(Tool):
    def key_press(self, key: str) -> ToolOutput:
        pyautogui.press(key)
        return ToolOutput(state_change=True, output=f"key_press {key} ok", error="")

    def key_type(self, text: str) -> ToolOutput:
        pyautogui.typewrite(text)
        return ToolOutput(state_change=True, output=f"key_type {len(text)} chars ok", error="")

    def key_hotkey(self, keys: list[str]) -> ToolOutput:
        pyautogui.hotkey(*keys)
        return ToolOutput(state_change=True, output=f"key_hotkey {'+'.join(keys)} ok", error="")

    def dispatch(self, name: str, kwargs: dict[str, object]) -> ToolOutput:
        if name == "key_press":
            try:
                return self.key_press(str(kwargs["key"]))
            except Exception as e:
                return ToolOutput(state_change=False, output="", error=str(e))
        elif name == "key_type":
            try:
                return self.key_type(str(kwargs["text"]))
            except Exception as e:
                return ToolOutput(state_change=False, output="", error=str(e))
        elif name == "key_hotkey":
            try:
                keys = [str(k) for k in kwargs["keys"]]
                return self.key_hotkey(keys)
            except Exception as e:
                return ToolOutput(state_change=False, output="", error=str(e))
        else:
            return ToolOutput(state_change=False, output="", error=f"unknown tool: {name}")

    def tool_schemas(self) -> dict[str, ChatCompletionFunctionToolParam]:
        return {
            "key_press": {
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
            "key_type": {
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
            "key_hotkey": {
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
        }
