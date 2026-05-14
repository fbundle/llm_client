import json

from py_mini_racer import MiniRacer

from tools.tool import Tool, ToolOutput


class JSRuntimeTool(Tool):
    def __init__(self) -> None:
        self._ctx = MiniRacer()

    def js_eval(self, code: str) -> ToolOutput:
        try:
            result = self._ctx.eval(code)
            return ToolOutput(state_change=False, output=json.dumps(result), error="")
        except Exception as e:
            return ToolOutput(state_change=False, output="", error=str(e))

    def call(self, name: str, args: str) -> ToolOutput:
        if name != "js_eval":
            return ToolOutput(state_change=False, output="", error=f"unknown tool: {name}")
        try:
            kwargs = json.loads(args)
            return self.js_eval(str(kwargs["code"]))
        except Exception as e:
            return ToolOutput(state_change=False, output="", error=str(e))

    def openai_tools(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "js_eval",
                    "description": (
                        "Execute JavaScript code and return the result. "
                        "Use this for calculations, coordinate math, or any computation. "
                        "The last expression's value is returned."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "JavaScript code to execute. The return value is the last expression.",
                            },
                        },
                        "required": ["code"],
                    },
                },
            },
        ]
