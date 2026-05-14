import json

from py_mini_racer import MiniRacer

from tools.tool import ChatCompletionFunctionToolParam, Tool, ToolOutput


class JSRuntimeTool(Tool):
    def __init__(self) -> None:
        self._ctx = MiniRacer()

    def js_eval(self, code: str) -> ToolOutput:
        try:
            result = self._ctx.eval(code)
            return ToolOutput(state_change=False, output=json.dumps(result), error="")
        except Exception as e:
            return ToolOutput(state_change=False, output="", error=str(e))

    def dispatch(self, name: str, kwargs: dict[str, object]) -> ToolOutput:
        if name != "js_eval":
            return ToolOutput(state_change=False, output="", error=f"unknown tool: {name}")
        try:
            return self.js_eval(str(kwargs["code"]))
        except Exception as e:
            return ToolOutput(state_change=False, output="", error=str(e))

    def tool_schemas(self) -> dict[str, ChatCompletionFunctionToolParam]:
        return {
            "js_eval": {
                "type": "function",
                "function": {
                    "name": "js_eval",
                    "description": "Execute JavaScript code. The value of the last expression is returned. Use for calculations, coordinate math, or any computation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "JavaScript code to evaluate.",
                            },
                        },
                        "required": ["code"],
                    },
                },
            },
        }
