"""OpenAI-compatible FastAPI server on top of tq_mlx_engine.

Handles: HTTP, SSE streaming, tool call parsing, model caching.
Delegates raw generation to engine.MlxEngine.
"""

import json
import logging
import re
import time
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from moka_py import Moka
from tq_mlx_engine.engine import MlxEngine

# Regex for Qwen XML tool call format:
#   <tool_call>
#   <function=NAME>
#   <parameter=KEY>VALUE</parameter>
#   </function>
#   </tool_call>
# Also supports JSON format:
#   <tool_call>{"name": "...", "arguments": {...}}</tool_call>
_TC_RE = re.compile(
    r"<tool_call>\s*"
    r"(.*?)"
    r"</tool_call>",
    re.DOTALL,
)
_FUNC_RE = re.compile(
    r"<function=([^>]+)>\s*(.*?)\s*</function>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def _make_tc_delta(index: int, name: str, arguments: dict) -> dict:
    """Build an OpenAI-format tool_call delta dict."""
    return {
        "index": index,
        "id": f"call_{index}_{name}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Extract tool calls from *text*, returning (cleaned_text, tool_calls).

    Supports two formats:

    **Qwen XML:**
        <tool_call>
        <function=NAME><parameter=KEY>VALUE</parameter></function>
        </tool_call>

    **JSON:**
        <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """
    tool_calls: list[dict] = []
    cleaned = text

    for tc_match in _TC_RE.finditer(text):
        block = tc_match.group(1).strip()
        cleaned = cleaned.replace(tc_match.group(0), "")

        # Try JSON format first
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                name = parsed.get("name", "")
                args = parsed.get("arguments", parsed.get("input", {}))
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(_make_tc_delta(0, name, args))
                continue
        except json.JSONDecodeError:
            pass

        # Qwen XML format
        for fn_match in _FUNC_RE.finditer(block):
            name = fn_match.group(1).strip()
            params_block = fn_match.group(2)
            kwargs: dict = {}
            for pm in _PARAM_RE.finditer(params_block):
                key = pm.group(1).strip()
                val = pm.group(2).strip()
                try:
                    kwargs[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    kwargs[key] = val
            tool_calls.append(_make_tc_delta(0, name, kwargs))

    return cleaned.strip(), tool_calls


class TQServer:
    """OpenAI-compatible chat completion server with TurboQuant KV cache."""

    def __init__(self, model_path: str, adapter_path: str | None = None):
        self.fastapi = FastAPI(title="TurboQuant MLX Server")
        self.default_model = model_path
        self.adapter_path = adapter_path

        # Cache up to 3 models, evict after 10 min idle, 24h max
        self._engines: Moka[str, MlxEngine] = Moka(capacity=3)

        self.fastapi.router.add_api_route(
            path="/v1/chat/completions",
            endpoint=self.chat_completions,
            methods=["POST"],
            response_model=None,
        )

        self.fastapi.router.add_api_route(
            path="/v1/models",
            endpoint=self.list_models,
            methods=["GET"],
        )

    def _get_engine(self, model_path: str) -> MlxEngine:
        return self._engines.get_with(
            key=model_path,
            initializer=lambda: MlxEngine(model_path, self.adapter_path),
            tti=60 * 10,
            ttl=60 * 60 * 24,
        )

    def _stream_response(self, engine: MlxEngine, messages: list[dict],
                         max_tokens: int, temperature: float, top_p: float,
                         top_k: int, min_p: float, repetition_penalty: float,
                         tools: list[dict] | None,
                         chat_template_kwargs: dict | None) -> Iterator[str]:
        """Yield SSE event strings from raw engine text."""
        t0 = time.time()
        full_text = ""

        for token in engine.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        ):
            full_text += token
            delta: dict = {}
            if token:
                delta["content"] = token

            chunk = {
                "id": f"chatcmpl-{int(t0 * 1000)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": engine.model_path,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }

            # Check for complete tool call blocks on every token
            _, tool_calls = parse_tool_calls(full_text)
            if tool_calls:
                chunk["choices"][0]["delta"]["tool_calls"] = tool_calls
                chunk["choices"][0]["finish_reason"] = "tool_calls"

            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    def _sync_response(self, engine: MlxEngine, messages: list[dict],
                       max_tokens: int, temperature: float, top_p: float,
                       top_k: int, min_p: float, repetition_penalty: float,
                       tools: list[dict] | None,
                       chat_template_kwargs: dict | None) -> dict:
        """Non-streaming chat completion."""
        t0 = time.time()
        full_text = ""
        finish_reason = "stop"

        for token in engine.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        ):
            full_text += token

        clean_text, tool_calls = parse_tool_calls(full_text)
        if tool_calls:
            finish_reason = "tool_calls"

        message: dict = {"role": "assistant", "content": clean_text or None}
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": f"chatcmpl-{int(t0 * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": engine.model_path,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {"completion_tokens": len(full_text.split())},
        }

    async def chat_completions(self, request: dict) -> dict | StreamingResponse:
        """OpenAI-compatible chat completions endpoint."""
        model_path: str = request.get("model", self.default_model)
        engine = self._get_engine(model_path)

        messages: list[dict] = request.get("messages", [])
        tools: list[dict] | None = request.get("tools")
        stream: bool = request.get("stream", False)
        max_tokens: int = request.get("max_tokens", 512)
        temperature: float = request.get("temperature", 0.0)
        top_p: float = request.get("top_p", 1.0)
        top_k: int = request.get("top_k", 0)
        min_p: float = request.get("min_p", 0.0)
        repetition_penalty: float = request.get("repetition_penalty", 1.0)
        chat_template_kwargs: dict | None = request.get("chat_template_kwargs")

        if stream:
            return StreamingResponse(
                self._stream_response(
                    engine, messages, max_tokens, temperature, top_p, top_k,
                    min_p, repetition_penalty, tools, chat_template_kwargs,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            return self._sync_response(
                engine, messages, max_tokens, temperature, top_p, top_k,
                min_p, repetition_penalty, tools, chat_template_kwargs,
            )

    async def list_models(self) -> dict:
        return {
            "object": "list",
            "data": [{"id": self.default_model, "object": "model"}],
        }
