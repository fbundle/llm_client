"""OpenAI-compatible FastAPI server on top of tq_mlx_engine.

Handles HTTP, SSE streaming, model caching.
Delegates raw generation to engine.MlxEngine and compatibility to compat.
"""

import json
import time
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from moka_py import Moka
from tq_mlx_engine.compat import build_prompt, parse_tool_calls
from tq_mlx_engine.engine import MlxEngine


class TQServer:
    """OpenAI-compatible chat completion server with TurboQuant KV cache."""

    def __init__(self, model_path: str, adapter_path: str | None = None,
                 max_context: int = 32768, **engine_kwargs):
        self.fastapi = FastAPI(title="TurboQuant MLX Server")
        self.default_model = model_path
        self.adapter_path = adapter_path
        self.max_context = max_context
        self._engine_kwargs = engine_kwargs

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
            initializer=lambda: MlxEngine(model_path, self.adapter_path,
                                          **self._engine_kwargs),
            tti=60 * 10,
            ttl=60 * 60 * 24,
        )

    def _stream_response(self, engine: MlxEngine,
                         messages: list[dict], max_tokens: int,
                         temperature: float, top_p: float, top_k: int,
                         min_p: float,
                         tools: list[dict] | None,
                         chat_template_kwargs: dict | None) -> Iterator[str]:
        """Yield SSE event strings from raw engine tokens."""
        prompt = build_prompt(engine, messages, tools, max_tokens,
                              self.max_context, chat_template_kwargs)
        prompt_tokens = engine.tokenizer.encode(prompt)

        t0 = time.time()
        generated: list[int] = []
        prev_text = ""

        for token_id in engine.generate(
            prompt=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        ):
            generated.append(token_id)
            full_text = engine.tokenizer.decode(generated)
            delta_text = full_text[len(prev_text):]
            prev_text = full_text

            delta: dict = {}
            if delta_text:
                delta["content"] = delta_text

            chunk = {
                "id": f"chatcmpl-{int(t0 * 1000)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": engine.model_path,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }

            _, tool_calls = parse_tool_calls(full_text)
            if tool_calls:
                chunk["choices"][0]["delta"]["tool_calls"] = tool_calls
                chunk["choices"][0]["finish_reason"] = "tool_calls"

            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    def _sync_response(self, engine: MlxEngine,
                       messages: list[dict], max_tokens: int,
                       temperature: float, top_p: float, top_k: int,
                       min_p: float,
                       tools: list[dict] | None,
                       chat_template_kwargs: dict | None) -> dict:
        """Non-streaming chat completion."""
        prompt = build_prompt(engine, messages, tools, max_tokens,
                              self.max_context, chat_template_kwargs)
        prompt_tokens = engine.tokenizer.encode(prompt)

        t0 = time.time()
        generated: list[int] = []
        finish_reason = "stop"

        for token_id in engine.generate(
            prompt=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        ):
            generated.append(token_id)

        full_text = engine.tokenizer.decode(generated)
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
            "usage": {"completion_tokens": len(generated)},
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
        chat_template_kwargs: dict | None = request.get("chat_template_kwargs")

        if stream:
            return StreamingResponse(
                self._stream_response(
                    engine, messages, max_tokens, temperature, top_p,
                    top_k, min_p, tools, chat_template_kwargs,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            return self._sync_response(
                engine, messages, max_tokens, temperature, top_p,
                top_k, min_p, tools, chat_template_kwargs,
            )

    async def list_models(self) -> dict:
        return {
            "object": "list",
            "data": [{"id": self.default_model, "object": "model"}],
        }
