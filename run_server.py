#!/usr/bin/env python3
"""OpenAI-compatible FastAPI server with TurboQuant KV cache compression.

Usage:
    uv run python run_server.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
    uv run python run_server.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --tq-bits 3 --tq-fused
"""

import argparse
import json
import logging
import time
from typing import Iterator

import mlx.core as mx
import mlx_lm
import mlx_lm.models.cache as cache_module
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from mlx_lm.models.cache import KVCache
from moka_py import Moka
from turboquant_mlx import apply_patch
from turboquant_mlx.adaptive import make_adaptive_cache


def _patch_cache(bits: int, fp16_layers: int, fused: bool) -> None:
    """Monkey-patch mlx_lm's make_prompt_cache to use TurboQuantKVCache."""

    def _turboquant_make_prompt_cache(model, max_kv_size=None):
        if hasattr(model, "make_cache"):
            default = model.make_cache()
            if default and not isinstance(default[0], KVCache):
                return default

        num_layers = len(model.layers)
        return make_adaptive_cache(
            num_layers,
            bits=bits,
            fp16_layers=fp16_layers,
            fused=fused,
            model=model,
        )

    cache_module.make_prompt_cache = _turboquant_make_prompt_cache


def _flatten_content(content: str | list) -> str:
    """Flatten OpenAI multimodal content to a plain string."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            kind = part.get("type", "")
            if kind == "text":
                parts.append(part.get("text", ""))
            elif kind == "image_url":
                parts.append("[image]")
        elif isinstance(part, str):
            parts.append(part)
    return "\n".join(parts)


def _normalize_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Normalize OpenAI tool_calls so arguments is a dict, not a JSON string.

    OpenAI API sends ``function.arguments`` as a JSON string, but tokenizer
    Jinja2 templates apply the ``items()`` filter which requires a dict.
    """
    normalized: list[dict] = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            fn = {**fn, "arguments": json.loads(args)}
        normalized.append({**tc, "function": fn})
    return normalized


def _format_messages(messages: list[dict]) -> list[dict]:
    """Normalize OpenAI message format for the tokenizer's chat template."""
    formatted: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        entry: dict = {
            "role": role,
            "content": _flatten_content(msg.get("content", "")),
        }
        if role == "assistant" and "tool_calls" in msg:
            entry["tool_calls"] = _normalize_tool_calls(msg["tool_calls"])
        if role == "tool" and "tool_call_id" in msg:
            entry["tool_call_id"] = msg["tool_call_id"]
        formatted.append(entry)
    return formatted


class MlxEngine:
    """Holds a loaded MLX model + tokenizer and handles chat generation."""

    def __init__(self, model_path: str, adapter_path: str | None = None):
        self.model_path = model_path
        logging.info(f"Loading model: {model_path}")
        self.model, self.tokenizer = mlx_lm.load(
            path_or_hf_repo=model_path,
            adapter_path=adapter_path,
        )
        logging.info("Model loaded")

    def _build_prompt(self, messages: list[dict],
                      chat_template_kwargs: dict | None = None) -> str:
        formatted = _format_messages(messages)
        if chat_template_kwargs is None:
            chat_template_kwargs = {}
        return self.tokenizer.apply_chat_template(
            conversation=formatted,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

    def stream_generate(self, messages: list[dict], max_tokens: int,
                        temperature: float, top_p: float, top_k: int,
                        min_p: float, repetition_penalty: float,
                        chat_template_kwargs: dict | None = None) -> Iterator[str]:
        """Yield SSE event strings for a chat completion stream."""
        prompt = self._build_prompt(messages, chat_template_kwargs)

        sampler = mlx_lm.sample_utils.make_sampler(
            temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
        )
        logits_processors = mlx_lm.sample_utils.make_logits_processors(
            repetition_penalty=repetition_penalty,
        )

        t0 = time.time()
        for response in mlx_lm.stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            delta: dict = {}
            if response.text:
                delta["content"] = response.text

            chunk = {
                "id": f"chatcmpl-{int(t0 * 1000)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_path,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }

            if response.finish_reason:
                chunk["choices"][0]["finish_reason"] = response.finish_reason

            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    def generate(self, messages: list[dict], max_tokens: int,
                 temperature: float, top_p: float, top_k: int,
                 min_p: float, repetition_penalty: float,
                 chat_template_kwargs: dict | None = None) -> dict:
        """Non-streaming chat completion."""
        prompt = self._build_prompt(messages, chat_template_kwargs)

        sampler = mlx_lm.sample_utils.make_sampler(
            temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
        )
        logits_processors = mlx_lm.sample_utils.make_logits_processors(
            repetition_penalty=repetition_penalty,
        )

        t0 = time.time()
        full_text = ""
        finish_reason = "stop"

        for response in mlx_lm.stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            full_text += response.text
            if response.finish_reason:
                finish_reason = response.finish_reason

        return {
            "id": f"chatcmpl-{int(t0 * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_path,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": finish_reason,
            }],
            "usage": {"completion_tokens": len(full_text.split())},
        }


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
            tti=60 * 10,   # evict 10 min after last access
            ttl=60 * 60 * 24,  # max 24 hours
        )

    async def chat_completions(self, request: dict) -> dict | StreamingResponse:
        """OpenAI-compatible chat completions endpoint."""
        model_path: str = request.get("model", self.default_model)
        engine = self._get_engine(model_path)

        messages: list[dict] = request.get("messages", [])
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
                engine.stream_generate(
                    messages, max_tokens, temperature, top_p, top_k, min_p,
                    repetition_penalty, chat_template_kwargs,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            return engine.generate(
                messages, max_tokens, temperature, top_p, top_k, min_p,
                repetition_penalty, chat_template_kwargs,
            )

    async def list_models(self) -> dict:
        return {
            "object": "list",
            "data": [{"id": self.default_model, "object": "model"}],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant MLX HTTP Server")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the MLX model weights, tokenizer, and config")
    parser.add_argument("--adapter-path", type=str,
                        help="Optional path for trained adapter weights")

    # Server
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1334)

    # TurboQuant
    parser.add_argument("--tq-bits", type=int, default=3,
                        help="TurboQuant bit width for KV cache (1-4, default: 3)")
    parser.add_argument("--tq-fused", action="store_true",
                        help="Use fused Metal attention kernels")
    parser.add_argument("--tq-fp16-layers", type=int, default=4,
                        help="First and last N layers to keep in FP16 (default: 4)")

    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Apply TurboQuant patches before loading the model
    _patch_cache(args.tq_bits, args.tq_fp16_layers, args.tq_fused)

    if args.tq_fused:
        apply_patch()

    logging.info(
        f"TurboQuant: {args.tq_bits}-bit, "
        f"{args.tq_fp16_layers}+{args.tq_fp16_layers} FP16 layers"
        f"{' (fused)' if args.tq_fused else ''}"
    )

    # Metal memory
    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)
        logging.info(f"Metal wired limit: {wired_limit / (1024**3):.1f} GB")

    server = TQServer(args.model, args.adapter_path)

    # Pre-load the default model so the first request doesn't block
    server._get_engine(args.model)

    import uvicorn
    logging.info(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(server.fastapi, host=args.host, port=args.port,
                log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
