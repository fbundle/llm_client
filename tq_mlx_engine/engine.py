"""Pure MLX model generation with TurboQuant KV cache compression.

No HTTP, no tool call parsing — just messages in, text out.
"""

import json
import logging
from typing import Iterator

import mlx_lm
import mlx_lm.models.cache as cache_module
from mlx_lm.models.cache import KVCache
from tq_mlx_engine.turboquant_mlx import apply_patch
from tq_mlx_engine.turboquant_mlx.adaptive import make_adaptive_cache


def patch_cache(bits: int, fp16_layers: int, fused: bool) -> None:
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


def format_messages(messages: list[dict]) -> list[dict]:
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
    """Holds a loaded MLX model + tokenizer. Streams raw text for chat prompts."""

    def __init__(self, model_path: str, adapter_path: str | None = None):
        self.model_path = model_path
        logging.info(f"Loading model: {model_path}")
        self.model, self.tokenizer = mlx_lm.load(
            path_or_hf_repo=model_path,
            adapter_path=adapter_path,
        )
        logging.info("Model loaded")

    def build_prompt(self, messages: list[dict],
                     tools: list[dict] | None = None,
                     chat_template_kwargs: dict | None = None) -> str:
        formatted = format_messages(messages)
        if chat_template_kwargs is None:
            chat_template_kwargs = {}
        return self.tokenizer.apply_chat_template(
            conversation=formatted,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

    def generate(self, messages: list[dict],
                 max_tokens: int = 512,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 top_k: int = 0,
                 min_p: float = 0.0,
                 repetition_penalty: float = 1.0,
                 tools: list[dict] | None = None,
                 chat_template_kwargs: dict | None = None) -> Iterator[str]:
        """Yield raw text tokens from the model for the given messages."""
        prompt = self.build_prompt(messages, tools, chat_template_kwargs)

        sampler = mlx_lm.sample_utils.make_sampler(
            temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
        )
        logits_processors = mlx_lm.sample_utils.make_logits_processors(
            repetition_penalty=repetition_penalty,
        )

        for response in mlx_lm.stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            yield response.text
