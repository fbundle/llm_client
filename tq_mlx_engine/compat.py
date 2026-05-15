"""Message formatting, prompt truncation, and tool call parsing.

Bridges the gap between OpenAI API format and the model's chat template.
"""

import json
import logging
import re

# ------------------------------------------------------------------
# Message formatting
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------


def build_prompt(engine, messages: list[dict],
                tools: list[dict] | None = None,
                max_tokens: int = 512,
                max_context: int = 0,
                chat_template_kwargs: dict | None = None) -> str:
    """Format messages, apply chat template, truncate to fit context.

    Returns the prompt string ready for the model's generate().
    """
    formatted = format_messages(messages)
    prompt = engine.tokenizer.apply_chat_template(
        conversation=formatted,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        **(chat_template_kwargs or {}),
    )

    model_max = getattr(engine.tokenizer, "model_max_length", 0)
    if not model_max and engine.config:
        model_max = engine.config.get("model_max_length", 0)
        if not model_max:
            derived = getattr(engine.config, "max_position_embeddings", 0)
            model_max = derived
    if max_context > 0:
        model_max = min(max_context, model_max) if model_max > 0 else max_context
    if model_max <= 0:
        return prompt

    budget = model_max - max_tokens
    if budget <= 0:
        raise ValueError(
            f"max_tokens ({max_tokens}) exceeds model context length "
            f"({model_max}). Reduce max_tokens."
        )

    tokens = engine.tokenizer.encode(prompt)
    if len(tokens) <= budget:
        return prompt

    logging.warning(
        f"Truncating prompt: {len(tokens)} tokens > {budget} budget "
        f"(model_max={model_max}, max_tokens={max_tokens})"
    )
    A = budget // 2
    return engine.tokenizer.decode(tokens[:A] + tokens[-A:])


# ------------------------------------------------------------------
# Tool call parsing
# ------------------------------------------------------------------

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
