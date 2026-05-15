#!/usr/bin/env python3
"""OpenAI-compatible server with TurboQuant KV cache compression.

Usage:
    python run_server.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 1334
    python run_server.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --tq-bits 3 --tq-fused
"""

import argparse
import logging

import mlx.core as mx
import mlx_lm.models.cache as cache_module
from mlx_lm.models.cache import KVCache
from mlx_lm.server import ModelProvider, run
from turboquant_mlx import apply_patch
from turboquant_mlx.adaptive import make_adaptive_cache


def _parse_size(s: str) -> int:
    """Parse a size string like '2GB' or '500MB' into bytes."""
    s = s.strip().upper()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(s)


def _patch_cache(bits: int, fp16_layers: int, fused: bool) -> None:
    """Monkey-patch mlx_lm's make_prompt_cache to use TurboQuantKVCache."""

    def _turboquant_make_prompt_cache(model, max_kv_size=None):
        # Delegate to model if it has its own cache factory (MLA, SSM, etc.)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant MLX HTTP Server")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the MLX model weights, tokenizer, and config")
    parser.add_argument("--adapter-path", type=str,
                        help="Optional path for trained adapter weights")
    parser.add_argument("--draft-model", type=str,
                        help="Model for speculative decoding")
    parser.add_argument("--num-draft-tokens", type=int, default=3,
                        help="Number of draft tokens for speculative decoding")

    # Server
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1334)
    parser.add_argument("--allowed-origins", type=lambda x: x.split(","), default="*")

    # Tokenizer
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--chat-template", type=str, default="")
    parser.add_argument("--use-default-chat-template", action="store_true")
    parser.add_argument("--chat-template-args", type=str, default="{}",
                        help="JSON string of extra args for apply_chat_template, e.g. '{\"enable_thinking\":false}'")

    # Sampling defaults
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)

    # Concurrency / batching
    parser.add_argument("--decode-concurrency", type=int, default=32)
    parser.add_argument("--prompt-concurrency", type=int, default=8)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--prompt-cache-size", type=int, default=10)
    parser.add_argument("--prompt-cache-bytes", type=_parse_size)

    # Distributed
    parser.add_argument("--pipeline", action="store_true",
                        help="Use pipelining instead of tensor parallelism")

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

    # -- Apply TurboQuant ---------------------------------------------------
    _patch_cache(args.tq_bits, args.tq_fp16_layers, args.tq_fused)

    if args.tq_fused:
        apply_patch()

    logging.info(
        f"TurboQuant: {args.tq_bits}-bit, "
        f"{args.tq_fp16_layers}+{args.tq_fp16_layers} FP16 layers"
        f"{' (fused)' if args.tq_fused else ''}"
    )

    # -- Run server ---------------------------------------------------------
    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)
        logging.info(f"Metal wired limit: {wired_limit / (1024**3):.1f} GB")

    run(args.host, args.port, ModelProvider(args))


if __name__ == "__main__":
    main()
