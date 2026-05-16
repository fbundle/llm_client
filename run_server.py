#!/usr/bin/env python3
"""MLX HTTP Server — thin entry point.

Usage:
    uv run python run_server.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""

import argparse
import logging

import mlx.core as mx
from mlx_engine.server import Server


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX HTTP Server")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the MLX model weights, tokenizer, and config")
    parser.add_argument("--adapter-path", type=str,
                        help="Optional path for trained adapter weights")

    # Server
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1334)

    # Context
    parser.add_argument("--max-context", type=int, default=32768,
                        help="Maximum context length in tokens (default: 32768). "
                             "Caps the model's theoretical max to fit GPU memory.")

    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Metal memory
    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)
        logging.info(f"Metal wired limit: {wired_limit / (1024**3):.1f} GB")

    server = Server(args.model, args.adapter_path,
                      max_context=args.max_context)

    # Pre-load the default model so the first request doesn't block
    server._get_engine(args.model)

    import uvicorn
    logging.info(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(server.fastapi, host=args.host, port=args.port,
                log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
