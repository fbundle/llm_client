#!/usr/bin/env python3
"""TurboQuant MLX HTTP Server — thin entry point.

Usage:
    uv run python run_server.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
    uv run python run_server.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --tq-bits 3 --tq-fused
"""

import argparse
import logging

import mlx.core as mx
from tq_mlx_engine.engine import patch_cache
from tq_mlx_engine.server import TQServer
from tq_mlx_engine.turboquant_mlx import apply_patch


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
    patch_cache(args.tq_bits, args.tq_fp16_layers, args.tq_fused)

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
