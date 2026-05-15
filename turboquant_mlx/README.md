# turboquant-mlx

TurboQuant KV cache compression for MLX on Apple Silicon.

PolarQuant (randomized Hadamard rotation + Lloyd-Max quantization) compresses
KV cache values to 3-4 bit with fused Metal kernels. Drop-in replacement for
mlx-lm's KVCache.

**Key finding:** K quantization destroys greedy decode at 4-bit and below;
V quantization is safe at 3-bit. Mixed-precision (K8 + V4) preserves identical
output while saving ~18% memory.

## Source

Forked from [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx)
(Apache-2.0 license, v0.3.0)

## Setup

    uv sync --extra server
    uv run python run_server.py --model mlx-community/Qwen2.5-7B-Instruct-4bit

## Features

- TurboQuantKVCache (3-4 bit V quantization)
- VOnlyTurboQuantCache (V-only compression, no mlx-lm fork needed)
- Adaptive layer isolation (FP16 for first/last N layers)
- Fused Metal kernels with butterfly-pulled-out optimization
- SIMD-group reductions (no tree_reduce dependency)
