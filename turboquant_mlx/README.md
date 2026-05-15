# turboquant-mlx

TurboQuant KV cache compression for MLX on Apple Silicon.  
PolarQuant (randomized Hadamard rotation + Lloyd-Max quantization) compresses
KV cache values to 3-4 bit with fused Metal kernels.

## Source

Forked from [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx)  
(Apache-2.0 license)

## Setup

    uv sync --extra tools
    uv pip install mlx-lm
