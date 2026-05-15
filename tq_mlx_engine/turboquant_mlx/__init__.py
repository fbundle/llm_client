from .cache import TurboQuantKVCache
from .v_only_cache import VOnlyTurboQuantCache
from .adaptive import make_adaptive_cache
from .patch import apply_patch, remove_patch
from .quantizer import PolarQuantizer
from .metal import fused_quantize, dequant_fp16
from .kernels import packed_dequantize, packed_fused_qk_scores
from .metal_kernels_v4 import (
    prerotate_query,
    prerot_fused_qk_scores,
    prerot_packed_dequantize,
)
from .sparse_v import sparse_v_matvec
from .flash_attention import flash_attention_turboquant
from .fused_attention import turboquant_attention
from .packing import pack_indices, unpack_indices, packed_dim
from .rotation import (
    walsh_hadamard_transform,
    randomized_hadamard_transform,
    inverse_randomized_hadamard,
    random_diagonal_sign,
)

__version__ = "0.3.0"
