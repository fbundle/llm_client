import logging
from typing import Iterator

import mlx.core as mx
import mlx_lm
from tq_mlx_engine.generate import StreamGenerationIteration, stream_generate
from tq_mlx_engine.prefix_dict import PrefixDict
from tq_mlx_engine.turboquant_mlx import apply_patch
from tq_mlx_engine.turboquant_mlx.cache import TurboQuantKVCache

type Cache = list[TurboQuantKVCache]

def get_cache_prompt_length(cache: Cache) -> int:
    return cache[0].offset

class MlxEngine:
    def __init__(self, model_path: str, adapter_path: str | None = None,
                 tq_bits: int = 3, tq_fused: bool = False):
        if tq_fused:
            apply_patch()

        self.model_path = model_path
        self._tq_bits = tq_bits
        self._tq_fused = tq_fused

        logging.info(f"Loading model: {model_path}")
        self.model, self.tokenizer, self.config = mlx_lm.load(      # type: ignore
            path_or_hf_repo=model_path,
            adapter_path=adapter_path,
            return_config=True,
        )
        logging.info("Model loaded")
        self.cache_dict: PrefixDict[Cache] = PrefixDict()


    def _make_cache(self) -> Cache:
        return [
            TurboQuantKVCache(bits=self._tq_bits, fused=self._tq_fused)
            for _ in range(len(self.model.layers))                   # type: ignore
        ]

    def model_func(self, cache: Cache, tokens: mx.array) -> tuple[list, mx.array]:
        # tokens[None] changes shape from [n] to [1, n]
        logits = self.model(tokens[None], cache=cache)
        return cache, logits[:, -1, :]


    def generate(self,
        prompt: list[int],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
    ) -> Iterator[int]:
        # manage cache
        prev_state: Cache | None = self.cache_dict.pop(prompt)
        if prev_state is None:
            prev_state = self._make_cache()
            suffix = prompt
        else:
            prefix_len = get_cache_prompt_length(prev_state)
            suffix = prompt[prefix_len:]

        i: Iterator[StreamGenerationIteration[Cache]] = stream_generate(
            new_token_list=mx.array(suffix),
            prev_state=prev_state,
            model_func=self.model_func,
            sample_func=mlx_lm.sample_utils.make_sampler(                   # type: ignore
                temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
            ),
            eos_token_set=set(self.tokenizer.eos_token_ids),
            max_completion_length=max_tokens,
        )

        completion_token_list: list[int] = []
        new_state: Cache = prev_state
        for o in i:
            new_state = o.state
            completion_token_list.append(o.token)
            yield o.token
        
        new_prompt = prompt + completion_token_list
        assert get_cache_prompt_length(new_state) == len(new_prompt)
        self.cache_dict.push(new_prompt, new_state)
