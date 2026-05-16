import logging
from typing import Iterator

import mlx.core as mx
import mlx_lm
from tqdm import tqdm
from mlx_engine.generate import StreamGenerationIteration, stream_generate
from mlx_engine.prefix_dict import PrefixDict
from mlx_lm.models.cache import KVCache


type Cache = list[KVCache]

def get_cache_prompt_length(cache: Cache) -> int:
    return cache[0].offset

class MlxEngine:
    def __init__(self,
        model_path: str,
        adapter_path: str | None = None,
        cache_capacity: int = 1,
    ):
        self.model_path = model_path

        logging.info(f"Loading model: {model_path}")
        self.model, self.tokenizer, self.config = mlx_lm.load(      # type: ignore
            path_or_hf_repo=model_path,
            adapter_path=adapter_path,
            return_config=True,
        )
        self.cache_dict: PrefixDict[Cache] = PrefixDict(capacity=cache_capacity)

        self._make_cache = lambda: [KVCache() for _ in range(len(self.model.layers))] # type: ignore

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
            logging.info("_make_cache")
            prev_state = self._make_cache()
            suffix = prompt
        else:
            prefix_len = get_cache_prompt_length(prev_state)
            suffix = prompt[prefix_len:]

        sample_func = lambda logits: int(mlx_lm.sample_utils.make_sampler(                   # type: ignore
            temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
        )(logits)[0])

        i: Iterator[StreamGenerationIteration[Cache]] = stream_generate(
            new_token_list=mx.array(suffix),
            prev_state=prev_state,
            model_func=self.model_func,
            sample_func=sample_func,
            eos_token_set=set(self.tokenizer.eos_token_ids),
            max_completion_length=max_tokens,
        )

        completion_token_list: list[int] = []
        new_state: Cache = prev_state
        for o in tqdm(i, desc=f"generating {len(prompt)} + ...", unit="tok"):
            new_state = o.state
            completion_token_list.append(o.token)
            yield o.token
        
        new_prompt = prompt + completion_token_list
        assert get_cache_prompt_length(new_state) == len(new_prompt)

        # self.cache_dict.push(new_prompt, new_state)
        # completion_token_list containg reasoning_content and will be cut off in the next call
        for kv in new_state:                                                                                
            kv.trim(len(completion_token_list))
        
        self.cache_dict.push(prompt, new_state)

        


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model_path = "mlx-community/Qwen3-0.6B-4bit"
    engine = MlxEngine(model_path)

    # Round 1
    messages = [{"role": "user", "content": "Say 'hello' and nothing else."}]
    prompt1 = engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokens1 = engine.tokenizer.encode(prompt1) # type: ignore

    print("--- Round 1 ---")
    out1: list[int] = []
    for t in engine.generate(tokens1, max_tokens=32):
        out1.append(t)
    
    text1 = engine.tokenizer.decode(out1)  # type: ignore
    print(f"  prompt tokens: {len(tokens1)}")
    print(f"  output:        {text1}")

    # Round 2 — conversation continues, prompt includes round 1 history
    messages.append({"role": "assistant", "content": text1})
    messages.append({"role": "user", "content": "What did you just say?"})
    prompt2 = engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokens2 = engine.tokenizer.encode(prompt2)  # type: ignore

    print("--- Round 2 ---")
    out2: list[int] = []
    for t in engine.generate(tokens2, max_tokens=32):
        out2.append(t)
    text2 = engine.tokenizer.decode(out2)  # type: ignore
    print(f"  prompt tokens: {len(tokens2)}")
    print(f"  cached prefix: {len(tokens2) - len(tokens1)} suffix tokens (reused {len(tokens1)} from round 1)")
    print(f"  output:        {text2}")
