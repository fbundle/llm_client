from dataclasses import dataclass
from typing import Iterator, Callable

import mlx.core as mx

type Token = int
type TokenList = mx.array  # (n,)
type Logits = mx.array     # (d,)

def make_token_list(token: Token) -> TokenList:
    return mx.array([token])

type SampleFunc = Callable[[Logits], Token]
type ModelFunc[T] = Callable[[T, TokenList], tuple[T, Logits]]


@dataclass
class StreamGenerationIteration[T]:
    state: T
    token: Token
    logits: Logits


def stream_generate[T](
    new_token_list: TokenList,
    prev_state: T,
    model_func: ModelFunc[T],
    sample_func: SampleFunc,
    eos_token_set: set[Token],
    max_completion_length: int,
) -> Iterator[StreamGenerationIteration[T]]:

    # put new_token_list into the model
    # get logits for the first sampling step
    state, logits = model_func(prev_state, new_token_list)

    for _ in range(max_completion_length):
        # sample a new token
        token = sample_func(logits)
        # put the new token into the model
        state, next_logits = model_func(state, make_token_list(token))
        # yield the state after the new token
        yield StreamGenerationIteration(state=state, token=token, logits=logits)
        # stop condition
        if token in eos_token_set:
            break
        # prepare for the next sampling step
        logits = next_logits
