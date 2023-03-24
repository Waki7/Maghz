from __future__ import annotations

from gym.spaces import Box

from mgz.typing import *


class Sentence(Box):
    def __init__(self, vocab_size: int,
                 shape: Tuple[SrcSeqLen],
                 dtype=torch.IntType):
        self.vocab_size = vocab_size
        self.dim_names: List[str] = [str(SrcSeqLen)]
        super(Sentence, self).__init__(low=0, high=vocab_size, shape=shape,
                                       dtype=dtype)
