from __future__ import annotations

from gym.spaces import Box

from mgz.typing import *


class Sentence(Box):
    def __init__(self, vocab_size: int,
                 sequence_len: Union[SrcSeqLen, TgtSeqLen],
                 dtype=torch.IntType):
        self.vocab_size = vocab_size
        self.dim_names: List[str]  # TODO = [str(SrcSeqLen)]
        super(Sentence, self).__init__(low=0, high=vocab_size,
                                       shape=(sequence_len,),
                                       dtype=dtype)


class SentenceQuery():
    def __init__(self, vocab_size: int,
                 sequence_len: SrcSeqLen,
                 query_len: TgtSeqLen,
                 dtype=torch.IntType):
        self.vocab_size = vocab_size
        self.dim_names: List[str]  # TODO = [str(SrcSeqLen)]
        super(SentenceQuery, self).__init__(low=0, high=vocab_size,
                                            shape=(sequence_len, query_len),
                                            dtype=dtype)
