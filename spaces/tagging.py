from __future__ import annotations

from gym.spaces import MultiDiscrete

from mgz.typing import *

'''
 Most generic form, the simple form of tagging would just be a MultiDiscrete([2, 2, 2])
'''


class Tagging(MultiDiscrete):
    def __init__(self, nvec: Iterable[int],
                 dtype=torch.IntType):
        self.dim_names: List[str]  # TODO = [str(SrcSeqLen)]
        super(Tagging, self).__init__(nvec=nvec, dtype=dtype)


class BinaryTagging(Tagging):
    def __init__(self,
                 dtype=torch.IntType):
        self.dim_names: List[str]  # TODO = [str(SrcSeqLen)]
        super(Tagging, self).__init__(nvec=[2], dtype=dtype)
