from mgz.typing import *
from gym.spaces import Box

import numpy as np


class GenericBox(Box):
    def __init__(self, low: float,
                 high: float,
                 shape: List[int] = None,
                 dtype=np.float32,
                 dim_names: List[str] = None):
        if dim_names is not None:
            dim_names = [str(dim) for dim in shape]
        self.dim_names = dim_names
        super(GenericBox, self).__init__(low=low, high=high, shape=(0,),
                                    dtype=dtype)
