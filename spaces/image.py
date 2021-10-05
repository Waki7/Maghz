from mgz.typing import *
from gym.spaces import Box

import numpy as np


class Image(Box):
    def __init__(self, low: float,
                 high: float,
                 shape: Tuple[int],
                 dtype=np.float32,
                 dim_names: List[str] = None):
        if dim_names is None:
            dim_names = [str(dim) for dim in shape]
        self.dim_names = dim_names
        super(Image, self).__init__(low=low, high=high, shape=shape,
                                    dtype=dtype)
