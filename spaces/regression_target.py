from mgz.typing import *
from spaces import Box

import numpy as np

class RegressionTarget(Box):
    def __init__(self, low: Union[np.ndarray, float],
                 high: Union[np.ndarray, float],
                 shape: Optional[List[int]] = None,
                 dtype=np.float32,
                 dim_names: List[str] = None):
        if dim_names is None:
            dim_names = [str(dim) for dim in shape]
        self.dim_names = dim_names
        super(Image, self).__init__(low=low, high=high, shape=shape,
                                    dtype=dtype)
