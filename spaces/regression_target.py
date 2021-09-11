from mgz.typing import *
from gym.spaces import Box
from spaces import Image
import numpy as np

__all__ = ['RegressionTarget', 'RegressionTargetT']


class RegressionTarget(Box):
    def __init__(self, low: Union[np.ndarray, float],
                 high: Union[np.ndarray, float],
                 shape: Optional[List[int]] = None,
                 dtype=np.float32,
                 dim_names: List[str] = None):
        if dim_names is not None:
            dim_names = [str(dim) for dim in shape]
        self.dim_names = dim_names
        super(RegressionTarget, self).__init__(low=low, high=high, shape=shape,
                                               dtype=dtype)

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class RegressionTargetT(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """
    pass