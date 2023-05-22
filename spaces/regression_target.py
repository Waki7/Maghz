from gym.spaces import Box

from mgz.typing import *

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
        return x.shape == self.shape and np.all(x >= self.low) and np.all(
            x <= self.high)


RegressionTargetT = Union[TensorT, torch.Tensor]
