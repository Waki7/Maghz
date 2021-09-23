from typing import TypeVar, Generic, Tuple, Union, Optional
import numpy as np

Time = TypeVar("Time")
Batch = TypeVar("Batch")
Height = TypeVar("Height")
Width = TypeVar("Width")

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class NDArray(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """
    pass


