from typing import *
from torch import Tensor


class DataSample(object):
    def __init__(self, features: Tuple[Tensor], labels: Tuple[Tensor]):
        self.features = features
        self.labels = labels
