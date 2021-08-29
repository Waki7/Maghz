import spaces as sp
from mgz.datasets.base_dataset import StandardDataset
from mgz.typing import *


class ImageDataset(StandardDataset):
    def __init__(self):
        super(ImageDataset, self).__init__()
        self.img_index: Dict[str, str] = {}

    @property
    def in_space(self) -> sp.Image:
        raise NotImplementedError

    @property
    def pred_space(self) -> sp.RegressionTarget:
        raise NotImplementedError
