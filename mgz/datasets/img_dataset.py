from mgz.typing import *
from mgz.datasets.base_dataset import StandardDataset
import spaces as sp
from mgz.generators.base_generator import BaseGenerator


class ImageDataset(StandardDataset):
    def __init__(self, generator: BaseGenerator, in_space: sp.Image,
                 pred_space: sp.RegressionTarget):
        super(ImageDataset, self).__init__()
