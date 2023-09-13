from mgz.typing import *
from mgz.models.base_model import BaseModel
from mgz.ds.base_dataset import BaseDataset
import spaces as sp
from mgz.models.base_model import BaseModel
from mgz.models.mobile_net import MobileNetV2
from mgz.version_control.model_index import Indexer
from mgz.version_control.model_node import ModelNode


class Manager():
    def __init__(self, index: Indexer):
        self.index = index
        self.current_pointer = None

    def query_by_model(self, model: BaseModel):
        pass

    def query_by_dataset(self, dataset: BaseDataset) -> ModelNode:
        pass

    def query_by_space(self, space: sp.Space):
        pass

    def query_by_metric(self):
        pass

    def joint_query(self):
        pass
