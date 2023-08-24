from abc import ABCMeta, abstractmethod

from mgz.ds.base_dataset import BaseDataset
from mgz.model_vc import ModelEdge, ModelNode


class BaseProtocol(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    def train(self, model_node: ModelNode, ds: BaseDataset,
              model_edge: ModelEdge,
              batch_size=8, device=None, distributed: bool = False,
              turn_off_shuffle=False) -> ModelNode:
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass
