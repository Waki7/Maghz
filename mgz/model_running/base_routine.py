from __future__ import annotations

from abc import ABCMeta, abstractmethod

import mgz.model_vc as vc
from mgz.ds.base_dataset import BaseDataset


class BaseProtocol(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    def train(self, model_node: vc.ModelNode, ds: BaseDataset,
              model_edge: vc.ModelEdge,
              batch_size=8, device=None, distributed: bool = False,
              turn_off_shuffle=False) -> vc.ModelNode:
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass
