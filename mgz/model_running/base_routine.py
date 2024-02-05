from __future__ import annotations

import json

import torch.utils.data

import mgz.settings as settings
import mgz.version_control as vc
from mgz.ds.base_dataset import BaseDataset
from mgz.ds.sentence_datasets.sentence_datasets import Sent2TagMetaTaskBatch, \
    TagQAMetaTaskBatch
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.typing import *

if TYPE_CHECKING:
    from mgz.version_control import ModelTransitionEdge, ModelNode, Metrics


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

    def to_json(self, as_str=False) -> Union[dict, str]:
        obj_dict = {}
        for k, v in sorted(self.__dict__.items()):
            obj_dict[k] = v
        return json.dumps(obj_dict, indent=4,
                          separators=(',', ': ')) if as_str else obj_dict


class BaseProtocol(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _check(self, ds: BaseDataset):
        raise NotImplementedError

    @abstractmethod
    def run_batch(self, model: BaseTransformer,
                  batch: Union[Sent2TagMetaTaskBatch, TagQAMetaTaskBatch],
                  model_edge: ModelTransitionEdge,
                  gpu_max_batch_size=4) -> \
            Tuple[FloatTensorT['1'], float]:
        raise NotImplementedError

    @abstractmethod
    def train_epoch(self,
                    model_node: ModelNode,
                    data_loader: torch.utils.data.DataLoader[
                        TagQAMetaTaskBatch],
                    val_data_loader: torch.utils.data.DataLoader[
                        TagQAMetaTaskBatch],
                    model_edge: ModelTransitionEdge,
                    log_interval=5,
                    val_interval=50,
                    gradient_accumulation_steps=8,
                    debug=False,
                    ) -> ModelNode:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def val_model(self,
                  val_data_loader: torch.utils.data.DataLoader[
                      TagQAMetaTaskBatch],
                  model_node: ModelNode,
                  training_edge: ModelTransitionEdge = None
                  ) -> Dict[Metrics, Union[float, List[float]]]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, model_node: vc.ModelNode):
        raise NotImplementedError

    def train(self, model_node: vc.ModelNode, ds: BaseDataset,
              model_edge: vc.ModelTransitionEdge, device=None,
              distributed: bool = False,
              turn_off_shuffle=False,
              val_ds: BaseDataset = None, n_epochs=1) -> vc.ModelNode:
        model_node.model.train()
        self._check(ds)

        if val_ds is None:
            val_ds = ds.gen_validation_data()
        else:
            val_ds = val_ds.load_validation_data()
        train_ds = ds.load_training_data()

        if device is None:
            device = settings.DEVICE

        train_dl = train_ds.create_dataloaders(device, batch_size=1,
                                               is_distributed=distributed,
                                               turn_off_shuffle=turn_off_shuffle)
        val_dl = val_ds.create_dataloaders(device, batch_size=1,
                                           is_distributed=distributed,
                                           turn_off_shuffle=turn_off_shuffle)
        for i in range(0, n_epochs):
            self.train_epoch(
                model_node,
                data_loader=train_dl,
                val_data_loader=val_dl,
                model_edge=model_edge
            )
        return model_node

    def evaluate(self, model_node: ModelNode, val_ds: BaseDataset,
                 device=None, ) -> Dict[Metrics, Union[float, List[float]]]:
        if device is None:
            device = settings.DEVICE
        val_ds.load_validation_data()
        val_dl = val_ds.create_dataloaders(device, batch_size=1,
                                           is_distributed=False,
                                           turn_off_shuffle=True)
        metrics: Dict[Metrics, Union[float, List[float]]] = self.val_model(
            val_dl, model_node)
        return metrics
