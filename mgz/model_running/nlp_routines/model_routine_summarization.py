from __future__ import annotations

import mgz.version_control as vc
import settings
import spaces as sp
# from mgz.ds import DataState
from mgz.ds.base_dataset import BaseDataset
from mgz.model_running.base_routine import BaseProtocol
from mgz.model_running.run_ops import run_epoch


class SummarizationRoutine(BaseProtocol):
    def __init__(self):
        super().__init__()
        self.train_init = False
        self.eval_init = False
        self.predict_init = False

    def _check(self, ds: BaseDataset):
        assert isinstance(ds.input_space, sp.Sentence)
        assert isinstance(ds.target_space, sp.Sentence)

    def train(self, model_node: vc.ModelNode, ds: BaseDataset,
              model_edge: vc.ModelTransitionEdge,
              batch_size=8, device=None, distributed: bool = False,
              turn_off_shuffle=False) -> vc.ModelNode:
        model_node.model.train()
        if model_node.mean_metrics is None:
            pass
        self._check(ds)

        val_ds = ds.gen_validation_data()
        train_ds = ds.load_training_data()

        if device is None:
            device = settings.DEVICE
        train_dl = train_ds.create_dataloaders(device, batch_size,
                                               is_distributed=distributed,
                                               turn_off_shuffle=turn_off_shuffle)
        val_dl = val_ds.create_dataloaders(device, batch_size,
                                           is_distributed=distributed,
                                           turn_off_shuffle=turn_off_shuffle)
        run_epoch(
            data_loader=train_dl,
            val_data_loader=val_dl,
            model=model_node.model, tokenizer=model_node.tokenizer,
            loss_fn=model_edge.loss_fn,
            optimizer=model_edge.optimizer,
            train_state=model_edge.train_state,
        )
        return model_node

    def evaluate(self, model_node: vc.ModelNode):
        pass

    def predict(self, model_node: vc.ModelNode):
        pass
