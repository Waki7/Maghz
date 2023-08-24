import settings
import spaces as sp
from mgz.ds import DataSplit
from mgz.ds.base_dataset import BaseDataset
from mgz.model_running.base_routine import BaseProtocol
from mgz.model_vc import ModelNode


class TaggingRoutine(BaseProtocol):
    def __init__(self):
        super().__init__()
        self.train_init = False
        self.eval_init = False
        self.predict_init = False

    def _check(self, ds: BaseDataset):
        assert isinstance(ds.input_space, sp.Sentence)
        assert isinstance(ds.target_space, sp.Tagging)

    def train(self, model_node: ModelNode, ds: BaseDataset,
              batch_size=8, device=None, distributed: bool = False,
              turn_off_shuffle=False) -> ModelNode:
        val_ds = ds.gen_validation_data()
        train_ds = ds.load_training_data()
        if device is None:
            device = settings.DEVICE
        train_dl = train_ds.create_dataloaders(device, batch_size,
                                               distributed=distributed,
                                               turn_off_shuffle=turn_off_shuffle)
        val_dl = val_ds.create_dataloaders(device, batch_size,
                                           distributed=distributed,
                                           turn_off_shuffle=turn_off_shuffle)

        assert self.ds.data_state == DataSplit.TRAIN

    def evaluate(self, model_node: ModelNode):
        pass

    def predict(self, model_node: ModelNode):
        pass
