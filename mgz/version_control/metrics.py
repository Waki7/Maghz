from enum import Enum


class Metrics(str, Enum):
    TRAIN_LOSS_MEAN = 'train/loss'
    TRAIN_ACC_MEAN = 'train/accuracy'

    VAL_ACC = 'val/accuracy'

    def __str__(self):
        return self.value
