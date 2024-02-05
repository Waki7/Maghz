from enum import Enum


class Metrics(str, Enum):
    TRAIN_LOSS_MEAN = 'train/loss'
    TRAIN_ACC_MEAN = 'train/accuracy'
    TRAIN_AVG_PRED = 'train/avg_pred'

    VAL_ACC_ALL = 'val/accuracy_all'
    VAL_ACC_MEAN = 'val/accuracy_mean'
    VAL_AVG_PRED = 'val/avg_pred'

    def __str__(self):
        return self.value
