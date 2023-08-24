import torch

from mgz.model_running.run_ops import TrainState
from mgz.model_vc import ModelNode


class ModelEdge:
    def __init__(self, orig_model: ModelNode, loss_fn,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None):
        self.parent = orig_model
        self.child = None
        self.train_state = TrainState()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
