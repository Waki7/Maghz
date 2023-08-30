from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mgz.model_running.run_ops import TrainState

if TYPE_CHECKING:
    import mgz.model_vc as vc


class ModelEdge:
    def __init__(self, orig_model: vc.ModelNode, loss_fn,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None):
        self.parent = orig_model
        self.child = None
        self.train_state = TrainState()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
