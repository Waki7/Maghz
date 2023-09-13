from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import mgz.model_running.run_ops as run_ops

if TYPE_CHECKING:
    import mgz.version_control as vc


class ModelEdge:
    def __init__(self, orig_model: vc.ModelNode, loss_fn,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None):
        self.parent = orig_model
        self.child = None
        self.train_state = run_ops.TrainState()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
