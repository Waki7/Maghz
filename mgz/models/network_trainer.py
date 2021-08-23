import os
import typing as typ

import torch
import torch.nn as nn

import mgz.settings as settings
from mgz import utils as model_utils
from mgz.utils.paths import Networks


class NetworkTrainer(object):
    def __init__(self,
                 lr: float = .001,
                 gradient_clip: float = 2.0,
                 optim_name: str = 'Adam'):
        self.updates_locked = False
        self.gradient_clip = gradient_clip
        self.lr = lr
        self.optim_name = optim_name
        self.optimizer = None
        self.params = None

    def init_optimizer(self, parameters: typ.Iterator[
        torch.nn.Parameter]) -> torch.optim.Optimizer:
        # floating point precision, so need to set epislon
        return getattr(torch.optim,
                       self.optim_name)(parameters, lr=self.lr, eps=1.e-4)

    def add_network(self, network: nn.Module):
        self.params = list(network.parameters())
        self.optimizer: torch.optim.Optimizer = self.init_optimizer(self.params)
        model_utils.module_dtype_init(network)

    def add_layer_to_optimizer(self, layer: nn.Module):
        self.params = list(layer.parameters()) + list(self.params)
        self.optimizer = self.init_optimizer(self.params)
        model_utils.module_dtype_init(layer)

    def lock_updates(self):
        self.updates_locked = True

    def unlock_updates(self):
        self.updates_locked = False

    def update_parameters(self, override_lock=False):
        if (not self.updates_locked) or override_lock:
            torch.nn.utils.clip_grad_value_(self.params, self.gradient_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def store_optimizer(self, model_folder):
        # todo, maybe load optimizer?
        optimizer_filename = os.path.join(model_folder,
                                          Networks.OPTIMIZER_FILENAME)
        torch.save(self.optimizer.state_dict(), optimizer_filename)

    def save(self, model_folder):
        self.store_optimizer(model_folder)
