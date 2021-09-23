import logging
import os
from typing import *
from mgz.models.network_trainer import NetworkTrainer
import mgz.utils.storage_utils as storage_utils
from mgz.utils.paths import Networks as paths
import spaces as sp
import torch
import torch.nn as nn
import mgz.settings as settings


class BaseModel(nn.Module):
    # def __init__(self, in_space: sp.Tuple, out_space: sp.Tuple, **kwargs):
    #     super(BaseModel, self).__init__()
    #     if not (isinstance(in_space, sp.Space) and
    #             isinstance(out_space, sp.Space)):
    #         logging.warning(
    #             'please pass in your shapes as a list of '
    #             'tuples as if the network input is multi modal')
    #     self.extra_parameters = nn.ParameterList()
    #
    #     self.in_space: sp.Space = in_space
    #     self.out_space: sp.Space = out_space
    #     print('obs space ', self.in_space)
    #     print('out space ', self.out_space)
    #
    #     self.trainer: Optional[NetworkTrainer] = None
    #     self.temp_predictor = nn.Identity()
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        self.extra_parameters = nn.ParameterList()

        self.trainer: Optional[NetworkTrainer] = None
        self.temp_in_layer = nn.Identity()
        self.temp_predictor = nn.Identity()

    def create_optimizer(self) -> NetworkTrainer:
        self.trainer = NetworkTrainer(self.cfg)
        self.trainer.add_network(self)
        return self.trainer

    def forward(self, *input):
        raise NotImplementedError

    def set_predictor(self, predictor):
        self.temp_predictor = predictor
        # self.trainer.add_layer_to_optimizer(predictor)

    def set_in_layer(self, in_layer):
        self.temp_in_layer = in_layer
        # self.trainer.add_layer_to_optimizer(in_layer)

    # loading and saving

    def get_config_filename(self, model_folder):
        return os.path.join(model_folder, paths.CONFIG_FILENAME)

    def store_config(self, model_dir_path):
        storage_utils.save_config(self.cfg,
                                  self.get_config_filename(model_dir_path),
                                  filename=paths.CONFIG_FILENAME)

    def load_config(self, model_dir_path):
        return storage_utils.load_config(
            self.get_config_filename(model_dir_path))

    def get_weights_filepath(self, model_dir_path):
        return os.path.join(model_dir_path, paths.WEIGHTS_FILENAME)

    def store_weights(self, model_folder):
        torch.save(self.state_dict(), self.get_weights_filepath(model_folder))

    def load(self, load_folder):
        self.cfg = self.load_config(load_folder)
        self.load_state_dict(torch.load(self.get_weights_filepath(load_folder),
                                        map_location=settings.DEVICE))

    def save(self, save_folder):
        self.temp_predictor = nn.Identity()
        self.trainer.store_optimizer(save_folder)
        self.store_weights(save_folder)
        self.store_config(save_folder)
