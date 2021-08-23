import logging
import os
from typing import *
from mgz.models.network_trainer import NetworkTrainer
import mgz.utils.storage_utils as storage_utils
from mgz.utils.paths import Networks as paths
import gym.spaces as rs
import torch
import torch.nn as nn
import mgz.settings as settings


class BaseModel(nn.Module):
    def __init__(self, obs_space: rs.Tuple, out_space: rs.Tuple, **kwargs):
        super(BaseModel, self).__init__()
        if not (isinstance(obs_space, rs.Space) and
                isinstance(out_space, rs.Space)):
            logging.warning(
                'please pass in your shapes as a list of '
                'tuples as if the network input is multi modal')
        self.extra_parameters = nn.ParameterList()

        self.obs_space: rs.Space = obs_space
        self.out_space: rs.Space = out_space
        print('obs space ', self.obs_space)
        print('out space ', self.out_space)

        self.trainer: Optional[NetworkTrainer] = None
        self.temp_predictor = nn.Identity()

    def create_optimizer(self) -> NetworkTrainer:
        self.trainer = NetworkTrainer(self.cfg)
        self.trainer.add_network(self)
        return self.trainer

    def forward(self, *input):
        raise NotImplementedError

    def add_temp_predictor(self, predictor):
        self.temp_predictor = predictor
        self.trainer.add_layer_to_optimizer(predictor)

    def get_in_shapes(self) -> List[int]:
        return self.in_shapes

    def get_out_shapes(self) -> List[int]:
        return self.out_shapes

    def get_out_features(self):
        return self.out_features

    def get_in_features(self):
        return self.in_features

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
