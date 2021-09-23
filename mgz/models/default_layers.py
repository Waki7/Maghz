import spaces as sp
import torch.nn as nn


def get_default_in_layer(in_space: sp.Space, out_space: sp.Space):
    if isinstance(in_space, sp.Image) and isinstance(out_space, sp.GenericBox):
        return nn.Identity(), in_space


def get_default_pred_layer(pred_space: sp.Space):
    if isinstance(sp.RegressionTarget):
        pass
