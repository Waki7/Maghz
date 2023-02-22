import spaces as sp
import torch.nn as nn

__all__ = ['get_default_in_layer', 'get_default_pred_layer']

def get_default_in_layer(in_space: sp.Space, out_space: sp.Space):
    if isinstance(in_space, sp.Image) and isinstance(out_space, sp.GenericBox):
        return nn.Identity(), in_space


def get_default_pred_layer(in_space: sp.Space, pred_space: sp.Space):
    if isinstance(pred_space, sp.RegressionTarget):
        return None
