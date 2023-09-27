from mgz.version_control.model_edge import ModelTransitionEdge
from mgz.models.base_model import BaseModel
import spaces as sp


class ModelOps:
    # cant back reference weights
    def train(self):
        pass

    # can back reference previous weights ftmp

    def input_space(self):
        pass

    def change_predictor(self, model: BaseModel,
                         new_out_space: sp.Space) -> ModelTransitionEdge:
        model.set_predictor()


