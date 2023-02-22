from mgz.model_vc.model_edge import ModelEdge
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
                         new_out_space: sp.Space) -> ModelEdge:
        model.set_predictor()


