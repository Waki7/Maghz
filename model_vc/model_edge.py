import spaces as sp
from mgz.models.base_model import BaseModel


class ModelEdge:
    def __init__(self, orig_model: BaseModel, in_space: sp.Space,
                 out_space: sp.Space):
        self.parent = orig_model
        self.child = None
