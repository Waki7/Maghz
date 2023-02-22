from mgz.models.base_model import BaseModel
from mgz.model_vc.model_edge import ModelEdge


class ModelNode:
    def __init__(self, model: BaseModel):
        self.model_cls = str(model.__class__)

        self.edges = []
        self.edge_data = []

    def add_edge(self, model_node: ModelEdge):
        pass
