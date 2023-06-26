from mgz.models.base_model import BaseModel
from mgz.model_vc.model_edge import ModelEdge
from mgz.typing import *
from mgz.models.nlp.base_transformer import BaseTransformer


class ModelNode:
    def __init__(self, model: Union[BaseModel, BaseTransformer]):
        self.model_cls = str(model.__class__)

        self.edges = []
        self.edge_data = []

    def add_edge(self, model_node: ModelEdge):
        pass
