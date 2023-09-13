from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizer

from mgz.models.base_model import BaseModel
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.typing import *

if TYPE_CHECKING:
    import mgz.version_control as vc

def create_accuray_metric()->Dict:
    return {
        'name': 'accuracy',
        'correct': 0,
        'correct_per_class': [],
        'total': 0,


    }



class ModelNode:
    def __init__(self, model: Union[BaseModel, BaseTransformer],
                 tokenizer: PreTrainedTokenizer):
        self.model_cls = str(model.__class__)
        self.model = model
        self.tokenizer = tokenizer
        self.edges = []
        self.edge_data = []
        self.metrics = []

    def add_edge(self, model_node: vc.ModelEdge):
        pass

    def train_seq_to_seq(self, in_seq: Sequence, out_seq: Sequence):
        pass

    def train_meta_tagging(self, in_seq: Sequence, out_seq: Sequence):
        pass

    def train_relevance_tagging(self, in_seq: Sequence, out_seq: Sequence):
        pass
