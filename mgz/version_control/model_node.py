from __future__ import annotations

import json
import os

from transformers import PreTrainedTokenizer, BitsAndBytesConfig

import mgz.version_control as vc
from mgz.models.base_model import BaseModel
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.typing import *
from mgz.version_control.metrics import Metrics

if TYPE_CHECKING:
    from mgz.models.nlp.led import LEDModel, LEDForSequenceClassification
    from mgz.version_control.model_edge import ModelTransitionEdge


def x() -> Dict:
    return {
        'name': 'accuracy',
        'correct': 0,
        'correct_per_class': [],
        'total': 0,

    }


class ModelNode:
    # Union thing is messy, what's a better way, probably pass a type to
    # ModelNode
    def __init__(self, model: Union[
        BaseModel, BaseTransformer, LEDModel, LEDForSequenceClassification],
                 tokenizer: PreTrainedTokenizer, model_id: str,
                 metrics: Dict[Metrics, Union[List[float], float]] = None):
        """
        Currently model_id is last in case we want to introduce the ability
        to have an auto populated ID.
        """
        self.model_id = model_id
        self.model_cls = str(model.__class__)
        self.model = model
        self.tokenizer = tokenizer
        self.edges_out: List[ModelTransitionEdge] = []
        self.edge_data = []

        self.metrics: Dict[Metrics, Union[float, List[float]]] = {}

    def to_json(self) -> str:
        obj_dict = {'model_id': self.model_id, 'model_cls': self.model_cls,
                    'tokenizer': self.tokenizer.__class__.__name__,
                    'metrics': self.metrics,
                    'edges': [edge.to_json() for edge in self.edges_out]}
        return json.dumps(obj_dict, indent=4, separators=(',', ': '))

    @classmethod
    def load_from_id(cls, model_cls: Type[BaseTransformer], model_id: str,
                     tokenizer_id: str = None,
                     quantization_config: BitsAndBytesConfig = None):
        node = vc.lookup_or_init_model(model_cls, model_id, tokenizer_id,
                                       quantization_config=quantization_config)
        model_dir = vc.CACHED_INDEXER.path_from_id(node.model_id)
        file_path = os.path.join(model_dir, 'metrics.json')
        with open(file_path, "r") as json_file:
            node.metrics = json.load(json_file)
        return node

    def store_metrics(self,
                      metrics: Dict[Metrics, Union[List[float], float]] = None):
        if metrics is not None:
            for k, v in metrics.items():
                self.metrics[k] = v
        model_dir = vc.CACHED_INDEXER.path_from_id(self.model_id)
        file_path = os.path.join(model_dir, 'metrics.json')

        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        data["metrics"] = {str(k): v for k, v in
                           self.metrics.items()}
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def has_initial_metrics(self):
        model_dir = vc.CACHED_INDEXER.path_from_id(self.model_id)
        file_path = os.path.join(model_dir, 'metrics.json')
        if not os.path.exists(file_path):
            return False
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return len(data) > 0

    def store_model_node(self, path: DirPath = None):
        model_dir = vc.CACHED_INDEXER.path_from_id(self.model_id)
        if path is not None:
            model_dir = path
        self.model.save(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        self.store_metrics()
