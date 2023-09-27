from __future__ import annotations

import json
import os

from transformers import PreTrainedTokenizer

import mgz.version_control as vc
from mgz.models.base_model import BaseModel
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.typing import *
from mgz.version_control import Metrics

if TYPE_CHECKING:
    from mgz.models.nlp.led import LEDModel
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
    def __init__(self, model: Union[BaseModel, BaseTransformer, LEDModel],
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

        self.mean_metrics: Dict[
            Metrics, float] = {} if metrics is None else metrics
        self.all_metrics: Dict[Metrics, List[float]] = {}

        # State
        # this should only be set by model_edge
        self.transitioning = False

    def to_json(self) -> str:
        obj_dict = {'model_id': self.model_id, 'model_cls': self.model_cls,
                    'tokenizer': self.tokenizer.__class__.__name__,
                    'metrics': self.mean_metrics,
                    'edges': [edge.to_json() for edge in self.edges_out]}
        return json.dumps(obj_dict, indent=4, separators=(',', ': '))

    @classmethod
    def load_from_id(cls, model_cls: Type[BaseTransformer], model_id: str,
                     tokenizer_id: str = None):
        node = vc.lookup_model(model_cls, model_id, tokenizer_id)
        node.load_metrics()
        return node

    # These act as a lock
    def start_transition(self):
        self.transitioning = True

    def end_transition(self):
        self.transitioning = False

    def store_metrics(self, summary_metrics: Dict[Metrics, float] = None,
                      iter_metrics: Dict[Metrics, List[float]] = None):
        if summary_metrics is not None:
            for k, v in summary_metrics.items():
                self.mean_metrics[k] = v
        if iter_metrics is not None:
            for k, v in iter_metrics.items():
                v = v if isinstance(v, list) else [v]
                if k in self.all_metrics:
                    self.all_metrics[k].extend(v)
                else:
                    self.all_metrics[k] = v
        model_dir = vc.CACHED_INDEXER.path_from_id(self.model_id)
        with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
            json.dump({"mean_metrics": {str(k): v for k, v in
                                        self.mean_metrics.items()},
                       "all_metrics": {str(k): v for k, v in
                                       self.all_metrics.items()}}, f)

    def has_initial_metrics(self):
        return len(self.mean_metrics) > 0

    def load_metrics(self):
        assert len(self.mean_metrics) == 0, "Metrics already loaded"
        model_dir = vc.CACHED_INDEXER.path_from_id(self.model_id)
        metric_file_path = os.path.join(model_dir, 'metrics.json')
        if os.path.exists(metric_file_path):
            with open(metric_file_path) as file_object:
                data: Dict[str, Dict[str, Union[float, List[float]]]] = \
                    json.load(file_object)
            for key, value in data.get("mean_metrics", {}).items():
                self.mean_metrics[Metrics(key)] = value
            for key, values in data.get("all_metrics", {}).items():
                self.all_metrics[Metrics(key)] = values

    def store_model_node(self):
        model_dir = vc.CACHED_INDEXER.path_from_id(self.model_id)
        # bug1 - the weights of the original model aren't copied so we only
        # have the changing out, this should be changed, I'm not sure if
        # saving both is necessary/optimal. Maybe we should load it fresh
        # from the disk.
        assert not self.transitioning, "At the moment we don't copy over weights, " \
                                       "so once you start training you can't add " \
                                       "store your original node, this can be " \
                                       "changed in the future by copying the " \
                                       "weights once you start trainign"
        self.model.save(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        self.store_metrics()
