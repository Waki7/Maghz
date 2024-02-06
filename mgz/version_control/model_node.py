from __future__ import annotations

import json
import os

from accelerate.utils import BnbQuantizationConfig
from torch.nn import Parameter
from transformers import PreTrainedTokenizer, BitsAndBytesConfig

import mgz.version_control as vc
from mgz.models.base_model import BaseModel
from mgz.models.nlp.base_transformer import BaseTransformer, DecoderTransformer
from mgz.typing import *
from mgz.version_control.metrics import Metrics

if TYPE_CHECKING:
    from mgz.models.nlp.led import LEDModel, LEDForConditionalGeneration
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
        BaseModel, BaseTransformer, LEDModel, LEDForConditionalGeneration, DecoderTransformer],
                 tokenizer: PreTrainedTokenizer, model_id: str,
                 metrics: Dict[Metrics, Union[List[float], float]] = None,
                 quantization_config: Optional[BnbQuantizationConfig] = None):
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
        self.quantization_config = quantization_config
        self.metrics: Dict[Metrics, Union[float, List[float]]] = {}

    def get_path(self) -> DirPath:
        return vc.CACHED_INDEXER.path_from_id(self.model_id)

    def to_json(self) -> str:
        obj_dict = {'model_id': self.model_id, 'model_cls': self.model_cls,
                    'tokenizer': self.tokenizer.__class__.__name__,
                    'metrics': self.metrics,
                    'edges': [edge.to_json() for edge in self.edges_out]}
        return json.dumps(obj_dict, indent=4, separators=(',', ': '))

    def freeze_parameters_except_for(self,
                                     exempt_search_string: Optional[List[str]] = None):
        if exempt_search_string is None:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for name, param in self.model.named_parameters():
                if any([s in name for s in exempt_search_string]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def split_parameters_group(self, match_string: str):
        params_match = []
        params_no_match = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if match_string in name:
                    params_match.append(param)
                else:
                    params_no_match.append(param)
        return params_match, params_no_match

    def get_optimizer(self, quantize: bool, lr: float,
                      params: List[
                          Union[Dict[str, Union[
                              List[Parameter], float]], Parameter]] = None,
                      weight_decay: float = 0.0, eps=1e-4,
                      betas: Tuple[float, float] = (0.9, 0.999)
                      ):
        if params is None:
            params = [p for n, p in self.model.named_parameters() if
                      p.requires_grad]
        import bitsandbytes
        optimizer = \
            bitsandbytes.optim.PagedAdam(params,
                                         lr=lr, weight_decay=weight_decay,
                                         betas=betas, eps=eps)
        # else:
        #     optimizer = \
        #         torch.optim.Adam(params, lr=lr,
        #                          weight_decay=weight_decay,
        #                          eps=eps, betas=betas)
        return optimizer

    @classmethod
    def load_from_id(cls, model_cls: Type[BaseTransformer], model_id: str,
                     tokenizer_id: str = None,
                     quantization_config: BitsAndBytesConfig = None,
                     quantize_8bit: bool = False):
        if quantize_8bit and quantization_config is None:
            try:
                from accelerate.utils import BnbQuantizationConfig
                import bitsandbytes
                quantization_config = BnbQuantizationConfig(
                    load_in_8bit=quantize_8bit, )
            except ImportError:
                print("Module 'some_module' is not installed.")
                quantization_config = None

        assert tokenizer_id is not None, 'NLP Model needs tokenizer id'
        node = \
            vc.CACHED_INDEXER.lookup_or_init(model_id,
                                             model_cls=model_cls,
                                             quantization_config=quantization_config)
        node.model.eval()
        node.model.verify_tokenizer(node.tokenizer)

        model_dir = vc.CACHED_INDEXER.path_from_id(node.model_id)
        file_path = os.path.join(model_dir, 'metrics.json')
        if os.path.exists(file_path):
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

        data = {}
        if os.path.exists(file_path):
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
        self.model.save(model_dir, self.quantization_config)
        self.tokenizer.save_pretrained(model_dir)
        self.store_metrics()
