from __future__ import annotations

import json
import os
from pathlib import Path

import transformers as hug
from accelerate.utils import BnbQuantizationConfig
from torch.nn import Parameter
from transformers import PreTrainedTokenizer, BitsAndBytesConfig

from mgz import settings
from mgz.models.base_model import BaseModel
from mgz.models.nlp.base_transformer import BaseTransformer, DecoderTransformer
from mgz.typing import *
from mgz.version_control.metrics import Metrics

if TYPE_CHECKING:
    from mgz.models.nlp.led import LEDModel, LEDForConditionalGeneration
    from mgz.version_control.model_edge import ModelTransitionEdge


class ModelDatabase:
    from mgz.models.nlp.led import LEDForConditionalGeneration
    '''
    Just some quick access base models. For easier book keeping for now.
    '''

    @staticmethod
    def get_led_model(
            model_name: str = 'allenai/led-base-16384-multi_lexsum-source-long') -> ModelNode:
        return ModelNode.load_from_id(ModelDatabase.LEDForConditionalGeneration,
                                      model_name,
                                      model_name)

    @staticmethod
    def mistral_openchat(
            model_name: str = "openchat/openchat-3.5-0106",
            quantized=False) -> ModelNode:
        # AdaptLLM/law-chat
        # OpenHermes-2.5-Mistral-7B
        from mgz.models.nlp.mistral_hug import MistralForCausalLMHug
        quantize = False
        quantization_cfg = None

        if quantize and settings.DEVICE != torch.device('cpu'):
            try:
                from accelerate.utils import BnbQuantizationConfig
                import bitsandbytes
                quantization_cfg = BnbQuantizationConfig(
                    load_in_8bit=quantize,
                    torch_dtype=torch.float16)

            except ImportError:
                print("Module 'some_module' is not installed.")
                quantization_cfg = None
        node = ModelNode.load_from_id(MistralForCausalLMHug,
                                      model_name,
                                      model_name,
                                      quantization_config=quantization_cfg)
        node.model = torch.compile(node.model)
        return node

    @staticmethod
    def get_quantized_model(
            model_name: str,
            quantize=False) -> ModelNode:
        # AdaptLLM/law-chat
        # OpenHermes-2.5-Mistral-7B
        from mgz.models.nlp.mistral_hug import MistralForCausalLMHug

        quantization_cfg = None
        if quantize and settings.DEVICE != torch.device('cpu'):
            try:
                from accelerate.utils import BnbQuantizationConfig
                import bitsandbytes
                quantization_cfg = BnbQuantizationConfig(
                    load_in_4bit=quantize, llm_int8_threshold=6,
                    torch_dtype=torch.float16)

            except ImportError:
                print("Module 'some_module' is not installed.")
                quantization_cfg = None
        node = ModelNode.load_from_id(MistralForCausalLMHug,
                                      model_name,
                                      model_name,
                                      quantization_config=quantization_cfg)
        node.model = torch.compile(node.model)
        return node


DEFAULT_INDEX_PATH = os.path.join(Path(__file__).resolve().parent.parent.parent,
                                  'index_dir/main').replace("\\",
                                                                         "/")


def x() -> Dict:
    return {
        'name': 'accuracy',
        'correct': 0,
        'correct_per_class': [],
        'total': 0,

    }


class ModelNode:
    root_path = DEFAULT_INDEX_PATH

    @classmethod
    def path_from_id(cls, model_id: str, create_if_not_exist=True):
        model_dir: DirPath = (
            os.path.join(cls.root_path, model_id).replace(
                "\\", "/"))
        if create_if_not_exist and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    @classmethod
    def lookup_or_init(cls, model_id: str,
                       model_cls: Type[BaseTransformer],
                       quantization_config: BitsAndBytesConfig = None) -> ModelNode:
        model_dir: DirPath = cls.path_from_id(model_id,
                                              create_if_not_exist=False)
        # TODO, cleaner way to distinguish the models that are available to load
        model: Optional[BaseTransformer] = \
            model_cls.load_model(model_dir, quantization_config)
        tokenizer: Optional[hug.PreTrainedTokenizer] = model_cls.load_tokenizer(
            model_dir)
        loaded_successfully = model is not None and tokenizer is not None
        if not loaded_successfully:
            if model is None:
                logging.warning('Model was not found {}'.format(model_dir))
            if tokenizer is None:
                logging.warning('Tokenizer was not found {}'.format(model_dir))
            logging.warning(
                'Model was in roots but not found in directory, this may be '
                'an online model: ' + model_dir)

        if not loaded_successfully:
            # Now we try to load the model from some other source
            model_cls.initial_save(model_id, model_dir)
            model: Optional[BaseTransformer] = \
                model_cls.load_model(model_dir, quantization_config)
            tokenizer: Optional[
                hug.PreTrainedTokenizer] = model_cls.load_tokenizer(
                model_dir)
        if model is None or tokenizer is None:
            raise FileNotFoundError(
                'Model {} not found. Checked in directory {}'.format(model_id,
                                                                     os.path.abspath(
                                                                         model_dir)))
        return ModelNode(model, tokenizer, model_id,
                         quantization_config=quantization_config)

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
            cls.lookup_or_init(model_id,
                               model_cls=model_cls,
                               quantization_config=quantization_config)
        node.model.eval()
        node.model.verify_tokenizer(node.tokenizer)

        model_dir = cls.path_from_id(node.model_id)
        file_path = os.path.join(model_dir, 'metrics.json')
        if os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                node.metrics = json.load(json_file)
        return node

    # Union thing is messy, what's a better way, probably pass a type to
    # ModelNode
    def __init__(self, model: Union[
        BaseModel, BaseTransformer, LEDModel, LEDForConditionalGeneration, DecoderTransformer],
                 tokenizer: PreTrainedTokenizer, model_id: str,
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
        return self.path_from_id(self.model_id)

    def to_json(self) -> str:
        obj_dict = {'model_id': self.model_id, 'model_cls': self.model_cls,
                    'tokenizer': self.tokenizer.__class__.__name__,
                    'metrics': self.metrics,
                    'edges': [edge.to_json() for edge in self.edges_out]}
        return json.dumps(obj_dict, indent=4, separators=(',', ': '))

    def get_parameters_by_string_in_name(self, match_string: str) -> List[
        Parameter]:
        return [p for n, p in self.model.named_parameters() if
                match_string in n]

    def freeze_parameters(self, params: List[Parameter]):
        for param in params:
            param.requires_grad = False

    def freeze_parameters_except_for(self,
                                     exempt_search_string: Optional[
                                         List[str]] = None):
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

    def store_metrics(self,
                      metrics: Dict[Metrics, Union[List[float], float]] = None):
        if metrics is not None:
            for k, v in metrics.items():
                self.metrics[k] = v
        model_dir = self.path_from_id(self.model_id)
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
        model_dir = self.path_from_id(self.model_id)
        file_path = os.path.join(model_dir, 'metrics.json')
        if not os.path.exists(file_path):
            return False
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return len(data) > 0

    def store_model_node(self, path: DirPath = None):
        model_dir = self.path_from_id(self.model_id)
        if path is not None:
            model_dir = path
        self.model.save(model_dir, self.quantization_config)
        self.tokenizer.save_pretrained(model_dir)
        self.store_metrics()
