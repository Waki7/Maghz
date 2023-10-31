from __future__ import annotations

import json
# from mgz.models.nlp.bart_interface import BARTHubInterface
import os
from pathlib import Path

from transformers import PreTrainedTokenizer, BitsAndBytesConfig

from mgz.models.base_model import BaseModel
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.typing import *

os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import logging

from mgz.version_control import ModelNode
import transformers as hug


class ModelDatabase:
    from mgz.models.nlp.led import LEDForSequenceClassification
    '''
    Just some quick access base models. For easier book keeping for now.
    '''

    @staticmethod
    def led_source_to_long_id() -> str:
        return 'allenai/led-base-16384-multi_lexsum-source-long'

    @staticmethod
    def led_source_to_long() -> ModelNode:
        return lookup_or_init_model(ModelDatabase.LEDForSequenceClassification,
                                    ModelDatabase.led_source_to_long_id(),
                                    ModelDatabase.led_source_to_long_id())


# DEFAULTS
DEFAULT_ROOTS = {}
DEFAULT_INDEX_PATH = os.path.join(Path(__file__).resolve().parent.parent.parent,
                                  'index_dir/main/indexer.json').replace("\\",
                                                                         "/")


def lookup_or_init_model(model_cls: Type[BaseTransformer], model_id: str,
                         tokenizer_id: str = None,
                         quantization_config: BitsAndBytesConfig = None) -> ModelNode:
    assert tokenizer_id is not None, 'NLP Model needs tokenizer id'
    model_node = \
        CACHED_INDEXER.lookup_or_init(model_id,
                                      model_cls=model_cls,
                                      quantization_config=quantization_config)
    model_node.model.eval()
    model_node.model.verify_tokenizer(model_node.tokenizer)
    return model_node

def get_models_path()->DirPath:
    return CACHED_INDEXER.root_path


class Indexer:
    @staticmethod
    def get_default_index():
        return CACHED_INDEXER

    def __init__(self, dir: DirPath):
        self.root_path = dir
        self.roots: Dict[str, List[str]] = {}
        self.runtime_model_cache: Dict[str, BaseTransformer] = {}
        self.runtime_tokenizer_cache: Dict[str, PreTrainedTokenizer] = {}

    def to_json(self, as_str=False) -> Union[dict, str]:
        obj_dict = {}
        for k, v in sorted(self.__dict__.items()):
            obj_dict[k] = v
        return json.dumps(obj_dict, indent=4,
                          separators=(',', ': ')) if as_str else obj_dict

    def save_as_json(self):
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        with open(
                os.path.join(self.root_path, 'indexer.json').replace("\\", "/"),
                'w') as f:
            f.write(self.to_json(as_str=True))

    @staticmethod
    def load_from_json(path=DEFAULT_INDEX_PATH):
        if not os.path.exists(path):
            logging.warning(
                'Indexer json file does not exist at {}, creating new one.'.format(
                    os.path.abspath(path)))
            idxer = Indexer(os.path.dirname(path))
            idxer.save_as_json()
        with open(path) as file_object:
            # store file data in object
            data: Dict = json.load(file_object)
        idxer = Indexer(data['root_path'])
        for k, v in sorted(data.items()):
            idxer.__dict__[k] = v
        return idxer

    def find_parent_child(self, model_id: str) -> Tuple[str, str]:
        names = model_id.split('/')
        parent = '/'.join(names[:2])
        child = '/'.join(names[2:])
        return parent, child

    def check_state(self):
        if not os.path.exists(self.root_path):
            raise OSError(
                f'Indexer root path {os.path.abspath(self.root_path)} does '
                f'not exist, check if the indexer moved.')

    def path_from_id(self, model_id: str, create_if_not_exist=True):
        model_dir: DirPath = (
            os.path.join(self.root_path, model_id).replace(
                "\\", "/"))
        if create_if_not_exist and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def get_or_add_to_root(self, model_id: str):
        """ Warning, this will create a direcotry, even if something else
        fails downstream"""
        if model_id not in self.roots:
            self.roots[model_id] = []
            self.roots = dict(sorted(self.roots.items()))
            self.save_as_json()
        return self.path_from_id(model_id, create_if_not_exist=True)

    def lookup_or_init(self, model_id: str,
                       model_cls: Type[BaseTransformer],
                       quantization_config: BitsAndBytesConfig = None) -> ModelNode:
        self.check_state()
        model_dir: DirPath = self.get_or_add_to_root(model_id)

        # TODO, cleaner way to distinguish the models that are available to load
        model: Optional[BaseTransformer] = \
            model_cls.load_model(model_dir, quantization_config)
        tokenizer: Optional[hug.PreTrainedTokenizer] = model_cls.load_tokenizer(
            model_dir)
        loaded_successfully = model is not None and tokenizer is not None
        if not loaded_successfully:
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
                model_id)
        if model is None or tokenizer is None:
            raise FileNotFoundError(
                'Model {} not found. Checked in directory {}'.format(model_id,
                                                                     os.path.abspath(
                                                                         model_dir)))
        return ModelNode(model, tokenizer, model_id)

    def save_to_index(self, model_node: ModelNode) -> DirPath:
        self.check_state()
        model_dir: DirPath = self.get_or_add_to_root(model_node.model_id)
        return model_dir

    def get_model_for(self, model_id: str):
        '''
            TODO we want to be able to look up models by task, input/output space.
        '''
        pass

    def cache_runtime_model(self, model_id: str, model: BaseModel):
        self.runtime_model_cache[model_id] = model

    def get_cached_runtime_nlp_model(self, model_id: str, model_cls: Type[
        BaseTransformer] = None) -> Tuple[
        BaseTransformer, PreTrainedTokenizer]:
        '''
        Let's try to automate the whole finding model_cls later
        '''

        if model_id not in self.runtime_model_cache:
            if model_cls is None:
                return None, None
            else:
                logging.info('loading model {} into cache'.format(model_id))
                model_node = lookup_or_init_model(model_cls, model_id, model_id)
                self.runtime_model_cache[model_id] = model_node.model
                self.runtime_tokenizer_cache[model_id] = model_node.tokenizer
        return self.runtime_model_cache[model_id], \
            self.runtime_tokenizer_cache[model_id]


CACHED_INDEXER = Indexer.load_from_json()  # what's currently cached when the script is running
if len(CACHED_INDEXER.roots) == 0:
    CACHED_INDEXER.roots = DEFAULT_ROOTS
    CACHED_INDEXER.save_as_json()
