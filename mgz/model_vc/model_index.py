import json
import os
from pathlib import Path

from transformers import PreTrainedTokenizerBase

from mgz.ds.base_dataset import BaseDataset
from mgz.model_vc.model_node import ModelNode
from mgz.models.base_model import BaseModel
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.typing import *

# DEFAULTS
DEFAULT_ROOTS = {}
DEFAULT_INDEX_PATH = os.path.join(Path(__file__).resolve().parent.parent.parent,
                                  'index_dir/main/indexer.json').replace("\\",
                                                                         "/")


class Indexer:
    @staticmethod
    def get_default_index():
        return CACHED_INDEXER

    def __init__(self, dir: DirPath):
        self.root_path = dir
        self.roots: Dict[str, List[str]] = {}
        self.runtime_model_cache: Dict[str, BaseTransformer] = {}
        self.runtime_tokenizer_cache: Dict[str, PreTrainedTokenizerBase] = {}

    def to_json(self) -> str:
        obj_dict = {}
        for k, v in sorted(self.__dict__.items()):
            obj_dict[k] = v
        return json.dumps(obj_dict, indent=4, separators=(',', ': '))

    def save_as_json(self):
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        with open(
                os.path.join(self.root_path, 'indexer.json').replace("\\", "/"),
                'w') as f:
            f.write(self.to_json())

    @staticmethod
    def load_from_json(path=DEFAULT_INDEX_PATH):
        # obj_dict = {}
        # for k, v in sorted(dict):
        #     obj_dict[k] = v
        if not os.path.exists(path):
            logging.warning(
                'Indexer json file does not exist at {}, creating new one.'.format(
                    os.path.abspath(path)))
            idxer = Indexer(os.path.dirname(path))
            idxer.save_as_json()
        with open(path) as file_object:
            # store file data in object
            data = json.load(file_object)
        idxer = Indexer(data['root_path'])
        for k, v in sorted(data.items()):
            idxer.__dict__[k] = v
        return idxer

    def full_path(self, name):
        return os.path.join(self.root_path, name).replace("\\", "/")

    def find_parent_child(self, model_id: str) -> Tuple[str, str]:
        names = model_id.split('/')
        parent = '/'.join(names[:2])
        child = '/'.join(names[2:])
        return parent, child

    def check_state(self):
        if not os.path.exists(self.root_path):
            raise OSError(
                f'Indexer root path {os.path.abspath(self.root_path)} does not exist, check if the indexer moved.')

    def add_root(self, model: str):
        self.roots[model] = []
        self.roots = dict(sorted(self.roots.items()))
        self.save_as_json()

    def lookup_or_init(self, model_id: str,
                       loader: Callable[[str], BaseDataset],
                       init_save: Callable[[str], None]) -> ModelNode:
        self.check_state()
        loaded_successfully = False
        # root_path, child_path = find_parent_child(model)
        # so this relationship, you have a root and you have a child, there is a relationship between the two, you can define the child by the parent somehow. In our case because we hacky we just using this to distinguish the model vs the weights. model is parent, weights is child.
        if model_id in self.roots:
            try:
                model = loader(self.full_path(model_id))
                loaded_successfully = True
            except FileNotFoundError as e:  # model has not been created yet and does not exist
                pass
        if not loaded_successfully:
            try:
                init_save(self.full_path(model_id))
                self.add_root(model_id)
                model = loader(self.full_path(model_id))
            except OSError as e:  # model has not been created yet and does not exist
                raise e
        return model

    def get_model_for(self, model_id: str):
        '''
            TODO we want to be able to look up models by task, input/output space.
        '''
        pass

    def cache_runtime_model(self, model_id: str, model: BaseModel):
        self.runtime_model_cache[model_id] = model

    def get_cached_runtime_nlp_model(self, model_id: str, model_cls: Type[
        BaseTransformer] = None) -> Tuple[
        BaseTransformer, PreTrainedTokenizerBase]:
        '''
        Let's try to automate the whole finding model_cls later
        '''

        if model_id not in self.runtime_model_cache:
            if model_cls is None:
                return None, None
            else:
                logging.info('loading model {} into cache'.format(model_id))
                model, tokenizer = model_cls.from_pretrained(model_id)
                self.runtime_model_cache[model_id] = model
                self.runtime_tokenizer_cache[model_id] = tokenizer
        return self.runtime_model_cache[model_id], \
            self.runtime_tokenizer_cache[model_id]


CACHED_INDEXER = Indexer.load_from_json()  # what's currently cached when the script is running
if len(CACHED_INDEXER.roots) == 0:
    CACHED_INDEXER.roots = DEFAULT_ROOTS
    CACHED_INDEXER.save_as_json()
