import json
import os
from pathlib import Path

import transformers as hug
from transformers import PreTrainedTokenizer

import mgz.version_control as vc
from mgz.models.base_model import BaseModel
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.typing import *


class ModelDatabase:
    from mgz.models.nlp.led import LEDForBinaryTagging
    '''
    Just some quick access base models. For easier book keeping for now.
    '''

    @staticmethod
    def led_source_to_long() -> vc.ModelNode:
        return lookup_model(ModelDatabase.LEDForBinaryTagging,
                            'allenai/led-base-16384-multi_lexsum-source-long',
                            'allenai/led-base-16384-multi_lexsum-source-long')


# DEFAULTS
DEFAULT_ROOTS = {}
DEFAULT_INDEX_PATH = os.path.join(Path(__file__).resolve().parent.parent.parent,
                                  'index_dir/main/indexer.json').replace("\\",
                                                                         "/")


def lookup_model(model_cls: Type[BaseTransformer], model_id: str,
                 tokenizer_id: str = None) -> vc.ModelNode:
    assert tokenizer_id is not None, 'NLP Model needs tokenizer id'
    model_node = \
        CACHED_INDEXER.lookup_or_init(model_id,
                                      load_model=model_cls.load_model,
                                      load_tokenizer=model_cls.load_tokenizer,
                                      init_save=model_cls.initial_save)
    model_node.model.eval()
    model_node.model.verify_tokenizer(model_node.tokenizer)
    return model_node


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
                f'Indexer root path {os.path.abspath(self.root_path)} does not exist, check if the indexer moved.')

    def path_from_id(self, model_id: str, create_if_not_exist=True):
        model_dir: DirPath = os.path.join(self.root_path, model_id).replace(
            "\\",
            "/")
        if create_if_not_exist and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def add_root(self, model_id: str):
        """ Warning, this will create a direcotry, even if something else
        fails downstream"""
        self.roots[model_id] = []
        self.roots = dict(sorted(self.roots.items()))
        self.save_as_json()
        return self.path_from_id(model_id, create_if_not_exist=True)

    def lookup_or_init(self, model_id: str,
                       load_model: Callable[[str], BaseTransformer],
                       load_tokenizer: Callable[[str], hug.PreTrainedTokenizer],
                       init_save: Callable[[str, str], None]) -> vc.ModelNode:
        self.check_state()
        loaded_successfully = False
        model_dir: DirPath = self.add_root(model_id)

        model = None
        tokenizer = None
        # so this relationship, you have a root and you have a child,
        # there is a relationship between the two, you can define the child
        # by the parent somehow. In our case because we hacky we just using
        # this to distinguish the model vs the weights. model is parent,
        # weights is child.
        if model_id in self.roots:
            try:
                model: BaseTransformer = load_model(model_dir)
                tokenizer: hug.PreTrainedTokenizer = load_tokenizer(model_dir)
                loaded_successfully = True
            except FileNotFoundError as e:  # model has not been created yet and does not exist
                os.rmdir(model_dir)
                raise e
        if not loaded_successfully:
            try:
                # try to change it later
                init_save(model_id, model_dir)
                model: BaseTransformer = load_model(model_dir)
                tokenizer: hug.PreTrainedTokenizer = load_tokenizer(model_id)
            except OSError as e:  # model has not been created yet and does not exist
                os.rmdir(model_dir)
                raise e
        return vc.ModelNode(model, tokenizer, model_id)

    def save_to_index(self, model_node: vc.ModelNode) -> DirPath:
        self.check_state()
        model_dir: DirPath = self.add_root(model_node.model_id)
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
                model_node = lookup_model(model_cls, model_id, model_id)
                self.runtime_model_cache[model_id] = model_node.model
                self.runtime_tokenizer_cache[model_id] = model_node.tokenizer
        return self.runtime_model_cache[model_id], \
            self.runtime_tokenizer_cache[model_id]


CACHED_INDEXER = Indexer.load_from_json()  # what's currently cached when the script is running
if len(CACHED_INDEXER.roots) == 0:
    CACHED_INDEXER.roots = DEFAULT_ROOTS
    CACHED_INDEXER.save_as_json()
