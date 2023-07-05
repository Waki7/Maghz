from mgz.typing import *
from mgz.typing import *
from mgz.models.base_model import BaseModel
from mgz.ds.base_dataset import BaseDataset
import spaces as sp
from mgz.models.base_model import BaseModel
from mgz.models.mobile_net import MobileNetV2
import json
from json import JSONDecoder, JSONEncoder
import os
from mgz.model_vc.model_node import ModelNode

# DEFAULTS
DEFAULT_ROOTS = {}


class Indexer:
    def __init__(self, dir: DirPath):
        self.root_path = dir
        self.roots: Dict[str, List[str]] = {}

    def to_json(self) -> str:
        obj_dict = {}
        for k, v in sorted(self.__dict__.items()):
            obj_dict[k] = v
        return json.dumps(obj_dict, indent=4, separators=(',', ': '))

    def save_as_json(self):
        with open(os.path.join(self.root_path, 'indexer.json'), 'w') as f:
            f.write(self.to_json())

    @staticmethod
    def load_from_json(path='../../../index_dir/main/indexer.json'):
        # obj_dict = {}
        # for k, v in sorted(dict):
        #     obj_dict[k] = v
        if not os.path.exists(path):
            raise OSError(
                f'Indexer json file {os.path.abspath(path)} does not exist.')
        with open(path) as file_object:
            # store file data in object
            data = json.load(file_object)
        idxer = Indexer(data['root_path'])
        for k, v in sorted(data.items()):
            idxer.__dict__[k] = v
        return idxer

    def full_path(self, name):
        return os.path.join(self.root_path, name)

    def find_parent_child(self, model: str) -> Tuple[str, str]:
        pass

    def check_state(self):
        if not os.path.exists(self.root_path):
            raise OSError(
                f'Indexer root path {os.path.abspath(self.root_path)} does not exist, check if the indexer moved.')

    def add_root(self, model: str):
        self.roots[model] = []
        self.roots = dict(sorted(self.roots.items()))
        self.save_as_json()

    def lookup(self, model_id: str,
               loader: Callable[[str], BaseDataset],
               init_save: Callable[[str], None]) -> ModelNode:
        self.check_state()
        # root_path, child_path = find_parent_child(model)
        # so this relationship, you have a root and you have a child, there is a relationship between the two, you can define the child by the parent somehow. In our case because we hacky we just using this to distinguish the model vs the weights. model is parent, weights is child.
        if model_id in self.roots:
            model = loader(self.full_path(model_id))
        else:
            try:
                init_save(self.full_path(model_id))
                self.add_root(model_id)
                model = loader(self.full_path(model_id))
            except OSError as e:  # model has not been created yet and does not exist
                raise e
        return model


CACHED_INDEXER = Indexer.load_from_json()  # what's currently cached when the script is running
if len(CACHED_INDEXER.roots) == 0:
    CACHED_INDEXER.roots = DEFAULT_ROOTS
    CACHED_INDEXER.save_as_json()
