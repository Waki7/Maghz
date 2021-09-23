from mgz.typing import *
from mgz.typing import *
from mgz.models.base_model import BaseModel
from mgz.datasets.base_dataset import BaseDataset
import spaces as sp
from mgz.models.base_model import BaseModel
from mgz.models.mobile_net import MobileNetV2
import json

ROOT_INDEX = {
    'MobileNetV2': MobileNetV2,
}

class Indexer:
    def __init__(self, dir: DirPath ):
        self.root_path = dir
        self.roots: List[str] = [MobileNetV2.__name__]

    def to_json(self) -> Dict:
        obj_dict = {}
        for k, v in sorted(self.__dict__.items()):
            obj_dict[k] = v

        return json.dumps(obj_dict, indent=4)

    def from_json(self, dict: Dict):
        obj_dict = {}
        for k, v in sorted(dict):
            obj_dict[k] = v
