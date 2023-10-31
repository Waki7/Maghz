import copy
import json
from enum import Enum
from functools import partial

import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import spaces as sp
from mgz.typing import *

LabelType = []
T = TypeVar("T")


class DataState(Enum):
    NOT_LOADED = 'not_loaded'
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class BaseDataset(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.meta_data = None

        self.in_space: sp.Space
        self.tgt_space: sp.Space

        # --- Initialization flags ---
        self.use_cuda = False
        # data_state must be set before loading sometimes
        self.data_state = DataState.NOT_LOADED
        self.loaded = False

    @property
    def input_space(self) -> sp.Sentence:
        raise NotImplementedError

    @property
    def target_space(self) -> Union[sp.Sentence, sp.RegressionTarget]:
        raise NotImplementedError

    def to_json(self, as_str=False, as_summary_str=False) -> Union[dict, str]:
        '''
        Warning: This method only shows a subset of the dataset's information.
        '''
        obj_dict = {
            "input_space": self.input_space.__repr__(),
            "target_space": self.target_space.__repr__(),
            "cls_name": self.__class__.__name__}
        if as_summary_str:
            # We probably want this to be a bit different in the future
            return "{}-".format(self.__class__.__name__)
        if as_str:
            json.dumps(obj_dict, indent=4, separators=(',', ': '))
        return obj_dict

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        raise NotImplementedError

    def __len__(self):
        'Denotes the total number of samples'
        raise NotImplementedError

    def gen(self) -> Generator[T, None, None]:
        raise NotImplementedError

    def __getitem__(self, index):
        'Generates one sample of data'
        raise NotImplementedError

    def _collate_fn(self, device: Union[int, torch.device],
                    batch: List[Tuple[GermanT, EnglishT]]):
        raise NotImplementedError

    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(self._collate_fn, device)

    def create_dataloaders(self,
                           device: Union[torch.device, int],
                           batch_size: int = 12,
                           is_distributed: bool = True,
                           turn_off_shuffle=False,
                           data_sampler: torch.utils.data.Sampler = None
                           ) -> (DataLoader, DataLoader):
        valid_sampler = (
            DistributedSampler(self) if is_distributed else data_sampler
        )
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=not turn_off_shuffle and (valid_sampler is None),
            sampler=valid_sampler,
            collate_fn=self.get_collate_fn(device)
        )
        return dataloader

    @abstractmethod
    def _load(self, train: bool = False, val: bool = False,
              test: bool = False):
        raise NotImplementedError

    def load_training_data(self):
        if self.data_state == DataState.TRAIN:
            return self
        self._load(train=True)
        return self

    def gen_training_data(self):
        assert self.data_state == DataState.NOT_LOADED
        return copy.deepcopy(self).load_training_data()

    def load_validation_data(self):
        if self.data_state == DataState.VAL:
            return self
        self._load(val=True)
        return self

    def gen_validation_data(self):
        assert self.data_state == DataState.NOT_LOADED
        return copy.deepcopy(self).load_validation_data()

    def load_testing_data(self):
        if self.data_state == DataState.TEST:
            return self
        self._load(test=True)
        return self

    def gen_testing_data(self):
        assert self.data_state == DataState.NOT_LOADED
        return copy.deepcopy(self).load_testing_data()
