from enum import Enum

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, ConcatDataset

import spaces as sp
from mgz.typing import *

LabelType = []
T = TypeVar("T")


class DataSplit(Enum):
    NOT_LOADED = 'not_loaded'
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class BaseDataset(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.img_index: Dict[str, str] = {}
        self.img_label_index: Dict[str, LabelType] = {}
        self.meta_data = None

        self.in_space: sp.Space
        self.tgt_space: sp.Space

        # --- Initialization flags ---
        self.use_cuda = False
        self.data_state = DataSplit.NOT_LOADED

    @abstractmethod
    @property
    def input_space(self) -> sp.Sentence:
        raise NotImplementedError

    @abstractmethod
    @property
    def target_space(self) -> Union[sp.Sentence, sp.RegressionTarget]:
        raise NotImplementedError

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        raise NotImplementedError

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_index)

    def gen(self) -> Generator[T, None, None]:
        raise NotImplementedError

    def __getitem__(self, index):
        'Generates one sample of data'
        raise NotImplementedError

    def create_dataloaders(self,
                           device: Union[torch.device, int],
                           batch_size: int = 12000,
                           is_distributed: bool = True,
                           ) -> (DataLoader, DataLoader):
        raise NotImplementedError

    def pad_idx(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def load_training_data(self):
        raise NotImplementedError

    @abstractmethod
    def gen_training_data(self):
        raise NotImplementedError

    @abstractmethod
    def load_validation_data(self):
        raise NotImplementedError

    @abstractmethod
    def gen_validation_data(self):
        raise NotImplementedError

    @abstractmethod
    def load_testing_data(self):
        raise NotImplementedError

    @abstractmethod
    def gen_testing_data(self):
        raise NotImplementedError

    # class _MapStyleDataset(torch.utils.data.Dataset):

    # def __init__(self, iter_data) -> None:
    #     # TODO Avoid list issue #1296
    #     self._data = list(iter_data)
    #
    # def __len__(self):
    #     return len(self._data)
    #
    # def __getitem__(self, idx):
    #     return self._data[idx]

    # def read_file(self, file: str) -> torch.Tensor:
    #     '''
    #     >>> img = self.read_file('test_path')
    #     >>> img.shape
    #     (3, 100, 100)
    #     :param file: full path to image
    #     :return:
    #     '''
    #     img = torch_io.read_image(file)
    #     new_shape = [img.shape[1] // self.downsample_ratio,
    #                  img.shape[2] // self.downsample_ratio]
    #     img = torch_f.resize(img=img,
    #                          interpolation=torch_f.InterpolationMode.BILINEAR,
    #                          size=new_shape)
    #     return img
    #
    # def __getitem__(self, idx) -> Tuple[NDArray, NDArray]:
    #     img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    #     image = read_image(img_path)
    #     label = self.img_labels.iloc[idx, 1]
    #     if self.transform:
    #         image = self.transform(image)
    #     if self.target_transform:
    #         label = self.target_transform(label)
    #     ds = DataSample(features=(image,), labels=(label,))
    #     return ds
