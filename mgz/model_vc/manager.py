from mgz.models.base_model import BaseModel
from mgz.datasets.base_dataset import BaseDataset
import spaces as sp


class Manager():
    def __init__(self):
        pass

    def query_by_model(self, model: BaseModel):
        pass

    def query_by_dataset(self, dataset: BaseDataset):
        pass

    def query_by_space(self, space: sp.Space):
        pass

    def query_by_metric(self):
        pass

    def joint_query(self):
        pass