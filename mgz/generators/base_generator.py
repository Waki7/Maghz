import torch

class BaseGenerator(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self):
        'Initialization'
        pass

  def __len__(self):
        'Denotes the total number of samples'
        raise NotImplementedError

  def __getitem__(self, index):
        'Generates one sample of data'
        raise NotImplementedError