import random

import numpy as np
import torch


# torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def to_cuda(tensor):
    return tensor.cuda()


def to_cpu(tensor):
    return tensor.cpu()


if torch.cuda.is_available():
    DEVICE_NUM = 0
    to_device = to_cuda
    print('{} gpus available, will be using gpu {}'.format(
        torch.cuda.device_count(), DEVICE_NUM))
    DEVICE = torch.device('cuda:{}'.format(DEVICE_NUM))
    torch.cuda.set_device(DEVICE)
else:
    print('Settings: using cpu')
    to_device = to_cpu
    DEVICE = torch.device('cpu')
DEVICE = torch.device('cpu')
DTYPE = torch.half
ARGS = {'device': DEVICE, 'dtype': DTYPE}

SEED = 23
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
