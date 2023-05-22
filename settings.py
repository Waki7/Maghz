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


_debug_use_cpu = False
_debug_use_cpu = True

if torch.cuda.is_available() and not _debug_use_cpu:
    DEVICE_NUM = 0
    to_device = to_cuda
    print('{} gpus available, will be using gpu {}'.format(
        torch.cuda.device_count(), DEVICE_NUM))
    DEVICE = torch.device('cuda:{}'.format(DEVICE_NUM))
    torch.cuda.set_device(DEVICE)
else:
    print('using cpu')
    to_device = to_cpu
    DEVICE = torch.device('cpu')

DTYPE_LONG = torch.long
DTYPE_X = torch.half  # torch.float torch.half
ARGS = {'device': DEVICE, 'dtype': DTYPE_X}

SEED = 23
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def print_gpu_usage():
    print("torch.cuda.memory_allocated: %f GB" % (
            torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    print("torch.cuda.memory_reserved: %f GB" % (
            torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
    print("torch.cuda.max_memory_reserved: %f GB" % (
            torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
