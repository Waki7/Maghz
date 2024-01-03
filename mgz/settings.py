import os
import random

import numpy as np
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128'
import torch


# torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def to_cuda(tensor):
    return tensor.cuda()


def to_cpu(tensor):
    return tensor.cpu()


def to_mps(tensor):
    return tensor.to(torch.device("mps"))


_debug_use_cpu = False
# _debug_use_cpu = True
DEVICE = torch.device('cpu')
print('cuda avail', torch.cuda.is_available())
if torch.backends.mps.is_available() and not _debug_use_cpu:
    to_device = to_mps
    print('MPS available, will be running on apple GPU')
    torch.backends.mps.is_available()
    DEVICE = torch.device("mps")
elif torch.cuda.is_available() and not _debug_use_cpu:
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
VRAM = 24
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def empty_cache():
    if DEVICE == torch.device('cpu'):
        return
    elif DEVICE == torch.device('mps'):
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_gpu_usage(print_tag=''):
    if DEVICE == torch.device('cpu'):
        print('using cpu, no gpu usage')
    elif DEVICE == torch.device('mps'):
        print(print_tag,
              torch.mps.current_allocated_memory() / 1e9,
              'gb')
    elif torch.cuda.is_available():
        print(print_tag, "torch.cuda.memory_allocated: %f GB" % (
                torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print(print_tag, "torch.cuda.memory_reserved: %f GB" % (
                torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print(print_tag, "torch.cuda.max_memory_reserved: %f GB" % (
                torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
def get_available_gpu_mem():
    if DEVICE == torch.device('cpu'):
        return 0
    elif DEVICE == torch.device('mps'):
        return torch.mps.current_allocated_memory() / 1e9
    elif torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a
        return f / 1e9

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(name, param.numel(), param.device, param.requires_grad,
                  param.dtype)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
