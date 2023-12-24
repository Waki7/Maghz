import bitsandbytes as bnb
import torch

torch.set_default_dtype(torch.float16)
model = torch.nn.ModuleList([
    bnb.nn.Linear4bit(
        4096, 4096, False, torch.float32, False, 'fp4'
    ) for _ in range(50)
])
model.cuda()  # An illegal memory access was encountered

del model

torch.cuda.empty_cache()
device = torch.cuda.current_device()  # where cuda is numba
