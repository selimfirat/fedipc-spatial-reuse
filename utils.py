import os
import random
import numpy as np
import torch
from torch import nn

def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def to_device(x, device):

    if issubclass(type(x), torch.Tensor) or issubclass(type(x), nn.Module):
        x = x.to(device)
    elif type(x) is dict:
        for k, v in x.items():
            x[k] = v.to(device)
    else:
        raise Exception("Unknown type cannot passed to device")

    return x
