
import torch

from config import DEVICE


def tensor(x, dtype=torch.float32, copy=False):
    if isinstance(x, torch.Tensor):
        return x
    if copy:
        return torch.tensor(x, device=DEVICE, dtype=dtype)
    else:
        return torch.as_tensor(x, device=DEVICE, dtype=dtype)


def to_device(x):
    return x.to(DEVICE)


def to_np(t):
    return t.cpu().detach().numpy()


def ones(*size, dtype=None):
    return torch.ones(*size, dtype=dtype, device=DEVICE)


def zeros(*size, dtype=None):
    return torch.zeros(*size, dtype=dtype, device=DEVICE)


def empty(*size, dtype=None):
    return torch.empty(*size, dtype=dtype, device=DEVICE)


def ones_like(x, dtype=None):
    return torch.ones_like(x, dtype=dtype, device=DEVICE)
