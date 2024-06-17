''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:56:21
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import os
from typing import Any, Iterable, Optional

import numpy as np
import random
import scipy.signal
import torch


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    r"""
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        numpy 1d vector x, [x0,  x1, x2]

    output:
        [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def set_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_torch_variable(device):
    device = device.lower()
    use_cpu = device == 'cpu'
    use_gpu = device.split(':')[0] == 'cuda'
    assert use_cpu or use_gpu, 'device must be either cpu or cuda:\{gpu_id\}'
    if not torch.cuda.is_available():
        os.environ["MODEL_DEVICE"] = 'cpu'
    else:
        os.environ["MODEL_DEVICE"] = device


def get_torch_device():
    device_name = os.environ.get("MODEL_DEVICE")
    try:
        return torch.device(device_name)
    except:
        raise ValueError("'MODEL_DEVICE' env variable has not been specified. Current 'MODEL_DEVICE' env variable is {device_name}")


def get_device_name():
    return os.environ.get("MODEL_DEVICE")


def to_tensor(
        item: Any,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        ignore_keys: list = [],
        transform_scalar: bool = True,
        squeeze=False
    ) -> torch.Tensor:
    device = get_torch_device() if device is None else device

    def squeeze_tensor(d):
        data = torch.tensor(d, dtype=dtype, device=device)
        if squeeze:
            return torch.squeeze(data)
        return data

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            if k in ignore_keys:
                new_data[k] = v
            else:
                new_data[k] = to_tensor(v, dtype, device, ignore_keys, transform_scalar, squeeze=squeeze)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        return squeeze_tensor(item)
    elif isinstance(item, np.ndarray):
        return squeeze_tensor(item)
    elif isinstance(item, bool) or isinstance(item, str):
        return item
    elif np.isscalar(item):
        if transform_scalar:
            return torch.as_tensor(item, device=device).to(dtype)
        else:
            return item
    elif item is None:
        return None
    elif isinstance(item, torch.Tensor):
        return item.to(dtype)
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_ndarray(item: Any, dtype: np.dtype=None) -> np.ndarray:
    def transform(d):
        if dtype is None:
            return np.array(d)
        else:
            return np.array(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            new_data[k] = to_ndarray(v, dtype)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        elif hasattr(item, '_fields'):  # namedtuple
            return type(item)(*[to_ndarray(t, dtype) for t in item])
        else:
            new_data = []
            for t in item:
                new_data.append(to_ndarray(t, dtype))
            return new_data
    elif isinstance(item, torch.Tensor):
        if item.device != 'cpu':
            item = item.detach().cpu()
        if dtype is None:
            return item.numpy()
        else:
            return item.numpy().astype(dtype)
    elif isinstance(item, np.ndarray):
        if dtype is None:
            return item
        else:
            return item.astype(dtype)
    elif isinstance(item, bool) or isinstance(item, str):
        return item
    elif np.isscalar(item):
        return np.array(item)
    elif item is None:
        return None
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_device(item: Any, device: str=None, ignore_keys: list = []) -> Any:
    if device is None:
        device = get_device_name()
    if isinstance(item, torch.nn.Module):
        return item.to(device)
    elif isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, dict):
        new_item = {}
        for k in item.keys():
            if k in ignore_keys:
                new_item[k] = item[k]
            else:
                new_item[k] = to_device(item[k], device)
        return new_item
    elif isinstance(item, np.ndarray) or isinstance(item, np.bool_):
        return item
    elif item is None or isinstance(item, str):
        return item
    elif isinstance(item, list):
        return [to_device(k, device) for k in item]
    elif isinstance(item, tuple):
        return tuple([to_device(k, device) for k in item])
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def to_dtype(item: Any, dtype: type) -> Any:
    if isinstance(item, torch.Tensor):
        return item.to(dtype=dtype)
    elif isinstance(item, dict):
        return {k: to_dtype(item[k], dtype) for k in item.keys()}
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


def CPU(var):
    return var.cpu().detach().numpy()


def kaiming_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def normal(x, mu, sigma_sq):
    pi = CUDA(Variable(torch.FloatTensor([np.pi])))
    a = (-1*(CUDA(Variable(x))-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b
