import contextlib
import warnings
from functools import wraps
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from exp.nn.swish import swish
from exp.nn.swish import Swish


# ========================================================================= #
# Random Sampling                                                           #
# ========================================================================= #


_SAMPLERS = {
    # torch samplers
    'normal':   lambda *shape, dtype=torch.float32, device=None: torch.randn(*shape, dtype=dtype, device=device),
    'uniform':  lambda *shape, dtype=torch.float32, device=None: torch.rand(*shape,  dtype=dtype, device=device),
    'suniform': lambda *shape, dtype=torch.float32, device=None: torch.rand(*shape,  dtype=dtype, device=device) * 2 - 1,
}

_NP_SAMPLERS = {
    # numpy samplers
    'normal':   lambda *shape: np.random.randn(*shape).astype('float32'),
    'uniform':  lambda *shape: np.random.rand(*shape).astype('float32'),
    'suniform': lambda *shape: np.random.rand(*shape).astype('float32') * 2 - 1,
}


def get_sampler(sampler: str = 'normal', tensor=True):
    return _SAMPLERS[sampler] if tensor else _NP_SAMPLERS[sampler]


# ========================================================================= #
# Normalised NormActivation                                                     #
# ========================================================================= #


def compute_activation_norms(activation_fn, num_samples=16384, device=None, dtype=torch.float32, sampler: Union[str, callable] = 'normal') -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        # sample values
        sampler = get_sampler(sampler)
        samples = sampler(num_samples, device=device, dtype=dtype)
        # feed forward & compute mean and std
        act = activation_fn(samples)
        return act.mean(), act.std(unbiased=False)


def to_norm_activation_fn(activation_fn=None, num_samples=16384, device=None, dtype=torch.float32):
    def decorator(act_fn):
        # compute normalisation values & cache
        mean, std = compute_activation_norms(activation_fn=act_fn, num_samples=num_samples, device=device, dtype=dtype)

        # actual activation function
        @wraps(act_fn)
        def _act_norm(x):
            return (act_fn(x) - mean) / std

        # rename activation function
        if activation_fn is not None:
            _act_norm.__name__ = f'norm_{_act_norm.__name__}'

        # return new activation function
        return _act_norm

    return decorator(activation_fn) if (activation_fn is not None) else decorator


# ========================================================================= #
# Activations                                                               #
# ========================================================================= #


# activation functions
def identity(x): return x


norm_tanh       = to_norm_activation_fn(torch.tanh)
norm_sigmoid    = to_norm_activation_fn(torch.sigmoid)
norm_relu       = to_norm_activation_fn(F.relu)
norm_relu6      = to_norm_activation_fn(F.relu6)
norm_elu        = to_norm_activation_fn(F.elu)
norm_swish      = to_norm_activation_fn(swish)
norm_leaky_relu = to_norm_activation_fn(F.leaky_relu)


ACTIVATIONS = {
    'identity':        identity,
    # builtin
    'tanh':            torch.tanh,
    'sigmoid':         torch.sigmoid,
    'relu':            F.relu,
    'relu6':           F.relu6,
    'leaky_relu':      F.leaky_relu,
    'elu':             F.elu,
    'swish':           swish,
    # custom
    'norm_tanh':       norm_tanh,
    'norm_sigmoid':    norm_sigmoid,
    'norm_relu':       norm_relu,
    'norm_relu6':      norm_relu6,
    'norm_leaky_relu': norm_leaky_relu,
    'norm_elu':        norm_elu,
    'norm_swish':      norm_swish,
}


def activate(x, mode='norm_swish'):
    return ACTIVATIONS[mode](x)


ACTIVATION_CLASSES = {
    'identity':   nn.Identity,
    # builtin
    'tanh':       nn.Tanh,
    'sigmoid':    nn.Sigmoid,
    'relu':       nn.ReLU,
    'relu6':      nn.ReLU6,
    'leaky_relu': nn.LeakyReLU,
    'elu':        nn.ELU,
    'swish':      Swish,
}


class GenericActivation(nn.Module):

    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn

    def __repr__(self):
        return f'GenericActivation({getattr(self.activation_fn, "__name__", self.activation_fn)})'

    def forward(self, x):
        return self.activation_fn(x)


def Activation(activation: str = 'norm_tanh') -> nn.Module:
    if activation in ACTIVATION_CLASSES:
        return ACTIVATION_CLASSES[activation]()
    else:
        return GenericActivation(ACTIVATIONS[activation])


def ActivationMaker(activation: str = 'norm_tanh'):
    return lambda: Activation(activation)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
