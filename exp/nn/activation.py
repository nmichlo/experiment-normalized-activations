import contextlib
import warnings
from functools import wraps
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
from torch import nn
from torch.nn import functional as F

from exp.nn.swish import swish
from exp.nn.swish import Swish


# ========================================================================= #
# Random Sampling                                                           #
# ========================================================================= #


_SAMPLERS = {
    'normal':   lambda *shape, dtype=torch.float32, device=None: torch.randn(*shape, dtype=dtype, device=device),
    'uniform':  lambda *shape, dtype=torch.float32, device=None: torch.rand(*shape,  dtype=dtype, device=device),
    'suniform': lambda *shape, dtype=torch.float32, device=None: torch.rand(*shape,  dtype=dtype, device=device) * 2 - 1
}


def get_sampler(sampler: Union[str, callable] = 'normal'):
    if isinstance(sampler, str):
        sampler = _SAMPLERS[sampler]
    else:
        assert callable(sampler)
    return sampler


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
# Norm NormActivation Class                                                     #
# ========================================================================= #


class NormActivationWrapper(nn.Module):

    def __init__(self, activation: nn.Module, init_samples: Optional[int] = 1024, init_sampler: Optional[str] = 'normal', can_optimize: bool = True):
        super().__init__()
        # save the activation
        assert isinstance(activation, nn.Module), f'activation is not an instance of nn.Module, got: {type(activation)}'
        self._activation = activation
        # can_optimize
        # compute mean and std
        if (init_samples is None) or (init_sampler is None):
            mean, std = 0, 1
        else:
            mean, std = compute_activation_norms(activation, num_samples=init_samples, dtype=torch.float32, sampler=init_sampler)
        # initialise values
        mean = nn.Parameter(torch.as_tensor(mean, dtype=torch.float32), requires_grad=False)
        std  = nn.Parameter(torch.as_tensor(std,  dtype=torch.float32), requires_grad=False)
        # save values
        self._can_optimize = can_optimize
        if can_optimize:
            self.register_parameter('_mean', mean)
            self.register_parameter('_std', std)
        else:
            # buffers are saved along with the model, but cannot receive a gradient
            self.register_buffer('_mean', mean, persistent=True)
            self.register_buffer('_std', std, persistent=True)
        # checks
        assert self._mean.shape == ()
        assert self._std.shape == ()

    def forward(self, x):
        return (self._activation(x) - self._mean) / self._std

    @property
    def can_optimize(self):
        return self._can_optimize

    # @property
    # def __name__(self):
    #     return f"norm_{getattr(self._activation, '__name__', self._activation)}".lower()
    #
    # def __str__(self):
    #     return self.__name__

    def __repr__(self):
        activation = getattr(self._activation, '__name__', self._activation)
        return f'{self.__class__.__name__}(activation={activation}, can_optimize={self.can_optimize})'

    @property
    def values(self):
        return [self._mean.item(), self._std.item()]

    @classmethod
    def recursive_get_from_module(cls, model: nn.Module, mode: str = 'all'):
        allowed_states = {'all': [True, False], 'unoptimizable': [False], 'optimizable': [True]}[mode]
        layers = []

        def _collect(m):
            if isinstance(m, cls):
                if m.can_optimize in allowed_states:
                    layers.append(m)

        model.apply(_collect)
        return layers


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


def NormActivation(activation: str = 'norm_tanh', norm_samples: int = 1024, norm_sampler: str = 'normal') -> NormActivationWrapper:
    # should normalise or not
    sample = False
    if activation.startswith('norm_'):
        sample = True
        activation = activation[len('norm_'):]
    # make new activation class
    activation = ACTIVATION_CLASSES[activation]()
    # instantiate NormActivation
    if sample:
        return NormActivationWrapper(activation, init_samples=norm_samples, init_sampler=norm_sampler, can_optimize=True)
    else:
        return NormActivationWrapper(activation, init_samples=None,         init_sampler=None,         can_optimize=False)


def NormActivationMaker(activation: str = 'norm_swish', norm_samples: int = 1024, norm_sampler: str = 'normal') -> Callable[[int, int, int], NormActivationWrapper]:
    return lambda i, inp, out: NormActivation(activation=activation, norm_samples=norm_samples, norm_sampler=norm_sampler)


# ========================================================================= #
# Norm Layers Hooks                                                         #
# ========================================================================= #


@contextlib.contextmanager
def norm_layers_context(model: nn.Module, mode='all', set_optimizable=False):
    # get layers
    norm_layers = tuple(NormActivationWrapper.recursive_get_from_module(model, mode=mode))
    # set optimizable
    for layer in norm_layers:
        layer.requires_grad_(set_optimizable)
    # return layers
    try:
        yield norm_layers
    finally:
        # unset optimizable
        for layer in norm_layers:
            layer.requires_grad_(False)


@contextlib.contextmanager
def forward_capture_context(layers: Sequence[nn.Module]) -> List[torch.Tensor]:
    # initialise capture hook
    stack = []
    def hook(module, input, output) -> None:
        if len(stack) >= len(layers):
            raise RuntimeError('stack was not cleared before next feed forward')
        stack.append(output)
    # register hooks
    handles = [layer.register_forward_hook(hook) for layer in layers]
    # yield stack
    try:
        yield stack
    finally:
        if len(stack) != 0:
            warnings.warn('stack was not cleared before context was exited.')
        stack.clear()
        # unregister hooks
        for handle in handles:
            handle.remove()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
