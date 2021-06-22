import numpy as np
import torch
from torch import nn


# ========================================================================= #
# Weight Initialisation Modes                                               #
# ========================================================================= #


_WEIGHT_INIT_MODES = {
    'none': lambda x: None,
    'default': lambda x: None,
    'uniform': torch.nn.init.uniform_,
    'normal': torch.nn.init.normal_,
    'ones': torch.nn.init.ones_,
    'zeros': torch.nn.init.zeros_,
    'xavier_uniform': torch.nn.init.xavier_uniform_,
    'xavier_normal': torch.nn.init.xavier_normal_,
    'kaiming_uniform': torch.nn.init.kaiming_uniform_,
    'kaiming_normal': torch.nn.init.kaiming_normal_,
    'orthogonal': torch.nn.init.orthogonal_,
}


def register_weight_initializer(name, function=None):
    """
    register a new weight initializer
    """
    def decorator(fn):
        if name in _WEIGHT_INIT_MODES:
            raise RuntimeError(f'weight initializer with name: {repr(name)} already exists!')
        _WEIGHT_INIT_MODES[name] = fn
        return fn
    # create decorator or decorate directly
    if function is None:
        return decorator
    else:
        return decorator(function)


# ========================================================================= #
# Weight Initializers                                                       #
# ========================================================================= #


def init_tensor(tensor: torch.Tensor, mode='xavier_normal') -> torch.Tensor:
    if mode in _WEIGHT_INIT_MODES:
        _WEIGHT_INIT_MODES[mode](tensor)
    else:
        raise KeyError(f'invalid weight init mode: {repr(mode)}')
    return tensor


def _init_layer_weights(layer: nn.Module, mode='xavier_normal', verbose=False):
    # initialise common layers layer
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        init_tensor(layer.weight, mode=mode)
        init_tensor(layer.bias, mode='zeros')
        print(f'\033[92m{mode} initialised\033[0m: {type(layer).__name__}')
    else:
        if verbose:
            print(f'\033[91skipped initialising\033[0m: {type(layer).__name__}')
    return layer


def init_weights(model: nn.Module, mode='xavier_normal', verbose=False) -> nn.Module:
    def _initialise(m):
        _init_layer_weights(m, mode=mode, verbose=verbose)
    return model.apply(_initialise)


# ========================================================================= #
# Custom Weight Initializers                                                #
# ========================================================================= #


@register_weight_initializer('custom')
def _custom_weight_init(weight: torch.Tensor):
    assert weight.ndim == 2, f'custom initializer only works with weights with 2 dimensions, got: {weight.shape}'
    out, inp = weight.shape
    with torch.no_grad():
        weight.normal_(0., np.sqrt(1 / inp))


@register_weight_initializer('custom_alt')
def _custom_alt_weight_init(weight: torch.Tensor):
    assert weight.ndim == 2, f'custom_alt initializer only works with weights with 2 dimensions, got: {weight.shape}'
    out, inp = weight.shape
    with torch.no_grad():
        weight.normal_(0., 1.)
        weight -= weight.mean(dim=-1, keepdim=True)
        weight /= weight.std(dim=-1, unbiased=False, keepdim=True)
        weight *= np.sqrt(1 / inp)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
