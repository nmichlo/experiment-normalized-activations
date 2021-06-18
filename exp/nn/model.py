from typing import Any
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch import nn

from exp.nn.activation import forward_capture_context
from exp.nn.activation import get_sampler
from exp.nn.activation import norm_layers_context
from exp.nn.activation import NormActivationMaker
from exp.nn.weights import init_weights
from exp.util.utils import iter_pairs
from exp.util.utils import print_stats


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def get_ae_layer_sizes(start, mid, r=1) -> np.ndarray:
    assert r >= 0
    down = np.int32(np.linspace(start, mid, r + 1))[:-1]
    return np.array([*down, mid, *down[::-1]], dtype='int')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


def norm_activations_optimize(
    model,
    obs_shape,
    sampler='normal',
    steps=100,
    batch_size=256,
    lr=1e-1,
):
    assert steps > 0 and batch_size > 0 and lr > 0
    sampler = get_sampler(sampler)
    # hook into the outputs of the layers
    # enabling gradients for the norms
    with norm_layers_context(model, mode='optimizable', set_optimizable=True) as layers, forward_capture_context(layers) as outputs:
        # exit early if model has no activation norm layers
        if not layers:
            return np.array([]), np.array([])
        # make the optimizer
        optimizer = torch.optim.Adam(nn.ModuleList(layers).parameters(), lr=lr, eps=1e-5, weight_decay=1e-7)
        # optimize
        loss_hist = []
        for i in range(steps):
            x = sampler(batch_size, *obs_shape)
            # feed forward
            y = model(x)
            # get outputs statistics
            means = torch.stack([y.mean() for y in outputs], dim=0)
            stds = torch.stack([y.std() for y in outputs], dim=0)
            outputs.clear()
            # compute loss
            loss_mean = ((0 - means)**2).mean()
            loss_std = ((1 - stds)**2).mean()
            loss = loss_mean + loss_std
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # save values
            loss_hist.append(loss.item())
        # get mean & std values
        loss_hist = np.array(loss_hist, dtype='float32')
        values = np.array([m.values for m in layers], dtype='float32').T
    # return everything!
    return values, loss_hist


def norm_activations_analyse(model, obs_shape, sampler='normal', steps=100, batch_size=512) -> Tuple[np.ndarray, np.ndarray]:
    assert steps > 0 and batch_size > 0
    sampler = get_sampler(sampler)
    # feed forward
    with torch.no_grad():
        with norm_layers_context(model) as layers, forward_capture_context(layers) as outputs:
            stats = []
            for i in range(steps):
                x = sampler(batch_size, *obs_shape)
                # feed forward
                y = model(x)
                # save results
                stats.append([[y.mean().item(), y.std().item()] for y in outputs])
                outputs.clear()
            # get mean & std values
            values = np.array([m.values for m in layers], dtype='float32').T
            stats = np.array(stats).mean(axis=0).T
    # return
    return values, stats


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


def make_sequential_nn(
    sizes: Sequence[int],
    activations: Union[list, str, callable] = 'norm_tanh',
    init_mode='default',
    norm_samples: int = 1024,      # ignored if `activations` is not a string
    norm_sampler: str = 'normal',  # ignored if `activations` is not a string
) -> nn.Module:
    """
    activation can be a `str` or `Callable[[int, int, int], nn.Module]`
    """
    assert len(sizes) >= 2
    # get activation maker
    if isinstance(activations, str):
        activations = NormActivationMaker(activations, norm_samples=norm_samples, norm_sampler=norm_sampler)
    # make activations
    if callable(activations):
        acts = [activations(i, inp, out) for i, (inp, out) in enumerate(iter_pairs(sizes[:-1]))]
    else:
        acts = list(activations)
    # make fully connected layers
    linears = [nn.Linear(inp, out) for inp, out in iter_pairs(sizes)]
    assert len(acts) == len(linears) - 1
    # interleave layers
    layers = [None] * (len(acts) + len(linears))
    layers[0::2] = linears
    layers[1::2] = acts
    # make model
    model = nn.Sequential(*layers)
    # initialise
    init_weights(model, mode=init_mode, verbose=False)
    return model


def normalize_model(
    model: nn.Module,
    obs_shape: Sequence[int],
    # test activations
    test_steps: int = 128,
    test_batch_size: int = 32,
    # normalise activations
    norm_sampler: str = 'normal',
    norm_steps: int = 100,
    norm_batch_size: int = 32,
    norm_lr: float = 1e-2,
) -> Union[nn.Module, Tuple[Any, ...]]:
    # get the activations
    before_values, before_stats = norm_activations_analyse(model,  obs_shape=obs_shape, sampler=norm_sampler, steps=test_steps, batch_size=test_batch_size)
    after_values, loss_hist     = norm_activations_optimize(model, obs_shape=obs_shape, sampler=norm_sampler, steps=norm_steps, batch_size=norm_batch_size, lr=norm_lr)
    after_values, after_stats   = norm_activations_analyse(model,  obs_shape=obs_shape, sampler=norm_sampler, steps=test_steps, batch_size=test_batch_size)
    # return results
    return model, (before_values, before_stats, after_values, after_stats, loss_hist)


def make_normalized_model(
    sizes,
    activation: str = 'norm_tanh',
    init_mode='default',
    # test activations
    test_steps: int = 128,
    test_batch_size: int = 32,
    # normalise activations
    norm_samples: int = 1024,
    norm_sampler: str = 'normal',
    norm_steps: int = 100,
    norm_batch_size: int = 32,
    norm_lr: float = 1e-2,
    # print settings
    stats: bool = False,
    log: bool = False,
) -> Union[nn.Module, Tuple[Any, ...]]:
    # make the model
    model = make_sequential_nn(
        sizes,
        activations=activation,
        init_mode=init_mode,
        norm_samples=norm_samples,
        norm_sampler=norm_sampler,
    )

    # normalise the model
    model, (before_values, before_stats, after_values, after_stats, loss_hist) = normalize_model(
        model=model,
        obs_shape=[sizes[0]],
        test_steps=test_steps,
        test_batch_size=test_batch_size,
        norm_sampler=norm_sampler,
        norm_steps=norm_steps,
        norm_batch_size=norm_batch_size,
        norm_lr=norm_lr,
    )
    # print everything
    if log:
        title = f'{min(sizes):3d} {max(sizes):3d} {(len(sizes) - 1) // 2:3d} | {activation:18s} | {norm_sampler:15s} | {init_mode:15s}'
        print_stats(before_stats,               title=f'{title} | BFR')
        print_stats(after_stats,                title=f'{title} | AFT')
        print_stats(before_values-after_values, title=f'{title} | DIF', centers=(0.0, 0.0), min_radius=(None, None))
        print()
    # return the  optimized model and stats
    if stats:
        return model, (before_values, before_stats, after_values, after_stats, loss_hist)
    else:
        return model


# ========================================================================= #
# Wrapper                                                                   #
# ========================================================================= #


# class CommonWrapperModule(nn.Module):
#
#     def __init__(self, model: nn.Module):
#         super().__init__()
#         self._model = model
#
#     def forward(self, x):
#         (B, *shape) = x.shape
#         # do normalize
#         if self.hparams.normalize_input:
#             x = (x - 0.5) / 0.5
#         x = x.reshape(B, -1)
#         # feed forward
#         x = self._model(x)
#         # denormalize
#         x = x.reshape(B, *shape)
#         if self.hparams.unnormalize_output:
#             x = (x * 0.5) + 0.5
#         # get values
#         return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
