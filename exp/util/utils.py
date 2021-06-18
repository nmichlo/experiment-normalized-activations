from itertools import chain
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt


def iter_pairs(items):
    items = list(items)
    return list(zip(items[:-1], items[1:]))


def color_plot_center(values, c=0):
    string = str_plot_center(values, center=c)
    pre, mid, aft = string[:2], string[2:-2], string[-2:]
    mid = ''.join(
        f'\033[91m{s}' if (v < c) else (f'\033[92m{s}' if (v > c) else f'{s}') for v, s in zip(values, mid)
    ) + '\033[0m'
    return ''.join([pre, mid, aft])


def str_plot_center(values, center: float = None, min_radius=None):
    if len(values) == 0:
        line = ''
    else:
        line = str_plot(values, center=center, min_radius=min_radius, lines=1, min_is_zero=True)
    return f'[▸{line}◂]'


def str_plot(values, center: float = None, min_radius=None, min=None, max=None, lines=1, min_is_zero=True):
    # check values
    bounded = False
    if (center is not None) or (min_radius is not None):
        assert (min is None) and (max is None)
    if (min is not None) and (max is not None):
        assert (center is None) and (min_radius is None)
        assert min < max
        bounded = True
    # consts
    symbols = ' ▁▂▃▄▅▆▇█'
    n = len(symbols)
    # get min and max
    if center is None:
        if bounded:
            m, M = min, max
        else:
            m, M = np.min(values), np.max(values)
        center, r = (m + M) / 2, M - m
    else:
        d = np.abs(center - np.min(values))
        D = np.abs(center - np.max(values))
        r = np.maximum(d, D)
    # adjust min radius
    if min_radius is not None:
        r = np.maximum(min_radius, r)
    r = np.maximum(r, 1e-20)
    # scale and clamp values
    values = np.clip((values - (center - r)) / (2 * r), 0, 1)
    # generate indices
    if min_is_zero:
        idx = 0 + np.int32(np.around(values * lines * (n - 1)))
    else:
        idx = 1 + np.int32(np.around(values * lines * (n - 2)))
    # normalise indices
    n_block_pad = idx // n
    n_space_pad = lines - (n_block_pad + 1)
    symbol_idx = idx % n
    # generate columns
    cols = [
        chain([' '] * ns, [symbols[i]], [symbols[-1]] * nb)
        for ns, i, nb in zip(n_space_pad, symbol_idx, n_block_pad)
    ]
    # generate rows
    return '\n'.join(''.join(row) for row in zip(*cols))


def print_stats(stats, stat_names=('μ', 'σ'), centers=(0.0, 1.0), min_radius=(0.05, 0.5), title=''):
    means, stds = stats
    if title:
        print(f'[{title}] - ', end='')
    print(
        f'{stat_names[0]}: {str_plot_center(means, center=centers[0], min_radius=min_radius[0])}',
        f'{stat_names[1]}: {str_plot_center(stds,  center=centers[1], min_radius=min_radius[1])}',
        '|',
        f'{stat_names[0]}: [' + ', '.join(f'{v:6.3f}' for v in means) + ']',
        f'{stat_names[1]}: [' + ', '.join(f'{v:6.3f}' for v in stds) + ']',
    )


def plot_batch(*batches: torch.Tensor, clip=False, mean_std: Optional[Tuple[float, float]] = None):
    fig, axs = plt.subplots(1, len(batches), squeeze=False)
    for batch, ax in zip(batches, axs.reshape(-1)):
        assert batch.ndim == 4
        assert batch.dtype == torch.float32
        if mean_std is not None:
            mean, std = mean_std
            batch = (batch * std) + mean
        if clip:
            batch = torch.clip(batch, 0, 1)
        image = torch.moveaxis(torchvision.utils.make_grid(batch, nrow=3), 0, -1)
        ax.imshow(image)
        ax.set_axis_off()
    fig.tight_layout()
    plt.show()
