import dataclasses
from argparse import Namespace
from functools import partial
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
import pytorch_lightning as pl
import torch
import wandb
from flash.core.classification import ClassificationTask
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch_optimizer import RAdam

from nfnets import ScaledStdConv2d, WSConv2d

from exp.data import make_image_data_module
from exp.nn.model import get_ae_layer_sizes
from exp.nn.model import get_layer_sizes
from exp.nn.swish import Swish
from exp.nn.weights import init_weights
from exp.util.nn_utils import find_submodules, replace_submodules
from exp.util.nn_utils import in_out_capture_context
from exp.util.pl_utils import pl_quick_train


# ===================================================================== #
# Layer Handling
# ===================================================================== #
from exp.util.utils import iter_pairs


@torch.no_grad()
def compute_gamma(activation_fn, batch_size=1024, samples=256):
    # from appendix D: https://arxiv.org/pdf/2101.08692.pdf
    y = activation_fn(torch.randn(batch_size, samples))
    gamma = torch.mean(torch.var(y, dim=1))**-0.5
    return gamma


class MeanShift(nn.Module):
    def __init__(self, wrapped_module: nn.Module):
        super().__init__()
        self.mean_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.wrapped_module = wrapped_module
        # mark as shifted
        assert not hasattr(wrapped_module, '_is_mean_shift_wrapped_'), f'module has already been mean shifted: {wrapped_module}'
        setattr(wrapped_module, '_is_mean_shift_wrapped_', True)

    def forward(self, x):
        return self.wrapped_module(x) - self.mean_shift


def wrap_mean_shift(model: nn.Module, visit_type=(nn.Conv2d, nn.Linear), visit_instance_of=False):
    """
    Normal layers sometimes struggle to normalise the activations
    if the mean is shifted.

    Wrap layers with a learnable parameter that can counteract this.
    """
    return replace_submodules(
        model,
        visit_type=visit_type,
        modify_fn=lambda c, k, p: MeanShift(c),
        visit_instance_of=visit_instance_of,
    )


# ===================================================================== #
# Layer Handling
# ===================================================================== #


# class WandbLayerCallback(pl.Callback):
#     def __init__(self, layers: Sequence[nn.Module], log_period: int = 500):
#         self._layers = layers
#         self._layer_handler = None
#         self._log_period = log_period
#     # logger
#     def batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, dataloader_idx, log=False):
#         inp_stds, out_stds, inp_means, out_means = self._layer_handler.step()
#         if log and (trainer.global_step % self._log_period == 0):
#             self._layer_handler.wandb_log_plots(inp_stds, out_stds, inp_means, out_means)
#     # hooks
#     def on_fit_start(self, trainer, pl_module): self._layer_handler = LayerStatsHandler(self._layers).start()
#     def on_fit_end(self,   trainer, pl_module): self._layer_handler.end()
#     def on_train_batch_end(self, *args, **kwargs):      self.batch_end(*args, **kwargs, log=False)
#     def on_test_batch_end(self, *args, **kwargs):       self.batch_end(*args, **kwargs, log=False)
#     def on_validation_batch_end(self, *args, **kwargs): self.batch_end(*args, **kwargs, log=False)


class LayerStatsHandler(object):

    """
    The layer handler manages an `in_out_capture_context` around
    model layers, and computes the mean & std of inputs and outputs
    to those layers.

    A single method `step` is provided that should be called to
    reset the stack of captured outputs after a call to the models
    `forward` method.

    *NB* wrapped nn.Modules should NEVER directly call `forward`,
         rather they should always use the `__call__` interface which
        passed variables to registered hooks required by this handler.
    """

    def __init__(self, layers: Sequence[nn.Module]):
        self._context_manager = in_out_capture_context(layers=layers, mode='in_out')
        self._layers = layers
        self._inp_stack = None
        self._out_stack = None

    def start(self) -> 'LayerStatsHandler':
        self._inp_stack, self._out_stack = self._context_manager.__enter__()
        return self

    def end(self) -> None:
        self._context_manager.__exit__(None, None, None)
        return None

    def _get_tensor(self, value) -> torch.Tensor:
        if isinstance(value, tuple):
            assert len(value) == 1
            return value[0]
        return value

    def step(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inp_stds, out_stds, inp_means, out_means = [], [], [], []
        # handle each layer
        for i in range(len(self._layers)):
            # compute values
            inp_stds.append(self._get_tensor(self._inp_stack[i]).std(unbiased=False))
            out_stds.append(self._get_tensor(self._out_stack[i]).std(unbiased=False))
            inp_means.append(self._get_tensor(self._inp_stack[i]).mean())
            out_means.append(self._get_tensor(self._out_stack[i]).mean())
        # aggregate
        inp_stds  = torch.stack(inp_stds,  dim=0)
        out_stds  = torch.stack(out_stds,  dim=0)
        inp_means = torch.stack(inp_means, dim=0)
        out_means = torch.stack(out_means, dim=0)
        # clear stacks
        self._inp_stack.clear()
        self._out_stack.clear()
        # return values
        return inp_stds, out_stds, inp_means, out_means

    @classmethod
    def wandb_log_plots(cls, inp_stds, out_stds, inp_means, out_means):
        entries = [
            Namespace(values=inp_stds,  label='inp_std', color='red'),
            Namespace(values=out_stds,  label='out_std', color='red'),
            Namespace(values=inp_means, label='inp_mean', color='red'),
            Namespace(values=out_means, label='out_mean', color='red'),
        ]
        df = pd.DataFrame([
            dict(x=x, y=y, label=entry.label)
            for entry in entries
            for x, y in enumerate(entry.values.detach().cpu().numpy().tolist())
        ])
        wandb.log({'layer_stats': px.line(df, x='x', y='y', color='label', title='Layer Input Standard Deviation')})


# ===================================================================== #
# Layer Regularisation
# ===================================================================== #


def std_slope(n: int, a: float, y_0: float = 0.001, y_n: float = 1.0, zero_is_flat=False, tensor=True):
    # check bounds
    if not ((-1 < a < 0) or (0 < a < 1)):
        raise ValueError('a must be in the range (-1, 0) or (0, 1)')
    # scale slope values
    if zero_is_flat:
        # scale slope value -> 0 is flat
        b = (1 / (1 - a)) if (a > 0) else (1 + a)
    else:
        # scale slope value -> 0 is sharp
        b = (1 / a) if (a > 0) else (-a)
    # compute values
    x = np.arange(n, dtype='float32')
    m = (y_n - y_0) / (b**(n - 1) - 1)
    y = m * (b**x - 1) + y_0
    # return values!
    if tensor:
        return torch.as_tensor(y, dtype=torch.float32)
    return y


@dataclasses.dataclass
class SlopeArgs:
    #     a,   y_0,   y_1
    # 1.  0.9   0.01 0.25
    # 2.  0.999 0.01 0.5
    # 3. -0.9   0.01 0.25
    # 4. -0.5   0.01 0.25
    a: float = 0.999
    y_0: float = 0.01
    y_n: float = 0.5

    def to_slope(self, n: int) -> torch.Tensor:
        return std_slope(n=n, a=self.a, y_0=self.y_0, y_n=self.y_n, zero_is_flat=False, tensor=True)


MeanStdTargetHint = Union[float, torch.Tensor, SlopeArgs]


def to_mean_or_std_target(n: int, target: MeanStdTargetHint) -> torch.Tensor:
    if isinstance(target, torch.Tensor):
        pass
    elif isinstance(target, SlopeArgs):
        target = target.to_slope(n)
    else:
        target = torch.full([n], fill_value=float(target))
    assert target.shape == (n,)
    return target


# ===================================================================== #
# Layer Regularisation
# ===================================================================== #


def normalize_targets(y: torch.Tensor, y_targ: torch.Tensor, model_type='classification') -> torch.Tensor:
    if model_type == 'classification':
        assert y.ndim == 2
        assert y_targ.ndim == 1
        assert y.shape[0] == y_targ.shape[0]
        return F.one_hot(y_targ, num_classes=y.shape[-1]).float()
    elif model_type == 'regression':
        assert y.shape == y_targ.shape
        return y_targ, {}
    else:
        raise KeyError(f'invalid normalize target model type: {repr(model_type)}')


class RegularizeLayerStatsTask(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[nn.Module],
        model_type='regression',
        # mean & std settings
        target_mean: MeanStdTargetHint = 0.,
        target_std: MeanStdTargetHint = 1.,
        reg_model_output: bool = True,
        # early stopping
        early_stop_value: float = 0.001,
        early_stop_exp_weight: float = 0.98,
        # optimizer settings
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict = None,
        learning_rate: float = 1e-3,
        # logging settings
        log_wandb_period: int = None,
        log_data_stats: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'layers'])
        self.hparams.optimizer_kwargs = self.hparams.optimizer_kwargs if self.hparams.optimizer_kwargs else {}
        # initialise
        self._model = model
        self._layer_handler = None
        self._layers = layers
        # normalise targets & store as paramter
        self._target_mean: torch.Tensor
        self._target_std: torch.Tensor
        self.register_buffer('_target_mean', to_mean_or_std_target(len(layers), target_mean))
        self.register_buffer('_target_std', to_mean_or_std_target(len(layers), target_std))
        # early stopping values
        self._early_stop_moving_ave = None

    def on_train_epoch_start(self, *args, **kwargs) -> None:
        self._layer_handler = LayerStatsHandler(self._layers).start()

    def on_train_epoch_end(self, *args, **kwargs) -> None:
        self._layer_handler = self._layer_handler.end()

    def _early_stop_update(self, loss: float):
        # update the exponential moving average
        if self._early_stop_moving_ave is None:
            self._early_stop_moving_ave = loss
        else:
            self._early_stop_moving_ave = self.hparams.early_stop_exp_weight * self._early_stop_moving_ave \
                                          + (1 - self.hparams.early_stop_exp_weight) * loss
        # log values
        self.log('early_stop_value', self._early_stop_moving_ave)
        self.log('early_stop_target', self.hparams.early_stop_value)
        # stop early if we have succeeded!
        if self._early_stop_moving_ave < self.hparams.early_stop_value:
            self.trainer.should_stop = True

    def training_step(self, x, batch_idx):
        x, y_targ = (x, x) if isinstance(x, torch.Tensor) else x

        # feed forward for layer hook
        y = self._model(x)
        inp_std, out_std, inp_mean, out_mean = self._layer_handler.step()
        y_targ = normalize_targets(y, y_targ, model_type=self.hparams.model_type)

        # compute data stats
        y_mean, y_std = y.mean(), y.std(unbiased=True)
        y_targ_mean, y_targ_std = y_targ.mean(), y_targ.std(unbiased=True)

        if self.hparams.log_data_stats:
            self.log('μ_x', x.mean(),              prog_bar=True)
            self.log('σ_x', x.std(unbiased=True),  prog_bar=True)
            self.log('μ_y', y_mean,                prog_bar=True)
            self.log('σ_y', y_std,                 prog_bar=True)
            self.log('μ_t', y_targ_mean,           prog_bar=True)
            self.log('σ_t', y_targ_std,            prog_bar=True)

        # compute target regularize loss
        loss_targ_mean, loss_targ_std = 0, 0
        if self.hparams.reg_model_output:
            loss_targ_mean  = F.mse_loss(y_mean, y_targ_mean)
            loss_targ_std   = F.mse_loss(y_std,  y_targ_std)
        loss_layer_mean = F.mse_loss(out_mean, self._target_mean)
        loss_layer_std  = F.mse_loss(out_std,  self._target_std)
        loss = loss_layer_mean + loss_layer_std + loss_targ_mean + loss_targ_std

        # update early stopping moving average
        self._early_stop_update(loss.item())

        # log everything
        self.log('l_σ', loss_layer_std,  prog_bar=True)
        self.log('l_μ',  loss_layer_mean, prog_bar=True)
        self.log('t_σ', loss_targ_std,   prog_bar=True)
        self.log('t_μ',  loss_targ_mean,  prog_bar=True)
        self.log('train_loss', loss)

        # log plots
        if self.hparams.log_wandb_period is not None:
            if self.trainer.global_step % self.hparams.log_wandb_period == 0:
                self._layer_handler.wandb_log_plots(inp_std, out_std, inp_mean, out_mean)

        return loss

    def configure_optimizers(self):
        return self.hparams.optimizer(
            self._model.parameters(),
            lr=self.hparams.learning_rate,
            **self.hparams.optimizer_kwargs,
        )


def has_kwarg(fn, name: str):
    import inspect
    params = inspect.signature(fn).parameters
    return name in params


# ===================================================================== #
# Models
# ===================================================================== #


def _make_linear_layers(in_shape, out_shape, hidden_sizes, ActType):
    # get sizes
    sizes = [int(np.prod(in_shape)), *hidden_sizes, int(np.prod(out_shape))]
    pairs = iter_pairs(sizes)
    # get linear layers
    layers = [m for inp, out in pairs for m in [nn.Linear(inp, out), ActType()]][:-1]
    # make layers
    return nn.Sequential(
        nn.Flatten(),
        *layers,
        nn.Unflatten(dim=-1, unflattened_size=out_shape)
    )


def make_model(name: str, Conv2dType=nn.Conv2d, ActType=nn.ReLU):
    # activations occur inplace
    if has_kwarg(ActType, 'inplace'):
        ActType = partial(ActType, inplace=True)
    # create the model
    if name == 'mnist_simple_fc_deep':
        return _make_linear_layers(in_shape=(1, 28, 28), out_shape=(10,), hidden_sizes=get_layer_sizes(128, 16, r=10), ActType=ActType)
    if name == 'mnist_simple_fc_deep_wide':
        return _make_linear_layers(in_shape=(1, 28, 28), out_shape=(10,), hidden_sizes=get_layer_sizes(256, 64, r=10), ActType=ActType)
    elif name == 'mnist_simple_fc':
        return _make_linear_layers(in_shape=(1, 28, 28), out_shape=(10,), hidden_sizes=get_layer_sizes(128, 16, r=2), ActType=ActType)
    elif name == 'mnist_simple_fc_wide':         # 850 K e:4=97.9%
        return _make_linear_layers(in_shape=(1, 28, 28), out_shape=(10,), hidden_sizes=get_layer_sizes(512, 128, r=2), ActType=ActType)
    # elif name == 'mnist_simple_ae_deep':
    #     return _make_linear_layers(in_shape=(1, 28, 28), out_shape=(1, 28, 28), hidden_sizes=get_ae_layer_sizes(128, 16, r=10), ActType=ActType)
    elif name == 'mnist_simple_conv':
        return nn.Sequential(
            Conv2dType(1,  32, kernel_size=3, padding=1), ActType(),
            Conv2dType(32, 16, kernel_size=3, padding=1), ActType(),
                nn.AvgPool2d(kernel_size=2),  # 28x28 -> 14x14
            Conv2dType(16, 32, kernel_size=3, padding=1), ActType(),
            Conv2dType(32,  8, kernel_size=3, padding=1), ActType(),
                nn.AvgPool2d(kernel_size=2),  # 14x14 -> 7x7
                nn.Flatten(),
            nn.Linear(7*7*8, 128), nn.Dropout(p=0.5), ActType(),
            nn.Linear(128, 10),
        )
    elif name == 'mnist_simple_conv_large':
        return nn.Sequential(
            Conv2dType(1,  32, kernel_size=3, padding=1), ActType(),
            Conv2dType(32, 64, kernel_size=3, padding=1), ActType(),
                nn.AvgPool2d(kernel_size=2),  # 28x28 -> 14x14
            Conv2dType(64, 64, kernel_size=3, padding=1), ActType(),
            Conv2dType(64, 96, kernel_size=3, padding=1), ActType(),
                nn.AvgPool2d(kernel_size=2),  # 14x14 -> 7x7
            Conv2dType(96, 128, kernel_size=3, padding=1), ActType(),
            Conv2dType(128, 96,  kernel_size=3, padding=1), ActType(),
                nn.AvgPool2d(kernel_size=2),  # 7x7 -> 3x3
                nn.Flatten(),
            nn.Linear(3*3*96, 256), nn.Dropout(p=0.5), ActType(),
            nn.Linear(256, 10),
        )
    elif name == 'mini_imagenet_simple_conv':
        return nn.Sequential(
            Conv2dType(3,  16, kernel_size=3, padding=1), ActType(),
            Conv2dType(16, 32, kernel_size=3, padding=1), ActType(),
                nn.AvgPool2d(kernel_size=3),  # 84x84 -> 28x28
            Conv2dType(32, 64, kernel_size=3, padding=1), ActType(),
            Conv2dType(64, 32, kernel_size=3, padding=1), ActType(),
                nn.AvgPool2d(kernel_size=2),  # 28x28 -> 14x14
            Conv2dType(32, 64, kernel_size=3, padding=1), ActType(),
            Conv2dType(64, 16, kernel_size=3, padding=1), ActType(),
                nn.AvgPool2d(kernel_size=2),  # 14x14 -> 7x7
                nn.Flatten(),
            nn.Linear(7*7*16, 384), nn.Dropout(p=0.5), ActType(),
            nn.Linear(384, 100),
        )
    else:
        raise KeyError(f'invalid model name: {repr(name)}')


def make_conv_model_and_reg_layers(name: str, Conv2dType=nn.Conv2d, ActType=nn.ReLU, init_mode: str = None, mean_shift_activations: bool = None, include_last: bool = False):
    # make model
    model = make_model(name, ActType=ActType, Conv2dType=Conv2dType)
    # relu needs mean_shift
    if (mean_shift_activations is None) and (ActType == nn.ReLU):
        mean_shift_activations = True
    # mean shift
    if mean_shift_activations:
        print('[mean shift]:', wrap_mean_shift(model))
    # initialise model weights
    if init_mode:
        init_weights(model, mode=init_mode)
    # get layers
    layers = find_submodules(model, (Conv2dType(1, 1, 1).__class__, nn.Linear), visit_instance_of=True)
    if not include_last:
        layers = layers[:-1]
    # return values
    return model, layers


# ===================================================================== #
# Main
# ===================================================================== #


def __main__(
    wandb_enabled = False,
    # model settings
    model: str = 'simple_fc_deep',
    model_ActType=Swish,
    model_Conv2dType=nn.Conv2d,
    model_init_mode: str = 'xavier_normal',
    # dataset
    dataset: str = 'mnist',
    # regularisation settings
    regularize: bool = True,
    reg_target_mean: MeanStdTargetHint = 0.0,
    reg_target_std: MeanStdTargetHint = 1.0,
    reg_with_noise=True,
    reg_model_output=True,
    # optimizers
    reg_lr=1e-3,
    reg_batch_size=128,
    train_lr=1e-3,
    train_batch_size=128,
    train_epochs=30,
    train_optimizer=Adam,
    # schedule
    train_scheduler=None,
    train_scheduler_kwargs=None,
):
    hparams = locals()

    model, layers = make_conv_model_and_reg_layers(
        name=f'{dataset}_{model}',
        ActType=model_ActType,
        Conv2dType=model_Conv2dType,
        init_mode=model_init_mode,
        include_last=not reg_model_output,
    )

    print(f'REG LAYERS: {len(layers)}')

    trainer = None
    if regularize:
        reg_data = make_image_data_module(
            dataset=f'noise_{dataset}' if reg_with_noise else f'{dataset}',
            batch_size=reg_batch_size,
            shift_mean_std=True,
            return_labels=True,
        )
        # regularize the network
        _, _, trainer = pl_quick_train(
            system=RegularizeLayerStatsTask(
                target_mean=reg_target_mean,
                target_std=reg_target_std,
                model=model,
                model_type='classification',
                layers=layers,
                optimizer=train_optimizer,
                log_wandb_period=500 if wandb_enabled else None,
                learning_rate=reg_lr,
                reg_model_output=reg_model_output,
                log_data_stats=False,
            ),
            data=reg_data,
            train_epochs=100,
            wandb_enabled=wandb_enabled,
            wandb_project='weights-test-2',
            hparams=hparams,
        )
        del reg_data, _

    # train the network
    _, _, _ = pl_quick_train(
        system=ClassificationTask(
            model,
            loss_fn=F.cross_entropy,
            optimizer=train_optimizer,
            learning_rate=train_lr,
            scheduler=train_scheduler,
            scheduler_kwargs=train_scheduler_kwargs,
        ),
        data=make_image_data_module(
            dataset=f'{dataset}',
            batch_size=train_batch_size,
            shift_mean_std=True,
            return_labels=True,
        ),
        train_epochs=train_epochs,
        wandb_enabled=wandb_enabled,
        wandb_project='weights-test-2',
        logger=trainer.logger if (trainer is not None) else None,
    )
    del _
    return model


if __name__ == '__main__':

    # 99% validation accuracy: Epoch 4

    gamma = compute_gamma(Swish())

    __main__(
        wandb_enabled=False,
        # regularisation settings
        # NOTE: no reg seems to be better unfortunately -- just use ScaledStdConv2d with xavier_normal init
        regularize=False,
        reg_target_std=1.0,  # SlopeArgs(a=0.999, y_0=0.1, y_n=1.0),
        reg_with_noise=True,
        reg_model_output=False,
        # model settings
        model='simple_conv_large',
        model_ActType=Swish,
        model_Conv2dType=lambda *args, **kwargs: ScaledStdConv2d(*args, **kwargs, gamma=gamma, use_layernorm=True),
        model_init_mode='xavier_normal',
        # training settings
        reg_lr=1e-3,
        train_lr=3e-4,
        # scheduler
        train_scheduler=torch.optim.lr_scheduler.MultiStepLR,
        train_scheduler_kwargs=dict(gamma=0.5, milestones=[4, 8], verbose=True),
    )
