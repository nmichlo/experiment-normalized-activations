from argparse import Namespace
from functools import partial
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

import pandas as pd
import plotly.express as px
import pytorch_lightning as pl
import torch
import wandb
from flash.core.classification import ClassificationTask
from torch import nn
from torch.nn import functional as F
from torch_optimizer import RAdam

from exp.data import make_image_data_module
from exp.nn.swish import Swish
from exp.nn.weights import init_weights
from exp.util.nn_utils import find_submodules, replace_submodules
from exp.util.nn_utils import in_out_capture_context
from exp.util.nn_utils import replace_conv_and_bn
from exp.util.pl_utils import pl_quick_train


# ===================================================================== #
# Layer Handling
# ===================================================================== #


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
        target_mean: Union[float, torch.Tensor] = 0.,
        target_std: Union[float, torch.Tensor] = 1.,
        # early stopping
        early_stop_value: float = 0.001,
        early_stop_exp_weight: float = 0.98,
        # optimizer settings
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict = None,
        learning_rate: float = 1e-3,
        # logging settings
        log_wandb_period: int = None,
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
        if not isinstance(target_mean, torch.Tensor): target_mean = torch.full([len(layers)], fill_value=target_mean)
        if not isinstance(target_std, torch.Tensor): target_std = torch.full([len(layers)], fill_value=target_std)
        self.register_buffer('_target_mean', target_mean)
        self.register_buffer('_target_std', target_std)
        assert self._target_mean.shape == (len(layers),)
        assert self._target_std.shape == (len(layers),)
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
        # TODO: this is dataset specific, special handling modes can be added?
        x, y_targ = (x, x) if isinstance(x, torch.Tensor) else x

        # feed forward for layer hook
        y = self._model(x)
        inp_std, out_std, inp_mean, out_mean = self._layer_handler.step()
        y_targ = normalize_targets(y, y_targ, model_type=self.hparams.model_type)

        # compute target regularize loss
        loss_targ_mean  = F.mse_loss(y.mean(), y_targ.mean())
        loss_targ_std   = F.mse_loss(y.std(unbiased=True), y_targ.std(unbiased=True))
        loss_layer_mean = F.mse_loss(out_mean, self._target_mean)
        loss_layer_std  = F.mse_loss(out_std,  self._target_std)
        loss = loss_layer_mean + loss_layer_std + loss_targ_mean + loss_targ_std

        # update early stopping moving average
        self._early_stop_update(loss.item())

        # log everything
        self.log('t_mu',  loss_targ_mean,  prog_bar=True)
        self.log('t_std', loss_targ_std,   prog_bar=True)
        self.log('l_mu',  loss_layer_mean, prog_bar=True)
        self.log('l_std', loss_layer_std,  prog_bar=True)
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
# Main
# ===================================================================== #


def make_model(name: str, Conv2dType=nn.Conv2d, ActType=nn.ReLU):
    # activations occur inplace
    if has_kwarg(ActType, 'inplace'):
        ActType = partial(ActType, inplace=True)
    # create the model
    if name == 'mnist_simple_conv':
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
    layers = find_submodules(model, (Conv2dType, nn.Linear))
    if not include_last:
        layers = layers[:-1]
    # return values
    return model, layers


def __main__(
    ActType=nn.ReLU,
    Conv2dType=nn.Conv2d,
    init_mode: str = None,
    WANDB = False,
    train_epochs=30,
    model: str = 'simple_conv',
    dataset: str = 'mini_imagenet'
):
    hparams = locals()

    model, layers = make_conv_model_and_reg_layers(
        name=f'{dataset}_{model}',
        ActType=ActType,
        Conv2dType=Conv2dType,
        init_mode=init_mode,
    )

    # regularize the network
    _, _, trainer = pl_quick_train(
        system=RegularizeLayerStatsTask(
            model=model,
            model_type='classification',
            layers=layers,
            optimizer=RAdam,
            log_wandb_period=500 if WANDB else None,
            learning_rate=3e-3,
        ),
        data=make_image_data_module(
            dataset=f'noise_{dataset}',
            batch_size=128,
            normalise=True,
            return_labels=True,
        ),
        train_epochs=100,
        wandb_enabled=WANDB,
        wandb_project='weights-test-2',
        hparams=hparams,
    )

    # train the network
    _, _, _ = pl_quick_train(
        system=ClassificationTask(
            model,
            loss_fn=F.cross_entropy,
            optimizer=RAdam,
            learning_rate=1e-3,
        ),
        data=make_image_data_module(
            dataset=f'{dataset}',
            batch_size=128,
            normalise=True,
            return_labels=True,
        ),
        train_epochs=train_epochs,
        wandb_enabled=WANDB,
        wandb_project='weights-test-2',
        logger=trainer.logger,
    )

    return model


if __name__ == '__main__':
    __main__(WANDB=False)
