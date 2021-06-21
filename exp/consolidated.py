from argparse import Namespace
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

import pandas as pd
import plotly.express as px
import pytorch_lightning as pl
import torch
import types
import wandb
from flash.core.classification import ClassificationTask
from nfnets import ScaledStdConv2d
from torch import nn
from torch.nn import functional as F
from torch_optimizer import RAdam

from exp.data import make_image_data_module
from exp.nn.swish import Swish
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


def inject_mean_shift(model: nn.Module):
    """
    Normal layers sometimes struggle to normalise the activations
    if the mean is shifted.

    This function injects a mean_shift parameter onto the models
    TODO: this should probably create a wrapper nn.Module instead and use that.
    """
    def _modify(m: nn.Module, key, parent):
        assert not hasattr(m, 'mean_shift')
        assert not hasattr(m, 'old_forward')
        m.register_parameter('mean_shift', nn.Parameter(torch.zeros(1), requires_grad=True))
        m.old_forward = m.forward
        m.forward = types.MethodType(lambda self, x: self.old_forward(x) - self.mean_shift, m)
    return replace_submodules(model, (nn.Conv2d, nn.Linear), _modify)


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
        return y_targ
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
        early_stop_exp_weight: float = 0.975,
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


# ===================================================================== #
# Main
# ===================================================================== #


if __name__ == '__main__':

    # config
    WANDB = False

    # create the model
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        nn.AvgPool2d(kernel_size=2),

        nn.Conv2d(16, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 8, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        nn.AvgPool2d(kernel_size=2),

        nn.Flatten(),

        nn.Linear(7*7*8, 128),
        # nn.Dropout(p=0.5),
        nn.ReLU(inplace=True),
        nn.Linear(128, 10),
    )

    gamma = compute_gamma(Swish())
    # print(replace_conv_and_bn(model, lambda *args, **kwargs: ScaledStdConv2d(*args, **kwargs, gamma=gamma)))
    print(replace_conv_and_bn(model, ScaledStdConv2d))
    print(replace_submodules(model, nn.ReLU, Swish))
    # init_weights(model, mode=init_mode)

    CONV_TYPE = ScaledStdConv2d
    ACT_TYPE = Swish

    ClassificationTask

    # print(inject_mean_shift(model))

    # create the task
    # classifier = ClassificationTask(
    #     model,
    #     loss_fn=F.cross_entropy,
    #     optimizer=torch.optim.Adam,
    #     learning_rate=1e-3,
    # )

    layers = find_submodules(model, (CONV_TYPE, nn.Linear))[:-1]

    # make the data
    data = make_image_data_module(
        dataset='noise_mnist',
        batch_size=128,
        normalise=True,
        return_labels=True,
    )

    system = RegularizeLayerStatsTask(
        model=model,
        model_type='classification',
        layers=layers,
        optimizer=RAdam,
        log_wandb_period=500 if WANDB else None,
        learning_rate=1e-2,
    )

    # train the network
    pl_quick_train(
        system, data,
        train_epochs=10,
        wandb_enabled=WANDB,
        wandb_project='weights-test-2',
        train_kwargs=dict(val_check_interval=1),
    )
