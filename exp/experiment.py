import dataclasses
import re
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F

from exp.data import make_image_data_module
from exp.nn.model import get_ae_layer_sizes
from exp.nn.model import make_normalized_model
from exp.nn.model import norm_activations_analyse
from exp.util.utils import plot_batch
from exp.util.utils import print_stats


# ========================================================================= #
# Standard Deviation Slope                                                  #
# ========================================================================= #


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


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


class SimpleSystem(pl.LightningModule):

    def __init__(
        self,
        model_hidden_sizes: Sequence[int] = (128, 32, 128),
        model_activation: str = 'tanh',
        model_init_mode: str = 'xavier_normal',
        model_batch_norm: bool = False,
        model_obs_shape: Tuple[int, int, int] = (1, 28, 28),
        # optimizer
        lr: float = 1e-3,
        # normalising
        norm_samples: Optional[int] = None,
        norm_sampler: str = 'normal',
        norm_steps: int = 2500,
        norm_batch_size: int = 256,
        norm_lr=1e-2,
        norm_targets_mean: Union[float, torch.Tensor] = 0.0,
        norm_targets_std: Union[float, torch.Tensor] = 1.0,
        # extra hparams
        **extra_hparams,
    ):
        super().__init__()
        self.save_hyperparameters()
        # compute params
        self.hparams.num_layers = len(self.hparams.model_hidden_sizes) + 1
        self.hparams.hidden_beg_size = self.hparams.model_hidden_sizes[0]
        self.hparams.hidden_mid_size = self.hparams.model_hidden_sizes[len(self.hparams.model_hidden_sizes)//2]
        self.hparams.hidden_end_size = self.hparams.model_hidden_sizes[-1]
        # make model
        self._base_model, _, self.title = make_normalized_model(
            model_sizes=[np.prod(model_obs_shape), *model_hidden_sizes, np.prod(model_obs_shape)],
            model_activation=model_activation,
            model_init_mode=model_init_mode,
            model_batch_norm=model_batch_norm,
            test_steps=128,
            test_batch_size=32,
            norm_samples=norm_samples,
            norm_sampler=norm_sampler,
            norm_steps=norm_steps,
            norm_batch_size=norm_batch_size,
            norm_lr=norm_lr,
            norm_targets_mean=norm_targets_mean,
            norm_targets_std=norm_targets_std,
            stats=True,
            log=True,
        )
        # init model
        self._model = nn.Sequential(
            nn.Flatten(start_dim=1),
            self._base_model,
            nn.Unflatten(dim=1, unflattened_size=model_obs_shape),
        )

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch, batch_idx):
        y = self(batch)
        loss = F.mse_loss(y, batch, reduction='mean')
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self._model.parameters(), lr=self.hparams.lr)

    @classmethod
    def quick_mnist_train(
        cls,
        # trainer
        dataset: str = 'mnist',
        epochs: int = 100,
        batch_size: int = 128,
        # model settings
        model_hidden_sizes: Sequence[int] = (128, 32, 128),
        model_activation: str = 'tanh',
        model_init_mode: str = 'xavier_normal',
        model_batch_norm: bool = False,
        # optimizer
        lr: float = 1e-3,
        # normalising
        norm_samples: Optional[int] = None,
        norm_sampler: str = 'normal',
        norm_steps: int = 2500,
        norm_batch_size: int = 256,
        norm_lr=1e-2,
        norm_targets_mean: Union[float, torch.Tensor] = 0.0,
        norm_targets_std: Union[SlopeArgs, float, torch.Tensor] = 1.0,
        # wandb settings
        log_wandb: bool = False,
        wandb_name: str = '',
        wandb_tags: Optional[Sequence[str]] = None,
        wandb_project: str = 'weight-init',
    ):
        extra_hparams = dict(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
        )

        # normalise parameters & extend hparams if needed
        if isinstance(norm_targets_std, SlopeArgs):
            extra_hparams.update(dict(
                targ_std_slope=norm_targets_std.a,
                targ_std_y_0=norm_targets_std.y_0,
                targ_std_y_n=norm_targets_std.y_n,
            ))
            norm_targets_std = std_slope(
                n=len(model_hidden_sizes),
                a=norm_targets_std.a,
                y_0=norm_targets_std.y_0,
                y_n=norm_targets_std.y_n,
            )

        # get the data
        data = make_image_data_module(dataset=dataset, batch_size=batch_size, normalise=True)

        # make the model
        model = cls(
            model_hidden_sizes=model_hidden_sizes,
            model_activation=model_activation,
            model_init_mode=model_init_mode,
            model_batch_norm=model_batch_norm,
            model_obs_shape=data.obs_shape,
            lr=lr,
            norm_samples=norm_samples,
            norm_sampler=norm_sampler,
            norm_steps=norm_steps,
            norm_batch_size=norm_batch_size,
            norm_lr=norm_lr,
            norm_targets_mean=norm_targets_mean,
            norm_targets_std=norm_targets_std,
            # log extra hparams
            **extra_hparams,
        )

        # train the model
        trainer = pl.Trainer(
            max_epochs=epochs,
            weights_summary='full',
            checkpoint_callback=False,
            logger=False if (not log_wandb) else WandbLogger(
                name=(wandb_name + re.sub(r'[\s|]+', '_', model.title).strip('_')).lower(),
                project=wandb_project,
                tags=wandb_tags
            ),
            gpus=1 if torch.cuda.is_available() else 0,
        )
        trainer.fit(model, data)

        # get data and vis batch
        with torch.no_grad():
            if torch.cuda.is_available():
                model = model.cuda()
            img_batch = data.sample_display_batch(9).cpu()
            out_batch = model(img_batch.to(model.device)).cpu()
            plot_batch(img_batch, out_batch, mean_std=data.norm_mean_std, clip=True)

        # print final stats
        train_values, train_stats = norm_activations_analyse(model._base_model, obs_shape=[28 * 28], sampler=norm_sampler, steps=128, batch_size=32)
        print_stats(train_stats,  title=f'{model.title} | TRN STAT')
        print_stats(train_values, title=f'{model.title} | TRN VALS')

        # FINISHED
        if log_wandb:
            wandb.finish()

        # return everything
        return model, data


if __name__ == '__main__':

    # ===================================================================== #
    # SETTINGS
    # ===================================================================== #

    _SHARED_KWARGS = dict(
        model_hidden_sizes=get_ae_layer_sizes(start=128, mid=32, r=10),
        epochs=15,
        log_wandb=True,
        wandb_name='TEMP_',
        wandb_tags=[],
        dataset='mnist',
    )

    # ===================================================================== #
    # 4. BASE RUNS
    # ===================================================================== #

    for activation in [
        'swish',
        'tanh',
        'relu',
    ]:
        for init_mode in [
            'xavier_normal',
            'kaiming_normal',
            'xavier_uniform',
            'kaiming_uniform',
        ]:
            SimpleSystem.quick_mnist_train(
                model_activation=activation,
                model_init_mode=init_mode,
                # targets
                norm_targets_mean=0,
                norm_targets_std=1,
                # shared
                **{
                    **_SHARED_KWARGS,
                    **dict(
                        epochs=30,
                        wandb_name=f'base:_',
                        wandb_tags=('BASE',)
                    ),
                },
            )

    # ===================================================================== #
    # 3. BEST REPEATS
    # ===================================================================== #

    for i in range(3):
        for s in [
            SlopeArgs(a=0.9,   y_0=0.01, y_n=0.25),  # 1.
            SlopeArgs(a=0.999, y_0=0.01, y_n=0.5),   # 2.
            SlopeArgs(a=-0.9,  y_0=0.01, y_n=0.25),  # 3.
            SlopeArgs(a=-0.5,  y_0=0.01, y_n=0.25),  # 4.
        ]:
            SimpleSystem.quick_mnist_train(
                model_activation='norm_swish',
                model_init_mode='xavier_normal',
                # targets
                norm_targets_mean=0,
                norm_targets_std=s,
                # shared
                **{
                    **_SHARED_KWARGS,
                    **dict(
                        epochs=30,
                        wandb_name=f'rerun:slope={s.a}:beg={s.y_0}:end={s.y_n}_',
                        wandb_tags=('RERUN',)
                    ),
                },
            )

    # # ===================================================================== #
    # # 2. SEARCH
    # # ===================================================================== #
    #
    # for y_n in [0.99, 0.75, 0.5, 0.25, 0.01]:                # should add 0.1
    #     for y_0 in [0.01, 0.25, 0.5, 0.75, 0.99]:            # should add 0.1
    #         for a in [0.999, 0.9, 0.5, -0.5, -0.9, -0.999]:  # should add 0.25, 0.1, -0.1, -0.25
    #             try:
    #                 SimpleSystem.quick_mnist_train(
    #                     model_activation='norm_swish',
    #                     model_init_mode='xavier_normal',
    #                     # targets
    #                     norm_targets_mean=0,
    #                     norm_targets_std=SlopeArgs(a=a, y_0=y_0, y_n=y_n),
    #                     # shared
    #                     **{**_SHARED_KWARGS, 'wandb_name': f'T:slope={a}:beg={y_0}:end={y_n}_'},
    #                 )
    #             except:
    #                 print('Failed')
    #             try:
    #                 wandb.finish()
    #             except:
    #                 pass
    #
    # # ===================================================================== #
    # # 1. TANH - BASE
    # # ===================================================================== #
    #
    # SimpleSystem.quick_mnist_train(
    #     model_activation='tanh',
    #     model_init_mode='custom',
    #     # targets
    #     norm_targets_mean=0,
    #     norm_targets_std=1,
    #     # shared
    #     **_SHARED_KWARGS,
    # )
    #
    # TANH : BEFORE TRAIN - custom
    # μ: [-0.001, -0.001, -0.001,  0.000, -0.001, -0.000,  0.001,  0.000, -0.000, -0.000, -0.000, -0.001, -0.000, -0.000,  0.000, -0.000, -0.000, -0.000, -0.000,  0.000,  0.000]
    # σ: [ 0.627,  0.486,  0.403,  0.355,  0.314,  0.291,  0.272,  0.259,  0.240,  0.233,  0.218,  0.205,  0.196,  0.182,  0.184,  0.180,  0.172,  0.169,  0.164,  0.155,  0.153]
    #
    # TANH : AFTER TRAIN - custom
    # μ: [ 0.001, -0.001,  0.001,  0.001,  0.001, -0.002, -0.002,  0.001, -0.003, -0.008, -0.001, -0.000,  0.001, -0.018,  0.029, -0.014,  0.008, -0.008, -0.032,  0.013,  0.032]
    # σ: [ 0.647,  0.513,  0.450,  0.419,  0.390,  0.380,  0.363,  0.346,  0.332,  0.324,  0.327,  0.312,  0.315,  0.328,  0.377,  0.438,  0.500,  0.547,  0.557,  0.554,  0.500]
    #
    # NORMTANH : AFTER TRAIN - custom
    # μ: [-0.015, -0.006, -0.007,  0.003, -0.004,  0.001,  0.000, -0.001, -0.008,  0.015, -0.011, -0.017, -0.009,  0.003,  0.006,  0.005,  0.000, -0.001, -0.005, -0.002,  0.018]
    # σ: [ 1.011,  1.016,  1.031,  1.058,  1.065,  1.086,  1.105,  1.130,  1.152,  1.149,  1.107,  1.084,  1.080,  1.087,  1.098,  1.149,  1.149,  1.183,  1.227,  1.205,  0.924]
    #
    # # ===================================================================== #
    # # 1. SETTINGS FROM TANH FOR SWISH (even applied to swish) WORK WELL!
    # # ===================================================================== #
    #
    # # TANH : BEFORE TRAIN - custom
    # targs_mean = [-0.001, -0.001, -0.001,  0.000, -0.001, -0.000,  0.001,  0.000, -0.000, -0.000, -0.000, -0.001, -0.000, -0.000,  0.000, -0.000, -0.000, -0.000, -0.000,  0.000,  0.000]
    # targs_std  = [ 0.627,  0.486,  0.403,  0.355,  0.314,  0.291,  0.272,  0.259,  0.240,  0.233,  0.218,  0.205,  0.196,  0.182,  0.184,  0.180,  0.172,  0.169,  0.164,  0.155,  0.153]
    #
    # # reverse -- large at end -- works well
    # targs_mean = targs_mean[::-1]
    # targs_std  = targs_std[::-1]
    #
    # # to tensor
    # targs_mean = torch.as_tensor(targs_mean)
    # targs_std = torch.as_tensor(targs_std)
    #
    # # adjust -- works well (but not needed)
    # # targs_mean = targs_mean ** 2
    # # targs_std = targs_std ** 2
    #
    # SimpleSystem.quick_mnist_train(
    #     model_activation='norm_swish',
    #     model_init_mode='xavier_normal',
    #     # targets
    #     norm_targets_mean=targs_mean,
    #     norm_targets_std=targs_std,
    #     # shared
    #     **_SHARED_KWARGS,
    # )
    #
    # # LOGS:
    # # normalising model: 100%|██████████| 1500/1500 [00:20<00:00, 72.33it/s, loss=0.00249]
    # # [ 32 784  11 | norm_swish         | normal          | xavier_normal   | BFR] - μ: [▸█▆▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄◂] σ: [▸▃▂▁▁                 ◂] | μ: [ 0.325,  0.169,  0.075,  0.029,  0.002,  0.001,  0.003,  0.001, -0.000,  0.000, -0.000,  0.000,  0.000, -0.000,  0.000, -0.000, -0.000,  0.000,  0.000, -0.000, -0.000] σ: [ 0.755,  0.470,  0.282,  0.155,  0.079,  0.040,  0.021,  0.010,  0.005,  0.003,  0.001,  0.001,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000]
    # # [ 32 784  11 | norm_swish         | normal          | xavier_normal   | AFT] - μ: [▸▄▄▄▅▃▄▄▄▃▅▄▄▄▄▄▄▄▄▄▄▄◂] σ: [▸▁                 ▁▁▂◂] | μ: [-0.000,  0.004, -0.000,  0.008, -0.018, -0.001,  0.006, -0.005, -0.014,  0.007,  0.002, -0.005, -0.006, -0.001, -0.003, -0.003,  0.002,  0.002, -0.001,  0.002,  0.004] σ: [ 0.242,  0.062,  0.026,  0.028,  0.017,  0.033,  0.035,  0.037,  0.040,  0.042,  0.052,  0.057,  0.060,  0.070,  0.081,  0.090,  0.106,  0.136,  0.175,  0.255,  0.427]
    # # [ 32 784  11 | norm_swish         | normal          | xavier_normal   | DIF] - μ: [▸ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄◂] σ: [▸ ▂▄▅▄▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅◂] | μ: [-0.325, -0.006, -0.001,  0.004, -0.017,  0.001,  0.002, -0.004, -0.004,  0.007,  0.003, -0.003, -0.003, -0.005, -0.007,  0.004,  0.005,  0.002, -0.008, -0.005,  0.002] σ: [-2.118, -1.011, -0.245,  0.538,  0.093,  0.600,  0.521,  0.503,  0.482,  0.474,  0.500,  0.596,  0.588,  0.612,  0.564,  0.535,  0.554,  0.620,  0.636,  0.662,  0.707]
    # # Epoch 99: 100%|██████| 469/469 [00:14<00:00, 32.09it/s, loss=0.0341, v_num=yo9x]
    # # [ 32 784  11 | norm_swish         | normal          | xavier_normal   | TRN] - μ: [▸▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▂ ◂] σ: [▸▂                ▁▁▁▂◂] | μ: [ 0.113,  0.036, -0.009, -0.016, -0.012, -0.021, -0.015, -0.006, -0.008, -0.001,  0.004, -0.003, -0.008, -0.009, -0.014, -0.021, -0.025, -0.025, -0.042, -0.129, -0.341] σ: [ 0.425,  0.149,  0.079,  0.115,  0.080,  0.109,  0.113,  0.110,  0.100,  0.089,  0.082,  0.080,  0.086,  0.107,  0.125,  0.137,  0.153,  0.194,  0.274,  0.390,  0.541]
    #
    # # ===================================================================== #
    # # 1. SWISH - REFERENCE
    # # ===================================================================== #
    #
    # SimpleSystem.quick_mnist_train(
    #     model_activation='swish',
    #     model_init_mode='xavier_normal',
    #     # targets
    #     norm_steps=1,
    #     norm_targets_mean=0,
    #     norm_targets_std=1,
    #     # shared
    #     **_SHARED_KWARGS,
    # )
    #
    # # LOGS:
    # # [ 32 784  11 | swish              | normal          | xavier_normal   | BFR] - μ: [▸█▆▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄◂] σ: [▸▃▂▁▁                 ◂] | μ: [ 0.325,  0.137,  0.042,  0.034,  0.015,  0.002,  0.002,  0.000, -0.001,  0.000,  0.000, -0.000,  0.000, -0.000,  0.000, -0.000, -0.000, -0.000, -0.000,  0.000,  0.000] σ: [ 0.757,  0.465,  0.261,  0.144,  0.082,  0.041,  0.021,  0.011,  0.006,  0.003,  0.002,  0.001,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000]
    # # [ 32 784  11 | swish              | normal          | xavier_normal   | AFT] - μ: [▸█▆▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄◂] σ: [▸▃▂▁▁                 ◂] | μ: [ 0.324,  0.137,  0.042,  0.034,  0.015,  0.002,  0.002,  0.000, -0.001,  0.000,  0.000,  0.000,  0.000, -0.000,  0.000, -0.000, -0.000, -0.000, -0.000,  0.000,  0.000] σ: [ 0.755,  0.466,  0.262,  0.145,  0.082,  0.041,  0.021,  0.011,  0.006,  0.003,  0.002,  0.001,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000]
    # # [ 32 784  11 | swish              | normal          | xavier_normal   | DIF] - μ: [▸▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄◂] σ: [▸▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄◂] | μ: [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000] σ: [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000]
    # # Epoch 99: 100%|███████| 469/469 [00:13<00:00, 33.85it/s, loss=0.077, v_num=y08c]
    # # [ 32 784  11 | swish              | normal          | xavier_normal   | TRN] - μ: [▸█▇▆▆▆▅▅▅▅▅▆▅▅▄▄▄▄▄▄▄▄◂] σ: [▸██▇▆▅▄▄▄▄▄▄▃▃▃▃▂▂▂▂▂▂◂] | μ: [ 1.349,  1.081,  0.764,  0.653,  0.512,  0.467,  0.388,  0.398,  0.361,  0.432,  0.519,  0.270,  0.177,  0.082,  0.084,  0.060,  0.052,  0.031,  0.037,  0.036, -0.090] σ: [ 2.425,  2.486,  1.942,  1.603,  1.296,  1.144,  1.040,  0.995,  1.013,  0.956,  0.969,  0.686,  0.541,  0.449,  0.448,  0.442,  0.431,  0.436,  0.415,  0.424,  0.264]
