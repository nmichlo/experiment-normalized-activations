import re
from typing import Optional
from typing import Sequence
from typing import Union

from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from exp.data.mnist import MnistDataModule
from exp.nn.model import get_ae_layer_sizes
from exp.nn.model import make_normalized_model
from exp.nn.model import norm_activations_analyse
from exp.util.utils import plot_batch


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #
from exp.util.utils import print_stats


class SimpleSystem(pl.LightningModule):

    def __init__(
        self,
        model_sizes: Sequence[int] = (28 * 28, 128, 32, 128, 28 * 28),
        model_activation: str = 'tanh',
        model_init_mode: str = 'xavier_normal',
        # optimizer
        lr: float = 1e-3,
        # normalising
        norm_samples: Optional[int] = None,
        norm_sampler: str = 'normal',
        norm_steps: int = 200,
        norm_batch_size: int = 32,
        norm_lr=5e-3,
        norm_targets_mean: Union[float, torch.Tensor] = 0.0,
        norm_targets_std: Union[float, torch.Tensor] = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        # compute params
        self.hparams.num_layers = len(self.hparams.model_sizes) - 1
        self.hparams.hidden_beg_size = self.hparams.model_sizes[1]
        self.hparams.hidden_mid_size = self.hparams.model_sizes[len(self.hparams.model_sizes)//2]
        self.hparams.hidden_end_size = self.hparams.model_sizes[-2]
        # make model
        self._base_model, _, self.title = make_normalized_model(
            sizes=model_sizes,
            activation=model_activation,
            init_mode=model_init_mode,
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
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),
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
        epochs: int = 100,
        batch_size: int = 128,
        # model settings
        model_hidden_sizes: Sequence[int] = (128, 32, 128),
        model_activation: str = 'tanh',
        model_init_mode: str = 'xavier_normal',
        # optimizer
        lr: float = 1e-3,
        # normalising
        norm_samples: Optional[int] = None,
        norm_sampler: str = 'normal',
        norm_steps: int = 500,
        norm_batch_size: int = 128,
        norm_lr=1e-2,
        norm_targets_mean: Union[float, torch.Tensor] = 0.0,
        norm_targets_std: Union[float, torch.Tensor] = 1.0,
        # wandb settings
        log_wandb: bool = False
    ):
        # make the model
        model = cls(
            model_sizes=[28*28, *model_hidden_sizes, 28*28],
            model_activation=model_activation,
            model_init_mode=model_init_mode,
            lr=lr,
            norm_samples=norm_samples,
            norm_sampler=norm_sampler,
            norm_steps=norm_steps,
            norm_batch_size=norm_batch_size,
            norm_lr=norm_lr,
            norm_targets_mean=norm_targets_mean,
            norm_targets_std=norm_targets_std,
        )
        # get the data
        data = MnistDataModule(batch_size=batch_size, normalise=True)
        # train the model
        trainer = pl.Trainer(
            max_epochs=epochs,
            weights_summary='full',
            checkpoint_callback=False,
            logger=WandbLogger(name=re.sub(r'[\s|]+', '_', model.title).strip('_'), project='weight-init') if log_wandb else False,
            gpus=1 if torch.cuda.is_available() else 0,
        )
        trainer.fit(model, data)
        # get data and vis batch
        with torch.no_grad():
            if torch.cuda.is_available():
                model = model.cuda()
            img_batch = data.sample_batch(9).cpu()
            out_batch = model(img_batch.to(model.device)).cpu()
            # plot everything
            plot_batch(img_batch, out_batch, mean_std=data.norm_mean_std, clip=True)
        # print final stats
        before_values, before_stats = norm_activations_analyse(model._base_model, obs_shape=[28*28], sampler=norm_sampler, steps=128, batch_size=32)
        print_stats(before_stats, title=f'{model.title} | TRN')
        # return everything
        return model, data


if __name__ == '__main__':

    # ===================================================================== #

    # BAD

    targs_mean = 0
    targs_std  = 1

    SimpleSystem.quick_mnist_train(
        model_hidden_sizes=get_ae_layer_sizes(start=128, mid=32, r=10),
        epochs=5,
        model_activation='swish',
        model_init_mode='xavier_normal',
        norm_sampler='normal',
        log_wandb=False,
        # targets
        norm_batch_size=128,
        norm_steps=1,
        norm_lr=5e-3,
        norm_targets_mean=targs_mean,
        norm_targets_std=targs_std,
    )

    # LOGS:
    # [ 32 784  11 | swish              | normal          | xavier_normal   | BFR] - μ: [▸█▆▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄◂] σ: [▸▃▂▁▁                 ◂] | μ: [ 0.325,  0.137,  0.042,  0.034,  0.015,  0.002,  0.002,  0.000, -0.001,  0.000,  0.000, -0.000,  0.000, -0.000,  0.000, -0.000, -0.000, -0.000, -0.000,  0.000,  0.000] σ: [ 0.757,  0.465,  0.261,  0.144,  0.082,  0.041,  0.021,  0.011,  0.006,  0.003,  0.002,  0.001,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000]
    # [ 32 784  11 | swish              | normal          | xavier_normal   | AFT] - μ: [▸█▆▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄◂] σ: [▸▃▂▁▁                 ◂] | μ: [ 0.324,  0.137,  0.042,  0.034,  0.015,  0.002,  0.002,  0.000, -0.001,  0.000,  0.000,  0.000,  0.000, -0.000,  0.000, -0.000, -0.000, -0.000, -0.000,  0.000,  0.000] σ: [ 0.755,  0.466,  0.262,  0.145,  0.082,  0.041,  0.021,  0.011,  0.006,  0.003,  0.002,  0.001,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000]
    # [ 32 784  11 | swish              | normal          | xavier_normal   | DIF] - μ: [▸▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄◂] σ: [▸▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄◂] | μ: [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000] σ: [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000]
    # Epoch 99: 100%|███████| 469/469 [00:13<00:00, 33.85it/s, loss=0.077, v_num=y08c]
    # [ 32 784  11 | swish              | normal          | xavier_normal   | TRN] - μ: [▸█▇▆▆▆▅▅▅▅▅▆▅▅▄▄▄▄▄▄▄▄◂] σ: [▸██▇▆▅▄▄▄▄▄▄▃▃▃▃▂▂▂▂▂▂◂] | μ: [ 1.349,  1.081,  0.764,  0.653,  0.512,  0.467,  0.388,  0.398,  0.361,  0.432,  0.519,  0.270,  0.177,  0.082,  0.084,  0.060,  0.052,  0.031,  0.037,  0.036, -0.090] σ: [ 2.425,  2.486,  1.942,  1.603,  1.296,  1.144,  1.040,  0.995,  1.013,  0.956,  0.969,  0.686,  0.541,  0.449,  0.448,  0.442,  0.431,  0.436,  0.415,  0.424,  0.264]

    # ===================================================================== #

    # SETTINGS FROM TANH (even applied to swish) WORK REALLY WELL!

    # TANH : BEFORE TRAIN - custom
    targs_mean = [-0.001, -0.001, -0.001,  0.000, -0.001, -0.000,  0.001,  0.000, -0.000, -0.000, -0.000, -0.001, -0.000, -0.000,  0.000, -0.000, -0.000, -0.000, -0.000,  0.000,  0.000]
    targs_std  = [ 0.627,  0.486,  0.403,  0.355,  0.314,  0.291,  0.272,  0.259,  0.240,  0.233,  0.218,  0.205,  0.196,  0.182,  0.184,  0.180,  0.172,  0.169,  0.164,  0.155,  0.153]

    # reverse -- large at end -- works well
    targs_mean = targs_mean[::-1]
    targs_std  = targs_std[::-1]

    # to tensor
    targs_mean = torch.as_tensor(targs_mean)
    targs_std = torch.as_tensor(targs_std)

    # adjust -- works well (but not needed)
    targs_mean = targs_mean ** 2
    targs_std = targs_std ** 2

    SimpleSystem.quick_mnist_train(
        model_hidden_sizes=get_ae_layer_sizes(start=128, mid=32, r=10),
        epochs=5,
        model_activation='norm_swish',
        model_init_mode='xavier_normal',
        norm_sampler='normal',
        log_wandb=False,
        # targets
        norm_batch_size=128,
        norm_steps=1500,
        norm_lr=5e-3,
        norm_targets_mean=targs_mean,
        norm_targets_std=targs_std,
    )

    # LOGS:
    # normalising model: 100%|██████████| 1500/1500 [00:20<00:00, 72.33it/s, loss=0.00249]
    # [ 32 784  11 | norm_swish         | normal          | xavier_normal   | BFR] - μ: [▸█▆▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄◂] σ: [▸▃▂▁▁                 ◂] | μ: [ 0.325,  0.169,  0.075,  0.029,  0.002,  0.001,  0.003,  0.001, -0.000,  0.000, -0.000,  0.000,  0.000, -0.000,  0.000, -0.000, -0.000,  0.000,  0.000, -0.000, -0.000] σ: [ 0.755,  0.470,  0.282,  0.155,  0.079,  0.040,  0.021,  0.010,  0.005,  0.003,  0.001,  0.001,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000]
    # [ 32 784  11 | norm_swish         | normal          | xavier_normal   | AFT] - μ: [▸▄▄▄▅▃▄▄▄▃▅▄▄▄▄▄▄▄▄▄▄▄◂] σ: [▸▁                 ▁▁▂◂] | μ: [-0.000,  0.004, -0.000,  0.008, -0.018, -0.001,  0.006, -0.005, -0.014,  0.007,  0.002, -0.005, -0.006, -0.001, -0.003, -0.003,  0.002,  0.002, -0.001,  0.002,  0.004] σ: [ 0.242,  0.062,  0.026,  0.028,  0.017,  0.033,  0.035,  0.037,  0.040,  0.042,  0.052,  0.057,  0.060,  0.070,  0.081,  0.090,  0.106,  0.136,  0.175,  0.255,  0.427]
    # [ 32 784  11 | norm_swish         | normal          | xavier_normal   | DIF] - μ: [▸ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄◂] σ: [▸ ▂▄▅▄▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅◂] | μ: [-0.325, -0.006, -0.001,  0.004, -0.017,  0.001,  0.002, -0.004, -0.004,  0.007,  0.003, -0.003, -0.003, -0.005, -0.007,  0.004,  0.005,  0.002, -0.008, -0.005,  0.002] σ: [-2.118, -1.011, -0.245,  0.538,  0.093,  0.600,  0.521,  0.503,  0.482,  0.474,  0.500,  0.596,  0.588,  0.612,  0.564,  0.535,  0.554,  0.620,  0.636,  0.662,  0.707]
    # Epoch 99: 100%|██████| 469/469 [00:14<00:00, 32.09it/s, loss=0.0341, v_num=yo9x]
    # [ 32 784  11 | norm_swish         | normal          | xavier_normal   | TRN] - μ: [▸▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▂ ◂] σ: [▸▂                ▁▁▁▂◂] | μ: [ 0.113,  0.036, -0.009, -0.016, -0.012, -0.021, -0.015, -0.006, -0.008, -0.001,  0.004, -0.003, -0.008, -0.009, -0.014, -0.021, -0.025, -0.025, -0.042, -0.129, -0.341] σ: [ 0.425,  0.149,  0.079,  0.115,  0.080,  0.109,  0.113,  0.110,  0.100,  0.089,  0.082,  0.080,  0.086,  0.107,  0.125,  0.137,  0.153,  0.194,  0.274,  0.390,  0.541]

    # ===================================================================== #

    # # TANH : BEFORE TRAIN - custom
    # targs_mean = [-0.001, -0.001, -0.001,  0.000, -0.001, -0.000,  0.001,  0.000, -0.000, -0.000, -0.000, -0.001, -0.000, -0.000,  0.000, -0.000, -0.000, -0.000, -0.000,  0.000,  0.000]
    # targs_std  = [ 0.627,  0.486,  0.403,  0.355,  0.314,  0.291,  0.272,  0.259,  0.240,  0.233,  0.218,  0.205,  0.196,  0.182,  0.184,  0.180,  0.172,  0.169,  0.164,  0.155,  0.153]
    #
    # # reverse -- large at end -- works well
    # targs_mean = targs_mean[::-1]
    # targs_std  = targs_std[::-1]
    #
    # # reverse half
    # # targs_mean = targs_mean[0::2] + targs_mean[1::2][::-1]
    # # targs_std  = targs_std[0::2]  + targs_std[1::2][::-1]
    #
    # # swap half
    # # targs_mean = targs_mean[0::2] + targs_mean[1::2]
    # # targs_std  = targs_std[0::2]  + targs_std[1::2]
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
    #     model_hidden_sizes=get_ae_layer_sizes(start=128, mid=32, r=10),
    #     epochs=10,
    #     model_activation='norm_tanh',
    #     model_init_mode='custom',
    #     norm_sampler='normal',
    #     log_wandb=True,
    #     # targets
    #     norm_steps=1500,
    #     norm_lr=5e-3,
    #     norm_targets_mean=targs_mean,
    #     norm_targets_std=targs_std,
    # )
    #
    # # TANH : BEFORE TRAIN - custom
    # # μ: [-0.001, -0.001, -0.001,  0.000, -0.001, -0.000,  0.001,  0.000, -0.000, -0.000, -0.000, -0.001, -0.000, -0.000,  0.000, -0.000, -0.000, -0.000, -0.000,  0.000,  0.000]
    # # σ: [ 0.627,  0.486,  0.403,  0.355,  0.314,  0.291,  0.272,  0.259,  0.240,  0.233,  0.218,  0.205,  0.196,  0.182,  0.184,  0.180,  0.172,  0.169,  0.164,  0.155,  0.153]
    #
    # # TANH : AFTER TRAIN - custom
    # # μ: [ 0.001, -0.001,  0.001,  0.001,  0.001, -0.002, -0.002,  0.001, -0.003, -0.008, -0.001, -0.000,  0.001, -0.018,  0.029, -0.014,  0.008, -0.008, -0.032,  0.013,  0.032]
    # # σ: [ 0.647,  0.513,  0.450,  0.419,  0.390,  0.380,  0.363,  0.346,  0.332,  0.324,  0.327,  0.312,  0.315,  0.328,  0.377,  0.438,  0.500,  0.547,  0.557,  0.554,  0.500]
    #
    # # NORMTANH : AFTER TRAIN - custom
    # # μ: [-0.015, -0.006, -0.007,  0.003, -0.004,  0.001,  0.000, -0.001, -0.008,  0.015, -0.011, -0.017, -0.009,  0.003,  0.006,  0.005,  0.000, -0.001, -0.005, -0.002,  0.018]
    # # σ: [ 1.011,  1.016,  1.031,  1.058,  1.065,  1.086,  1.105,  1.130,  1.152,  1.149,  1.107,  1.084,  1.080,  1.087,  1.098,  1.149,  1.149,  1.183,  1.227,  1.205,  0.924]



# custom
# kaiming_normal
# xavier_normal





# if __name__ == '__main__':
#
#     for activation in [
#         None,
#         torch.sigmoid,
#         norm_sigmoid,
#         F.relu,
#         norm_relu,
#         torch.tanh,
#         norm_tanh,
#         swish,
#         norm_swish,
#     ]:
#         print('=' * 100)
#         print(getattr(activation, '__name__', activation))
#         print('=' * 100)
#         a, b = 64, 8
#         sample_activations(get_ae_layer_sizes(a, b, 5), init_mode='default', activation=activation)
#         sample_activations(get_ae_layer_sizes(a, b, 5), init_mode='default', activation=activation)
#         sample_activations(get_ae_layer_sizes(a, b, 5), init_mode='custom', activation=activation)
#         sample_activations(get_ae_layer_sizes(a, b, 5), init_mode='custom', activation=activation)
#         sample_activations(get_ae_layer_sizes(a, b, 5), init_mode='xavier_normal', activation=activation)
#         sample_activations(get_ae_layer_sizes(a, b, 5), init_mode='xavier_normal', activation=activation)
#
#
# if __name__ == '__main__':
#
#     sizes = get_ae_layer_sizes(128, 64, 10)
#     model = make_sequential_nn(sizes, 'norm_sigmoid')
#     loss_hist, (means, stds) = norm_activations_optimize(
#         model, obs_shape=[sizes[0]], batch_size=256, lr=1e-2, steps=99, sampler='normal'
#     )
#
#     plt.plot(loss_hist)
#     plt.show()
#
#     plt.plot(means)
#     plt.plot(stds)
#     plt.show()
#
#
# if __name__ == '__main__':
#
#     for img_batch in MnistDataModule(batch_size=9, shuffle=False, num_workers=0).setup().val_dataloader():
#         break
#
#
# if __name__ == '__main__':
#
#     plot_batch(img_batch)
#
#
# if __name__ == '__main__':
#
#     make_normalized_model(get_ae_layer_sizes(128, 64, 10), 'tanh');
#     make_normalized_model(get_ae_layer_sizes(128, 64, 10), 'norm_tanh');
#     make_normalized_model(get_ae_layer_sizes(128, 64, 50), 'norm_tanh');

#
# if __name__ == '__main__':
#
#     run('tanh', 'default', get_ae_layer_sizes(128, 32, r=10), batch_size=128, lr=1e-3, epochs=3, loss='mse');
#     run('tanh', 'custom', get_ae_layer_sizes(128, 32, r=10), batch_size=64, lr=3e-3, epochs=1, loss='mse');
#     run('norm_tanh', 'custom', get_ae_layer_sizes(128, 32, r=10), batch_size=64, lr=3e-3, epochs=1, loss='mse');
#
#
#     # QUESTION: why is norm + default init so good? I thought custom/xavier_normal would have been good?
#
#     for init in ['default', 'custom', 'xavier_normal']:
#         for act in ['tanh', 'norm_tanh', 'swish', 'norm_swish']:
#             run(act, init, get_ae_layer_sizes(128, 32, r=10), batch_size=64, lr=1e-3, epochs=2, loss='mse');
#
#     # QUESTION: why is norm + default init so good? I thought custom/xavier_normal would have been good?
#
#     for init in ['default', 'custom', 'xavier_uniform', 'xavier_normal']:
#         for act in ['norm_swish', 'swish']:
#             run(act, 'none', init, hidden_sizes=get_ae_layer_sizes(128, 32, r=10), batch_size=128, lr=1e-3, epochs=5, loss='mse');
