from typing import Sequence

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from exp.data.mnist import MnistDataModule
from exp.nn.model import get_ae_layer_sizes
from exp.nn.model import make_normalized_model
from exp.util.utils import plot_batch


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


class SimpleSystem(pl.LightningModule):

    def __init__(
        self,
        model_sizes: Sequence[int] = (28 * 28, 128, 32, 128, 28 * 28),
        model_activation: str = 'tanh',
        model_init_mode: str = 'xavier_normal',
        # optimizer
        lr: float = 1e-3,
        # normalising
        norm_samples: int = 128,
        norm_sampler: str = 'normal',
        norm_steps: int = 200,
        norm_batch_size: int = 32,
        norm_lr=5e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        # init model
        self._model = nn.Sequential(
            nn.Flatten(start_dim=1),
            make_normalized_model(
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
                stats=False,
                log=True,
            ),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),
        )

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch, batch_idx):
        y = self(batch)
        return F.mse_loss(y, batch, reduction='mean')

    # def validation_step(self, batch, batch_idx):
    #     return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

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
        norm_samples: int = 128,
        norm_sampler: str = 'normal',
        norm_steps: int = 200,
        norm_batch_size: int = 32,
        norm_lr=5e-3,
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
        )
        # get the data
        data = MnistDataModule(batch_size=batch_size, normalise=True)
        # train the model
        trainer = pl.Trainer(
            max_epochs=epochs,
            weights_summary='full',
            checkpoint_callback=False,
            logger=False,
            gpus=1 if torch.cuda.is_available() else 0
        )
        trainer.fit(model, data)
        # get data and vis batch
        with torch.no_grad():
            if torch.cuda.is_available():
                model = model.cuda()
            img_batch = data.sample_batch(9).cpu()
            out_batch = model(img_batch.to(model.device)).cpu()
            # plot everything
            plot_batch(img_batch, out_batch, mean_std=data.norm_mean_std)
        # return everything
        return model, data


if __name__ == '__main__':
    SimpleSystem.quick_mnist_train(
        epochs=5,
        model_activation='norm_tanh',
    )






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
