import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import AttributeDict
from tqdm import tqdm


def pl_quick_train(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    hparams: dict = None,
    # train settings
    train_epochs: int = None,
    train_steps: int = None,
    train_callbacks = None,
    # progress bar settings
    progress_ncols: int = 120,
    # trainer settings
    wandb_enabled: bool = False,
    wandb_name: str = None,
    wandb_project: str = 'default',
):
    extra_hparams = AttributeDict(train_epochs=train_epochs, train_steps=train_steps)
    if hasattr(data, 'hparams'):  extra_hparams.update(data.hparams)
    if hasattr(model, 'hparams'): extra_hparams.update(model.hparams)
    if hparams:                   extra_hparams.update(hparams)
    # create the logger
    if wandb_enabled:
        logger = WandbLogger(name=wandb_name, project=wandb_project)
        logger.log_hyperparams(extra_hparams)
    else:
        logger = False
    # progress bar
    class WiderProgressBar(ProgressBar):
        def init_train_tqdm(self):
            return tqdm(
                desc='Train', initial=self.train_batch_idx, position=(2 * self.process_position),
                disable=self.is_disabled, leave=True, ncols=progress_ncols, file=sys.stdout,
                smoothing=0, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            )
    # initialise the trainer
    trainer = pl.Trainer(
        max_epochs=train_epochs,
        max_steps=train_steps,
        weights_summary='full',
        checkpoint_callback=False,
        callbacks=[WiderProgressBar(), *(train_callbacks if train_callbacks else [])],
        logger=logger,
        gpus=1 if torch.cuda.is_available() else 0,
    )
    # train the model
    trainer.fit(model, data)
    # return everything
    return model, data
