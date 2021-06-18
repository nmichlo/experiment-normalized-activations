import os
from typing import Sequence
from typing import Tuple

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader


class MNIST(torchvision.datasets.MNIST, Sequence):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img


class MnistDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 128, num_workers=os.cpu_count(), shuffle=True, normalise=False):
        super().__init__()
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        # normalize settings
        self._normalise = normalise
        self._mean_std = (0.5, 0.5) if self._normalise else (0., 1.)

    @property
    def norm_mean_std(self) -> Tuple[float, float]:
        return self._mean_std

    def setup(self, stage=None):
        if self._normalise:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(*self.norm_mean_std)
            ])
        else:
            transform = torchvision.transforms.ToTensor()
        # get datasets
        self._data_trn = MNIST(root='data', transform=transform, train=True, download=True)
        self._data_val = MNIST(root='data', transform=transform, train=False, download=False)

    def train_dataloader(self):
        return DataLoader(dataset=self._data_trn, batch_size=self._batch_size, shuffle=self._shuffle)

    def val_dataloader(self):
        return DataLoader(dataset=self._data_val, batch_size=self._batch_size, shuffle=False)

    def sample_batch(self, n=9, val=True, shuffle=False) -> torch.Tensor:
        return next(iter(DataLoader(self._data_val if val else self._data_trn, num_workers=0, batch_size=n, shuffle=shuffle)))
