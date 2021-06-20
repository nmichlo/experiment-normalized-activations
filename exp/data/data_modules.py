import itertools
import os
import pickle
import warnings
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import random_split
from torch.utils.data.dataset import T_co
from torchvision.datasets import MNIST


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #
from exp.nn.activation import get_sampler


def split_sizes(length, n):
    k, m = divmod(length, n)
    return k + (np.arange(n) < m)


def split_bounds(length, n):
    x1 = np.cumsum(split_sizes(length, n))
    x0 = np.concatenate([[0], x1[:-1]])
    return list(zip(x0.tolist(), x1.tolist()))


def split_ratios(length, ratios=(0.7, 0.15, 0.15)):
    assert np.allclose(np.sum(ratios), 1.0)
    sizes, error = [], 0.
    for r in ratios:
        size = round(length * (r + error))
        error += r - size / length
        sizes.append(size)
    assert sum(sizes) == length
    return sizes


# ========================================================================= #
# Base Data Module                                                          #
# ========================================================================= #


class BaseImageModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int = 128,
        normalise: Union[bool, Tuple[float, float]] = False,
        shuffle: bool = True,
        num_workers: int = os.cpu_count(),
    ):
        super().__init__()
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._normalise = normalise
        self._num_workers = num_workers
        # datasets
        self._data_trn: Dataset = None
        self._data_tst: Dataset = None
        self._data_val: Dataset = None
        # has setup
        self._has_setup = False

    def setup(self, stage=None):
        if self._has_setup:
            warnings.warn(f'dataset {self.__class__.__name__} has been setup more than once... skipping step!')
            return
        self._has_setup = True
        # get image normalise transform
        if self._normalise is False:
            transform = torchvision.transforms.ToTensor()
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(*self.norm_mean_std)
            ])
        # load data
        self._data_trn, self._data_tst, self._data_val = self._setup(transform=transform)

    def _setup(self, transform) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        raise NotImplementedError

    @property
    def norm_mean_std(self) -> Tuple[float, float]:
        if isinstance(self._normalise, bool):
            mean, std = (0.5, 0.5) if self._normalise else (0., 1.)
        else:
            mean, std = self._normalise
        # return values
        return float(mean), float(std)

    @property
    def img_shape(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    @property
    def obs_shape(self) -> Tuple[int, int, int]:
        H, W, C = self.img_shape
        return C, H, W

    @property
    def obs_size(self) -> int:
        return int(np.prod(self.img_shape))

    def _make_dataloader(self, name, data):
        if data is None:
            raise ValueError(f'{name} was not initialised by: {self.__class__.__name__}.setup')
        if isinstance(data, IterableDataset):
            # WARNING: iterable dataset cannot make use of shuffle!
            return DataLoader(dataset=data, batch_size=self._batch_size)
        else:
            return DataLoader(dataset=data, batch_size=self._batch_size, shuffle=self._shuffle)

    def train_dataloader(self):
        return self._make_dataloader('data_trn', self._data_trn)

    def test_dataloader(self):
        return self._make_dataloader('data_tst', self._data_tst)

    def val_dataloader(self):
        return self._make_dataloader('data_val', self._data_val)

    def sample_display_batch(self, n=9) -> torch.Tensor:
        return next(iter(DataLoader(self._data_val, num_workers=0, batch_size=n, shuffle=False)))


# ========================================================================= #
# NOISE                                                                     #
# ========================================================================= #


class NoiseDataset(IterableDataset):

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        sampler: str = 'normal',
        return_labels: bool = False,
        num_labels: int = 2,
        length: Optional[int] = 60000,
    ):
        self._sampler = get_sampler(sampler)
        self._obs_shape = obs_shape
        self._return_labels = return_labels
        self._num_labels = num_labels
        self._length = length

    def __iter__(self) -> Iterator[torch.Tensor]:
        counter = itertools.count() if (self._length is None) else range(self._length)
        # yield all values!
        for i in counter:
            obs = self._sampler(*self._obs_shape, dtype=torch.float32, device=None)
            if self._return_labels:
                yield obs, np.random.randint(0, self._num_labels)
            else:
                yield obs


class NoiseImageDataModule(BaseImageModule):

    def __init__(
        self,
        obs_shape: Tuple[int, int, int] = (1, 28, 28),
        sampler: str = 'normal',
        batch_size: int = 128,
        normalise: Union[bool, Tuple[float, float]] = False,
        shuffle: bool = True,
        num_workers: int = os.cpu_count(),
        return_labels: bool = False,
        num_labels: int = 2,
        length: Optional[int] = 60000,
    ):
        super().__init__(batch_size=batch_size, normalise=normalise, shuffle=shuffle, num_workers=num_workers)
        self._dataset = NoiseDataset(obs_shape=obs_shape, sampler=sampler, return_labels=return_labels, num_labels=num_labels, length=length)
        C, H, W = obs_shape
        self._img_shape = (H, W, C)

    def _setup(self, transform) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        return (self._dataset, self._dataset, self._dataset)

    @property
    def img_shape(self) -> Tuple[int, int, int]:
        return self._img_shape


# ========================================================================= #
# MNIST                                                                     #
# ========================================================================= #


class ImageMNIST(MNIST, Sequence):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img


class ImageMnistDataModule(BaseImageModule):

    def __init__(
        self,
        batch_size: int = 128,
        normalise: Union[bool, Tuple[float, float]] = False,
        shuffle: bool = True,
        num_workers: int = os.cpu_count(),
        return_labels=False
    ):
        super().__init__(batch_size=batch_size, normalise=normalise, shuffle=shuffle, num_workers=num_workers)
        self._mnist_cls = MNIST if return_labels else ImageMNIST

    @property
    def img_shape(self) -> Tuple[int, int, int]:
        return (28, 28, 1)

    def _setup(self, transform) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        return (
            self._mnist_cls(root='data', transform=transform, train=True, download=True),    # train
            None,                                                                       # test
            self._mnist_cls(root='data', transform=transform, train=False, download=False),  # val
        )


class MnistDataModule(BaseImageModule):

    @property
    def img_shape(self) -> Tuple[int, int, int]:
        return (28, 28, 1)

    def _setup(self, transform) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        return (
            MNIST(root='data', transform=transform, train=True, download=True),    # train
            None,                                                                  # test
            MNIST(root='data', transform=transform, train=False, download=False),  # val
        )


# ========================================================================= #
# Imagenet Mini                                                             #
# ========================================================================= #


class ImageNetMiniFile(Dataset):

    def __init__(self, file: str, return_labels=False, transform=None, transform_label=None):
        self._file = file
        self._return_labels = return_labels
        self._transform = transform
        self._transform_label = transform_label
        # load the file
        with open(file, 'rb') as f:
            data = pickle.load(f)
            self.raw_labels = data['class_dict']
            self.data = data['image_data']
        # check the data
        assert self.data.ndim == 4
        assert self.data.dtype == np.uint8
        assert self.data.shape[1:] == (84, 84, 3)
        # check the labels
        assert len(self.raw_labels) <= 255
        assert all(len(v) == 600 for v in self.raw_labels.values())
        assert sum(len(v) for v in self.raw_labels.values()) == len(self.data)
        # get the classes
        self.label_names = sorted(self.raw_labels.keys())
        self.label_ids = np.array(list(range(len(self.label_names))), dtype='uint8')
        # generate the label mapping
        self.labels = np.full(len(self.data), fill_value=255, dtype='uint8')  # 255 is intended to give an error if not overwritten
        for cls_idx, cls_name in zip(self.label_ids, self.label_names):
            self.labels[self.raw_labels[cls_name]] = cls_idx
        # check generated labels
        values, counts = np.unique(self.labels, return_counts=True)
        assert len(values) == len(self.label_names)
        assert np.all(values < len(self.label_names))
        assert np.all(counts == 600)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # load and process image
        img = self.data[idx]
        if self._transform is not None:
            img = self._transform(img)
        # return single
        if not self._return_labels:
            return img
        # load and process label
        label = self.labels[idx]
        if self._transform_label is not None:
            label = self._transform(label)
        # return both
        return img, label


class ConcatImageNetMiniFiles(Dataset):

    """
    mini-ImageNet:
    https://github.com/yaoyao-liu/mini-imagenet-tools

    DOWNLOAD LINKS:
    https://drive.google.com/drive/folders/137M9jEv8nw0agovbUiEN_fPl_waJ2jIj
    """

    FILE_NAME_TRN = 'mini-imagenet-cache-train.pkl'
    FILE_NAME_TST = 'mini-imagenet-cache-test.pkl'
    FILE_NAME_VAL = 'mini-imagenet-cache-val.pkl'

    def __init__(self, data_root: str = 'data', return_labels=False, transform=None, transform_label=None):
        self._data_dir = os.path.join(data_root, 'imagenet-mini')
        self._return_labels = return_labels
        self._transform = transform
        self._transform_label = transform_label
        # load files ~ 2GB
        dataset_trn = ImageNetMiniFile(os.path.join(self._data_dir, self.FILE_NAME_TRN))
        dataset_tst = ImageNetMiniFile(os.path.join(self._data_dir, self.FILE_NAME_TST))
        dataset_val = ImageNetMiniFile(os.path.join(self._data_dir, self.FILE_NAME_VAL))
        # check sizes
        assert dataset_trn.data.shape == (64*600, 84, 84, 3)
        assert dataset_tst.data.shape == (20*600, 84, 84, 3)
        assert dataset_val.data.shape == (16*600, 84, 84, 3)
        # combine data
        self.data = np.concatenate([
            dataset_trn.data,
            dataset_tst.data,
            dataset_val.data,
        ], axis=0)
        # combine labels
        self.label_names = [
            *dataset_trn.label_names,
            *dataset_tst.label_names,
            *dataset_val.label_names,
        ]
        self.label_ids = np.concatenate([
            dataset_trn.label_ids,
            dataset_tst.label_ids + len(dataset_trn.label_names),
            dataset_val.label_ids + len(dataset_trn.label_names) + len(dataset_tst.label_names),
        ])
        self.labels = np.concatenate([
            dataset_trn.labels,
            dataset_tst.labels + len(dataset_trn.label_names),
            dataset_val.labels + len(dataset_trn.label_names) + len(dataset_tst.label_names),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # load and process image
        img = self.data[idx]
        if self._transform is not None:
            img = self._transform(img)
        # return single
        if not self._return_labels:
            return img
        # load and process label
        label = self.labels[idx]
        if self._transform_label is not None:
            label = self._transform(label)
        # return both
        return img, label


class ImageNetMiniDataModule(BaseImageModule):

    @property
    def img_shape(self) -> Tuple[int, int, int]:
        return (84, 84, 3)

    def __init__(
        self,
        batch_size: int = 128,
        normalise: Union[bool, Tuple[float, float]] = False,
        shuffle: bool = True,
        num_workers: int = os.cpu_count(),
        data_root: str = 'data',
        return_labels=False,
    ):
        super().__init__(batch_size=batch_size, normalise=normalise, shuffle=shuffle, num_workers=num_workers)
        self._data_root = data_root
        self._return_labels = return_labels

    def _setup(self, transform) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        dataset = ConcatImageNetMiniFiles(
            data_root=self._data_root,
            return_labels=self._return_labels,
            transform=transform,
        )
        # shuffle and split the data
        subset_trn, subset_tst, subset_val = random_split(
            dataset, [48000, 6000, 6000],  # 80%, 10%, 10%
            generator=torch.Generator().manual_seed(42),
        )
        # return values
        return subset_trn, subset_tst, subset_val


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


_DATA_MODULE_CLASSES = {
    'mnist': ImageMnistDataModule,
    'mini_imagenet': ImageNetMiniDataModule,
    'noise_mnist':                 lambda **kwargs: NoiseImageDataModule(obs_shape=(1, 28, 28), length=60000, sampler='normal',   num_labels=10,  **kwargs),  # alias for normal
    'noise_mnist_normal':          lambda **kwargs: NoiseImageDataModule(obs_shape=(1, 28, 28), length=60000, sampler='normal',   num_labels=10,  **kwargs),
    'noise_mnist_uniform':         lambda **kwargs: NoiseImageDataModule(obs_shape=(1, 28, 28), length=60000, sampler='uniform',  num_labels=10,  **kwargs),
    'noise_mini_imagenet':         lambda **kwargs: NoiseImageDataModule(obs_shape=(3, 84, 84), length=60000, sampler='normal',   num_labels=100, **kwargs),  # alias for normal
    'noise_mini_imagenet_normal':  lambda **kwargs: NoiseImageDataModule(obs_shape=(3, 84, 84), length=60000, sampler='normal',   num_labels=100, **kwargs),
    'noise_mini_imagenet_uniform': lambda **kwargs: NoiseImageDataModule(obs_shape=(3, 84, 84), length=60000, sampler='uniform',  num_labels=100, **kwargs),
}


def make_image_data_module(
    dataset: str = 'mnist',
    batch_size: int = 128,
    normalise: Union[bool, Tuple[float, float]] = False,
    num_workers: int = os.cpu_count(),
    return_labels: bool = False,
) -> BaseImageModule:
    dataset_cls = _DATA_MODULE_CLASSES[dataset]
    return dataset_cls(
        batch_size=batch_size,
        normalise=normalise,
        num_workers=num_workers,
        return_labels=return_labels,
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
