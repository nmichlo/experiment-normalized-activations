import copy
import inspect
import itertools
import os
import pickle
import warnings
from argparse import Namespace
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.core.saving import ALLOWED_CONFIG_TYPES
from pytorch_lightning.core.saving import PRIMITIVE_TYPES
from pytorch_lightning.utilities import AttributeDict
from pytorch_lightning.utilities.parsing import save_hyperparameters
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import random_split
from torchvision.datasets import MNIST

from exp.nn.activation import get_sampler


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


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


class HparamDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self._hparams = AttributeDict()
        self._hparams_initial = AttributeDict()

    def save_hyperparameters(self, *args, ignore: Optional[Union[Sequence[str], str]] = None) -> None:
        """Copied from pl.LightningModule"""
        frame = inspect.currentframe().f_back
        save_hyperparameters(self, *args, ignore=ignore, frame=frame)

    @property
    def hparams(self) -> Union[AttributeDict, dict, Namespace]:
        """Copied from pl.LightningModule"""
        if not hasattr(self, "_hparams"):
            self._hparams = AttributeDict()
        return self._hparams

    @property
    def hparams_initial(self) -> AttributeDict:
        """Copied from pl.LightningModule"""
        if not hasattr(self, "_hparams_initial"):
            return AttributeDict()
        # prevent any change
        return copy.deepcopy(self._hparams_initial)

    def _set_hparams(self, hp: Union[dict, Namespace, str]) -> None:
        """Copied from pl.LightningModule"""
        if isinstance(hp, Namespace):
            hp = vars(hp)
        if isinstance(hp, dict):
            hp = AttributeDict(hp)
        elif isinstance(hp, PRIMITIVE_TYPES):
            raise ValueError(f"Primitives {PRIMITIVE_TYPES} are not allowed.")
        elif not isinstance(hp, ALLOWED_CONFIG_TYPES):
            raise ValueError(f"Unsupported config type of {type(hp)}.")

        if isinstance(hp, dict) and isinstance(self.hparams, dict):
            self.hparams.update(hp)
        else:
            self._hparams = hp


# ========================================================================= #
# Base Data Module                                                          #
# ========================================================================= #


class ImageDataModule(HparamDataModule):

    def __init__(
        self,
        batch_size: int = 128,
        shift_mean_std: Tuple[float, float] = (0., 1.),
        shuffle: bool = True,
        num_workers: int = os.cpu_count(),
    ):
        # initialise
        super().__init__()
        # checks
        _, _ = shift_mean_std
        # save hyper-parameters
        self.save_hyperparameters(ignore=['has_trn', 'has_tst', 'has_val'])
        # extra hparams
        self.hparams.img_shape = self.img_shape
        self.hparams.obs_shape = self.obs_shape
        self.hparams.dataset = self.__class__.__name__
        # datasets
        self._data_trn: Dataset = None
        self._data_tst: Dataset = None
        self._data_val: Dataset = None
        # has setup
        self._has_setup = False

    def setup(self, stage=None):
        if self._has_setup:
            return
        self._has_setup = True
        # get image normalise transform
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*self.shift_mean_std)
        ])
        # load data
        self._data_trn, self._data_tst, self._data_val = self._setup(transform=transform)

    def _setup(self, transform) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        raise NotImplementedError

    @property
    def shift_mean_std(self) -> Tuple[float, float]:
        return tuple(self.hparams.shift_mean_std)

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
        assert name in ('trn', 'tst', 'val')
        if data is None:
            return None
        if isinstance(data, IterableDataset):
            return DataLoader(dataset=data, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        else:
            return DataLoader(dataset=data, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=self.hparams.shuffle if (name == 'trn') else False)

    def train_dataloader(self): return self._make_dataloader('trn', self._data_trn)
    def test_dataloader(self): return self._make_dataloader('tst', self._data_tst)
    def val_dataloader(self): return self._make_dataloader('val', self._data_val)

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
        transform=None,
        transform_label=None,
    ):
        C, H, W = obs_shape
        self._img_shape = (H, W, C)
        self._sampler = get_sampler(sampler, tensor=False)
        self._return_labels = return_labels
        self._num_labels = num_labels
        self._length = length
        self._transform = transform
        self._transform_label = transform_label
        # warnings
        if self._transform is not None:
            warnings.warn(f'`transform` applied to {self.__class__.__name__} is probably incorrect, rather adjust the `sampler`')

    def __iter__(self) -> Iterator[torch.Tensor]:
        counter = itertools.count() if (self._length is None) else range(self._length)
        # yield all values!
        for i in counter:
            # handle obs
            obs = self._sampler(*self._img_shape)
            if self._transform is not None:
                obs = self._transform(obs)
            # return obs
            if not self._return_labels:
                yield obs
            # handle labels
            label = np.random.randint(0, self._num_labels)
            if self._transform_label is not None:
                label = self._transform_label(label)
            # return both
            yield obs, label


class NoiseImageDataModule(ImageDataModule):

    def __init__(
        self,
        obs_shape: Tuple[int, int, int] = (1, 28, 28),
        sampler: str = 'normal',
        batch_size: int = 128,
        shift_mean_std: Tuple[float, float] = (0., 1.),
        shuffle: bool = True,
        num_workers: int = os.cpu_count(),
        return_labels: bool = False,
        num_labels: int = 2,
        # sizes
        trn_length: Optional[int] = 60000,
        val_length: Optional[int] = 10000,
        tst_length: Optional[int] = 10000,
    ):
        C, H, W = obs_shape
        self._img_shape = (H, W, C)
        # initialise
        super().__init__(batch_size=batch_size, shift_mean_std=shift_mean_std, shuffle=shuffle, num_workers=num_workers)
        self.save_hyperparameters()

    def _setup(self, transform) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        trn_dataset = None if (self.hparams.trn_length is None) else NoiseDataset(obs_shape=self.hparams.obs_shape, sampler=self.hparams.sampler, return_labels=self.hparams.return_labels, num_labels=self.hparams.num_labels, length=self.hparams.trn_length, transform=transform)
        tst_dataset = None if (self.hparams.val_length is None) else NoiseDataset(obs_shape=self.hparams.obs_shape, sampler=self.hparams.sampler, return_labels=self.hparams.return_labels, num_labels=self.hparams.num_labels, length=self.hparams.val_length, transform=transform)
        val_dataset = None if (self.hparams.tst_length is None) else NoiseDataset(obs_shape=self.hparams.obs_shape, sampler=self.hparams.sampler, return_labels=self.hparams.return_labels, num_labels=self.hparams.num_labels, length=self.hparams.tst_length, transform=transform)
        return (trn_dataset, tst_dataset, val_dataset)

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


class ImageMnistDataModule(ImageDataModule):

    def __init__(
        self,
        batch_size: int = 128,
        shift_mean_std: Tuple[float, float] = (0., 1.),
        shuffle: bool = True,
        num_workers: int = os.cpu_count(),
        return_labels=False
    ):
        super().__init__(batch_size=batch_size, shift_mean_std=shift_mean_std, shuffle=shuffle, num_workers=num_workers)
        self.save_hyperparameters()
        # initialise
        self._mnist_cls = MNIST if return_labels else ImageMNIST

    @property
    def img_shape(self) -> Tuple[int, int, int]:
        return (28, 28, 1)

    def _setup(self, transform) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        return (
            self._mnist_cls(root='data', transform=transform, train=True, download=True),    # train
            None,                                                                            # test
            self._mnist_cls(root='data', transform=transform, train=False, download=False),  # val
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
        return img, label.astype('int')


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
        return img, label.astype('int')


class ImageNetMiniDataModule(ImageDataModule):

    @property
    def img_shape(self) -> Tuple[int, int, int]:
        return (84, 84, 3)

    def __init__(
        self,
        batch_size: int = 128,
        shift_mean_std: Tuple[float, float] = (0., 1.),
        shuffle: bool = True,
        num_workers: int = os.cpu_count(),
        data_root: str = 'data',
        return_labels=False,
        has_test_data=False,
    ):
        super().__init__(batch_size=batch_size, shift_mean_std=shift_mean_std, shuffle=shuffle, num_workers=num_workers)
        self.save_hyperparameters()
        # initialise
        self._data_root = data_root
        self._return_labels = return_labels

    def _setup(self, transform) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        dataset = ConcatImageNetMiniFiles(
            data_root=self._data_root,
            return_labels=self._return_labels,
            transform=transform,
        )
        # shuffle and split the data
        if self.hparams.has_test_data:
            subset_trn, subset_tst, subset_val = random_split(
                dataset, [45000, 7500, 7500],  # 75%, 12.5%, 12.5%
                generator=torch.Generator().manual_seed(42),
            )
            return subset_trn, subset_tst, subset_val
        else:
            subset_trn, subset_val = random_split(
                dataset, [51000, 9000],  # (6/7) ~= 85%, (1/7) ~= 15% # mnist ratios
                generator=torch.Generator().manual_seed(42),
            )
            return subset_trn, None, subset_val


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


_DATA_MEAN_STD = {
    'mnist':                       (0.1306604762738431, 0.3081078038564622),  # TODO: verify this!
    'noise_mnist':                 (0., 1.),
    'noise_mnist_normal':          (0., 1.),
    'noise_mnist_uniform':         (0., 1.),
    'mini_imagenet':               (0.5, 0.5),  # TODO: THIS IS WRONG!
    'noise_mini_imagenet':         (0., 1.),
    'noise_mini_imagenet_normal':  (0., 1.),
    'noise_mini_imagenet_uniform': (0., 1.),
}

_DATA_MODULE_CLASSES = {
    'mnist':                       ImageMnistDataModule,
    'noise_mnist':                 lambda **kwargs: NoiseImageDataModule(obs_shape=(1, 28, 28), trn_length=60000, tst_length=None, val_length=10000, sampler='normal',   num_labels=10,  **kwargs),  # alias for normal
    'noise_mnist_normal':          lambda **kwargs: NoiseImageDataModule(obs_shape=(1, 28, 28), trn_length=60000, tst_length=None, val_length=10000, sampler='normal',   num_labels=10,  **kwargs),
    'noise_mnist_uniform':         lambda **kwargs: NoiseImageDataModule(obs_shape=(1, 28, 28), trn_length=60000, tst_length=None, val_length=10000, sampler='uniform',  num_labels=10,  **kwargs),
    'mini_imagenet':               ImageNetMiniDataModule,
    'noise_mini_imagenet':         lambda **kwargs: NoiseImageDataModule(obs_shape=(3, 84, 84), trn_length=51000, tst_length=None, val_length=9000,  sampler='normal',   num_labels=100, **kwargs),  # alias for normal
    'noise_mini_imagenet_normal':  lambda **kwargs: NoiseImageDataModule(obs_shape=(3, 84, 84), trn_length=51000, tst_length=None, val_length=9000,  sampler='normal',   num_labels=100, **kwargs),
    'noise_mini_imagenet_uniform': lambda **kwargs: NoiseImageDataModule(obs_shape=(3, 84, 84), trn_length=51000, tst_length=None, val_length=9000,  sampler='uniform',  num_labels=100, **kwargs),
}


def make_image_data_module(
    dataset: str = 'mnist',
    batch_size: int = 128,
    shift_mean_std: Union[bool, Tuple[float, float]] = True,
    num_workers: int = os.cpu_count(),
    return_labels: bool = True,
    **kwargs,
) -> ImageDataModule:
    dataset_cls = _DATA_MODULE_CLASSES[dataset]
    # shift mean and std
    if isinstance(shift_mean_std, bool):
        if shift_mean_std:
            try:
                shift_mean_std = _DATA_MEAN_STD[dataset]
            except KeyError:
                raise KeyError('no default `shift_mean_std` for: {repr(dataset)}, set `shift_mean_std=False`')
        else:
            shift_mean_std = (0., 1.)
    # create data
    return dataset_cls(
        batch_size=batch_size,
        shift_mean_std=shift_mean_std,
        num_workers=num_workers,
        return_labels=return_labels,
        **kwargs,
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
