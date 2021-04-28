
import os

import torch
import numpy as np
from torch.utils.data import Dataset

from utility.pytorch_utils import tensor, to_device
from config import DATA_DIR

OFFSET = 100
VALID_RATIO = 0.04


class DEMDataset(Dataset):
    def __init__(
            self, dem_data, mask, idx, window_size
    ):
        super().__init__()

        lw = window_size // 2
        rw = window_size - lw

        idx0, idx1 = idx
        s0, s1 = dem_data.shape
        in_window = (idx0 >= OFFSET) & (idx0 < (s0 - OFFSET)) \
                    & (idx1 >= OFFSET) & (idx1 < (s1 - OFFSET))

        dem_data[np.isnan(dem_data)] = 0.0
        self.dem_data = tensor(dem_data, dtype=torch.float32)
        self.mask = tensor(mask, dtype=torch.bool)
        self.idx0 = tensor(idx0[in_window], dtype=torch.long).view(-1, 1)
        self.idx1 = tensor(idx1[in_window], dtype=torch.long).view(-1, 1)

        self.lw, self.rw = lw, rw
        self.num_data = len(self.idx0)

        w_sum = self.rw + self.lw
        i0 = torch.arange(-self.lw, self.rw).view(1, -1, 1)
        i1 = i0.clone().view(1, 1, -1)
        self.i0_window = to_device(i0.expand(-1, -1, w_sum).reshape(1, -1))
        self.i1_window = to_device(i1.expand(-1, w_sum, -1).reshape(1, -1))

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        b = len(index)
        i0 = (self.i0_window.expand(b, -1) + self.idx0[index]).flatten()
        i1 = (self.i1_window.expand(b, -1) + self.idx1[index]).flatten()
        dem = self.dem_data[i0, i1].view(b, -1, 1)
        mask = self.mask[i0, i1].view(b, -1)
        target = self.dem_data[self.idx0[index, 0], self.idx1[index, 0]].view(b, 1, 1)
        return dem, mask, target, self.idx0[index], self.idx1[index]


class DataLoader:
    def __init__(self, dataset, batch_size, epoch_split=None):
        self.dataset = dataset
        if epoch_split is None:
            self.num_data = len(dataset)
        else:
            self.num_data = int(len(dataset) / epoch_split)
        self.batch_size = batch_size
        self.idx = np.arange(self.num_data)

    def __len__(self):
        return self.num_data

    def __iter__(self):
        i = 0
        idx = np.random.permutation(self.idx)
        while i < self.num_data:
            idx_batch = idx[i:i+self.batch_size]
            i += self.batch_size
            yield self.dataset[idx_batch]


def get_dataset(
        dataset_name, window_size, batch_size, mask_fname, mask_size,
        epoch_split, recon_nodata=False
):
    # load data, mask
    dem_data = np.load(os.path.join(DATA_DIR, dataset_name))
    mask_fname = os.path.join(DATA_DIR, mask_fname)
    if os.path.isfile(mask_fname):
        d = np.load(mask_fname)
        train_mask = d['train_mask']
        valid_mask = d['valid_mask']
        test_mask = d['test_mask']
    else:
        train_mask, valid_mask, test_mask = \
            sample_random_mask(dem_data.shape, mask_size)
        np.savez(
            mask_fname,
            train_mask=train_mask,
            valid_mask=valid_mask,
            test_mask=test_mask
        )
    mask = np.logical_not(np.isnan(dem_data))

    # mask matrix to index
    if not recon_nodata:
        mask_all = np.logical_and(mask, train_mask)
        train_idx = np.where(mask_all)
        valid_idx = np.where(np.logical_and(mask, valid_mask))
        test_idx = np.where(np.logical_and(mask, test_mask))
    else:
        mask_all = mask
        train_idx = valid_idx = test_idx = np.where(np.isnan(dem_data))

    # dataset
    train_dataset = DEMDataset(dem_data, mask_all, train_idx, window_size)
    valid_dataset = DEMDataset(dem_data, mask_all, valid_idx, window_size)
    test_dataset = DEMDataset(dem_data, mask_all, test_idx, window_size)

    # loader
    train_loader = DataLoader(train_dataset, batch_size, epoch_split)
    valid_loader = DataLoader(valid_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)

    return train_loader, valid_loader, test_loader


def sample_random_mask(mask_shape, mask_size):
    train_mask = np.ones(mask_shape, dtype='bool')
    valid_mask = np.zeros(mask_shape, dtype='bool')
    test_mask = np.zeros(mask_shape, dtype='bool')

    num_total_pixels = train_mask.size
    num_eval_pixels = 0
    while num_eval_pixels / num_total_pixels < VALID_RATIO:
        i0 = np.random.randint(OFFSET, mask_shape[0]-OFFSET)
        i1 = np.random.randint(OFFSET, mask_shape[1]-OFFSET)
        l0 = l1 = mask_size // 2
        if (train_mask[i0 - l0:i0 + l0 + 1, i1 - l1:i1 + l1 + 1]).all():
            train_mask[i0 - l0:i0 + l0 + 1, i1 - l1:i1 + l1 + 1] = False
            if num_eval_pixels / num_total_pixels < VALID_RATIO/2:
                valid_mask[i0 - l0:i0 + l0 + 1, i1 - l1:i1 + l1 + 1] = True
            else:
                test_mask[i0 - l0:i0 + l0 + 1, i1 - l1:i1 + l1 + 1] = True
            num_eval_pixels += (2*l0+1) * (2*l1+1)
    return train_mask, valid_mask, test_mask
