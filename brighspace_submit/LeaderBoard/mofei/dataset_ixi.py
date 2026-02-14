"""
Dataset for IXI synthetic pairs (pretraining).

Loads npz files created by prepare_ixi.py containing paired
(low_upsampled, high_resampled) volumes at TARGET_SHAPE.

Supports both 2D and 2.5D modes.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import rotate

TARGET_SHAPE = (179, 221, 200)


class IXISliceDataset(Dataset):
    """2D slice dataset from IXI synthetic pairs."""

    def __init__(self, data_dir, augment=True, max_subjects=None):
        self.augment = augment
        self.entries = []

        npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if max_subjects:
            npz_files = npz_files[:max_subjects]

        print(f"Loading {len(npz_files)} IXI subjects...")
        for f in npz_files:
            d = np.load(f)
            low, high = d["low"], d["high"]
            for z in range(TARGET_SHAPE[2]):
                self.entries.append({
                    "low_slice": low[:, :, z],
                    "high_slice": high[:, :, z],
                })

        print(f"  {len(self.entries)} slices loaded")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        low = entry["low_slice"].copy()
        high = entry["high_slice"].copy()

        if self.augment:
            if np.random.random() > 0.5:
                low = low[:, ::-1].copy()
                high = high[:, ::-1].copy()
            if np.random.random() > 0.6:
                angle = np.random.uniform(-8, 8)
                low = rotate(low, angle, reshape=False, order=1, mode='reflect')
                high = rotate(high, angle, reshape=False, order=1, mode='reflect')
            if np.random.random() > 0.5:
                scale = np.random.uniform(0.9, 1.1)
                bias = np.random.uniform(-0.05, 0.05)
                low = np.clip(low * scale + bias, 0, 1)
            if np.random.random() > 0.6:
                sigma = np.random.uniform(0.005, 0.02)
                low = np.clip(low + np.random.randn(*low.shape).astype(np.float32) * sigma, 0, 1)

        return (
            torch.from_numpy(low[np.newaxis].copy()).float(),
            torch.from_numpy(high[np.newaxis].copy()).float(),
        )


class IXISliceDataset25D(Dataset):
    """2.5D slice dataset from IXI synthetic pairs."""

    def __init__(self, data_dir, n_adj=5, augment=True, max_subjects=None):
        self.augment = augment
        self.n_adj = n_adj
        self.half = n_adj // 2
        self.entries = []

        npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if max_subjects:
            npz_files = npz_files[:max_subjects]

        print(f"Loading {len(npz_files)} IXI subjects (2.5D, n_adj={n_adj})...")
        for f in npz_files:
            d = np.load(f)
            low, high = d["low"], d["high"]
            n_slices = TARGET_SHAPE[2]
            for z in range(n_slices):
                self.entries.append({
                    "low_vol": low,
                    "high_slice": high[:, :, z],
                    "z": z,
                    "n_slices": n_slices,
                })

        print(f"  {len(self.entries)} slices loaded")

    def _get_adjacent(self, entry):
        vol = entry["low_vol"]
        z, n = entry["z"], entry["n_slices"]
        slices = []
        for dz in range(-self.half, self.half + 1):
            idx = z + dz
            if idx < 0: idx = -idx
            elif idx >= n: idx = 2 * (n - 1) - idx
            idx = max(0, min(idx, n - 1))
            slices.append(vol[:, :, idx])
        return np.stack(slices, axis=0)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        low_adj = self._get_adjacent(entry)
        high = entry["high_slice"].copy()

        if self.augment:
            do_flip = np.random.random() > 0.5
            do_rotate = np.random.random() > 0.6
            angle = np.random.uniform(-8, 8) if do_rotate else 0

            for c in range(self.n_adj):
                if do_flip: low_adj[c] = low_adj[c][:, ::-1].copy()
                if do_rotate: low_adj[c] = rotate(low_adj[c], angle, reshape=False, order=1, mode='reflect')
            if do_flip: high = high[:, ::-1].copy()
            if do_rotate: high = rotate(high, angle, reshape=False, order=1, mode='reflect')

            if np.random.random() > 0.5:
                scale = np.random.uniform(0.9, 1.1)
                bias = np.random.uniform(-0.05, 0.05)
                low_adj = np.clip(low_adj * scale + bias, 0, 1)

            if np.random.random() > 0.6:
                sigma = np.random.uniform(0.005, 0.02)
                low_adj = np.clip(low_adj + np.random.randn(*low_adj.shape).astype(np.float32) * sigma, 0, 1)

        return (
            torch.from_numpy(low_adj.copy()).float(),
            torch.from_numpy(high[np.newaxis].copy()).float(),
        )


def get_ixi_dataloader(data_dir, batch_size=32, num_workers=4, augment=True,
                       mode="2d", n_adj=5, max_subjects=None):
    """Get IXI dataloader for pretraining."""
    if mode == "25d":
        ds = IXISliceDataset25D(data_dir, n_adj=n_adj, augment=augment,
                                max_subjects=max_subjects)
    else:
        ds = IXISliceDataset(data_dir, augment=augment, max_subjects=max_subjects)

    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)
