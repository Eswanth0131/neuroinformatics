"""
2.5D Dataset — returns adjacent slices as multi-channel input.

Input:  (n_adj, H, W) — N adjacent slices centered on target
Output: (1, H, W)     — center high-field slice

Handles edge cases by reflecting/padding at volume boundaries.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate, gaussian_filter

from dataset import TARGET_SHAPE, augment_slice, preprocess_train, preprocess_test


class SliceDataset25D(Dataset):
    """
    Dataset that returns N adjacent low-field slices + center high-field slice.

    Args:
        volumes: list of dicts from preprocess_train()
        n_adj: number of adjacent slices (must be odd). Default 5 = 2+1+2
        augment: whether to apply augmentation
    """
    def __init__(self, volumes: list[dict], n_adj: int = 5, augment: bool = False):
        assert n_adj % 2 == 1, "n_adj must be odd"
        self.n_adj = n_adj
        self.half = n_adj // 2
        self.augment = augment
        self.entries = []

        for vol in volumes:
            n_slices = TARGET_SHAPE[2]
            for z in range(n_slices):
                self.entries.append({
                    "low_vol": vol["low"],    # full volume reference
                    "high_slice": vol["high"][:, :, z],
                    "z": z,
                    "n_slices": n_slices,
                })

    def _get_adjacent_slices(self, entry):
        """Extract n_adj slices centered on z, reflecting at boundaries."""
        vol = entry["low_vol"]
        z = entry["z"]
        n = entry["n_slices"]
        slices = []
        for dz in range(-self.half, self.half + 1):
            idx = z + dz
            # Reflect at boundaries
            if idx < 0:
                idx = -idx
            elif idx >= n:
                idx = 2 * (n - 1) - idx
            idx = max(0, min(idx, n - 1))  # safety clamp
            slices.append(vol[:, :, idx])
        return np.stack(slices, axis=0)  # (n_adj, H, W)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        low_adj = self._get_adjacent_slices(entry)  # (n_adj, H, W)
        high = entry["high_slice"]                   # (H, W)

        if self.augment:
            # Apply same spatial augmentation to all channels
            # First handle flip and rotation (spatial transforms)
            do_flip = np.random.random() > 0.5
            do_rotate = np.random.random() > 0.6
            angle = np.random.uniform(-8, 8) if do_rotate else 0

            for c in range(self.n_adj):
                if do_flip:
                    low_adj[c] = low_adj[c][:, ::-1].copy()
                if do_rotate:
                    low_adj[c] = rotate(low_adj[c], angle, reshape=False, order=1, mode='reflect')

            if do_flip:
                high = high[:, ::-1].copy()
            if do_rotate:
                high = rotate(high, angle, reshape=False, order=1, mode='reflect')

            # Input-only augmentations (apply to all input channels)
            if np.random.random() > 0.5:
                scale = np.random.uniform(0.9, 1.1)
                bias = np.random.uniform(-0.05, 0.05)
                low_adj = np.clip(low_adj * scale + bias, 0, 1)

            if np.random.random() > 0.6:
                sigma = np.random.uniform(0.005, 0.02)
                noise = np.random.randn(*low_adj.shape).astype(np.float32) * sigma
                low_adj = np.clip(low_adj + noise, 0, 1)

        return (
            torch.from_numpy(low_adj.copy()).float(),     # (n_adj, H, W)
            torch.from_numpy(high[np.newaxis].copy()).float(),  # (1, H, W)
        )


def get_dataloaders_25d(
    data_dir: str,
    cache_dir: str = None,
    val_subjects: int = 2,
    val_start: int = -1,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
    n_adj: int = 5,
):
    """Same as get_dataloaders but returns 2.5D datasets."""
    volumes = preprocess_train(data_dir, cache_dir)

    if val_subjects > 0:
        if val_start < 0:
            val_start = len(volumes) - val_subjects
        val_end = val_start + val_subjects
        val_vols = volumes[val_start:val_end]
        train_vols = volumes[:val_start] + volumes[val_end:]
    else:
        train_vols = volumes
        val_vols = []

    print(f"Train: {len(train_vols)} subjects ({len(train_vols)*200} slices)")
    print(f"Mode: 2.5D with {n_adj} adjacent slices")
    if val_vols:
        val_ids = [v["sample_id"] for v in val_vols]
        print(f"Val:   {len(val_vols)} subjects ({val_ids})")

    train_ds = SliceDataset25D(train_vols, n_adj=n_adj, augment=augment)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    val_loader = None
    if val_vols:
        val_ds = SliceDataset25D(val_vols, n_adj=n_adj, augment=False)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    return train_loader, val_loader
