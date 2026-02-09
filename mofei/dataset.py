"""
Dataset for low-field → high-field MRI enhancement.

Preprocessing: trilinear upsample low-field (112,138,40) → (179,221,200),
then train a 2D network on paired axial slices.
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom


TARGET_SHAPE = (179, 221, 200)


def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI volume as a float32 numpy array."""
    return nib.load(path).get_fdata().astype(np.float32)


def upsample_volume(vol: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    """Trilinear upsample a volume to target_shape."""
    factors = [t / s for t, s in zip(target_shape, vol.shape)]
    return zoom(vol, factors, order=1).astype(np.float32)  # order=1 = trilinear


def normalize(vol: np.ndarray) -> np.ndarray:
    """Min-max normalize a volume to [0, 1]."""
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)


def preprocess_train(data_dir: str, cache_dir: str = None) -> list[dict]:
    """
    Load and preprocess all training pairs.
    Returns list of dicts: {'low': (179,221,200), 'high': (179,221,200), 'sample_id': str}
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "train_volumes.npz")
        if os.path.exists(cache_path):
            print(f"Loading cached training data from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return list(data["volumes"])

    low_dir = os.path.join(data_dir, "train", "low_field")
    high_dir = os.path.join(data_dir, "train", "high_field")

    low_files = sorted(glob.glob(os.path.join(low_dir, "*.nii*")))
    volumes = []

    for lf_path in low_files:
        sample_id = os.path.basename(lf_path).split("_lowfield")[0]
        hf_path = os.path.join(high_dir, f"{sample_id}_highfield.nii.gz")
        if not os.path.exists(hf_path):
            hf_path = hf_path.replace(".nii.gz", ".nii")

        print(f"Processing {sample_id}...")
        low_vol = normalize(load_nifti(lf_path))
        high_vol = normalize(load_nifti(hf_path))

        low_up = upsample_volume(low_vol)
        # Ensure high-field is exactly target shape
        assert high_vol.shape == TARGET_SHAPE, \
            f"High-field shape {high_vol.shape} != {TARGET_SHAPE}"

        volumes.append({
            "low": low_up,
            "high": high_vol,
            "sample_id": sample_id,
        })

    if cache_dir:
        print(f"Caching training data to {cache_path}")
        np.savez_compressed(cache_path, volumes=volumes)

    return volumes


def preprocess_test(data_dir: str, cache_dir: str = None) -> list[dict]:
    """Load and preprocess test volumes (low-field only)."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "test_volumes.npz")
        if os.path.exists(cache_path):
            print(f"Loading cached test data from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return list(data["volumes"])

    low_dir = os.path.join(data_dir, "test", "low_field")
    low_files = sorted(glob.glob(os.path.join(low_dir, "*.nii*")))
    volumes = []

    for lf_path in low_files:
        sample_id = os.path.basename(lf_path).split("_lowfield")[0]
        print(f"Processing {sample_id}...")
        low_vol = normalize(load_nifti(lf_path))
        low_up = upsample_volume(low_vol)

        volumes.append({
            "low": low_up,
            "sample_id": sample_id,
        })

    if cache_dir:
        print(f"Caching test data to {cache_path}")
        np.savez_compressed(cache_path, volumes=volumes)

    return volumes


class SliceDataset(Dataset):
    """
    Dataset of 2D axial slices from preprocessed volumes.

    Args:
        volumes: list of dicts from preprocess_train()
        augment: whether to apply data augmentation
    """

    def __init__(self, volumes: list[dict], augment: bool = False):
        self.slices = []
        self.augment = augment

        for vol in volumes:
            for z in range(TARGET_SHAPE[2]):
                self.slices.append({
                    "low": vol["low"][:, :, z],
                    "high": vol["high"][:, :, z],
                })

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        s = self.slices[idx]
        low = s["low"][np.newaxis]   # (1, 179, 221)
        high = s["high"][np.newaxis]

        if self.augment and torch.rand(1).item() > 0.5:
            # Horizontal flip
            low = low[:, :, ::-1].copy()
            high = high[:, :, ::-1].copy()

        return torch.from_numpy(low), torch.from_numpy(high)


def get_dataloaders(
    data_dir: str,
    cache_dir: str = None,
    val_subjects: int = 2,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
):
    """
    Build train and validation DataLoaders with subject-level split.

    Args:
        data_dir: path to dataset root (contains train/ and test/)
        val_subjects: number of subjects held out for validation
        batch_size: batch size
        num_workers: DataLoader workers
        augment: whether to augment training data
    """
    volumes = preprocess_train(data_dir, cache_dir)

    train_vols = volumes[:-val_subjects] if val_subjects > 0 else volumes
    val_vols = volumes[-val_subjects:] if val_subjects > 0 else []

    print(f"Train: {len(train_vols)} subjects ({len(train_vols)*200} slices)")
    if val_vols:
        print(f"Val:   {len(val_vols)} subjects ({len(val_vols)*200} slices)")

    train_ds = SliceDataset(train_vols, augment=augment)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    val_loader = None
    if val_vols:
        val_ds = SliceDataset(val_vols, augment=False)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    return train_loader, val_loader
