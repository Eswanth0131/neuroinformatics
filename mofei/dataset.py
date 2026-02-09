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
from scipy.ndimage import zoom, rotate, gaussian_filter, map_coordinates


TARGET_SHAPE = (179, 221, 200)


def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)


def upsample_volume(vol: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    factors = [t / s for t, s in zip(target_shape, vol.shape)]
    return zoom(vol, factors, order=1).astype(np.float32)


def normalize(vol: np.ndarray) -> np.ndarray:
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)


def preprocess_train(data_dir: str, cache_dir: str = None) -> list[dict]:
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
        volumes.append({"low": low_up, "sample_id": sample_id})

    if cache_dir:
        print(f"Caching test data to {cache_path}")
        np.savez_compressed(cache_path, volumes=volumes)

    return volumes


# ─── Augmentation ─────────────────────────────────────────────────────────────

def augment_slice(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply identical random augmentations to a low/high slice pair.
    Input/output shape: (H, W) each.
    """
    # Horizontal flip (50%)
    if np.random.random() > 0.5:
        low = low[:, ::-1].copy()
        high = high[:, ::-1].copy()

    # Random rotation ±8 degrees (40%)
    if np.random.random() > 0.6:
        angle = np.random.uniform(-8, 8)
        low = rotate(low, angle, reshape=False, order=1, mode='reflect')
        high = rotate(high, angle, reshape=False, order=1, mode='reflect')

    # Intensity scaling — only on low-field input (50%)
    if np.random.random() > 0.5:
        scale = np.random.uniform(0.9, 1.1)
        bias = np.random.uniform(-0.05, 0.05)
        low = np.clip(low * scale + bias, 0, 1)

    # Additive Gaussian noise — only on low-field input (40%)
    if np.random.random() > 0.6:
        sigma = np.random.uniform(0.005, 0.02)
        low = np.clip(low + np.random.randn(*low.shape).astype(np.float32) * sigma, 0, 1)

    # Gaussian blur — only on low-field input (30%)
    if np.random.random() > 0.7:
        s = np.random.uniform(0.3, 1.0)
        low = gaussian_filter(low, sigma=s).astype(np.float32)

    return low, high


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SliceDataset(Dataset):
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
        low, high = s["low"], s["high"]

        if self.augment:
            low, high = augment_slice(low, high)

        return (
            torch.from_numpy(low[np.newaxis].copy()),
            torch.from_numpy(high[np.newaxis].copy()),
        )


def get_dataloaders(
    data_dir: str,
    cache_dir: str = None,
    val_subjects: int = 2,
    val_start: int = -1,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
):
    """
    val_start: index of first val subject. -1 means last val_subjects.
    e.g. val_start=0, val_subjects=2 holds out subjects 0,1
         val_start=8, val_subjects=2 holds out subjects 8,9
         val_start=-1, val_subjects=2 holds out last 2 (default)
    """
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
    if val_vols:
        val_ids = [v["sample_id"] for v in val_vols]
        print(f"Val:   {len(val_vols)} subjects ({val_ids})")

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
