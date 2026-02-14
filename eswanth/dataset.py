import os
import re
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import nibabel as nib
except ImportError as e:
    raise ImportError("Please install nibabel: pip install nibabel") from e

# Final shape that all volumes will be resized to
TARGET_SHAPE = (179, 221, 200)

LOW_FOLDER_NAME = "low_field"
HIGH_FOLDER_NAME = "high_field"

# Used to extract subject id like "sample_001"
SAMPLE_RE = re.compile(r"(sample_\d+)")


def _sorted_nii_files(folder: str) -> List[str]:
    # Return sorted list of .nii or .nii.gz files
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]
    files.sort()
    return files


def _extract_sample_id(path: str) -> str:
    # Extract sample id from filename
    base = os.path.basename(path)
    m = SAMPLE_RE.search(base)
    if not m:
        raise ValueError(f"Could not parse sample id from filename: {base}")
    return m.group(1)


def load_nifti(path: str) -> np.ndarray:
    # Load 3D MRI volume as float32
    img = nib.load(path)
    data = img.get_fdata()

    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data.shape}")

    return np.asarray(data, dtype=np.float32)


def normalize_minmax(vol: np.ndarray) -> np.ndarray:
    # Scale volume values to [0, 1]
    vmin = float(vol.min())
    vmax = float(vol.max())

    if vmax - vmin < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)

    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)


def resize_3d_to_target(
    vol: np.ndarray,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE
) -> np.ndarray:
    # Resize 3D volume using trilinear interpolation
    Ht, Wt, Zt = target_shape

    # Convert to torch shape (1,1,Z,H,W)
    t = torch.from_numpy(vol).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

    t = torch.nn.functional.interpolate(
        t,
        size=(Zt, Ht, Wt),
        mode="trilinear",
        align_corners=False,
    )

    # Convert back to (H,W,Z)
    out = t.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return out.astype(np.float32)


@dataclass(frozen=True)
class SubjectPair:
    # Stores paths for one subject
    sample_id: str
    low_path: str
    high_path: str


def find_subject_pairs(data_root: str) -> List[SubjectPair]:
    # Match low-field and high-field volumes by sample id
    low_dir = os.path.join(data_root, "train", LOW_FOLDER_NAME)
    high_dir = os.path.join(data_root, "train", HIGH_FOLDER_NAME)

    low_files = _sorted_nii_files(low_dir)
    pairs: List[SubjectPair] = []

    for lp in low_files:
        sid = _extract_sample_id(lp)

        hp_gz = os.path.join(high_dir, f"{sid}_highfield.nii.gz")
        hp_nii = os.path.join(high_dir, f"{sid}_highfield.nii")

        if os.path.exists(hp_gz):
            hp = hp_gz
        elif os.path.exists(hp_nii):
            hp = hp_nii
        else:
            raise FileNotFoundError(f"High-field file not found for {sid}")

        pairs.append(SubjectPair(sid, lp, hp))

    return pairs


class MRISlicePairDataset(Dataset):
    # Returns 2D slice pairs (low_slice, high_slice)

    def __init__(
        self,
        subjects: List[SubjectPair],
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
        cache_in_ram: bool = True,
        augment: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.subjects = subjects
        self.target_shape = target_shape
        self.cache_in_ram = cache_in_ram
        self.augment = augment
        self.rng = random.Random(seed)

        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._Z = target_shape[2]  # number of slices
        self._N = len(subjects) * self._Z  # total dataset size

    def __len__(self) -> int:
        return self._N

    def _load_subject(self, subj: SubjectPair):
        # Load and preprocess full 3D volumes
        if self.cache_in_ram and subj.sample_id in self._cache:
            return self._cache[subj.sample_id]

        low = normalize_minmax(load_nifti(subj.low_path))
        high = normalize_minmax(load_nifti(subj.high_path))

        # Upsample low-field to target resolution
        low_up = resize_3d_to_target(low, self.target_shape)

        # Resize high-field only if needed
        if tuple(high.shape) != tuple(self.target_shape):
            high = resize_3d_to_target(high, self.target_shape)

        if self.cache_in_ram:
            self._cache[subj.sample_id] = (low_up, high)

        return low_up, high

    def _maybe_augment(self, x, y):
        # Apply simple augmentation to input
        if not self.augment:
            return x, y

        # Horizontal flip
        if self.rng.random() < 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=1).copy()

        # Intensity scaling (input only)
        if self.rng.random() < 0.5:
            scale = 0.9 + 0.2 * self.rng.random()
            bias = -0.05 + 0.10 * self.rng.random()
            x = np.clip(scale * x + bias, 0.0, 1.0)

        return x.astype(np.float32), y.astype(np.float32)

    def __getitem__(self, idx: int):
        # Convert flat index into subject index + slice index
        subj_idx = idx // self._Z
        z = idx % self._Z

        low_up, high = self._load_subject(self.subjects[subj_idx])

        # Extract slice
        x = low_up[:, :, z]
        y = high[:, :, z]

        x, y = self._maybe_augment(x, y)

        return (
            torch.from_numpy(x).unsqueeze(0).float(),
            torch.from_numpy(y).unsqueeze(0).float(),
        )


def split_subjects(subjects, val_subjects=2, seed=42):
    # Split subjects into train and validation sets
    rng = random.Random(seed)
    subjects = subjects.copy()
    rng.shuffle(subjects)

    val = subjects[:val_subjects]
    train = subjects[val_subjects:]
    return train, val


def make_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 2,
    val_subjects: int = 2,
    cache_in_ram: bool = True,
    augment: bool = True,
    seed: int = 42,
):
    # Create train and validation DataLoaders
    all_subjects = find_subject_pairs(data_root)
    train_subj, val_subj = split_subjects(all_subjects, val_subjects, seed)

    train_ds = MRISlicePairDataset(
        train_subj, cache_in_ram=cache_in_ram, augment=augment, seed=seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if len(val_subj) > 0:
        val_ds = MRISlicePairDataset(
            val_subj, cache_in_ram=cache_in_ram, augment=False, seed=seed
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader
