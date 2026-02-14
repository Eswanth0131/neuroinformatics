"""
Synthetic Low-Field MRI Generation Pipeline

Takes high-field (3T/1.5T) brain T1 MRI volumes and degrades them to simulate
64mT low-field characteristics, creating paired (degraded, original) data.

Degradation steps (calibrated to match competition 64mT data):
1. Downsample to low-field resolution (112, 138, 40)
2. Add Rician noise (SNR matched to real 64mT)
3. Slight Gaussian blur (simulate lower gradient strength)
4. Contrast reduction (lower field → flatter contrast)
5. Upsample back to target shape (179, 221, 200) via trilinear

Usage:
    # Prepare IXI data for pretraining
    python prepare_ixi.py --ixi_dir data/ixi --output_dir data/ixi_pairs --max_subjects 500

    # Quick test with 10 subjects
    python prepare_ixi.py --ixi_dir data/ixi --output_dir data/ixi_pairs --max_subjects 10
"""

import argparse
import glob
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter
from concurrent.futures import ProcessPoolExecutor
from functools import partial


# Match competition dimensions exactly
LOW_FIELD_SHAPE = (112, 138, 40)
TARGET_SHAPE = (179, 221, 200)


def load_nifti(path):
    return nib.load(path).get_fdata().astype(np.float32)


def normalize(vol):
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)


def add_rician_noise(vol, snr=8.0):
    """
    Add Rician noise to simulate low-field MRI noise characteristics.
    Rician noise = sqrt((signal + noise_real)^2 + noise_imag^2)
    At low SNR, this creates the characteristic noise floor.
    """
    sigma = vol.max() / snr
    noise_real = np.random.randn(*vol.shape).astype(np.float32) * sigma
    noise_imag = np.random.randn(*vol.shape).astype(np.float32) * sigma
    noisy = np.sqrt((vol + noise_real) ** 2 + noise_imag ** 2)
    return noisy.astype(np.float32)


def reduce_contrast(vol, factor=0.7):
    """
    Reduce contrast to simulate lower field strength.
    Lower B0 → shorter T1 differences → flatter contrast.
    """
    mean_val = vol.mean()
    return (mean_val + (vol - mean_val) * factor).astype(np.float32)


def simulate_lowfield(high_vol, snr_range=(6, 12), blur_range=(0.5, 1.2),
                      contrast_range=(0.6, 0.85)):
    """
    Full degradation pipeline: high-field → simulated 64mT low-field.

    Random parameters within ranges for augmentation diversity.
    Returns the degraded volume at LOW_FIELD_SHAPE resolution.
    """
    vol = high_vol.copy()

    # 1. Random contrast reduction
    contrast = np.random.uniform(*contrast_range)
    vol = reduce_contrast(vol, contrast)

    # 2. Gaussian blur (simulate lower gradient performance)
    blur_sigma = np.random.uniform(*blur_range)
    vol = gaussian_filter(vol, sigma=blur_sigma)

    # 3. Downsample to low-field resolution
    factors = [l / h for l, h in zip(LOW_FIELD_SHAPE, vol.shape)]
    vol = zoom(vol, factors, order=1)

    # 4. Add Rician noise at low resolution (where it matters most)
    snr = np.random.uniform(*snr_range)
    vol = add_rician_noise(vol, snr=snr)

    # 5. Clip and re-normalize
    vol = np.clip(vol, 0, None)
    vol = normalize(vol)

    return vol


def process_subject(nifti_path, output_dir, target_shape=TARGET_SHAPE):
    """
    Process a single IXI subject:
    1. Load high-field volume
    2. Resample to target shape (= competition high-field resolution)
    3. Generate degraded low-field version
    4. Upsample low-field to target shape (= competition preprocessing)
    5. Save paired data
    """
    basename = os.path.basename(nifti_path).replace('.nii.gz', '').replace('.nii', '')

    try:
        raw_vol = load_nifti(nifti_path)
    except Exception as e:
        print(f"  SKIP {basename}: {e}")
        return None

    # Skip very small or corrupted volumes
    if raw_vol.size < 100000 or raw_vol.max() < 1e-6:
        print(f"  SKIP {basename}: too small or empty")
        return None

    # Resample high-field to competition target shape
    high_vol = normalize(raw_vol)
    high_factors = [t / s for t, s in zip(target_shape, high_vol.shape)]
    high_resampled = zoom(high_vol, high_factors, order=3).astype(np.float32)
    high_resampled = normalize(high_resampled)

    # Generate synthetic low-field
    low_vol = simulate_lowfield(high_resampled)  # at LOW_FIELD_SHAPE

    # Upsample low-field to target shape (same as competition preprocessing)
    low_factors = [t / s for t, s in zip(target_shape, low_vol.shape)]
    low_upsampled = zoom(low_vol, low_factors, order=1).astype(np.float32)

    # Save
    out_path = os.path.join(output_dir, f"{basename}.npz")
    np.savez_compressed(out_path, low=low_upsampled, high=high_resampled)

    return basename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ixi_dir", required=True, help="Directory with IXI .nii.gz files")
    parser.add_argument("--output_dir", required=True, help="Output directory for paired data")
    parser.add_argument("--max_subjects", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefer_3t", action="store_true", default=True,
                        help="Prioritize 3T (Hammersmith) scans")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all NIfTI files
    patterns = [
        os.path.join(args.ixi_dir, "*.nii.gz"),
        os.path.join(args.ixi_dir, "*.nii"),
    ]
    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(p))
    all_files = sorted(all_files)
    print(f"Found {len(all_files)} NIfTI files")

    # Prioritize 3T scans (Hammersmith = "HH") since they're closest to competition
    if args.prefer_3t:
        hh_files = [f for f in all_files if "-HH-" in f]
        other_files = [f for f in all_files if "-HH-" not in f]
        all_files = hh_files + other_files
        print(f"  {len(hh_files)} from Hammersmith (3T), {len(other_files)} from other sites")

    files = all_files[:args.max_subjects]
    print(f"Processing {len(files)} subjects...")

    # Process in parallel
    success = 0
    fn = partial(process_subject, output_dir=args.output_dir)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, result in enumerate(executor.map(fn, files)):
            if result is not None:
                success += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(files)}] {success} successful")

    print(f"\nDone: {success}/{len(files)} subjects processed")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
