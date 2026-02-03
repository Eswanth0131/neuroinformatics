from extract_slices import create_submission_df, load_nifti, base64_to_slice
from metric import score
import os
import sys
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

class myDataset(Dataset):
    def __init__(self):
        super().__init__(self, )

if __name__ == "__main__":

    if len(sys.argv) < 2:
        sample = '01'
        fig, ax = plt.subplots(1, 2)
        low_path = f'./data/kaggle/train/low_field/sample_0{sample}_lowfield.nii'
        high_path = f'./data/kaggle/train/high_field/sample_0{sample}_highfield.nii'
        low_img = load_nifti(low_path)
        high_img = load_nifti(high_path)
        low_mid = low_img.shape[2] // 2
        high_mid = high_img.shape[2] // 2
        ax[0].imshow(low_img[:, :, low_mid])
        ax[1].imshow(high_img[:, :, high_mid])

        print(low_img.shape, high_img.shape, low_mid, high_mid)
        plt.show()
        sys.exit(0)

    
