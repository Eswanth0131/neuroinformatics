from extract_slices import create_submission_df, load_nifti, base64_to_slice
from metric import score
import os
import re
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import nibabel as nib
from bids import BIDSLayout
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_dataframe(train_folder_path = 'data/kaggle/train'):
    # NOTE: There are no target labels for the kaggle test dataset
    """
    For a given slice in the low field MRI, the pandas dataframe of this dataset matches with one picture in the high field MRI. There are more slices in the high field than the low field in the kaggle dataset. Run "python main.py" to see.

    Example: index z=0 in low field is matched with z=0 in heigh field, index z=1 in low field is matched with z=5 in heigh field, ... 
    """
    high_dir = os.path.join(train_folder_path, 'high_field')
    low_dir = os.path.join(train_folder_path, 'low_field')
    
    # high_paths = os.listdir(high_dir)
    low_files = os.listdir(low_dir)

    sample_l = []
    layer_l = []
    img_l = []

    sample_h = []
    layer_h = []
    img_h = []

    for low_file in tqdm(low_files, position=0):
        l_file_path = os.path.join(low_dir, low_file)
        # sample_l.append(re.findall(r'(\d+)', low_file)[0])
        temp_low_img = load_nifti(l_file_path)
        for low_z in tqdm(range(temp_low_img.shape[2]), position=1, leave=False):
            # Store the input in dataframe format
            sample_l.append(re.findall(r'(\d+)', low_file)[0])
            layer_l.append(low_z)
            img_l.append(temp_low_img[:, :, low_z])

            # Find corresponding high sample input
            high_file = low_file.replace('low', 'high')
            h_file_path = os.path.join(high_dir, high_file)
            temp_high_img = load_nifti(h_file_path)
            sample_h.append(re.findall(r'(\d+)', high_file)[0])
            # NOTE: This controls the mapping of input to target
            high_z = temp_high_img.shape[2] // temp_low_img.shape[2] * low_z
            layer_h.append(high_z)
            img_h.append(temp_high_img[:, :, high_z])

    train_df = {
        'sample_l' : sample_l,
        'layer_l' : layer_l,
        'img_l' : img_l,

        'sample_h' : sample_h,
        'layer_h' : layer_h,
        'img_h' : img_h,
    }

    return pd.DataFrame(train_df)

class myDataset(Dataset):
    def __init__(self, dataframe:pd.DataFrame, mode:str = 'train', train_test_split = [9, 1]):
        """
        mode : valid values is 'test' or 'train'
        """
        self.dataset = dataframe
        self.mode = mode
        self.train_test_split = train_test_split

    def __len__(self):
        # train 90 test 10 split
        if self.mode == 'train':
            return self.dataset.shape[0] // 10 * self.train_test_split[0]
        else:
            return self.dataset.shape[0] // 10 * self.train_test_split[1]

    def __getitem__(self, index):
        # input, target
        return self.dataset.iloc[index]['img_l'], self.dataset.iloc[index]['img_h']


if __name__ == "__main__":
    """
    'python main.py' to just see 1 low image vs 5 high images
    'python main.py 1' to just see 1 low image vs 1 high images randomly, after matching
    'python main.py 2' to just see 1 low image vs 1 high images randomly, with force reloading data
    'python main.py 3' to run piplines
    """
    print(sys.argv)
    if len(sys.argv) < 2:
        sample = '13'
        fig, ax = plt.subplots(2, 3)
        low_path = f'./data/kaggle/train/low_field/sample_0{sample}_lowfield.nii'
        high_path = f'./data/kaggle/train/high_field/sample_0{sample}_highfield.nii'
        low_img = load_nifti(low_path)
        high_img = load_nifti(high_path)
        # low_mid = low_img.shape[2] // 2
        # high_mid = high_img.shape[2] // 2
        low_mid = 10
        high_mid = int(high_img.shape[2] / low_img.shape[2] * low_mid)
        # ax[0, 0].imshow(low_img[:, :, low_mid], cmap='gray')
        ax[0, 0].imshow(low_img[:, :, low_mid])
        for i in range(5):
            # ax[((1 + i) // 3) % 2, (1 + i) % 3].imshow(high_img[:, :, high_mid + i], cmap='gray')
            ax[((1 + i) // 3) % 2, (1 + i) % 3].imshow(high_img[:, :, high_mid + i])

        print(f"low field image shape: {low_img.shape}\nhigh field image shape: {high_img.shape}\nlow field index: {low_mid}\nhigh field index: {high_mid}")
        plt.show()
        sys.exit(0)

    elif len(sys.argv) == 2:
        # NOTE: There are no target labels for the kaggle test dataset
        data_path = './data.pkl'
        print(os.getcwd(), not os.path.exists(data_path))
        if not os.path.exists(data_path) or sys.argv[1] == 2:
            data_df = create_dataframe()
            data_df.to_pickle(data_path)
        else:
            data_df = pd.read_pickle(data_path)

        if sys.argv[1] == 1 or sys.argv[1] == 2:
            train_dataset = myDataset(data_df)
            test_dataset = myDataset(data_df, 'test')
            
            train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

            imgs, targets = next(iter(train_loader))
            fig, ax = plt.subplots(2)
            ax[0].imshow(imgs[0])
            ax[1].imshow(targets[0])
            plt.show()

        elif sys.argv[1] == 3:

            pass