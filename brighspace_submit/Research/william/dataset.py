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

import torch.nn.functional as F

def whole_img_df(strength_type:str, image_file_paths):
    """
    Kaggle
    Create the dataframe from a directory 
    image_file_paths: (data/kaggle/train/low_field, data/kaggle/train/high_field)
    strength_type: field strength type(low, high)
    """
    sample = []
    img = []
    for f in tqdm(image_file_paths):
        # print(f)
        sample.append(
            re.findall(r'(\d+)', f)[0]
        )
        img.append(
            load_nifti(f)
        )

    # print(len(sample), len(img))

    train_df = {
        'sample' : sample,
        f'img_{strength_type}' : img,
    }
    return pd.DataFrame(train_df)

def create_3D_Image_DF(
        input_path='data/kaggle/train/low_field',
        target_path='data/kaggle/train/high_field', 
        input_to_target = ('low', 'high'),
    ):
    """
    Kaggle
    Combines the dataframes, input and target dataframes
    """
    input_files = os.listdir(input_path)
    input_file_paths = []
    target_file_paths = []
    for f in input_files:
        input_file_paths.append(os.path.join(input_path, f))
        target_file_paths.append(os.path.join(target_path, f.replace(*input_to_target)))
        # print(f"{f}\n{f.replace(*input_to_target)}\n")

    input_df = whole_img_df('low', input_file_paths)
    target_df = whole_img_df('high', target_file_paths)

    final_df = input_df.merge(target_df,  on='sample')
    return final_df

def create_dataframe(train_folder_path = 'data/kaggle/train'):
    # NOTE: There are no target labels for the kaggle test dataset
    """
    ONLY FOR THE KAGGLE DATASET

    For a given slice in the low field MRI, the pandas dataframe of this dataset matches with one picture in the high field MRI. There are more slices in the high field than the low field in the kaggle dataset. Run "python main.py gen" to see.

    Example: index z=0 in low field is matched with z=0 in heigh field, index z=1 in low field is matched with z=5 in heigh field, ... 

    update: For every x,y input image, we get x,y,5 size target image as well
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
            img_h.append(temp_high_img[:, :, high_z:high_z + 5])

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
        self.last_input = None
        self.last_target = None

    def __len__(self):
        # train 90 test 10 split
        if self.mode == 'train':
            return self.dataset.shape[0] // 10 * self.train_test_split[0]
        else:
            return self.dataset.shape[0] // 10 * self.train_test_split[1]

    def __getitem__(self, index):
        data_row = self.dataset.iloc[index]
        # input, target
        input_img = data_row['img_l']
        target_img = data_row['img_h']
        input_img = input_img[np.newaxis, :]
        # print(target_img.shape)
        # NOTE: TRANSPOSE HERE: purpose for submission format vs model input
        target_img = target_img.transpose((2, 0, 1))

        return input_img.astype(np.float32), target_img.astype(np.float32)

class SuperFormer_Dataset(Dataset):
    def __init__(self, dataframe:pd.DataFrame, mode:str = 'train', train_test_split = [.9, .1]):
        """
        mode : valid values is 'test' or 'train'
        """
        self.dataset = dataframe
        self.mode = mode
        self.train_test_split = train_test_split

    def __len__(self):
        # train 90 test 10 split
        if self.mode == 'train':
            return int(self.dataset.shape[0] * self.train_test_split[0])
        else:
            return int(self.dataset.shape[0] * self.train_test_split[1])

    def __getitem__(self, index):
        flag = True
        if flag:
            data_row = self.dataset.iloc[index]
            # input, target
            input_img = data_row['img_low']
            target_img = data_row['img_high']

            input_img = input_img.astype(np.float32)
            target_img = target_img.astype(np.float32)

            # Normalize input
            input_img = ((input_img - input_img.min()) / (input_img.max() - input_img.min()))
            target_img = ((target_img - target_img.min()) / (target_img.max() - target_img.min()))

            # input_img = input_img[np.newaxis, :]
            # target_img = target_img[np.newaxis, :]
            # print(target_img.shape)
            # NOTE: TRANSPOSE HERE: purpose for submission format vs model input
            # target_img = target_img.transpose((2, 0, 1))

            # print(input_img.shape)
            # input_img = F.interpolate(input_img, size=(179, 221), mode='bicubic')
            
            # input_img = input_img[:, :, np.newaxis,]
            # target_img = target_img[:, :, np.newaxis,]

            input_img = torch.from_numpy(input_img)
            target_img = torch.from_numpy(target_img)
            # print(f"input_img: {input_img.size()}\ntarget_img: {target_img.size()}")

            return input_img, target_img
        # else:
        #     image_df = self.dataset[self.dataset['sample_l'] == f'{index:03d}']
        #     print(image_df)

        #     final_temp_image = np.array([])
        #     for i in range(image_df.shape[0]):
        #         final_temp_image.
        #     pass

class image_3D_dataset(Dataset):
    def __init__(self, 
                 dataframe:pd.DataFrame, 
                 mode:str = 'train', 
                 train_test_split = [.9, .1],
                channels = 1
    ):
        """
        mode : valid values is 'test' or 'train'
        """
        self.dataset = dataframe
        self.mode = mode
        self.train_test_split = train_test_split
        self.channels = channels

    def __len__(self):
        return int(self.dataset.shape[0])
        # train 90 test 10 split
        if self.mode == 'train':
            return int(self.dataset.shape[0] * self.train_test_split[0])
        else:
            return int(self.dataset.shape[0] * self.train_test_split[1])
        

    def __getitem__(self, index):
        flag = True
        if flag:
            data_row = self.dataset.iloc[index]
            # input, target
            input_img = data_row['img_low']
            target_img = data_row['img_high']

            # Change Datatype
            input_img = input_img.astype(np.float32)
            target_img = target_img.astype(np.float32)

            # Normalize input
            input_img = ((input_img - input_img.min()) / (input_img.max() - input_img.min()))
            target_img = ((target_img - target_img.min()) / (target_img.max() - target_img.min()))

            # Add channels
            if self.channels:
                input_img = input_img[np.newaxis, ...]
                target_img = target_img[np.newaxis, ...]

            # From Numpy to torch
            input_img = torch.from_numpy(input_img)
            target_img = torch.from_numpy(target_img)
            # print(f"input_img: {input_img.size()}\ntarget_img: {target_img.size()}")

            return input_img, target_img

if __name__ == "__main__":
    remote = True
    data_df = None
    # NOTE: There are no target labels for the kaggle test dataset

    if sys.argv[1] == 'gen1':
        data_path = './data/misc/data.pkl'
        data_df = create_dataframe()
        data_df.to_pickle(data_path)

    if sys.argv[1] == 'gen2':
        data_path = './data/misc/data3D.pkl'
        data_df = create_3D_Image_DF()
        data_df.to_pickle(data_path)

    elif sys.argv[1] == 'view':
        data_df = pd.read_pickle(data_path)

        if sys.argv[2] == '1':
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

            plt.savefig('./images/many_slice.png') if remote else plt.show()

        if sys.argv[2] == '2':
            train_dataset = myDataset(data_df)
            test_dataset = myDataset(data_df, 'test')
            
            train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

            imgs, targets = next(iter(train_loader))
            fig, ax = plt.subplots(2)
            ax[0].imshow(imgs[0][0, :, :])
            ax[1].imshow(targets[0][0, :, :])

            plt.savefig('./images/dataloader_img.png') if remote else plt.show()

        if sys.argv[2] == 'result':
            pass

    elif sys.argv[1] == 'info':
        data_df = pd.read_pickle(data_path)

        train_dataset = myDataset(data_df)
        test_dataset = myDataset(data_df, 'test')
        
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
        info = enumerate(train_loader)
        for _ in range(3):
            batch, (X, y) = next(info)
            # X, y = z
            print(batch)
            print(f"X Shape (input): {X.shape}\ny Target (target): {y.shape}")
            print(f"type: {type(X)}")
            # test_sample = ((X - X.min()) / (X.max() - X.min()))
            test_sample = X
            # test_sample = torch.nn.functional.normalize(X, dim=(0, 1, 2, 3))
            # test_index = test_sample >= 240
            # print(test_sample[test_index])
            print(test_sample.max(), test_sample.min())

    elif sys.argv[1] == 'super':
        # data_df = pd.read_pickle(data_path)
        data_df = create_3D_Image_DF()
        print(data_df.shape, data_df.columns)


        train_dataset = SuperFormer_Dataset(data_df)
        test_dataset = SuperFormer_Dataset(data_df, 'test')
        
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
        for batch, (X, y) in enumerate(train_loader):
            # X, y = z
            print(batch)
            print(f"X Shape (input): {X.shape}\ny Target (target): {y.shape}")
            print(f"type: {type(X)}")
            # test_sample = ((X - X.min()) / (X.max() - X.min()))
            test_sample = X
            # test_sample = torch.nn.functional.normalize(X, dim=(0, 1, 2, 3))
            # test_index = test_sample >= 240
            # print(test_sample[test_index])
            print(f"max: {test_sample.max()}, min: {test_sample.min()}")

    elif sys.argv[1] == 'resnet':
        # data_df = pd.read_pickle(data_path)
        data_df = create_3D_Image_DF()
        print(data_df.shape, data_df.columns)

        train_dataset = image_3D_dataset(data_df)
        test_dataset = image_3D_dataset(data_df, 'test')
        
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

        for batch, (X, y) in enumerate(train_loader):
            # X, y = z
            print(batch)
            print(f"X Shape (input): {X.shape}\ny Target (target): {y.shape}")
            print(f"type: {type(X)}")
            test_sample = X
            print(f"max: {test_sample.max()}, min: {test_sample.min()}")

    elif sys.argv[1] == 'df':
        data_df = pd.read_pickle(data_path)

        print(data_df.columns)
        print(data_df.head(6))
