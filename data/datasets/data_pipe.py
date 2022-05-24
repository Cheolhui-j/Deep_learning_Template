import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from utils.img2lmdb import ImageFolderLMDB
from torchvision.datasets import ImageFolder
from pathlib import Path
import bcolz
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from skimage import io, transform

# ----------------------------------------- Get train data -----------------------------------------

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data_info = pd.read_csv(csv_path, header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # image
        img_path = os.path.join(self.root_dir, self.image_arr[idx])
        image = io.imread(img_path)
        image = self.transform(image)

        #label
        label = self.label_arr[idx]

        return (image, label, img_path)  

def get_custom_data_from_image(csv_path, root_dir):
    """ get the train data and create the dataloader """

    # ============================================
    # train data transformation
    # ============================================
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((56, 56)), # ***
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # ============================================
    # load trainset
    # ============================================
    dataset = CustomDatasetFromImages(csv_path=csv_path, root_dir=root_dir, transform=data_transform)
    num_class = dataset[-1][1] + 1

    return dataset, num_class

def get_img_train_data(imgs_folder):
    """ get the train data and create the dataloader """

    # ============================================
    # train data transformation
    # ============================================
    train_data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # ============================================
    # load trainset
    # ============================================
    trainset = ImageFolder(imgs_folder, train_data_transform)
    num_class = trainset[-1][1] + 1

    return trainset, num_class

def get_lmdb_train_data(lmdb_folder):
    """ get the train data and create the dataloader """

    # ============================================
    # train data transformation
    # ============================================
    train_data_transform = transforms.Compose([
        #transforms.Resize((56, 56)), # ***
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # ============================================
    # load trainset
    # ============================================
    LMDB_path = lmdb_folder + '.lmdb'
    trainset = ImageFolderLMDB(LMDB_path, transform=train_data_transform)
    num_class = trainset[-1][1] + 1

    return trainset, num_class

# --------------------------------------------------------------------------------------------------


# --------------------------------------- Get validation data --------------------------------------

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = path/name, mode='r')
    issame = np.load(path/'{}_list.npy'.format(name))
    return carray, issame


def get_val_data(data_path, val_target):
    """ get the validation data and create the dataloader """

    data_path = Path(data_path)
    val_name, val_issame = get_val_pair(data_path, val_target)

    return val_name, val_issame

# --------------------------------------------------------------------------------------------------
