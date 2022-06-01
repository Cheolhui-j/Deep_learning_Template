import os
import os.path as osp
from PIL import Image
import six
import string

import lmdb
import pickle
import tqdm
import pyarrow as pa
import pandas as pd

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets

import numpy as np
from skimage import io, transform

class ImageFolderCSV(data.Dataset):
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

        return image, label  