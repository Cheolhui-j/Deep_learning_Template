
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from .datasets.img2lmdb import ImageFolderLMDB
from torchvision.datasets import ImageFolder
from pathlib import Path
import bcolz
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import sys
from skimage import io, transform

from .datasets.mnist import MNIST
from .transforms import build_transforms

sys.path.append('../')
from config import cfg

def build_mnist_dataset(transforms, mnist_path, is_train=True):
    datasets = MNIST(root=mnist_path, train=is_train, transform=transforms, download=True)
    num_class = datasets[-1][1] + 1
    return datasets, num_class

def build_lmdb_dataset(transforms, lmdb_path, lmdb_name, is_train=True):
    train_data_transform = transforms
    if lmdb_path is None:
        print("lmdb path is empty\n")
    lmdb_path = os.path.join(lmdb_path, lmdb_name + '.lmdb')
    datasets = ImageFolderLMDB(lmdb_path, train_data_transform)
    num_class = datasets[-1][1] + 1

    return datasets, num_class

def build_img_dataset(transforms, img_path, is_train=True):
    train_data_transform = transforms
    if img_path is None:
        print("lmdb path is empty\n")
    img_path = os.path.join(img_path, 'imgs')
    datasets = ImageFolder(img_path, train_data_transform)
    num_class = datasets[-1][1] + 1

    return datasets, num_class

def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.batch_size
        shuffle = True
    else:
        batch_size = cfg.batch_size
        shuffle = False
    
    transforms = build_transforms(cfg, is_train)

    datasets = None
    # train_data_path = os.path.join(cfg.train_dataset_dir, cfg.train_dataset)
    if cfg.train_dataset_type == "mnist" :
        datasets, num_class = build_mnist_dataset(transforms, "./", is_train)
    elif cfg.train_dataset_type == "lmdb" :
        datasets, num_class = build_lmdb_dataset(transforms, cfg.train_dataset_dir, cfg.train_dataset, is_train)
    else:
        datasets, num_class = build_img_dataset(transform, cfg.train_dataset_dir, is_train)

    num_workers = cfg.num_workers
    data_loader = DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader, num_class

# Get validation data 

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = path/name, mode='r')
    issame = np.load(path/'{}_list.npy'.format(name))
    return carray, issame


def get_val_data(data_path, val_target):
    """ get the validation data and create the dataloader """

    data_path = Path(data_path)
    val_name, val_issame = get_val_pair(data_path, val_target)

    return val_name, val_issame

if __name__ == "__main__":

    dataloader, num_class = make_data_loader(cfg, False)
    print(dataloader, num_class)

    val_dataset = []
    val_labels = []
    for val_name in cfg.val_dataset:
        val_data, val_label = get_val_data(cfg.val_dataset_dir, val_name)
        val_dataset.append(val_data)
        val_labels.append(val_label)
        print(val_name, val_data, val_label)   