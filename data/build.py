
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from datasets.img2lmdb import ImageFolderLMDB
from torchvision.datasets import ImageFolder
from pathlib import Path
import bcolz
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import sys
from skimage import io, transform

from datasets.mnist import MNIST
from transforms import build_transforms

sys.path.append('../')
from config import cfg

def build_mnist_dataset(transforms, mnist_path, is_train=True):
    datasets = MNIST(root=mnist_path, train=is_train, transform=transforms, download=True)
    return datasets

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
        datasets = build_mnist_dataset(transforms, "./", is_train)
    elif cfg.train_dataset_type == "lmdb" :
        datasets, _ = build_lmdb_dataset(transforms, cfg.train_dataset_dir, cfg.train_dataset, is_train)
    else:
        datasets, _ = build_img_dataset(transform, cfg.train_dataset_dir, is_train)

    num_workers = cfg.num_workers
    data_loader = DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader


if __name__ == "__main__":

    dataloader = make_data_loader(cfg, False)

    print(dataloader)