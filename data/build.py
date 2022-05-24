
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

from .datasets.mnist import MNIST
from .datasets.img2lmdb import ImageFolderLMDB

from .transforms import build_transforms

def build_mnist_dataset(transforms, mnist_path, is_train=True):
    datasets = MNIST(root=mnist_path, train=is_train, transform=transforms, download=True)
    return datasets

def build_lmdb_dataset(transforms, lmdb_path, is_train=True):
    train_data_transform = transforms
    if lmdb_path is None:
        print("lmdb path is empty\n")
    datasets = ImageFolderLMDB(lmdb_path, train_data_transform)
    num_class = datasets[-1][1] + 1

    return datasets, num_class

def build_img_dataset(transforms, img_path, is_train=True):
    train_data_transform = transforms
    if img_path is None:
        print("lmdb path is empty\n")
    datasets = ImageFolder(img_path, train_data_transform)
    num_class = datasets[-1][1] + 1

    return datasets, num_class

def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False
    
    transforms = build_transforms(cfg, is_train)

    datasets = None
    if cfg.DATASETS.TRAIN_DB_TYPE == "mnist" :
        datasets = build_mnist_dataset(transforms, "./", is_train)
    elif cfg.DATASETS.TRAIN_DB_TYPE == "lmdb" :
        datasets, _ = build_custom_dataset(transforms, cfg.DATASETS.TRAIN_LMDB_PATH, is_train)
    else:
        datasets, _ = build_img_dataset(transform, cfg.DATASETS.TRAIN_IMG_PATH, is_train)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader