
from torch.utils import data

from .datasets.mnist import MNIST
from .datasets.custom_dataloader_from_lmdb import ImageFolderLMDB

from .transforms import build_transforms

def build_dataset(transforms, mnist_path, is_train=True):
    datasets = MNIST(root=mnist_path, train=is_train, transform=transforms, download=True)
    return datasets

def build_custom_dataset(transforms, lmdb_path, is_train=True):
    train_data_transform = transforms
    if lmdb_path is None:
        print("lmdb path is empty\n")
    LMDB_path = lmdb_path
    datasets = ImageFolderLMDB(LMDB_path, train_data_transform)
    num_class = datasets[-1][1] + 1

    return datasets

def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False
    
    transforms = build_transforms(cfg, is_train)

    datasets = None
    if cfg.DATASETS.TRAIN_DB_TYPE == "custom":
        datasets = build_custom_dataset(transforms, "", is_train)
    else :
        datasets = build_dataset(transforms, "", is_train)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader