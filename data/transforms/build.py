import torchvision.transforms as T

from .transfroms import RandomErasing

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if is_train:
        transform = T.Compose(
            [
                T.Resize((112, 112)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize((112, 112)),
                T.ToTensor(),
                normalize_transform
            ]
        )

    return transform 