# TODO add ImageFolder dataset
# Add code for Mapping using dataframe containaing id and target.

import torchvision
from torchvision import datasets
from torch.utils.data import Dataset
import os
import torch
from PIL import Image

__all__ = [
    "create_fashion_mnist_dataset",
    "create_folder_dataset",
    "create_loaders",
    "CSVDataset",
]


def create_fashion_mnist_dataset(train_transforms, valid_transforms):
    """Creates Fashion MNIST train dataset and a test dataset.
    Args:
    train_transforms: Transforms to be applied to train dataset.
    test_transforms: Transforms to be applied to test dataset.
    """

    # This code can be re-used for other torchvision Image Dataset too.
    train_set = torchvision.datasets.FashionMNIST(
        "./data", download=True, train=True, transform=train_transforms
    )

    valid_set = torchvision.datasets.FashionMNIST(
        "./data", download=True, train=False, transform=valid_transforms
    )

    return train_set, valid_set


def create_cifar10_dataset(train_transforms, valid_transforms):
    """Creates CIFAR10 train dataset and a test dataset.
    Args:
    train_transforms: Transforms to be applied to train dataset.
    test_transforms: Transforms to be applied to test dataset.
    """
    # This code can be re-used for other torchvision Image Dataset too.
    train_set = torchvision.datasets.CIFAR10(
        "./data", download=True, train=True, transform=train_transforms
    )

    valid_set = torchvision.datasets.CIFAR10(
        "./data", download=True, train=False, transform=valid_transforms
    )

    return train_set, valid_set


def create_folder_dataset(root_dir, transforms, split: float = 0.8, **kwargs):
    """
    Creates Train and Validation Dataset from a Root folder
    Arrange dataset as follows: -
    root/class_a/image01.png
    root/class_b/image01.png

    Creates train and validation dataset from this root dir.
    This applies same transforms to both train and validation

    Args:
    root_dir : Root directory of the dataset to read from
    transforms: Transforms to be applied to train and validation datasets.
    split: Float number denoting percentage of train items

    """
    complete_dataset = datasets.ImageFolder(root_dir, transform=transforms)
    train_split = len(complete_dataset) * split
    valid_split = len(complete_dataset) * (1 - split)

    train_set, valid_set = torch.utils.data.random_split(
        complete_dataset, [train_split, valid_split]
    )

    return train_set, valid_set


def create_loaders(
    train_dataset,
    valid_dataset,
    train_batch_size=32,
    valid_batch_size=32,
    num_workers=1,
    **kwargs
):
    """
    Creates train loader and test loader from train and test datasets
    Args:
        train_dataset: Torchvision train dataset.
        valid_dataset: Torchvision valid dataset.
        train_batch_size (int) : Default 32, Training Batch size
        valid_batch_size (int) : Default 32, Validation Batch size
        num_workers (int) : Defualt 1, Number of workers for training and validation.
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset, train_batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, valid_batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, valid_loader


class CSVDataset(Dataset):
    """
    Creates Torchvision Dataset From CSV File.
    Args:
        df: DataFrame with a column 2 columns image_id and target
        data_dir: Directory from where data is to be read.
        target: target column name
        transform: Trasforms to apply while creating Dataset.
    """

    def __init__(self, df, data_dir, target, transform):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.image_id[idx]
        label = self.df.target[idx]

        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path)
        image = self.transform(image)

        return image, label


# if __name__ == "__main__":
#     train_set, valid_set = create_fashion_mnist_dataset(config.train_transforms, config.valid_transforms)
