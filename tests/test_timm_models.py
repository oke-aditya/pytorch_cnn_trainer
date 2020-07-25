import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytest
from tqdm import tqdm
import timm
import torchvision.transforms as T
from pytorch_cnn_trainer import dataset
from pytorch_cnn_trainer import model_factory
from pytorch_cnn_trainer import utils
from pytorch_cnn_trainer import engine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_models():
    # print("Setting Seed for the run, seed = {}".format(SEED))
    # utils.seed_everything(SEED)
    # We don't need seeds for tests

    MODEL_NAME = "efficientnet_b0"  # For now a very small model

    print("Creating Train and Validation Dataset")
    train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    valid_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    # Test defaults
    train_set, valid_set = dataset.create_cifar10_dataset(
        train_transforms, valid_transforms
    )
    print("Train and Validation Datasets Created")

    print("Creating DataLoaders")
    train_loader, valid_loader = dataset.create_loaders(train_set, train_set)

    print("Train and Validation Dataloaders Created")
    print("Creating Model")

    # Right method and complete check would be
    # for model in timm.list_models() and then do this. It willl go out of github actions limits.
    model = model_factory.create_timm_model(
        MODEL_NAME, num_classes=10, in_channels=3, pretrained=False,
    )

    if torch.cuda.is_available():
        print("Model Created. Moving it to CUDA")
    else:
        print("Model Created. Training on CPU only")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criterion = (
        nn.CrossEntropyLoss()
    )  # All classification problems we need Cross entropy loss.

    # We do not need early stopping too
    history = engine.sanity_fit(
        model,
        train_loader,
        valid_loader,
        criterion,
        device,
        num_batches=10,
        grad_penalty=True,
    )

    history2 = engine.fit(
        1,
        model,
        train_loader,
        valid_loader,
        criterion,
        device,
        optimizer,
        num_batches=10,
        grad_penalty=True,
    )
    print("Done")
    return 1
