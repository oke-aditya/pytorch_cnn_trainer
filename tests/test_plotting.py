import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytest
from tqdm import tqdm
import timm
import torchvision.transforms as T
from torch.cuda import amp
from pytorch_cnn_trainer import dataset
from pytorch_cnn_trainer import model_factory
from pytorch_cnn_trainer import utils
from pytorch_cnn_trainer import engine
from torch.optim.swa_utils import SWALR
from pytorch_cnn_trainer import plotter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_plotting():
    print("Creating Train and Validation Dataset")
    train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    valid_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    train_set, valid_set = dataset.create_cifar10_dataset(
        train_transforms, valid_transforms
    )
    print("Train and Validation Datasets Created")

    print("Creating DataLoaders")
    train_loader, valid_loader = dataset.create_loaders(train_set, train_set)

    print("Train and Validation Dataloaders Created")
    print("Creating Model")

    all_supported_models = [
        "resnet18",
        # "resnet34",
        # "resnet50",
        # "resnet101",
        # "resnet152",
        # "resnext50_32x4d",
        # "resnext101_32x8d",
        # "vgg11",
        # "vgg13",
        # "vgg16",
        # "vgg19",
        # "mobilenet",
        # "mnasnet0_5",
        # "mnasnet1_0",
    ]

    for model_name in all_supported_models:
        model = model_factory.create_torchvision_model(
            model_name, num_classes=10, pretrained=False
        )  # We don't need pretrained True, we just need a forward pass

        if torch.cuda.is_available():
            print("Model Created. Moving it to CUDA")
        else:
            print("Model Created. Training on CPU only")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        criterion = (
            nn.CrossEntropyLoss()
        )  # All classification problems we need Cross entropy loss

        # early_stopper = utils.EarlyStopping(
        #     patience=7, verbose=True, path=SAVE_PATH
        # )
        # We do not need early stopping too

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

        swa_scheduler = SWALR(
            optimizer, anneal_strategy="linear", anneal_epochs=20, swa_lr=0.05
        )
        swa_start = 2
        epoch = 5

    history = engine.fit(epoch, model, train_loader, valid_loader, criterion, device, optimizer)
    plotter.plot_results(history, metric1='loss', metric2='top5_acc')
    return 1
