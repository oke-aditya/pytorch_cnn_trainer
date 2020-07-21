import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as T
from pytorch_cnn_trainer import dataset
from pytorch_cnn_trainer import model_factory
import config
from pytorch_cnn_trainer import utils
from pytorch_cnn_trainer import engine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print("Setting Seed for the run, seed = {}".format(config.SEED))
    utils.seed_everything(config.SEED)

    print("Creating Train and Validation Dataset")
    train_set, valid_set = dataset.create_cifar10_dataset(
        config.train_transforms, config.valid_transforms
    )
    print("Train and Validation Datasets Created")

    print("Creating DataLoaders")
    train_loader, valid_loader = dataset.create_loaders(
        train_set,
        train_set,
        config.TRAIN_BATCH_SIZE,
        config.VALID_BATCH_SIZE,
        config.NUM_WORKERS,
    )

    print("Train and Validation Dataloaders Created")
    print("Creating Model")

    # model = model.create_timm_model(
    #     config.MODEL_NAME,
    #     num_classes=config.NUM_ClASSES,
    #     in_channels=config.IN_CHANNELS,
    #     pretrained=config.PRETRAINED,
    # )
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
            model_name, num_classes=config.NUM_ClASSES, pretrained=config.PRETRAINED,
        )

        if torch.cuda.is_available():
            print("Model Created. Moving it to CUDA")
        else:
            print("Model Created. Training on CPU only")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        # Optionially a schedulear
        # scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=1e-4, max_lr=1e-3, mode="min")

        criterion = (
            nn.CrossEntropyLoss()
        )  # All classification problems we need Cross entropy loss

        early_stopper = utils.EarlyStopping(
            patience=7, verbose=True, path=config.SAVE_PATH
        )
        # history = engine.fit(
        #     epochs=2,
        #     model=model,
        #     train_loader=train_loader,
        #     valid_loader=valid_loader,
        #     criterion=criterion,
        #     device=device,
        #     optimizer=optimizer,
        #     early_stopper=early_stopper,
        # )
        history = engine.santity_fit(model, train_loader, valid_loader, criterion, device, 128)

    print("Done !!")
