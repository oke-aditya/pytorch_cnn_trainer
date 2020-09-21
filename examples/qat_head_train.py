# This uses models from torchvision.quantization.models
# Directly possible to train with this

import torchvision.models.quantization as models
from pytorch_cnn_trainer import engine
import config
import torch
from pytorch_cnn_trainer import utils
import torch.optim as optim
from tqdm import tqdm
import time
from pytorch_cnn_trainer import dataset
import torch.nn as nn


def create_combined_model(num_classes):
    # Just as an example it creates resnet18
    model_fe = models.resnet18(pretrained=True, progress=True, quantize=True)
    num_ftrs = model_fe.fc.in_features

    # Step 1. Isolate the feature extractor.
    model_fe_features = nn.Sequential(
        model_fe.quant,  # Quantize the input
        model_fe.conv1,
        model_fe.bn1,
        model_fe.relu,
        model_fe.maxpool,
        model_fe.layer1,
        model_fe.layer2,
        model_fe.layer3,
        model_fe.layer4,
        model_fe.avgpool,
        model_fe.dequant,
    )

    new_head = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_ftrs, num_classes))

    # Step 3. Combine, and don't forget the quant stubs.
    new_model = nn.Sequential(
        model_fe_features,
        nn.Flatten(1),
        new_head,
    )

    return new_model


if __name__ == "__main__":
    print(f"Setting Seed for the run, seed = {config.SEED}")
    utils.seed_everything(config.SEED)

    print("Creating Train and Validation Dataset")
    train_set, valid_set = dataset.create_cifar10_dataset(config.train_transforms, config.valid_transforms)

    print("Train and Validation Datasets Created")

    print("Creating DataLoaders")
    train_loader, valid_loader = dataset.create_loaders(train_set, train_set,
                                                        config.TRAIN_BATCH_SIZE, config.VALID_BATCH_SIZE,
                                                        config.NUM_WORKERS,)

    q_model = create_combined_model(config.NUM_ClASSES)
    # Currently the quantized models can only be run on CPU.
    # However, it is possible to send the non-quantized parts of the model to a GPU.
    device = "cpu"
    q_model = q_model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Note that we are only training the head.
    optimizer = optim.SGD(q_model.parameters(), lr=0.01, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    early_stopper = utils.EarlyStopping(patience=3, verbose=True, path=config.SAVE_PATH)

    history = engine.fit(config.EPOCHS, q_model, train_loader,
                         valid_loader, criterion, device, optimizer,
                         exp_lr_scheduler, early_stopper, num_batches=50,)
    print("Done")
