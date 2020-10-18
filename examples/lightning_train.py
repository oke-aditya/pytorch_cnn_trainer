import pytorch_lightning as pl
import torch
import torchvision
from pytorch_cnn_trainer import dataset
from pytorch_cnn_trainer.lightning_trainer import CNN
import config

if __name__ == "__main__":
    pl.seed_everything(seed=42)
    # print(model)
    print("Creating Train and Validation Dataset")
    train_set, valid_set = dataset.create_cifar10_dataset(config.train_transforms, config.valid_transforms)
    print("Train and Validation Datasets Created")

    print("Creating DataLoaders")
    train_loader, valid_loader = dataset.create_loaders(train_set, train_set, config.TRAIN_BATCH_SIZE,
                                                        config.VALID_BATCH_SIZE,config.NUM_WORKERS,)

    print("Train and Validation Dataloaders Created")

    print("Creating Model")
    model = CNN("resnet18", num_classes=10, pretrained=True)
    # print(model)
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model, train_loader, valid_loader)



