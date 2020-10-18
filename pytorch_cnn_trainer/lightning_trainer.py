# # Do the same thing with Lightning
# # We can use the config, model and dataset
# # We need to redefine the engine and train

import pytorch_lightning as pl
import torch
from pytorch_cnn_trainer.model_factory import _create_torchvision_backbone
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning.metrics.functional import accuracy

__all__ = ["CNN"]


class CNN(pl.LightningModule):
    """
    Creates a CNN which can be fine-tuned.
    """
    def __init__(self, pretrained_backbone: str,
                 learning_rate: float = 0.0001, num_classes: int = 100,
                 pretrained: bool = True, **kwargs,):
        super().__init__()

        self.num_classes = num_classes
        self.bottom, self.out_channels = _create_torchvision_backbone(pretrained_backbone, num_classes, pretrained=pretrained)
        self.top = nn.Linear(self.out_channels, self.num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.bottom(x)
        x = self.top(x.view(-1, self.out_channels))
        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        train_loss = F.cross_entropy(outputs, targets, reduction='sum')
        # Possible we can compute top-1 and top-5 accuracy here.
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        val_loss = F.cross_entropy(outputs, targets, reduction='sum')
        # Possible we can compute top-1 and top-5 accuracy here.
        return {"loss": val_loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


