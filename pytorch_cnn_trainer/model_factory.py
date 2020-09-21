# We can use any model timm models or torchvision models.
# Model should have AdaptiveAvgPool2d layer and then Linear Layer
# We will remove the Linear Layer and create a new Linear Layer with num_classes

import timm
import torchvision
import torch.nn as nn

__all__ = ["create_timm_model", "create_torchvision_model", "fine_tune_torchvision"]


def create_timm_model(
    model_name: str,
    num_classes: int,
    in_channels: int = 3,
    pretrained: bool = True,
):
    """
    Creates a model from PyTorch Image Models repository.
    To know which models are supported print(timm.list_models())
    Args:
        model_name (str) : Name of the model. E.g. efficientnet_b3
        num_classes (int) : Number of classes for classification.
        in_channels (int) : Defualt 3. Number of channels of images, 1 for grayscale, 3 for RGB
        pretrained (bool) : If true uses modelwweights pretrained on ImageNet.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    return model


# Use this when adaptive avg_pooling is not available or you want to pass custom CNN
# This can work with adaptive case too. So it is a bit generic case.
def _create_backbone_generic(model, out_channels: int):
    # Here out_channels is a mandatory argument.
    modules_total = list(model.children())
    modules = modules_total[:-1]
    ft_backbone = nn.Sequential(*modules)
    ft_backbone.out_channels = out_channels
    return ft_backbone


# Use this when you have Adaptive Pooling layer in End.
# When Model.features is not applicable.
def _create_backbone_adaptive(model, out_channels: int = None):
    # Out channels is optional can pass it if user knows it
    # print(list(model.children())[-1].in_features)
    if out_channels is None:
        # Tries to infer the out_channels from layer before adaptive avg pool.
        modules_total = list(model.children())
        out_channels = modules_total[-1].in_features
    return _create_backbone_generic(model, out_channels=out_channels)


# Use this when model.features function is available as in mobilenet and vgg
# Again it is mandatory for out_channels here
def _create_backbone_features(model, out_channels: int):
    ft_backbone = model.features
    ft_backbone.out_channels = out_channels
    return ft_backbone


def _create_torchvision_backbone(
    model_name: str, num_classes: int, pretrained: bool = True
):
    """
    Creates CNN backbone from Torchvision.
    Args:
        model_name (str) : Name of the model. E.g. resnet18
        num_classes (int) : Number of classes for classification.
        pretrained (bool) : If true uses modelwweights pretrained on ImageNet.
    """
    if model_name == "mobilenet":
        mobile_net = torchvision.models.mobilenet_v2(pretrained=pretrained)
        out_channels = 1280
        ft_backbone = _create_backbone_features(mobile_net, 1280)
        return ft_backbone, out_channels

    elif model_name in [
        "vgg11",
        "vgg13",
        "vgg16",
        "vgg19",
    ]:
        out_channels = 512

        if model_name == "vgg11":
            vgg_net = torchvision.models.vgg11(pretrained=pretrained)
        elif model_name == "vgg13":
            vgg_net = torchvision.models.vgg13(pretrained=pretrained)
        elif model_name == "vgg16":
            vgg_net = torchvision.models.vgg16(pretrained=pretrained)
        elif model_name == "vgg19":
            vgg_net = torchvision.models.vgg19(pretrained=pretrained)

        ft_backbone = _create_backbone_features(vgg_net, out_channels)
        return ft_backbone, out_channels

    elif model_name in ["resnet18", "resnet34"]:
        out_channels = 512
        if model_name == "resnet18":
            resnet_net = torchvision.models.resnet18(pretrained=pretrained)
        elif model_name == "resnet34":
            resnet_net = torchvision.models.resnet34(pretrained=pretrained)

        ft_backbone = _create_backbone_adaptive(resnet_net, out_channels)
        return ft_backbone, out_channels

    elif model_name in [
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
    ]:
        out_channels = 2048
        if model_name == "resnet50":
            resnet_net = torchvision.models.resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            resnet_net = torchvision.models.resnet101(pretrained=pretrained)
        elif model_name == "resnet152":
            resnet_net = torchvision.models.resnet152(pretrained=pretrained)
        elif model_name == "resnext50_32x4d":
            resnet_net = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif model_name == "resnext101_32x8d":
            resnet_net = torchvision.models.resnext101_32x8d(pretrained=pretrained)

        ft_backbone = _create_backbone_adaptive(resnet_net, 2048)
        return ft_backbone, out_channels

    elif model_name in ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"]:
        out_channels = 1280
        if model_name == "mnasnet0_5":
            mnasnet_net = torchvision.models.mnasnet0_5(pretrained=pretrained)
        elif model_name == "mnasnet0_75":
            mnasnet_net = torchvision.models.mnasnet0_75(pretrained=pretrained)
        elif model_name == "mnasnet1_0":
            mnasnet_net = torchvision.models.mnasnet1_0(pretrained=pretrained)
        elif model_name == "mnasnet1_3":
            mnasnet_net = torchvision.models.mnasnet1_3(pretrained=pretrained)

        ft_backbone = _create_backbone_adaptive(mnasnet_net, 1280)
        return ft_backbone, out_channels

    else:
        raise ValueError("No such model implemented in torchvision")


class fine_tune_torchvision(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        # old_model = torchvision.models.resnet18(pretrained=True)
        # self.bottom = nn.Sequential(*list(old_model.children())[:-1])
        self.bottom, self.out_channels = _create_torchvision_backbone(
            model_name, num_classes, pretrained=pretrained
        )
        self.top = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        x = self.bottom(x)
        x = self.top(x.view(-1, self.out_channels))
        return x


# Wrapper function for consistency with timm models
# We do not pass in_channels here since torchvision models is not supported with it.
def create_torchvision_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
):
    """
    Creates CNN model from Torchvision. It replaces the top classification layer with num_classes you need.
    Args:
        model_name (str) : Name of the model. E.g. resnet18
        num_classes (int) : Number of classes for classification.
        pretrained (bool) : If true uses modelwweights pretrained on ImageNet.
    """
    model = fine_tune_torchvision(model_name, num_classes, pretrained)
    return model


# if __name__ == "__main__":
# ft_model = fine_tune_torchvision("mobilenet", 10)
# print(ft_model)
# print("Done")
