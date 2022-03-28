import torch.nn as nn
import torchvision.models as models

import happier.lib as lib


def get_backbone(name, pretrained=True, **kwargs):
    if name == 'resnet34':
        lib.LOGGER.info("using ResNet-34")
        out_dim = 512
        backbone = models.resnet34(pretrained=pretrained)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    elif name == 'resnet50':
        lib.LOGGER.info("using ResNet-50")
        out_dim = 2048
        backbone = models.resnet50(pretrained=pretrained)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    return (backbone, pooling, out_dim)
