"""Classification model implementation."""

import torch
import torchvision
from torch import nn


class ShorterClassification(nn.Module):
    """Shorter version of ResNet classification model.

    Output from layer two (512 layers) is concatenated to 512 layers and then passed to layer 3."""

    def __init__(self):
        super().__init__()

        pretrained_resnet = torchvision.models.resnet50(weights="DEFAULT")
        resnet = torchvision.models.resnet50(weights=None)

        self.extractor = nn.Sequential(
            pretrained_resnet.conv1,
            pretrained_resnet.bn1,
            pretrained_resnet.relu,
            pretrained_resnet.maxpool,
            pretrained_resnet.layer1,
            pretrained_resnet.layer2,
        )

        # Freeze extractor parameters
        for param in self.extractor.parameters():
            param.requires_grad = False

        self.final = nn.Sequential(
            resnet.layer4,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1, bias=True),
        )

        self.final.apply(self.init_weights)

    def forward(self, shoeprint, shoemark):
        shoeprint_embedding = self.extractor(shoeprint)
        shoemark_embedding = self.extractor(shoemark)

        concatenated = torch.cat((shoeprint_embedding, shoemark_embedding), dim=1)

        return self.final(concatenated)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class BottleneckClassification(nn.Module):
    """Bottleneck ResNet classification model.

    The output from layer 3 is concatenated for a shoeprint and shoemark, reduced in layers by a
    bottleneck layer and then passed into layer 4."""

    def __init__(self):
        super().__init__()

        pretrained_resnet = torchvision.models.resnet50(weights="DEFAULT")
        resnet = torchvision.models.resnet50(weights=None)

        self.extractor = nn.Sequential(
            pretrained_resnet.conv1,
            pretrained_resnet.bn1,
            pretrained_resnet.relu,
            pretrained_resnet.maxpool,
            pretrained_resnet.layer1,
            pretrained_resnet.layer2,
            pretrained_resnet.layer3,
        )

        # Freeze extractor parameters
        for param in self.extractor.parameters():
            param.requires_grad = False

        # TODO Test simpler bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(inplace=True),
        )

        # TODO Test frozen layer 4 with pretrained weights, potentially unfreezing
        self.final = nn.Sequential(
            resnet.layer4,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1, bias=True),
        )

        self.final.apply(self.init_weights)

    def forward(self, shoeprint, shoemark):
        shoeprint_embedding = self.extractor(shoeprint)
        shoemark_embedding = self.extractor(shoemark)

        concatenated = torch.cat((shoeprint_embedding, shoemark_embedding), dim=1)
        concatenated = self.bottleneck(concatenated)

        return self.final(concatenated)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
