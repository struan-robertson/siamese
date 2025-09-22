"""Siamese model implementation."""

import torch
import torchvision
from torch import nn


class SharedSiamese(nn.Module):
    """Siamese model with shared weights."""

    def __init__(
        self,
        embedding_size=128,
        *,
        pre_trained: bool = False,
        refreeze: bool = False,
        permafrost: int = 0,
    ):
        super().__init__()

        if pre_trained:
            model = torchvision.models.resnet50(weights="DEFAULT")
        else:
            model = torchvision.models.resnet50(weights=None)

        # Replace final FC layer with embedding layers
        model.fc = nn.Linear(model.fc.in_features, embedding_size)
        model.apply(self.init_weights)

        # Freeze model
        if pre_trained:
            for param in model.parameters():
                param.requires_grad = False
        self.permafrost = permafrost
        self.refreeze = refreeze

        self.batch_norm = nn.BatchNorm1d(embedding_size)
        self.model = model

    def unfreeze_idx(self, idx: int):
        layer_mappings = {
            0: self.model.fc,
            1: self.model.layer4,
            2: self.model.layer3,
            3: self.model.layer2,
            4: self.model.layer1,
            5: self.model.conv1,
        }

        if self.permafrost > 0 and idx > self.permafrost:
            return

        for param in layer_mappings[idx].parameters():  # pyright: ignore [reportAttributeAccessIssue]
            param.requires_grad = True

        if self.refreeze and idx >= 1 and idx <= 5:
            for param in layer_mappings[idx - 1].parameters():  # pyright: ignore [reportAttributeAccessIssue]
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
