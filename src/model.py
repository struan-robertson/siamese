"""Siamese model implementation."""

import torch
import torchvision
from torch import nn


class SharedSiamese(nn.Module):
    """Siamese model with shared weights."""

    def __init__(self, embedding_size=128):
        super().__init__()

        self.model = torchvision.models.resnet50(weights=None)

        # Replace final FC layer with embedding layers
        # self.model.conv1 = nn.Conv2d(
        #     1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        # )
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_size)

        self.model.apply(self.init_weights)

    def unfreeze_idx(self, idx):
        children = list(self.model.children())
        n = len(children)

        # Convert negative indices to positive equivalents
        valid_indices = set()

        if idx < 0:
            idx = n + idx  # -1 becomes n-1, -2 becomes n-2, etc.
        if 0 <= idx < n:
            valid_indices.add(idx)
        else:
            return

        # Unfreeze selected children
        for idx in valid_indices:
            for param in children[idx].parameters():
                param.requires_grad = True

    ## Completely unfreeze model
    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    ## Revert to initial state (only fc trainable)
    def freeze_except_fc(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = "fc" in name

    def forward(self, x):
        return self.model(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
