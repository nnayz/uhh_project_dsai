"""Encoder module placeholder."""

import torch.nn as nn


class Encoder(nn.Module):
    """Simple convolutional encoder placeholder."""

    def __init__(self, in_channels=1, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)
