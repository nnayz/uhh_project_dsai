import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict

def conv_block(
    in_channels: int,
    out_channels: int
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=1,
            stride=1
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class PrototypicalNetwork(nn.Module):
    def __init__(self):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 64)
        )

    def forward(self, x):
        (num_samples, seq_len, mel_bins) = x.shape
        x = x.view(-1, 1, seq_len, mel_bins)

        x = self.encoder(x)
        x = nn.MaxPool2d(2)(x)

        return x.view(x.size(0), -1)

    def conv3x3(in_planes, out_planes, stride=1):
        """
        3x3 convolution with padding
        """
