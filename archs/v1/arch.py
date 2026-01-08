"""
Prototypical Network architecture for few-shot bioacoustic classification.

This implements the baseline v1 architecture with a Conv4 encoder
and prototype-based classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

from utils.distance import Distance


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
        "elu": nn.ELU(inplace=True),
        "gelu": nn.GELU(),
    }
    if name not in activations:
        raise ValueError(
            f"Unknown activation: {name}. Available: {list(activations.keys())}"
        )
    return activations[name]


class ConvBlock(nn.Module):
    """Conv block matching the reference ProtoNet encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        with_bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=with_bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class Conv4Encoder(nn.Module):
    """
    4-block CNN encoder matching the reference ProtoNet model.

    Input: (B, 1, n_mels, T)
    Output: (B, D) flattened conv features
    """

    def __init__(
        self,
        in_channels: int = 1,
        emb_dim: int = 2048,
        conv_channels: List[int] = None,
        with_bias: bool = False,
        time_max_pool_dim: int = 4,
    ) -> None:
        super().__init__()

        if conv_channels is None:
            conv_channels = [64, 64, 64, 64]

        self.conv_blocks = nn.ModuleList()

        in_ch = in_channels
        for out_ch in conv_channels:
            self.conv_blocks.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    with_bias=with_bias,
                )
            )
            in_ch = out_ch
        self.final_pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)

        x = self.final_pool(x)
        x = self.flatten(x)
        return x


class ProtoNet(nn.Module):
    """
    Prototypical Network for few-shot classification on episodes.

    Expects episodic input:
      - support_x : (Ns, 1, n_mels, T)
      - support_y : (Ns,)
      - query_x   : (Nq, 1, n_mels, T)
      - query_y   : (Nq,)
    """

    def __init__(
        self,
        emb_dim: int = 2048,
        distance: Union[str, Distance] = Distance.EUCLIDEAN,
        in_channels: int = 1,
        conv_channels: List[int] = None,
        activation: str = "leaky_relu",
        with_bias: bool = False,
        drop_rate: float = 0.1,
        time_max_pool_dim: int = 4,
    ) -> None:
        super().__init__()

        if isinstance(distance, str):
            distance = Distance(distance.lower())

        self.encoder = Conv4Encoder(
            in_channels=in_channels,
            emb_dim=emb_dim,
            conv_channels=conv_channels,
            activation=activation,
            with_bias=with_bias,
            drop_rate=drop_rate,
            time_max_pool_dim=time_max_pool_dim,
        )
        self.distance = distance

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Squared Euclidean distance between all rows of x and y."""
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise ValueError("Embedding dimension mismatch")

        x_exp = x.unsqueeze(1).expand(n, m, d)
        y_exp = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x_exp - y_exp, 2).sum(dim=2)

    @staticmethod
    def cosine_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cosine distance = 1 - cosine similarity."""
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        sim = torch.matmul(x_norm, y_norm.t())
        return 1.0 - sim

    def _compute_prototypes(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute class prototypes by averaging embeddings per class.

        Args:
            embeddings: (N, D)
            labels: (N,)

        Returns:
            prototypes: (Nc, D)
            class_ids: (Nc,)
        """
        class_ids = torch.unique(labels)
        protos = []
        for c in class_ids:
            mask = labels == c
            protos.append(embeddings[mask].mean(dim=0))
        prototypes = torch.stack(protos, dim=0)
        return prototypes, class_ids

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run one few-shot episode.

        Returns:
            loss: scalar
            logits: (Nq, Nc)
        """
        device = next(self.parameters()).device
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

        emb_support = self.encoder(support_x)
        emb_query = self.encoder(query_x)

        prototypes, class_ids = self._compute_prototypes(emb_support, support_y)

        if self.distance == Distance.EUCLIDEAN:
            dists = self.euclidean_dist(emb_query, prototypes)
        else:
            dists = self.cosine_dist(emb_query, prototypes)

        logits = -dists

        label_map = {int(c.item()): i for i, c in enumerate(class_ids)}
        mapped_query_y = torch.tensor(
            [label_map[int(l.item())] for l in query_y], dtype=torch.long, device=device
        )

        loss = F.cross_entropy(logits, mapped_query_y)
        return loss, logits
