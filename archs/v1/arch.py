"""
Prototypical Network architecture for few-shot bioacoustic classification.

This implements the baseline v1 architecture with the ResNet-style encoder
used in the DCASE baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

from utils.distance import Distance


def conv3x3(in_planes: int, out_planes: int, with_bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=with_bias,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        drop_rate: float = 0.0,
        drop_block: bool = False,
        block_size: int = 1,
        with_bias: bool = False,
        non_linearity: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, with_bias=with_bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1) if non_linearity == "leaky_relu" else nn.ReLU()
        self.conv2 = conv3x3(planes, planes, with_bias=with_bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes, with_bias=with_bias)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.num_batches_tracked += 1

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out


class ResNetEncoder(nn.Module):
    """ResNet encoder matching the baseline ProtoNet implementation."""

    def __init__(
        self,
        embedding_dim: int = 2048,
        drop_rate: float = 0.1,
        with_bias: bool = False,
        non_linearity: str = "leaky_relu",
        time_max_pool_dim: int = 4,
        layer_4: bool = False,
    ) -> None:
        super().__init__()
        self.inplanes = 1
        self.keep_avg_pool = True
        self.features = type(
            "FeatureConfig",
            (),
            {
                "drop_rate": drop_rate,
                "with_bias": with_bias,
                "non_linearity": non_linearity,
                "time_max_pool_dim": time_max_pool_dim,
                "embedding_dim": embedding_dim,
                "layer_4": layer_4,
            },
        )()

        self.layer1 = self._make_layer(BasicBlock, 64, stride=2)
        self.layer2 = self._make_layer(BasicBlock, 128, stride=2)
        self.layer3 = self._make_layer(
            BasicBlock,
            64,
            stride=2,
            drop_block=True,
            block_size=5,
        )
        self.layer4 = self._make_layer(
            BasicBlock,
            64,
            stride=2,
            drop_block=True,
            block_size=5,
        )
        self.pool_avg = nn.AdaptiveAvgPool2d(
            (
                time_max_pool_dim,
                int(embedding_dim / (time_max_pool_dim * 64)),
            )
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=non_linearity
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        stride: int = 1,
        drop_block: bool = False,
        block_size: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layer = block(
            self.inplanes,
            planes,
            stride,
            downsample,
            self.features.drop_rate,
            drop_block,
            block_size,
            self.features.with_bias,
            self.features.non_linearity,
        )
        self.inplanes = planes * block.expansion
        return nn.Sequential(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_samples, seq_len, mel_bins = x.shape
        x = x.view(-1, 1, seq_len, mel_bins)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.features.layer_4:
            x = self.layer4(x)
        x = self.pool_avg(x)
        return x.view(x.size(0), -1)


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
        with_bias: bool = False,
        drop_rate: float = 0.1,
        non_linearity: str = "leaky_relu",
        time_max_pool_dim: int = 4,
        layer_4: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(distance, str):
            distance = Distance(distance.lower())

        self.encoder = ResNetEncoder(
            embedding_dim=emb_dim,
            drop_rate=drop_rate,
            with_bias=with_bias,
            non_linearity=non_linearity,
            time_max_pool_dim=time_max_pool_dim,
            layer_4=layer_4,
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
