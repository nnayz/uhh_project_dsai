from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from utils.config import Distance

class ConvEncoder(nn.Module):
    """
    Simple CNN encoder for log-mel spectrograms.

    Input: (B, 1, n_mels, T)
    Output: (B, emb_dim)
    """
    def __init__(
        self,
        in_channels: int = 1,
        emb_dim: int = 128
    ) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=-1)
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
        emb_dim: int = 128,
        distance: Distance = Distance.EUCLIDEAN
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(in_channels=1, emb_dim=emb_dim)
        self.distance = distance

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Squared Euclidean distance between all rows of x and y.
        """
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
        """
        Cosine distance = 1 - cosine similarity.
        """
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        sim = torch.matmul(x_norm, y_norm.t()) # (Nq, Nc)
        return 1.0 - sim

    def _compute_prototypes(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute class prototypes by averaging embeddings per class.

        embeddings: (N, D)
        labels:     (N,)
        returns:
            prototypes: (Nc, D) [Nc means number of classes]
            class_ids:  (Nc,)
        """
        class_ids = torch.unique(labels) # (Nc,)
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
        query_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run one few shot episode.

        Returns:
        loss: scalar
        logits: (Nq, Nc)
        """
        device = next(self.parameters()).device
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

        emb_support = self.encoder(support_x) # (Ns, D)
        emb_query = self.encoder(query_x) # (Nq, D)

        prototypes, class_ids = self._compute_prototypes(emb_support, support_y)

        if self.distance == Distance.EUCLIDEAN:
            dists = self.euclidean_dist(emb_query, prototypes) # (Nq, Nc)
        else:
            dists = self.cosine_dist(emb_query, prototypes) # (Nq, Nc)

        logits = -dists # closer = larger logit

        # map original labels -> [0..Nc-1]
        label_map = {
            int(c.item()): i for i, c in enumerate(class_ids)
        }
        mapped_query_y = torch.tensor(
            [label_map[int(l.item())] for l in query_y],
            dtype=torch.long,
            device=device
        )

        loss = F.cross_entropy(logits, mapped_query_y)
        return loss, logits