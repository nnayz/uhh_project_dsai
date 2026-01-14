"""
V3 Architecture: Audio Spectrogram Transformer (AST) encoder for ProtoNet.

This version replaces the ResNet encoder with a transformer over log-mel patches.
Trained from scratch (no pretrained weights).
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.distance import Distance


class PatchEmbed(nn.Module):
    """2D patch embedding for log-mel spectrograms."""

    def __init__(self, patch_size: Tuple[int, int], embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ASTEncoder(nn.Module):
    """Audio Spectrogram Transformer encoder."""

    def __init__(
        self,
        n_mels: int = 128,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        patch_freq: int = 8,
        patch_time: int = 2,
        max_time_bins: int = 32,
        dropout: float = 0.1,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.pooling = pooling

        self.patch_embed = PatchEmbed(
            patch_size=(patch_freq, patch_time),
            embed_dim=embed_dim,
        )

        grid_freq = max(1, n_mels // patch_freq)
        grid_time = max(1, max_time_bins // patch_time)
        self.grid_size = (grid_freq, grid_time)
        num_patches = grid_freq * grid_time

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _interpolate_pos_embed(self, grid_freq: int, grid_time: int) -> torch.Tensor:
        if (grid_freq, grid_time) == self.grid_size:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        patch_pos = patch_pos.reshape(
            1, self.grid_size[0], self.grid_size[1], -1
        ).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=(grid_freq, grid_time),
            mode="bilinear",
            align_corners=False,
        )
        embed_dim = patch_pos.shape[1]
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        _, _, grid_freq, grid_time = x.shape
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self._interpolate_pos_embed(grid_freq, grid_time)
        x = x + pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x)
        x = self.norm(x)

        if self.pooling == "mean":
            return x[:, 1:].mean(dim=1)
        return x[:, 0]


class ProtoNetV3(nn.Module):
    """Prototypical Network with AST encoder."""

    def __init__(
        self,
        emb_dim: int = 384,
        distance: Union[str, Distance] = Distance.EUCLIDEAN,
        n_mels: int = 128,
        patch_freq: int = 8,
        patch_time: int = 2,
        max_time_bins: int = 32,
        depth: int = 6,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        dropout: float = 0.1,
        pooling: str = "cls",
    ) -> None:
        super().__init__()

        if isinstance(distance, str):
            distance = Distance(distance.lower())

        self.encoder = ASTEncoder(
            n_mels=n_mels,
            embed_dim=emb_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            patch_freq=patch_freq,
            patch_time=patch_time,
            max_time_bins=max_time_bins,
            dropout=dropout,
            pooling=pooling,
        )
        self.distance = distance

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        sim = torch.matmul(x_norm, y_norm.t())
        return 1.0 - sim

    def _compute_prototypes(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
