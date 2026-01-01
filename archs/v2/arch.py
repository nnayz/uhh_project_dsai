"""
V2 Architecture: Enhanced Prototypical Network for Few-Shot Bioacoustic Classification.

Key improvements over V1:
1. Deeper ResNet-style encoder (vs simple Conv4)
2. Multi-head attention mechanisms (temporal + channel)
3. Learnable distance metric (vs fixed Euclidean)
4. Residual connections for better gradient flow
5. Dual pooling (GAP + GMP) for richer representations

Based on research:
- "Deep Residual Learning" (He et al., 2016)
- "Squeeze-and-Excitation Networks" (Hu et al., 2018)
- "Prototypical Networks for Few-shot Learning" (Snell et al., 2017)
- "PANNs: Large-Scale Pretrained Audio Neural Networks" (Kong et al., 2020)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

from utils.distance import Distance


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention mechanism.
    
    Learns to emphasize important frequency bands (channels) for classification.
    
    Args:
        in_channels: Number of input channels
        reduction: Reduction ratio for bottleneck
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Attention-weighted features
        """
        b, c, _, _ = x.size()
        
        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention.expand_as(x)


class ResNetBlock(nn.Module):
    """
    Residual block with batch normalization and optional downsampling.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for downsampling
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        
        # Channel attention
        self.channel_attention = ChannelAttention(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply channel attention
        out = self.channel_attention(out)
        
        # Add skip connection
        out += identity
        out = self.relu(out)
        
        return out


class TemporalAttention(nn.Module):
    """
    Simplified temporal attention using 1D convolutions.
    
    More efficient than multi-head attention and doesn't require
    knowing spatial dimensions in advance.
    
    Args:
        channels: Number of input channels
    """
    
    def __init__(self, channels: int):
        super().__init__()
        # Attention weights computed from channel-wise global pooling
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) where W is time dimension
        Returns:
            Attention-weighted features
        """
        B, C, H, W = x.shape
        
        # Global average pooling over frequency dimension
        # (B, C, H, W) -> (B, C, W)
        pooled = x.mean(dim=2)
        
        # Compute temporal attention weights
        # (B, C, W) -> (B, C, W)
        attn_weights = self.conv(pooled)
        
        # Reshape for broadcasting: (B, C, W) -> (B, C, 1, W)
        attn_weights = attn_weights.unsqueeze(2)
        
        # Apply attention
        out = x * attn_weights
        
        return out


class ResNetAttentionEncoder(nn.Module):
    """
    Deep ResNet-style encoder with attention mechanisms for audio classification.
    
    Architecture:
    - 4 ResNet blocks with progressive downsampling
    - Channel attention in each block
    - Temporal attention after ResNet blocks
    - Dual pooling (GAP + GMP) for richer representations
    
    Args:
        in_channels: Number of input channels (1 for mono spectrograms)
        emb_dim: Output embedding dimension
        channels: List of channel numbers for each block
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        emb_dim: int = 2048,
        channels: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if channels is None:
            channels = [64, 128, 256, 512]
        
        # Initial conv to expand channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        
        # ResNet blocks with progressive downsampling
        self.layer1 = self._make_layer(channels[0], channels[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(channels[0], channels[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(channels[1], channels[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(channels[2], channels[3], stride=2, dropout=dropout)
        
        # Temporal attention to focus on important time frames
        # Uses channel-wise attention over time dimension
        self.temporal_attention = TemporalAttention(channels=channels[3])
        
        # Global pooling (both average and max for richer representation)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        
        # Final projection
        self.fc = nn.Linear(channels[3] * 2, emb_dim)  # *2 for GAP + GMP
        self.dropout = nn.Dropout(dropout)
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout: float
    ) -> nn.Module:
        """Create a ResNet block."""
        return ResNetBlock(in_channels, out_channels, stride, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, freq, time) - log-mel or PCEN spectrogram
        Returns:
            Embedding: (B, emb_dim) - L2-normalized embedding
        """
        # Initial conv
        x = self.conv1(x)
        
        # ResNet blocks with downsampling
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Temporal attention (focus on important time frames)
        x = self.temporal_attention(x)
        
        # Dual pooling
        gap = self.gap(x).flatten(1)
        gmp = self.gmp(x).flatten(1)
        pooled = torch.cat([gap, gmp], dim=1)
        
        # Final projection
        x = self.dropout(pooled)
        x = self.fc(x)
        
        # L2 normalization for better distance-based classification
        x = F.normalize(x, p=2, dim=-1)
        
        return x


class LearnableDistanceMetric(nn.Module):
    """
    Learnable distance metric for comparing query and prototype embeddings.
    
    More flexible than fixed Euclidean distance - learns what differences matter.
    
    Args:
        emb_dim: Embedding dimension
    """
    
    def __init__(self, emb_dim: int):
        super().__init__()
        
        # MLP to compute distance from concatenated embeddings
        self.distance_net = nn.Sequential(
            nn.Linear(emb_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
    
    def forward(self, query_emb: torch.Tensor, proto_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute learnable distance between queries and prototypes.
        
        Args:
            query_emb: (Nq, D) query embeddings
            proto_emb: (Nc, D) prototype embeddings
        
        Returns:
            distances: (Nq, Nc) distance matrix
        """
        Nq, D = query_emb.shape
        Nc, _ = proto_emb.shape
        
        # Expand to compute all pairwise distances
        query_exp = query_emb.unsqueeze(1).expand(Nq, Nc, D)  # (Nq, Nc, D)
        proto_exp = proto_emb.unsqueeze(0).expand(Nq, Nc, D)  # (Nq, Nc, D)
        
        # Concatenate and compute distance
        combined = torch.cat([query_exp, proto_exp], dim=2)  # (Nq, Nc, 2D)
        distances = self.distance_net(combined).squeeze(2)  # (Nq, Nc)
        
        return distances


class ProtoNetV2(nn.Module):
    """
    V2 Prototypical Network with ResNet encoder and learnable distance metric.
    
    Key improvements over V1:
    - Deeper ResNet encoder with attention (vs simple Conv4)
    - Learnable distance metric (vs fixed Euclidean)
    - Channel and temporal attention mechanisms
    - Better regularization with residual connections
    
    Args:
        emb_dim: Embedding dimension
        distance: Distance metric ('euclidean', 'cosine', or 'learnable')
        in_channels: Input channels (1 for mono audio)
        channels: Channel progression for ResNet blocks
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        emb_dim: int = 2048,
        distance: Union[str, Distance] = "learnable",
        in_channels: int = 1,
        channels: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if channels is None:
            channels = [64, 128, 256, 512]
        
        # Enhanced encoder
        self.encoder = ResNetAttentionEncoder(
            in_channels=in_channels,
            emb_dim=emb_dim,
            channels=channels,
            dropout=dropout,
        )
        
        # Distance metric
        if isinstance(distance, str):
            distance_lower = distance.lower()
            if distance_lower == "learnable":
                self.distance_type = "learnable"
                self.distance_metric = LearnableDistanceMetric(emb_dim)
            elif distance_lower in ["euclidean", "cosine"]:
                self.distance_type = distance_lower
                self.distance_metric = None
            else:
                raise ValueError(f"Unknown distance: {distance}")
        else:
            # Handle Distance enum from utils
            self.distance_type = distance.value if hasattr(distance, 'value') else str(distance).lower()
            self.distance_metric = None
    
    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Squared Euclidean distance."""
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        
        x_exp = x.unsqueeze(1).expand(n, m, d)
        y_exp = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x_exp - y_exp, 2).sum(dim=2)
    
    @staticmethod
    def cosine_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cosine distance."""
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        sim = torch.matmul(x_norm, y_norm.t())
        return 1.0 - sim
    
    def _compute_prototypes(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute class prototypes by averaging embeddings.
        
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
        Forward pass for one few-shot episode.
        
        Args:
            support_x: (Ns, C, H, W) support spectrograms
            support_y: (Ns,) support labels
            query_x: (Nq, C, H, W) query spectrograms
            query_y: (Nq,) query labels
        
        Returns:
            loss: Scalar loss
            logits: (Nq, Nc) classification logits
        """
        device = next(self.parameters()).device
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)
        
        # Encode support and query
        emb_support = self.encoder(support_x)
        emb_query = self.encoder(query_x)
        
        # Compute prototypes
        prototypes, class_ids = self._compute_prototypes(emb_support, support_y)
        
        # Compute distances
        if self.distance_type == "learnable":
            dists = self.distance_metric(emb_query, prototypes)
        elif self.distance_type == "euclidean":
            dists = self.euclidean_dist(emb_query, prototypes)
        elif self.distance_type == "cosine":
            dists = self.cosine_dist(emb_query, prototypes)
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        
        # Convert distances to logits (negative distance)
        logits = -dists
        
        # Map labels to [0, Nc-1]
        label_map = {int(c.item()): i for i, c in enumerate(class_ids)}
        mapped_query_y = torch.tensor(
            [label_map[int(l.item())] for l in query_y],
            dtype=torch.long,
            device=device
        )
        
        # Compute loss
        loss = F.cross_entropy(logits, mapped_query_y)
        
        return loss, logits

