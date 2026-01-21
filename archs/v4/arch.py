"""
V4 Architecture: EfficientNet-B1 encoder for Prototypical Network.

This version replaces the ResNet encoder with EfficientNet-B1, which provides:
- Better parameter efficiency
- Compound scaling (depth/width/resolution)
- Built-in Squeeze-and-Excitation attention
- ImageNet pretrained weights for better initialization
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    raise ImportError(
        "efficientnet-pytorch is required for v4. Install with: pip install efficientnet-pytorch"
    )

from utils.distance import Distance


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B1 encoder for bioacoustic spectrograms.
    
    Adapts EfficientNet to work with variable-length spectrograms (time × frequency).
    Uses global pooling to handle variable time dimensions.
    
    Args:
        embedding_dim: Output embedding dimension (default: 2048)
        model_name: EfficientNet model variant (default: 'efficientnet-b1')
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        dropout: Dropout rate for final projection (default: 0.1)
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        model_name: str = "efficientnet-b1",
        pretrained: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        # Load pretrained EfficientNet
        self.backbone = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
        
        # Get the feature dimension from EfficientNet
        # EfficientNet-B1 outputs 1280 features after global pooling
        backbone_dim = self.backbone._fc.in_features
        
        # Remove the classification head
        self.backbone._fc = nn.Identity()
        
        # Projection head to desired embedding dimension
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, embedding_dim),
        )
        
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EfficientNet encoder.
        
        Args:
            x: Input spectrogram (B, T, n_mels) where B=batch, T=time, n_mels=128
            
        Returns:
            embeddings: (B, embedding_dim)
        """
        # Reshape from (B, T, n_mels) to (B, 1, T, n_mels) for 2D conv
        num_samples, seq_len, mel_bins = x.shape
        x = x.view(-1, 1, seq_len, mel_bins)
        
        # EfficientNet expects 3 channels, so we replicate the single channel
        # This is a common approach when adapting grayscale inputs to RGB models
        x = x.repeat(1, 3, 1, 1)
        
        # EfficientNet requires minimum input size to avoid kernel size issues
        # After multiple stride-2 downsampling layers, small inputs become too small
        # Use a smaller minimum (32x32)
        min_h, min_w = 32, 32
        _, _, h, w = x.shape
        
        # If input is too small, resize or pad to minimum size
        if h < min_h or w < min_w:
            # Calculate padding needed
            pad_h = max(0, min_h - h)
            pad_w = max(0, min_w - w)
            
            # For very small inputs where padding would exceed input size, use interpolation
            # Reflect padding requires padding < input_size, so use interpolation for very small inputs
            # Otherwise use constant padding (more memory efficient than reflect for small inputs)
            if (pad_h > 0 and pad_h >= h) or (pad_w > 0 and pad_w >= w):
                # Input too small for padding, use interpolation
                new_h = max(h, min_h)
                new_w = max(w, min_w)
                x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                # Can use padding - pad symmetrically: (pad_left, pad_right, pad_top, pad_bottom)
                if pad_h > 0 or pad_w > 0:
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    # Use constant padding (0) - works for any size and is memory efficient
                    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        # Forward through EfficientNet backbone
        # extract_features returns features before global pooling
        features = self.backbone.extract_features(x)
        
        # Global average pooling over spatial dimensions
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        
        # Project to desired embedding dimension
        embeddings = self.projection(features)
        
        return embeddings


class ProtoNetV4(nn.Module):
    """
    Prototypical Network for few-shot classification with EfficientNet encoder.

    Expects episodic input:
      - support_x : (Ns, T, n_mels)
      - support_y : (Ns,)
      - query_x   : (Nq, T, n_mels)
      - query_y   : (Nq,)
    """

    def __init__(
        self,
        emb_dim: int = 2048,
        distance: Union[str, Distance] = Distance.EUCLIDEAN,
        model_name: str = "efficientnet-b1",
        pretrained: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if isinstance(distance, str):
            distance = Distance(distance.lower())

        self.encoder = EfficientNetEncoder(
            embedding_dim=emb_dim,
            model_name=model_name,
            pretrained=pretrained,
            dropout=dropout,
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

