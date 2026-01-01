"""
V2 Architecture for Few-Shot Bioacoustic Classification.

Enhanced version with:
- Multi-representation audio frontend (PCEN + Log-mel)
- ResNet-style encoder with attention mechanisms
- Learnable prototypes with cross-attention
- Advanced data augmentation
"""

from .arch import ProtoNetV2
from .lightning_module import ProtoNetV2LightningModule

__all__ = ["ProtoNetV2", "ProtoNetV2LightningModule"]

