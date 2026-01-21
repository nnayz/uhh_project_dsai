"""V4 Architecture: EfficientNet-B1 encoder for Prototypical Network."""

from archs.v4.arch import EfficientNetEncoder, ProtoNetV4
from archs.v4.lightning_module import ProtoNetV4LightningModule

__all__ = [
    "EfficientNetEncoder",
    "ProtoNetV4",
    "ProtoNetV4LightningModule",
]

