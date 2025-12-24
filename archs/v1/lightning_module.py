"""
PyTorch Lightning Module for Prototypical Network.

This module wraps the ProtoNet model for training with PyTorch Lightning.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import lightning as L
from typing import Dict, List, Optional, Tuple, Union

from archs.v1.arch import ProtoNet
from utils.distance import Distance


class ProtoNetLightningModule(L.LightningModule):
    """Prototypical Network Lightning Module for few-shot classification."""
    
    def __init__(
        self,
        emb_dim: int = 2048,
        distance: Union[str, Distance] = "euclidean",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_gamma: float = 0.65,
        scheduler_step_size: int = 10,
        scheduler_type: str = "step",
        n_mels: int = 128,
        in_channels: int = 1,
        conv_channels: List[int] = None,
        activation: str = "leaky_relu",
        with_bias: bool = False,
        drop_rate: float = 0.1,
        time_max_pool_dim: int = 4,
    ) -> None:
        """
        Initialize the Prototypical Network Lightning Module.

        Args:
            emb_dim: Embedding dimension for the encoder output.
            distance: Distance metric ('euclidean' or 'cosine').
            lr: Learning rate.
            weight_decay: Weight decay (L2 regularization).
            scheduler_gamma: Multiplicative factor for LR scheduler.
            scheduler_step_size: Step size for StepLR scheduler.
            scheduler_type: Type of scheduler ('step', 'cosine', 'none').
            n_mels: Number of mel bands in input features.
            in_channels: Number of input channels.
            conv_channels: List of channel sizes for conv blocks.
            activation: Activation function name.
            with_bias: Whether to use bias in convolutions.
            drop_rate: Dropout rate.
            time_max_pool_dim: Time dimension for adaptive pooling.
        """
        super().__init__()
        self.save_hyperparameters()
        
        if conv_channels is None:
            conv_channels = [64, 64, 64, 64]
        
        self.model = ProtoNet(
            emb_dim=emb_dim,
            distance=distance,
            in_channels=in_channels,
            conv_channels=conv_channels,
            activation=activation,
            with_bias=with_bias,
            drop_rate=drop_rate,
            time_max_pool_dim=time_max_pool_dim,
        )
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_type = scheduler_type

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one few-shot episode."""
        return self.model(support_x, support_y, query_x, query_y)

    def _map_labels(
        self,
        support_y: torch.Tensor,
        query_y: torch.Tensor,
    ) -> torch.Tensor:
        """Map original labels to [0..Nc-1]."""
        class_ids = torch.unique(support_y)
        label_map = {int(c.item()): i for i, c in enumerate(class_ids)}
        return torch.tensor(
            [label_map[int(l.item())] for l in query_y],
            dtype=torch.long,
            device=query_y.device,
        )

    def _shared_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        stage: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shared logic for training, validation, and test steps."""
        support_x, support_y, query_x, query_y = [t.squeeze(0) for t in batch]
        
        loss, logits = self(support_x, support_y, query_x, query_y)
        mapped_query_y = self._map_labels(support_y, query_y)
        acc = (logits.argmax(dim=1) == mapped_query_y).float().mean()
        
        self.log_dict(
            {f"{stage}_loss": loss, f"{stage}_acc": acc},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        
        return loss, acc

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step."""
        loss, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step."""
        loss, _ = self._shared_step(batch, "val")
        return loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Test step."""
        loss, _ = self._shared_step(batch, "test")
        return loss

    def configure_optimizers(self) -> Dict:
        """Configure optimizer and learning rate scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        config = {"optimizer": optimizer}
        
        if self.scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        elif self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=50,
                eta_min=self.lr * 0.01,
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        
        return config
