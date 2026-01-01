"""
PyTorch Lightning Module for Prototypical Network.

This module wraps the ProtoNet model for training with PyTorch Lightning.
Enhanced with comprehensive evaluation metrics for bioacoustic classification.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import lightning as L
from typing import Dict, List, Optional, Tuple, Union

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

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
        num_classes: int = 10,
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
            num_classes: Number of classes for episode (n_way).
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
        
        # Initialize evaluation metrics
        # Validation metrics (updated every epoch)
        self.val_metrics = MetricCollection({
            'precision': MulticlassPrecision(num_classes=num_classes, average='macro'),
            'recall': MulticlassRecall(num_classes=num_classes, average='macro'),
            'f1': MulticlassF1Score(num_classes=num_classes, average='macro'),
        })
        
        # Per-class accuracy for detailed analysis
        self.val_per_class_acc = MulticlassAccuracy(
            num_classes=num_classes, 
            average='none'
        )
        
        # Confusion matrix
        self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        
        # Test metrics (separate from validation)
        self.test_metrics = MetricCollection({
            'precision': MulticlassPrecision(num_classes=num_classes, average='macro'),
            'recall': MulticlassRecall(num_classes=num_classes, average='macro'),
            'f1': MulticlassF1Score(num_classes=num_classes, average='macro'),
        })
        
        self.test_per_class_acc = MulticlassAccuracy(
            num_classes=num_classes,
            average='none'
        )
        
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

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
        preds = logits.argmax(dim=1)
        acc = (preds == mapped_query_y).float().mean()
        
        self.log_dict(
            {f"{stage}_loss": loss, f"{stage}_acc": acc},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        
        # Update metrics for validation and test
        if stage == "val":
            self.val_metrics.update(preds, mapped_query_y)
            self.val_per_class_acc.update(preds, mapped_query_y)
            self.val_confusion_matrix.update(preds, mapped_query_y)
        elif stage == "test":
            self.test_metrics.update(preds, mapped_query_y)
            self.test_per_class_acc.update(preds, mapped_query_y)
            self.test_confusion_matrix.update(preds, mapped_query_y)
        
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
    
    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at epoch end."""
        # Compute main metrics (precision, recall, f1)
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val_{name}", value, prog_bar=True)
        
        # Compute per-class accuracy
        per_class_acc = self.val_per_class_acc.compute()
        for i, acc in enumerate(per_class_acc):
            self.log(f"val_class_{i}_acc", acc)
        
        # Log average per-class accuracy (useful summary metric)
        self.log("val_avg_class_acc", per_class_acc.mean())
        
        # Reset all metrics for next epoch
        self.val_metrics.reset()
        self.val_per_class_acc.reset()
        self.val_confusion_matrix.reset()

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Test step."""
        loss, _ = self._shared_step(batch, "test")
        return loss
    
    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics at epoch end."""
        # Compute main metrics
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test_{name}", value)
        
        # Compute per-class accuracy
        per_class_acc = self.test_per_class_acc.compute()
        for i, acc in enumerate(per_class_acc):
            self.log(f"test_class_{i}_acc", acc)
        
        # Log average per-class accuracy
        self.log("test_avg_class_acc", per_class_acc.mean())
        
        # Compute and print confusion matrix
        cm = self.test_confusion_matrix.compute()
        print("\nTest Confusion Matrix:")
        print(cm.cpu().numpy())
        
        # Reset all metrics
        self.test_metrics.reset()
        self.test_per_class_acc.reset()
        self.test_confusion_matrix.reset()


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
