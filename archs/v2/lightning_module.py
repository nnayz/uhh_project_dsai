"""
PyTorch Lightning Module for V2 Prototypical Network.

Integrates:
- V2 architecture with ResNet + Attention
- SpecAugment data augmentation
- Enhanced evaluation metrics
- Learnable distance metric
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
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

from archs.v2.arch import ProtoNetV2
from archs.v2.augmentation import BioacousticAugmentation
from utils.distance import Distance


class ProtoNetV2LightningModule(L.LightningModule):
    """
    V2 Prototypical Network Lightning Module with enhanced features.
    
    New features over V1:
    - Integrated SpecAugment augmentation
    - Deeper ResNet-Attention encoder
    - Learnable distance metric
    - Same enhanced metrics as V1
    """
    
    def __init__(
        self,
        emb_dim: int = 2048,
        distance: Union[str, Distance] = "learnable",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_gamma: float = 0.65,
        scheduler_step_size: int = 10,
        scheduler_type: str = "step",
        n_mels: int = 128,
        in_channels: int = 1,
        channels: List[int] = None,
        dropout: float = 0.1,
        num_classes: int = 10,
        # Augmentation parameters
        use_augmentation: bool = True,
        use_spec_augment: bool = True,
        use_noise: bool = True,
        time_mask_pct: float = 0.15,
        freq_mask_pct: float = 0.15,
    ) -> None:
        """
        Initialize V2 Prototypical Network Lightning Module.
        
        Args:
            emb_dim: Embedding dimension.
            distance: Distance metric ('euclidean', 'cosine', 'learnable').
            lr: Learning rate.
            weight_decay: Weight decay (L2 regularization).
            scheduler_gamma: LR scheduler multiplicative factor.
            scheduler_step_size: LR scheduler step size.
            scheduler_type: Type of scheduler ('step', 'cosine', 'none').
            n_mels: Number of mel bands (not used directly, for compatibility).
            in_channels: Number of input channels.
            channels: Channel progression [64, 128, 256, 512].
            dropout: Dropout rate.
            num_classes: Number of classes for metrics (n_way).
            use_augmentation: Enable data augmentation.
            use_spec_augment: Enable SpecAugment.
            use_noise: Enable Gaussian noise augmentation.
            time_mask_pct: SpecAugment time masking percentage.
            freq_mask_pct: SpecAugment frequency masking percentage.
        """
        super().__init__()
        self.save_hyperparameters()
        
        if channels is None:
            channels = [64, 128, 256, 512]
        
        # V2 Model
        self.model = ProtoNetV2(
            emb_dim=emb_dim,
            distance=distance,
            in_channels=in_channels,
            channels=channels,
            dropout=dropout,
        )
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_type = scheduler_type
        
        # Data augmentation
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.augmentation = BioacousticAugmentation(
                use_spec_augment=use_spec_augment,
                use_mixup=False,  # Mixup complicates episodic learning
                use_noise=use_noise,
                time_mask_pct=time_mask_pct,
                freq_mask_pct=freq_mask_pct,
            )
        
        # Enhanced evaluation metrics (same as V1)
        # Validation metrics
        self.val_metrics = MetricCollection({
            'precision': MulticlassPrecision(num_classes=num_classes, average='macro'),
            'recall': MulticlassRecall(num_classes=num_classes, average='macro'),
            'f1': MulticlassF1Score(num_classes=num_classes, average='macro'),
        })
        
        self.val_per_class_acc = MulticlassAccuracy(
            num_classes=num_classes,
            average='none'
        )
        
        self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        
        # Test metrics
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
    
    def _apply_augmentation(
        self,
        support_x: torch.Tensor,
        query_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation to support and query sets.
        Only applied during training.
        """
        if not self.training or not self.use_augmentation:
            return support_x, query_x
        
        # Apply augmentation separately to support and query
        support_x, _ = self.augmentation(support_x)
        query_x, _ = self.augmentation(query_x)
        
        return support_x, query_x
    
    def _shared_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        stage: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shared logic for training, validation, and test steps."""
        support_x, support_y, query_x, query_y = [t.squeeze(0) for t in batch]
        
        # Apply augmentation (only during training)
        if stage == "train":
            support_x, query_x = self._apply_augmentation(support_x, query_x)
        
        # Forward pass
        loss, logits = self(support_x, support_y, query_x, query_y)
        mapped_query_y = self._map_labels(support_y, query_y)
        preds = logits.argmax(dim=1)
        acc = (preds == mapped_query_y).float().mean()
        
        # Log basic metrics
        self.log_dict(
            {f"{stage}_loss": loss, f"{stage}_acc": acc},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        
        # Update advanced metrics for val/test
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
        """Training step with augmentation."""
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
        # Compute main metrics
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val_{name}", value, prog_bar=True)
        
        # Compute per-class accuracy
        per_class_acc = self.val_per_class_acc.compute()
        for i, acc in enumerate(per_class_acc):
            self.log(f"val_class_{i}_acc", acc)
        
        # Log average per-class accuracy
        self.log("val_avg_class_acc", per_class_acc.mean())
        
        # Reset all metrics
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

