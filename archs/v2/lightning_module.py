"""
PyTorch Lightning Module for V2 Prototypical Network (reference training behavior).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from archs.v2.arch import ProtoNetV2
from utils.distance import Distance
from utils.loss import prototypical_loss, prototypical_loss_filter_negative
from archs.v2.augmentation import AudioAugmentation


class ProtoNetV2LightningModule(L.LightningModule):
    """
    Enhanced Prototypical Network with reference training loop.
    """

    def __init__(
        self,
        emb_dim: int = 1024,
        distance: Union[str, Distance] = "learnable",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_gamma: float = 0.65,
        scheduler_step_size: int = 10,
        scheduler_type: str = "cosine",
        n_mels: int = 128,
        in_channels: int = 1,
        channels: List[int] = None,
        dropout: float = 0.1,
        num_classes: int = 10,
        n_shot: int = 5,
        negative_train_contrast: bool = False,
        use_augmentation: bool = True,
        use_spec_augment: bool = True,
        use_noise: bool = True,
        time_mask_pct: float = 0.15,
        freq_mask_pct: float = 0.15,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if channels is None:
            channels = [64, 128, 256, 512]

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
        self.n_shot = n_shot
        self.negative_train_contrast = negative_train_contrast

        self.use_augmentation = use_augmentation
        self.augmentation = AudioAugmentation(
            use_spec_augment=use_spec_augment,
            use_noise=use_noise,
            time_mask_pct=time_mask_pct,
            freq_mask_pct=freq_mask_pct,
        )

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.use_augmentation:
            return x
        x, _ = self.augmentation(x)
        return x

    def _forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.permute(0, 2, 1).unsqueeze(1)
        x = self._apply_augmentation(x)
        return self.model.encoder(x)

    def _step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.negative_train_contrast:
            x, x_neg, y, y_neg, _ = batch
            x = torch.cat([x, x_neg], dim=0)
            y = torch.cat([y, y_neg], dim=0)
            loss_fn = prototypical_loss_filter_negative
        else:
            x, y, _ = batch
            loss_fn = prototypical_loss

        embeddings = self._forward_embed(x)
        loss, acc, dist_loss = loss_fn(embeddings, y, self.n_shot)
        return loss, acc, dist_loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, acc, dist_loss = self._step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss + dist_loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, acc, dist_loss = self._step(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss + dist_loss

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, acc, dist_loss = self._step(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        return loss + dist_loss

    def configure_optimizers(self) -> Dict:
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
