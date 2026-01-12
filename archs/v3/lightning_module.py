"""
PyTorch Lightning Module for V3 Prototypical Network (AST encoder).
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from archs.v3.arch import ProtoNetV3
from utils.distance import Distance
from utils.loss import prototypical_loss, prototypical_loss_filter_negative


class ProtoNetV3LightningModule(L.LightningModule):
    """Prototypical Network Lightning Module for AST-based encoder."""

    def __init__(
        self,
        emb_dim: int = 384,
        distance: Union[str, Distance] = "euclidean",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_gamma: float = 0.65,
        scheduler_step_size: int = 10,
        scheduler_type: str = "cosine",
        n_mels: int = 128,
        patch_freq: int = 8,
        patch_time: int = 2,
        max_time_bins: int = 32,
        depth: int = 6,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        dropout: float = 0.1,
        pooling: str = "cls",
        num_classes: int = 10,
        n_shot: int = 5,
        negative_train_contrast: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = ProtoNetV3(
            emb_dim=emb_dim,
            distance=distance,
            n_mels=n_mels,
            patch_freq=patch_freq,
            patch_time=patch_time,
            max_time_bins=max_time_bins,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            pooling=pooling,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_type = scheduler_type
        self.n_shot = n_shot
        self.negative_train_contrast = negative_train_contrast
        self.n_mels = n_mels

    def _forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if x.shape[1] == self.n_mels:
                x = x.unsqueeze(1)
            else:
                x = x.permute(0, 2, 1).unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] != 1:
            x = x.permute(0, 3, 1, 2)
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
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, on_step=True, on_epoch=True)
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
