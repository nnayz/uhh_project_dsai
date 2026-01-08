"""
PyTorch Lightning Module for Prototypical Network (reference training behavior).

Training loop:
- Batch-based prototypical loss over class-balanced batches
- Optional negative contrastive sampling
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from archs.v1.arch import ProtoNet
from utils.distance import Distance
from utils.loss import prototypical_loss, prototypical_loss_filter_negative


class ProtoNetLightningModule(L.LightningModule):
    """Prototypical Network Lightning Module with reference training loop."""

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
        with_bias: bool = False,
        drop_rate: float = 0.1,
        time_max_pool_dim: int = 4,
        non_linearity: str = "leaky_relu",
        layer_4: bool = False,
        num_classes: int = 10,
        n_shot: int = 5,
        negative_train_contrast: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = ProtoNet(
            emb_dim=emb_dim,
            distance=distance,
            with_bias=with_bias,
            drop_rate=drop_rate,
            time_max_pool_dim=time_max_pool_dim,
            non_linearity=non_linearity,
            layer_4=layer_4,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_type = scheduler_type
        self.n_shot = n_shot
        self.negative_train_contrast = negative_train_contrast

    def _forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Encode segments and return embeddings."""
        if x.dim() == 4:
            x = x.squeeze(1).permute(0, 2, 1)
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
