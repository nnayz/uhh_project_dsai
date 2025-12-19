import torch
import torch.nn.functional as F
from torch.optim import AdamW
import lightning as L
from typing import Tuple

from archs.v1.arch import ProtoNet
from utils.distance import Distance

class ProtoNetLightningModule(L.LightningModule):
    def __init__(
        self, 
        emb_dim: int = 128, 
        distance: Distance = Distance.EUCLIDEAN,
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ) -> None:
        """
        Prototypical Network Lightning Module.

        Args:
            emb_dim: Embedding dimension
            distance: Distance metric
            lr: Learning rate
            weight_decay: Weight decay
        """
        super().__init__()
        self.model = ProtoNet(emb_dim=emb_dim, distance=distance)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(
        self, 
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor
    ) -> torch.Tensor:
        """
        Run one few shot episode.

        Returns:
            loss: scalar
            logits: (Nq, Nc)
        """
        return self.model(support_x, support_y, query_x, query_y)

    def _map_labels(
        self,
        support_y: torch.Tensor,
        query_y: torch.Tensor
    ) -> torch.Tensor:
        """
        Map original labels to [0..Nc-1]. Match query labels to prototype indices (same logic as ProtoNet)

        Args:
            support_y: (Ns,)
            query_y: (Nq,)
        Returns:
            mapped_query_y: (Nq,)
        """
        class_ids = torch.unique(support_y)
        label_map = {
            int(c.item()): i for i, c in enumerate(class_ids)
        }
        return torch.tensor(
            [label_map[int(l.item())] for l in query_y],
            dtype=torch.long,
            device=query_y.device
        )

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: (support_x, support_y, query_x, query_y)
            batch_idx: batch index
        Returns:
            loss: scalar
        """

        support_x, support_y, query_x, query_y = [t.squeeze(0) for t in batch]
        loss, logits = self(support_x, support_y, query_x, query_y)
        mapped_query_y = self._map_labels(support_y, query_y)
        acc = (logits.argmax(dim=1) == mapped_query_y).float().mean()

        self.log_dict({
            "train_loss": loss,
            "train_acc": acc
        },
        prog_bar=True,
        on_step=False,
        on_epoch=True
        )

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch: (support_x, support_y, query_x, query_y)
            batch_idx: batch index
        Returns:
            loss: scalar
        """
        support_x, support_y, query_x, query_y = [t.squeeze(0) for t in batch]
        loss, logits = self(support_x, support_y, query_x, query_y)
        mapped_query_y = self._map_labels(support_y, query_y)
        acc = (logits.argmax(dim=1) == mapped_query_y).float().mean()

        self.log_dict({
            "val_loss": loss,
            "val_acc": acc
        },
        prog_bar=True,
        on_step=False,
        on_epoch=True
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer.

        Returns:
            optimizer: torch.optim.Optimizer
        """
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)