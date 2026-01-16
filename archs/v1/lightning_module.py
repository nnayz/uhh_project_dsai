"""
PyTorch Lightning Module for Prototypical Network (reference training behavior).

Training loop:
- Batch-based prototypical loss over class-balanced batches
- Optional negative contrastive sampling
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import os
import copy
from pathlib import Path
import numpy as np
import pandas as pd

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
        self.onset_offset = {}

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

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.is_global_zero:
            return
        self._run_val_event_eval()

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

    def _euclidean_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise ValueError("Embedding dimension mismatch")
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x - y, 2).sum(2)

    def _get_probability(self, proto_pos, proto_neg, query_emb):
        prototypes = torch.stack([proto_pos, proto_neg]).squeeze(1)
        dists = self._euclidean_dist(query_emb, prototypes)
        logits = -dists
        prob = torch.softmax(logits, dim=1)
        prob_pos = prob[:, 0]
        return prob_pos.detach().cpu().tolist()

    def _evaluate_prototypes(
        self, x_pos, x_neg, x_query, hop_seg, strt_index_query, audio_name
    ):
        device = self.device
        x_pos = torch.tensor(x_pos)
        x_neg = torch.tensor(x_neg)
        x_query = torch.tensor(x_query)

        pos_dataset = torch.utils.data.TensorDataset(
            x_pos, torch.zeros(x_pos.shape[0], dtype=torch.long)
        )
        pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_size=None)

        query_dataset = torch.utils.data.TensorDataset(
            x_query, torch.zeros(x_query.shape[0], dtype=torch.long)
        )
        q_loader = torch.utils.data.DataLoader(
            dataset=query_dataset,
            batch_size=self.trainer.datamodule.eval_param.query_batch_size,
            shuffle=False,
        )

        emb_dim = self.trainer.datamodule.features.embedding_dim
        pos_set_feat = torch.zeros(0, emb_dim).cpu()

        for batch in pos_loader:
            x, _ = batch
            if x.dim() == 2:
                x = x.unsqueeze(0)
            feat = self.model.encoder(x.to(device)).cpu()
            feat_mean = feat.mean(dim=0).unsqueeze(0)
            pos_set_feat = torch.cat((pos_set_feat, feat_mean), dim=0)
        pos_proto = pos_set_feat.mean(dim=0)

        prob_comb = []
        iterations = self.trainer.datamodule.eval_param.iterations
        samples_neg = self.trainer.datamodule.eval_param.samples_neg

        for _ in range(iterations):
            prob_pos_iter = []
            n_neg = min(len(x_neg), samples_neg)
            if n_neg == 0:
                continue
            neg_indices = torch.randperm(len(x_neg))[:n_neg]
            x_neg_ind = x_neg[neg_indices]
            if x_neg_ind.dim() == 2:
                x_neg_ind = x_neg_ind.unsqueeze(0)
            feat_neg = self.model.encoder(x_neg_ind.to(device)).detach().cpu()
            proto_neg = feat_neg.mean(dim=0)
            for batch in q_loader:
                x_q, _ = batch
                emb_q = self.model.encoder(x_q.to(device)).detach().cpu()
                prob_pos_iter.extend(self._get_probability(pos_proto, proto_neg, emb_q))
            prob_comb.append(prob_pos_iter)

        prob_final = np.mean(np.array(prob_comb), axis=0)
        thresh_list = np.arange(0, 1, 0.05)
        onset_offset_ret = {}
        for thresh in thresh_list:
            krn = np.array([1, -1])
            prob_thresh = np.where(prob_final > thresh, 1, 0)
            changes = np.convolve(krn, prob_thresh)
            onset_frames = np.where(changes == 1)[0]
            offset_frames = np.where(changes == -1)[0]

            str_time_query = (
                strt_index_query
                * self.trainer.datamodule.features.hop_mel
                / self.trainer.datamodule.features.sr
            )

            onset = (
                onset_frames
                * hop_seg
                * self.trainer.datamodule.features.hop_mel
                / self.trainer.datamodule.features.sr
            )
            onset = onset + str_time_query

            offset = (
                offset_frames
                * hop_seg
                * self.trainer.datamodule.features.hop_mel
                / self.trainer.datamodule.features.sr
            )
            offset = offset + str_time_query
            if len(onset) != len(offset):
                min_len = min(len(onset), len(offset))
                onset = onset[:min_len]
                offset = offset[:min_len]
            onset_offset_ret[thresh] = [onset, offset]
        return onset_offset_ret

    def _run_val_event_eval(self) -> None:
        from preprocessing.sequence_data import PrototypeAdaSeglenBetterNegTestSetV2
        from utils.evaluation import evaluate as eval_fn
        from utils.post_proc import post_processing as post_proc
        from utils.post_proc_new import post_processing as post_proc_new

        self.onset_offset = {}

        path = copy.deepcopy(self.trainer.datamodule.path)
        path.test_dir = None
        eval_dataset = PrototypeAdaSeglenBetterNegTestSetV2(
            path,
            self.trainer.datamodule.features,
            self.trainer.datamodule.train_param,
            self.trainer.datamodule.eval_param,
        )
        loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        for batch in loader:
            (
                (x_pos, x_neg, x_query, _, _, _, hop_seg, _, _, _),
                strt_index_query,
                audio_path,
                _,
            ) = batch
            x_pos = x_pos[0].cpu().numpy()
            x_neg = x_neg[0].cpu().numpy()
            x_query = x_query[0].cpu().numpy()
            hop_seg = hop_seg[0].item()
            strt_index_query = strt_index_query[0].item()
            audio_path = audio_path[0]

            onset_offset = self._evaluate_prototypes(
                x_pos,
                x_neg,
                x_query,
                hop_seg,
                strt_index_query,
                audio_path,
            )

            for k in onset_offset.keys():
                if k not in self.onset_offset:
                    self.onset_offset[k] = {
                        "name_arr": np.array([]),
                        "onset_arr": np.array([]),
                        "offset_arr": np.array([]),
                    }
                name = np.repeat(audio_path, len(onset_offset[k][0]))
                self.onset_offset[k]["name_arr"] = np.append(
                    self.onset_offset[k]["name_arr"], name
                )
                self.onset_offset[k]["onset_arr"] = np.append(
                    self.onset_offset[k]["onset_arr"], onset_offset[k][0]
                )
                self.onset_offset[k]["offset_arr"] = np.append(
                    self.onset_offset[k]["offset_arr"], onset_offset[k][1]
                )

        # Get log_dir from datamodule's config (trainer.log_dir may be None with MLFlowLogger)
        log_dir = self.trainer.datamodule.cfg.runtime.log_dir
        out_root = Path(log_dir) / "val_eval" / f"epoch_{self.current_epoch:03d}"
        out_root.mkdir(parents=True, exist_ok=True)

        best = None
        # best_label = None

        for k in self.onset_offset.keys():
            df_out = pd.DataFrame(
                {
                    "Audiofilename": [
                        os.path.basename(x) for x in self.onset_offset[k]["name_arr"]
                    ],
                    "Starttime": self.onset_offset[k]["onset_arr"],
                    "Endtime": self.onset_offset[k]["offset_arr"],
                }
            )
            alpha_dir = out_root / str(k)
            alpha_dir.mkdir(parents=True, exist_ok=True)
            eval_raw = alpha_dir / "Eval_raw.csv"
            df_out.to_csv(eval_raw, index=False)

            raw_scores, _, _, _ = eval_fn(
                str(eval_raw),
                self.trainer.datamodule.path.eval_dir + "/",
                "run_raw",
                "VAL",
                str(alpha_dir),
            )
            if best is None or raw_scores["fmeasure"] > best["fmeasure"]:
                best = raw_scores
                best_label = f"raw_{k}"

            for threshold in np.arange(0.2, 0.6, 0.1):
                out_csv = (
                    alpha_dir / f"Eval_VAL_threshold_ada_postproc_{threshold:.1f}.csv"
                )
                post_proc(
                    self.trainer.datamodule.path.eval_dir + "/",
                    str(eval_raw),
                    str(out_csv),
                    threshold=threshold,
                )
                scores, _, _, _ = eval_fn(
                    str(out_csv),
                    self.trainer.datamodule.path.eval_dir + "/",
                    f"run_minlen_{threshold:.1f}",
                    "VAL",
                    str(alpha_dir),
                )
                if scores["fmeasure"] > best["fmeasure"]:
                    best = scores
                    best_label = f"minlen_{k}_{threshold:.1f}"

            for threshold_length in np.arange(0.05, 0.25, 0.05):
                out_csv = (
                    alpha_dir
                    / f"Eval_VAL_threshold_fix_length_postproc_{threshold_length:.2f}.csv"
                )
                post_proc_new(
                    self.trainer.datamodule.path.eval_dir + "/",
                    str(eval_raw),
                    str(out_csv),
                    threshold_length=threshold_length,
                )
                scores, _, _, _ = eval_fn(
                    str(out_csv),
                    self.trainer.datamodule.path.eval_dir + "/",
                    f"run_fixed_{threshold_length:.2f}",
                    "VAL",
                    str(alpha_dir),
                )
                if scores["fmeasure"] > best["fmeasure"]:
                    best = scores
                    # best_label = f"fixed_{k}_{threshold_length:.2f}"

        if best is not None:
            # Log the best f-measure (for checkpointing and early stopping)
            self.log("val/fmeasure", best["fmeasure"], prog_bar=True)
