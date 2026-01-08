"""
PyTorch Lightning DataModule aligned with the reference training behavior.

This module uses:
- Dynamic segment sampling from precomputed feature arrays
- IdentityBatchSampler-based episodic batching
- Optional adaptive segment length test set
"""

from __future__ import annotations

from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from preprocessing.sequence_data import (
    IdentityBatchSampler,
    PrototypeDynamicArrayDataSet,
    PrototypeDynamicArrayDataSetVal,
    PrototypeDynamicArrayDataSetWithEval,
    PrototypeTestSet,
    PrototypeAdaSeglenBetterNegTestSetV2,
)


class DCASEFewShotDataModule(L.LightningDataModule):
    """DataModule for class-balanced episodic training."""

    def __init__(
        self,
        cfg,
        use_cache: Optional[bool] = None,
        force_recompute: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_param = cfg.train_param
        self.eval_param = cfg.eval_param
        self.path = cfg.path
        self.features = cfg.features
        self.use_cache = use_cache
        self.force_recompute = force_recompute

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.train_loader = None
        self.val_loader = None

        self.save_hyperparameters(ignore=["cfg"])
        self.init()

    def prepare_data(self) -> None:
        """Validate that required feature files exist."""
        from preprocessing.feature_export import validate_features

        missing = validate_features(self.cfg)
        if missing:
            sample = "\n".join(str(p) for p in missing[:10])
            raise RuntimeError(
                "Missing feature files (e.g., *_logmel.npy). "
                "Generate them before training.\n"
                f"Example missing files:\n{sample}"
            )

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def init(self, stage: Optional[str] = None) -> None:
        """Initialize datasets and loaders to match reference training behavior."""
        if self.train_param.use_validation_first_5:
            self.dataset = PrototypeDynamicArrayDataSetWithEval(
                path=self.path,
                features=self.features,
                train_param=self.train_param,
            )
        else:
            self.dataset = PrototypeDynamicArrayDataSet(
                path=self.path,
                features=self.features,
                train_param=self.train_param,
            )

        self.sampler = IdentityBatchSampler(
            self.train_param,
            self.dataset.train_eval_class_idxs,
            self.dataset.extra_train_class_idxs,
            batch_size=self.train_param.n_shot * 2,
            n_episode=int(
                len(self.dataset)
                / (self.train_param.k_way * self.train_param.n_shot * 2)
            ),
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_sampler=self.sampler, num_workers=2
        )

        self.val_dataset = PrototypeDynamicArrayDataSetVal(
            path=self.path,
            features=self.features,
            train_param=self.train_param,
        )
        self.val_sampler = IdentityBatchSampler(
            self.train_param,
            self.val_dataset.eval_class_idxs,
            [],
            batch_size=self.train_param.n_shot * 2,
            n_episode=int(
                len(self.val_dataset)
                / (self.train_param.k_way * self.train_param.n_shot * 2)
            ),
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_sampler=self.val_sampler, num_workers=2
        )

        if self.train_param.adaptive_seg_len:
            self.data_test = PrototypeAdaSeglenBetterNegTestSetV2(
                self.path,
                self.features,
                self.train_param,
                self.eval_param,
            )
        else:
            self.data_test = PrototypeTestSet(
                self.path,
                self.features,
                self.train_param,
                self.eval_param,
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )


def create_datamodule(
    cfg, use_cache: Optional[bool] = None, force_recompute: bool = False
):
    """Factory for creating the sequence-based datamodule."""
    return DCASEFewShotDataModule(
        cfg=cfg,
        use_cache=use_cache,
        force_recompute=force_recompute,
    )
