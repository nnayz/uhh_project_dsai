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
from omegaconf import DictConfig

from preprocessing.sequence_data import (
    IdentityBatchSampler,
    PrototypeDynamicArrayDataSet, 
    PrototypeDynamicArrayDataSetVal,
    PrototypeDynamicArrayDataSetWithEval,
    PrototypeTestSet,
    PrototypeAdaSeglenBetterNegTestSetV2,
)


class DCASEFewShotDataModule(L.LightningDataModule):
    """
    DataModule for DCASE few-shot learning task.

    Implements five key methods:
    - prepare_data: prep the data
    - setup: (things to do on every accelerator)
    - train_dataloader: return the train loader
    - val_dataloader: return the val loader
    - test_dataloader: return the test loader


    This allows you to share a full-dataset without explaining how to download, split, transform, and process the data. 
    """

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_param = cfg.train_param
        self.eval_param = cfg.eval_param
        self.path = cfg.path
        self.features = cfg.features

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.train_loader = None
        self.val_loader = None

        self._is_setup = False
        self.save_hyperparameters(ignore=["cfg"])

    def prepare_data(self) -> None:
        """Validate that required feature files exist."""
        from preprocessing.feature_export import validate_features


        # Check only the splits that will be used.
        splits = []
        if getattr(self.cfg, "train", False):
            splits.extend(["train", "val"])
        if getattr(self.cfg, "test", False):
            splits.append("val")
            if getattr(self.cfg.path, "test_dir", None):
                splits.append("test")
        splits = list(dict.fromkeys(splits))
        if not splits:
            return

        missing = validate_features(self.cfg, splits=splits)
        if missing:
            sample = "\n".join(str(p) for p in missing[:10])
            raise RuntimeError(
                f"Missing feature files (e.g., *_logmel.npy). Generate them before training.\nExample missing files:\n{sample}"
            )

    def setup(self, stage: Optional[str] = None) -> None:
        if self._is_setup and stage is None:
            return
        num_workers = int(getattr(self.cfg.runtime, "num_workers", 0))

        if stage in (None, "fit"):
            # Get the training dataset.
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
                self.dataset,
                batch_sampler=self.sampler,
                num_workers=num_workers,
            )

        if stage in (None, "fit", "validate"):
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
                self.val_dataset,
                batch_sampler=self.val_sampler,
                num_workers=num_workers,
            )

        if stage in (None, "test", "validate"):
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

        self._is_setup = True

    def train_dataloader(self):
        if self.train_loader is None:
            self.setup(stage="fit")
        return self.train_loader

    def val_dataloader(self):
        if self.val_loader is None:
            self.setup(stage="validate")
        return self.val_loader

    def test_dataloader(self):
        if self.data_test is None:
            self.setup(stage="test")
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=int(getattr(self.cfg.runtime, "num_workers", 0)),
            pin_memory=True,
            shuffle=False,
        )


def create_datamodule(cfg):
    """Factory for creating the sequence-based datamodule."""
    return DCASEFewShotDataModule(cfg=cfg)
