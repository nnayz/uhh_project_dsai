"""
PyTorch Lightning DataModule for DCASE Few-Shot Bioacoustic.

This DataModule encapsulates the entire data pipeline for training and evaluation,
supporting both cached features (Phase 2) and on-the-fly extraction (legacy mode).

The recommended workflow is:
    1. Run feature extraction once (Phase 1): g5 extract-features
    2. Training uses cached features (Phase 2): g5 train v1
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import lightning as L
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from .cached_dataset import (
    CachedFeatureDataset,
    CachedFewShotEpisodeDataset,
    create_cached_dataset,
    create_cached_episode_dataset,
)
from .feature_cache import (
    extract_and_cache_features,
    get_cache_dir,
    load_manifest,
    verify_cache_integrity,
)
from .dataset import DCASEEventDataset, FewShotEpisodeDataset

logger = logging.getLogger(__name__)


class DCASEFewShotDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for DCASE Few-Shot Bioacoustic classification.
    
    This DataModule supports two modes:
    
    1. Cached Mode (default, recommended):
        - Loads pre-extracted features from .npy files
        - Fast I/O, reproducible, no audio processing during training
        - Requires running feature extraction first
    
    2. On-the-fly Mode (legacy):
        - Extracts features from audio during training
        - Slower, but no preprocessing required
        - Useful for debugging or small experiments
    
    Example usage:
        ```python
        datamodule = DCASEFewShotDataModule(cfg)
        
        # Ensure features are cached
        datamodule.prepare_data()
        
        # Setup datasets
        datamodule.setup("fit")
        
        # Use in trainer
        trainer.fit(model, datamodule)
        ```
    
    Attributes:
        cfg: Hydra configuration.
        use_cache: Whether to use cached features.
        train_dataset: Training episode dataset.
        val_dataset: Validation episode dataset.
        test_dataset: Test episode dataset.
    """

    def __init__(
        self,
        cfg: DictConfig,
        use_cache: Optional[bool] = None,
        force_recompute: bool = False,
    ) -> None:
        """
        Initialize the DataModule.
        
        Args:
            cfg: Hydra DictConfig with all configuration.
            use_cache: Override for using cached features.
                If None, uses cfg.features.use_cache.
            force_recompute: Force re-extraction of features.
        """
        super().__init__()
        self.cfg = cfg
        self.force_recompute = force_recompute
        
        # Determine caching mode
        if use_cache is not None:
            self.use_cache = use_cache
        else:
            self.use_cache = cfg.features.use_cache
        
        # Datasets (set in setup())
        self.train_dataset: Optional[CachedFewShotEpisodeDataset] = None
        self.val_dataset: Optional[CachedFewShotEpisodeDataset] = None
        self.test_dataset: Optional[CachedFewShotEpisodeDataset] = None
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=["cfg"])

    def prepare_data(self) -> None:
        """
        Extract and cache features if needed.
        
        This method runs on a single process (rank 0) before setup().
        It handles Phase 1: .wav → features → .npy
        
        If use_cache is True and cache doesn't exist, features are extracted.
        If use_cache is False, this is a no-op.
        """
        if not self.use_cache:
            logger.info("Cache disabled, skipping feature extraction in prepare_data")
            return
        
        # Check and extract features for each split
        splits = [
            ("train", self.cfg.annotations.train_files),
            ("val", self.cfg.annotations.val_files),
            ("test", self.cfg.annotations.test_files),
        ]
        
        for split_name, annotation_paths in splits:
            if not annotation_paths:
                continue
            
            cache_dir = get_cache_dir(self.cfg, split_name)
            
            # Check if valid cache exists
            needs_extraction = self.force_recompute
            if not needs_extraction:
                if not cache_dir.exists():
                    needs_extraction = True
                    logger.info(f"Cache not found for {split_name}, will extract features")
                elif not verify_cache_integrity(cache_dir):
                    needs_extraction = True
                    logger.warning(f"Cache integrity check failed for {split_name}")
            
            if needs_extraction:
                logger.info(f"Extracting features for {split_name}...")
                extract_and_cache_features(
                    cfg=self.cfg,
                    split=split_name,
                    annotation_paths=annotation_paths,
                    force_recompute=self.force_recompute,
                )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for training, validation, or testing.
        
        This method runs on every process after prepare_data().
        It loads the appropriate datasets based on the stage.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage == "fit" or stage is None:
            # Training dataset
            if self.cfg.annotations.train_files:
                if self.use_cache:
                    self.train_dataset = create_cached_episode_dataset(
                        cfg=self.cfg,
                        split="train",
                        num_episodes=self.cfg.arch.episodes.episodes_per_epoch,
                    )
                else:
                    self.train_dataset = self._create_legacy_episode_dataset("train")
                
                logger.info(
                    f"Train dataset: {len(self.train_dataset)} episodes, "
                    f"{self.train_dataset.get_num_classes()} classes"
                )
            
            # Validation dataset
            if self.cfg.annotations.val_files:
                val_episodes = getattr(
                    self.cfg.arch.episodes, "val_episodes", 
                    self.cfg.arch.episodes.episodes_per_epoch // 10
                )
                if self.use_cache:
                    self.val_dataset = create_cached_episode_dataset(
                        cfg=self.cfg,
                        split="val",
                        num_episodes=val_episodes,
                    )
                else:
                    self.val_dataset = self._create_legacy_episode_dataset(
                        "val", num_episodes=val_episodes
                    )
                
                logger.info(
                    f"Val dataset: {len(self.val_dataset)} episodes, "
                    f"{self.val_dataset.get_num_classes()} classes"
                )
        
        if stage == "test" or stage is None:
            # Test dataset
            if self.cfg.annotations.test_files:
                test_episodes = getattr(
                    self.cfg.arch.episodes, "test_episodes",
                    self.cfg.arch.episodes.episodes_per_epoch // 10
                )
                if self.use_cache:
                    self.test_dataset = create_cached_episode_dataset(
                        cfg=self.cfg,
                        split="test",
                        num_episodes=test_episodes,
                    )
                else:
                    self.test_dataset = self._create_legacy_episode_dataset(
                        "test", num_episodes=test_episodes
                    )
                
                logger.info(
                    f"Test dataset: {len(self.test_dataset)} episodes, "
                    f"{self.test_dataset.get_num_classes()} classes"
                )

    def _create_legacy_episode_dataset(
        self,
        split: str,
        num_episodes: Optional[int] = None,
    ) -> FewShotEpisodeDataset:
        """
        Create a legacy FewShotEpisodeDataset (on-the-fly extraction).
        
        This is the fallback when cache is disabled.
        
        Args:
            split: Dataset split name.
            num_episodes: Override for number of episodes.
            
        Returns:
            FewShotEpisodeDataset instance.
        """
        # Get annotation paths for split
        if split == "train":
            annotation_paths = [Path(p) for p in self.cfg.annotations.train_files]
        elif split == "val":
            annotation_paths = [Path(p) for p in self.cfg.annotations.val_files]
        else:
            annotation_paths = [Path(p) for p in self.cfg.annotations.test_files]
        
        base_dataset = DCASEEventDataset(
            annotations=annotation_paths,
            cfg=self.cfg,
        )
        
        dataset = FewShotEpisodeDataset(
            base_dataset=base_dataset,
            cfg=self.cfg,
        )
        
        # Override num_episodes if specified
        if num_episodes is not None:
            dataset.num_episodes = num_episodes
        
        return dataset

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not set up. Call setup('fit') first.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.annotations.batch_size,
            num_workers=self.cfg.runtime.num_workers,
            shuffle=False,  # Episode dataset handles randomness
            pin_memory=True,
            prefetch_factor=getattr(self.cfg.runtime, "prefetch_factor", 2),
            persistent_workers=self.cfg.runtime.num_workers > 0,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create the validation DataLoader."""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.annotations.batch_size,
            num_workers=self.cfg.runtime.num_workers,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=getattr(self.cfg.runtime, "prefetch_factor", 2),
            persistent_workers=self.cfg.runtime.num_workers > 0,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Create the test DataLoader."""
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.annotations.batch_size,
            num_workers=self.cfg.runtime.num_workers,
            shuffle=False,
            pin_memory=True,
            prefetch_factor=getattr(self.cfg.runtime, "prefetch_factor", 2),
            persistent_workers=self.cfg.runtime.num_workers > 0,
        )

    def get_train_num_classes(self) -> int:
        """Get number of classes in training set."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not set up.")
        return self.train_dataset.get_num_classes()

    def get_cache_info(self) -> dict:
        """Get information about the feature cache."""
        info = {
            "use_cache": self.use_cache,
            "splits": {},
        }
        
        for split in ["train", "val", "test"]:
            cache_dir = get_cache_dir(self.cfg, split)
            if cache_dir.exists():
                manifest = load_manifest(cache_dir)
                if manifest:
                    info["splits"][split] = {
                        "cache_dir": str(cache_dir),
                        "num_samples": manifest.num_samples,
                        "num_classes": manifest.num_classes,
                        "version": manifest.version,
                    }
        
        return info


def create_datamodule(
    cfg: DictConfig,
    use_cache: Optional[bool] = None,
) -> DCASEFewShotDataModule:
    """
    Factory function to create a DataModule.
    
    Args:
        cfg: Hydra DictConfig.
        use_cache: Override for using cached features.
        
    Returns:
        DCASEFewShotDataModule instance.
    """
    return DCASEFewShotDataModule(cfg=cfg, use_cache=use_cache)

