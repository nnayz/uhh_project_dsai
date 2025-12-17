from pathlib import Path
from typing import Tuple, Optional

from torch.utils.data import DataLoader
from omegaconf import DictConfig

from .dataset import DCASEEventDataset, FewShotEpisodeDataset


def make_dcase_event_dataset(
    cfg: DictConfig,
    annotations: Optional[list] = None,
) -> DCASEEventDataset:
    """
    Create a DCASEEventDataset from annotation files.

    Args:
        cfg: Hydra DictConfig with annotations settings.
        annotations: Optional list of annotation paths. If None, uses cfg.annotations.train_files.

    Returns:
        DCASEEventDataset: The created dataset.

    Raises:
        ValueError: If no annotation files are provided.
    """
    ann_paths = annotations if annotations is not None else [Path(p) for p in cfg.annotations.train_files]
    
    if not ann_paths:
        raise ValueError(
            "No annotation files provided. "
            "Either pass 'annotations' argument or set cfg.annotations.train_files."
        )
    
    return DCASEEventDataset(
        annotations=ann_paths,
        cfg=cfg,
    )


def make_fewshot_dataloaders(
    cfg: DictConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and (optionally) validation dataloaders that yield few-shot episodes.

    All parameters are read from the cfg object:
        - annotations.train_files: Training annotation CSV files
        - annotations.val_files: Validation annotation CSV files
        - annotations.positive_label: Label value indicating a positive example
        - annotations.class_name: Explicit class name for annotations
        - annotations.min_duration: Minimum duration in seconds
        - annotations.max_frames: Maximum number of time frames
        - annotations.batch_size: Batch size for dataloaders
        - runtime.num_workers: Number of workers for dataloaders

    Args:
        cfg: Hydra DictConfig.

    Returns:
        Tuple[DataLoader, Optional[DataLoader]]: Train dataloader and optional validation dataloader.

    Raises:
        ValueError: If annotations.train_files is empty.
    """
    train_files = [Path(p) for p in cfg.annotations.train_files]
    
    if not train_files:
        raise ValueError(
            "cfg.annotations.train_files is empty. "
            "Training requires at least one annotation file path."
        )
    
    # Base flat dataset for training
    train_base = DCASEEventDataset(
        annotations=train_files,
        cfg=cfg,
    )
    train_episode_dataset = FewShotEpisodeDataset(
        base_dataset=train_base,
        cfg=cfg,
    )
    train_loader = DataLoader(
        train_episode_dataset,
        batch_size=cfg.annotations.batch_size,
        num_workers=cfg.runtime.num_workers,
        shuffle=False,  # episode dataset does its own randomness
        pin_memory=True,
    )

    val_loader = None
    val_files = [Path(p) for p in cfg.annotations.val_files] if cfg.annotations.val_files else []
    
    if val_files:
        val_base = DCASEEventDataset(
            annotations=val_files,
            cfg=cfg,
        )
        val_episode_dataset = FewShotEpisodeDataset(
            base_dataset=val_base,
            cfg=cfg,
        )
        val_loader = DataLoader(
            val_episode_dataset,
            batch_size=cfg.annotations.batch_size,
            num_workers=cfg.runtime.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    return train_loader, val_loader
