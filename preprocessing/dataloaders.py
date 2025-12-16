from pathlib import Path
from typing import Tuple, Optional

from torch.utils.data import DataLoader

from .dataset import DCASEEventDataset, FewShotEpisodeDataset
from utils.config import Config


def make_dcase_event_dataset(
    config: Config,
    annotations: Optional[list] = None,
) -> DCASEEventDataset:
    """
    Create a DCASEEventDataset from annotation files.

    Args:
        config: Configuration object (required).
        annotations: Optional list of annotation paths. If None, uses config.TRAIN_ANNOTATION_FILES.

    Returns:
        DCASEEventDataset: The created dataset.

    Raises:
        ValueError: If no annotation files are provided.
    """
    ann_paths = annotations if annotations is not None else config.TRAIN_ANNOTATION_FILES
    
    if not ann_paths:
        raise ValueError(
            "No annotation files provided. "
            "Either pass 'annotations' argument or set config.TRAIN_ANNOTATION_FILES."
        )
    
    return DCASEEventDataset(
        annotations=ann_paths,
        config=config,
    )


def make_fewshot_dataloaders(
    config: Config,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and (optionally) validation dataloaders that yield few-shot episodes.

    All parameters are read from the config object:
        - TRAIN_ANNOTATION_FILES: Training annotation CSV files
        - VAL_ANNOTATION_FILES: Validation annotation CSV files
        - POSITIVE_LABEL: Label value indicating a positive example
        - CLASS_NAME: Explicit class name for annotations
        - MIN_DURATION: Minimum duration in seconds
        - MAX_FRAMES: Maximum number of time frames
        - BATCH_SIZE: Batch size for dataloaders
        - NUM_WORKERS: Number of workers for dataloaders

    Args:
        config: Configuration object (required).

    Returns:
        Tuple[DataLoader, Optional[DataLoader]]: Train dataloader and optional validation dataloader.

    Raises:
        ValueError: If TRAIN_ANNOTATION_FILES is empty.
    """
    if not config.TRAIN_ANNOTATION_FILES:
        raise ValueError(
            "config.TRAIN_ANNOTATION_FILES is empty. "
            "Training requires at least one annotation file path."
        )
    
    # Base flat dataset for training
    train_base = DCASEEventDataset(
        annotations=config.TRAIN_ANNOTATION_FILES,
        config=config,
    )
    train_episode_dataset = FewShotEpisodeDataset(
        base_dataset=train_base,
        config=config,
    )
    train_loader = DataLoader(
        train_episode_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,  # episode dataset does its own randomness
        pin_memory=True,
    )

    val_loader = None
    if config.VAL_ANNOTATION_FILES and len(config.VAL_ANNOTATION_FILES) > 0:
        val_base = DCASEEventDataset(
            annotations=config.VAL_ANNOTATION_FILES,
            config=config,
        )
        val_episode_dataset = FewShotEpisodeDataset(
            base_dataset=val_base,
            config=config,
        )
        val_loader = DataLoader(
            val_episode_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
            pin_memory=True,
        )

    return train_loader, val_loader
