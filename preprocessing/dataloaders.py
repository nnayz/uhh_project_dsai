from pathlib import Path
from typing import List, Tuple, Optional, Union

from torch.utils.data import DataLoader

from .dataset import DCASEEventDataset, FewShotEpisodeDataset
from utils.config import Config


def make_dcase_event_dataset(
    annotation_files: List[Union[str, Path]],
    config: Optional[Config] = None,
    positive_label: str = "POS",
    class_name: Optional[str] = None,
    min_duration: Optional[float] = 0.5,
) -> DCASEEventDataset:
    """
    Create a DCASEEventDataset from annotation files.

    Args:
        annotation_files: List of paths to annotation CSV files.
        config: Configuration object. If None, uses default Config().
        positive_label: The label value that indicates a positive example.
        class_name: Optional explicit class name to use for all annotations.
        min_duration: Minimum duration in seconds (pads shorter segments).

    Returns:
        DCASEEventDataset: The created dataset.
    """
    if config is None:
        config = Config()

    # Convert frame/hop length from seconds to samples
    n_fft = int(config.FRAME_LENGTH * config.SAMPLING_RATE)
    hop_length = int(config.HOP_LENGTH * config.SAMPLING_RATE)
    win_length = n_fft

    ann_paths = [Path(a) for a in annotation_files]
    return DCASEEventDataset(
        annotations=ann_paths,
        positive_label=positive_label,
        class_name=class_name,
        target_sr=config.SAMPLING_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=config.N_MELS,
        min_duration=min_duration,
    )


def make_fewshot_dataloaders(
    train_annotation_files: List[Union[str, Path]],
    config: Optional[Config] = None,
    val_annotation_files: Optional[List[Union[str, Path]]] = None,
    positive_label: str = "POS",
    class_name: Optional[str] = None,
    min_duration: Optional[float] = 0.5,
    max_frames: int = 512,
    batch_size: int = 1,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and (optionally) validation dataloaders that yield few-shot episodes.

    Args:
        train_annotation_files: List of paths to training annotation CSV files.
        config: Configuration object. If None, uses default Config().
        val_annotation_files: Optional list of paths to validation annotation CSV files.
        positive_label: The label value that indicates a positive example.
        class_name: Optional explicit class name to use for all annotations.
        min_duration: Minimum duration in seconds (pads shorter segments).
        max_frames: Maximum number of time frames (pads/crops to this).
        batch_size: Batch size for dataloaders (usually 1 episode per batch).

    Returns:
        Tuple[DataLoader, Optional[DataLoader]]: Train dataloader and optional validation dataloader.
    """
    if config is None:
        config = Config()

    # Convert frame/hop length from seconds to samples
    n_fft = int(config.FRAME_LENGTH * config.SAMPLING_RATE)
    hop_length = int(config.HOP_LENGTH * config.SAMPLING_RATE)
    win_length = n_fft

    # Base flat dataset for training
    train_base = DCASEEventDataset(
        annotations=[Path(a) for a in train_annotation_files],
        positive_label=positive_label,
        class_name=class_name,
        target_sr=config.SAMPLING_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=config.N_MELS,
        min_duration=min_duration,
    )
    train_episode_dataset = FewShotEpisodeDataset(
        base_dataset=train_base,
        k_way=config.N_WAY,
        n_shot=config.K_SHOT,
        n_query=config.N_QUERY,
        max_frames=max_frames,
        num_episodes=config.EPISODES_PER_EPOCH,
    )
    train_loader = DataLoader(
        train_episode_dataset,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        shuffle=False,  # episode dataset does its own randomness
        pin_memory=True,
    )

    val_loader = None
    if val_annotation_files is not None and len(val_annotation_files) > 0:
        val_base = DCASEEventDataset(
            annotations=[Path(a) for a in val_annotation_files],
            positive_label=positive_label,
            class_name=class_name,
            target_sr=config.SAMPLING_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=config.N_MELS,
            min_duration=min_duration,
        )
        val_episode_dataset = FewShotEpisodeDataset(
            base_dataset=val_base,
            k_way=config.N_WAY,
            n_shot=config.K_SHOT,
            n_query=config.N_QUERY,
            max_frames=max_frames,
            num_episodes=config.EPISODES_PER_EPOCH,
        )
        val_loader = DataLoader(
            val_episode_dataset,
            batch_size=batch_size,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
            pin_memory=True,
        )

    return train_loader, val_loader
