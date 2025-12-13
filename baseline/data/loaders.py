# data/loaders.py

from pathlib import Path
from typing import List, Tuple, Optional

from torch.utils.data import DataLoader

from .fewshot_dataset import DCASEEventDataset, FewShotEpisodeDataset


def make_dcase_event_dataset(
    root_dir: str | Path,
    annotation_files: List[str | Path],
    audio_subdir: Optional[str] = None,
    positive_label: str = "POS",
    class_name: Optional[str] = None,
    # log-mel params
    target_sr: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 64,
    min_duration: Optional[float] = 0.5,
) -> DCASEEventDataset:
    root_dir = Path(root_dir)
    ann_paths = [Path(a) if not Path(a).is_absolute() else Path(a) for a in annotation_files]
    return DCASEEventDataset(
        root_dir=root_dir,
        annotations=ann_paths,
        audio_subdir=audio_subdir,
        positive_label=positive_label,
        class_name=class_name,
        target_sr=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        min_duration=min_duration,
    )


def make_fewshot_dataloaders(
    train_root: str | Path,
    train_annotation_files: List[str | Path],
    val_root: Optional[str | Path] = None,
    val_annotation_files: Optional[List[str | Path]] = None,
    audio_subdir: Optional[str] = None,
    k_way: int = 5,
    n_shot: int = 5,
    n_query: int = 10,
    batch_size: int = 1,
    num_workers: int = 4,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and (optionally) validation dataloaders that yield few-shot episodes.
    """
    if val_root is None:
        val_root = train_root

    # base flat dataset for training
    train_base = make_dcase_event_dataset(
        root_dir=train_root,
        annotation_files=train_annotation_files,
        audio_subdir=audio_subdir,
    )
    train_episode_dataset = FewShotEpisodeDataset(
        base_dataset=train_base,
        k_way=k_way,
        n_shot=n_shot,
        n_query=n_query,
    )
    train_loader = DataLoader(
        train_episode_dataset,
        batch_size=batch_size,   # usually 1 episode per batch
        num_workers=num_workers,
        shuffle=False,           # episode dataset does its own randomness
        pin_memory=True,
    )

    val_loader = None
    if val_annotation_files is not None and len(val_annotation_files) > 0:
        val_base = make_dcase_event_dataset(
            root_dir=val_root,
            annotation_files=val_annotation_files,
            audio_subdir=audio_subdir,
        )
        val_episode_dataset = FewShotEpisodeDataset(
            base_dataset=val_base,
            k_way=k_way,
            n_shot=n_shot,
            n_query=n_query,
        )
        val_loader = DataLoader(
            val_episode_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )

    return train_loader, val_loader
