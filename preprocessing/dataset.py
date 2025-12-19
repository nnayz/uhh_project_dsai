from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from omegaconf import DictConfig

from schemas import SegmentExample
from .ann_service import AnnotationService
from .preprocess import extract_logmel_segment


class DCASEEventDataset(Dataset):
    """
    Flat dataset of all labeled segments (events) across a DCASE-style directory.

    Handles:
      - multi-class CSVs with CLASS_x columns (POS/NEG/UNK)
      - single-class CSVs with a 'Q' column (POS/UNK)
      - fallback: CSVs with only Audiofilename/Starttime/Endtime (all rows are POS)
    """

    def __init__(
        self,
        annotations: List[Path],
        cfg: DictConfig,
    ) -> None:
        """
        Initialize the DCASEEventDataset.

        Args:
            annotations: List of paths to annotation CSV files.
            cfg: Hydra DictConfig with annotations and data settings.
        """
        super().__init__()

        self.cfg = cfg

        # Use AnnotationService to load and parse annotations
        self.annotation_service = AnnotationService(
            positive_label=cfg.annotations.positive_label,
            class_name=cfg.annotations.class_name,
        )

        annotation_paths = annotations
        self.examples: List[SegmentExample] = self.annotation_service.load_annotations(
            annotation_paths=annotation_paths,
        )
        self.class_to_idx: Dict[str, int] = self.annotation_service.get_class_to_idx()

        if not self.examples:
            raise RuntimeError(
                f"No positive events found in annotations. "
                f"Check CSV format (columns) and positive_label='{cfg.annotations.positive_label}'."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single example by index.

        Args:
            idx: Index of the example.

        Returns:
            Tuple of (tensor, label) where tensor is shape (1, n_mels, T).
        """
        ex = self.examples[idx]
        logmel = extract_logmel_segment(
            wav_path=ex.wav_path,
            start_time=ex.start_time,
            end_time=ex.end_time,
            cfg=self.cfg,
        )
        tensor = torch.from_numpy(logmel)[None, ...]  # (1, n_mels, T)
        label = ex.class_id
        return tensor, label

    def get_num_classes(self) -> int:
        """Return the number of unique classes."""
        return len(self.class_to_idx)

    def get_class_to_idx(self) -> Dict[str, int]:
        """Return the mapping from class names to indices."""
        return self.class_to_idx.copy()

    def get_idx_to_class(self) -> Dict[int, str]:
        """Return the mapping from indices to class names."""
        return {v: k for k, v in self.class_to_idx.items()}


class FewShotEpisodeDataset(Dataset):
    """
    Wraps a flat DCASEEventDataset and yields few-shot episodes.

    Each __getitem__ returns:
        support_x: (k_way * n_shot, 1, n_mels, T)
        support_y: (k_way * n_shot,)
        query_x:   (k_way * n_query, 1, n_mels, T)
        query_y:   (k_way * n_query,)
    """

    def __init__(
        self,
        base_dataset: DCASEEventDataset,
        cfg: DictConfig,
    ) -> None:
        """
        Initialize the FewShotEpisodeDataset.

        Args:
            base_dataset: The underlying DCASEEventDataset.
            cfg: Hydra DictConfig with episodes settings.
        """
        super().__init__()

        self.base_dataset = base_dataset
        self.k_way = cfg.arch.episodes.n_way
        self.n_shot = cfg.arch.episodes.k_shot
        self.n_query = cfg.arch.episodes.n_query
        self.max_frames = cfg.annotations.max_frames
        self.num_episodes = cfg.arch.episodes.episodes_per_epoch

        # Map from class_id to list of indices in base_dataset
        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, ex in enumerate(self.base_dataset.examples):
            self.class_to_indices.setdefault(ex.class_id, []).append(idx)

        self.class_ids = list(self.class_to_indices.keys())

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a few-shot episode.

        Args:
            idx: Episode index (used for seeding if needed).

        Returns:
            Tuple of (support_x, support_y, query_x, query_y).
        """
        import numpy as np

        # Sample k_way distinct classes
        if len(self.class_ids) < self.k_way:
            raise RuntimeError(
                f"Not enough classes ({len(self.class_ids)}) to sample k_way={self.k_way}"
            )

        chosen_classes = np.random.choice(
            self.class_ids, size=self.k_way, replace=False
        )

        support_x_list = []
        support_y_list = []
        query_x_list = []
        query_y_list = []

        for class_id in chosen_classes:
            indices = self.class_to_indices[class_id]
            needed = self.n_shot + self.n_query
            if len(indices) < needed:
                chosen = np.random.choice(indices, size=needed, replace=True)
            else:
                chosen = np.random.choice(indices, size=needed, replace=False)

            support_idx = chosen[: self.n_shot]
            query_idx = chosen[self.n_shot:]

            for i in support_idx:
                x, _ = self.base_dataset[i]
                support_x_list.append(x)
                support_y_list.append(class_id)

            for i in query_idx:
                x, _ = self.base_dataset[i]
                query_x_list.append(x)
                query_y_list.append(class_id)

        # Crop/pad to fixed max frames
        T_max = self.max_frames

        def crop_pad(t: torch.Tensor, T_max: int) -> torch.Tensor:
            """Crop or pad tensor to fixed time dimension."""
            T = t.shape[-1]
            if T > T_max:
                t = t[..., :T_max]
            elif T < T_max:
                diff = T_max - T
                t = F.pad(t, (0, diff))
            return t

        support_x_padded = [crop_pad(t, T_max) for t in support_x_list]
        query_x_padded = [crop_pad(t, T_max) for t in query_x_list]

        support_x = torch.stack(support_x_padded, dim=0)
        query_x = torch.stack(query_x_padded, dim=0)
        support_y = torch.tensor(support_y_list, dtype=torch.long)
        query_y = torch.tensor(query_y_list, dtype=torch.long)

        return support_x, support_y, query_x, query_y
