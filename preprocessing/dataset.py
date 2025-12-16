from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from schemas import SegmentExample
from .ann_service import AnnotationService
from .preprocess import extract_logmel_segment
from utils.config import Config


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
        annotations: List[Union[str, Path]],
        config: Optional[Config] = None,
        positive_label: str = "POS",
        class_name: Optional[str] = None,
        # log-mel params (override config if provided)
        target_sr: Optional[int] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        n_mels: Optional[int] = None,
        min_duration: Optional[float] = 0.5,
    ) -> None:
        """
        Initialize the DCASEEventDataset.

        Args:
            annotations: List of paths to annotation CSV files.
            config: Configuration object. If None, uses default Config().
            positive_label: The label value that indicates a positive example.
            class_name: Optional explicit class name to use for all annotations.
            target_sr: Target sampling rate for audio (overrides config).
            n_fft: FFT window size (overrides config).
            hop_length: Hop length for spectrogram (overrides config).
            win_length: Window length for spectrogram (overrides config).
            n_mels: Number of mel frequency bins (overrides config).
            min_duration: Minimum duration in seconds (pads shorter segments).
        """
        super().__init__()

        if config is None:
            config = Config()

        self.positive_label = positive_label
        self.class_name = class_name

        # Use config values as defaults, allow overrides
        _target_sr = target_sr if target_sr is not None else config.SAMPLING_RATE
        _n_fft = n_fft if n_fft is not None else int(config.FRAME_LENGTH * config.SAMPLING_RATE)
        _hop_length = hop_length if hop_length is not None else int(config.HOP_LENGTH * config.SAMPLING_RATE)
        _win_length = win_length if win_length is not None else _n_fft
        _n_mels = n_mels if n_mels is not None else config.N_MELS

        # log-mel params stored for use in __getitem__
        self.spec_params = dict(
            target_sr=_target_sr,
            n_fft=_n_fft,
            hop_length=_hop_length,
            win_length=_win_length,
            n_mels=_n_mels,
            min_duration=min_duration,
        )

        # Use AnnotationService to load and parse annotations
        self.annotation_service = AnnotationService(
            positive_label=positive_label,
            class_name=class_name,
        )

        annotation_paths = [Path(a) for a in annotations]
        self.examples: List[SegmentExample] = self.annotation_service.load_annotations(
            annotation_paths
        )
        self.class_to_idx: Dict[str, int] = self.annotation_service.get_class_to_idx()

        if not self.examples:
            raise RuntimeError(
                f"No positive events found in annotations. "
                f"Check CSV format (columns) and positive_label='{self.positive_label}'."
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
            **self.spec_params,
        )
        tensor = torch.from_numpy(logmel)[None, ...]  # (1, n_mels, T)
        label = ex.class_id
        return tensor, label

    def get_num_classes(self) -> int:
        """
        Return the number of unique classes.

        Returns:
            int: The number of unique classes.
        """
        return len(self.class_to_idx)

    def get_class_to_idx(self) -> Dict[str, int]:
        """
        Return the mapping from class names to indices.

        Returns:
            Dict[str, int]: The mapping from class names to indices.
        """
        return self.class_to_idx.copy()

    def get_idx_to_class(self) -> Dict[int, str]:
        """
        Return the mapping from indices to class names.

        Returns:
            Dict[int, str]: The mapping from indices to class names.
        """
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
        config: Optional[Config] = None,
        k_way: Optional[int] = None,
        n_shot: Optional[int] = None,
        n_query: Optional[int] = None,
        max_frames: int = 512,
        num_episodes: Optional[int] = None,
    ) -> None:
        """
        Initialize the FewShotEpisodeDataset.

        Args:
            base_dataset: The underlying DCASEEventDataset.
            config: Configuration object. If None, uses default Config().
            k_way: Number of classes per episode (overrides config.N_WAY).
            n_shot: Number of support examples per class (overrides config.K_SHOT).
            n_query: Number of query examples per class (overrides config.N_QUERY).
            max_frames: Maximum number of time frames (pads/crops to this).
            num_episodes: Number of episodes per epoch (overrides config.EPISODES_PER_EPOCH).
        """
        super().__init__()

        if config is None:
            config = Config()

        self.base_dataset = base_dataset
        self.k_way = k_way if k_way is not None else config.N_WAY
        self.n_shot = n_shot if n_shot is not None else config.K_SHOT
        self.n_query = n_query if n_query is not None else config.N_QUERY
        self.max_frames = max_frames
        self.num_episodes = num_episodes if num_episodes is not None else config.EPISODES_PER_EPOCH

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
            """
            Crop or pad tensor to fixed time dimension.

            Args:
                t: The tensor to crop or pad.
                T_max: The maximum time dimension.

            Returns:
                The cropped or padded tensor.
            """
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
