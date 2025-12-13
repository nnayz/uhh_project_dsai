# baseline/data/fewshot_dataset.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .preprocess import extract_logmel_segment


@dataclass
class SegmentExample:
    """Represents a single labeled event segment in a file."""
    wav_path: Path
    start_time: float
    end_time: float
    class_id: int


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
        root_dir: str | Path,
        annotations: List[Path],
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
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.audio_subdir = audio_subdir
        self.positive_label = positive_label
        self.class_name = class_name

        # log-mel params stored for use in __getitem__
        self.spec_params = dict(
            target_sr=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            min_duration=min_duration,
        )

        self.examples: List[SegmentExample] = []
        self.class_to_idx: Dict[str, int] = {}

        for ann_path in annotations:
            self._load_annotation_file(ann_path)

        if not self.examples:
            raise RuntimeError(
                f"No positive events found in annotations under {self.root_dir}. "
                f"Check CSV format (columns) and positive_label='{self.positive_label}'."
            )

    # --------- Helper: resolve WAV path robustly ---------
    def _resolve_wav_path(self, audio_dir: Path, wav_name: str | Path) -> Path:
        """
        Build a path to the audio file.

        Strategy:
        1. If there's a .wav with the same stem (e.g. 'a1.wav'), use that.
        2. Otherwise, use the exact name from the CSV.
        """
        wav_name = str(wav_name)
        p = Path(wav_name)

        # Prefer .wav with same stem if it exists
        alt_wav = audio_dir / (p.stem + ".wav")
        if alt_wav.is_file():
            return alt_wav

        # Fall back to the name as given
        candidate = audio_dir / wav_name
        return candidate

    def _load_annotation_file(self, ann_path: str | Path) -> None:
        ann_path = Path(ann_path)
        if not ann_path.is_file():
            return

        df = pd.read_csv(ann_path)

        audio_col = "Audiofilename"
        start_col = "Starttime"
        end_col = "Endtime"

        if audio_col not in df.columns:
            raise ValueError(f"{ann_path} has no '{audio_col}' column")

        # Directory where WAVs live: same folder as the CSV
        audio_dir = ann_path.parent

        # ---------- Case 1: explicit single-class with 'Q' (POS/UNK) ----------
        if "Q" in df.columns and self.class_name is None:
            label_col = "Q"
            # use CSV stem as class name, e.g. 'BV_file123'
            csv_class_name = ann_path.stem
            if csv_class_name not in self.class_to_idx:
                self.class_to_idx[csv_class_name] = len(self.class_to_idx)
            class_id = self.class_to_idx[csv_class_name]

            for _, row in df.iterrows():
                value = str(row[label_col]).upper()
                if value != self.positive_label:
                    continue

                wav_name = row[audio_col]
                wav_path = self._resolve_wav_path(audio_dir, wav_name)
                start_time = float(row[start_col])
                end_time = float(row[end_col])

                self.examples.append(
                    SegmentExample(
                        wav_path=wav_path,
                        start_time=start_time,
                        end_time=end_time,
                        class_id=class_id,
                    )
                )
            return

        # ---------- Case 2: single-class but class_name given explicitly ----------
        if self.class_name is not None and "Q" in df.columns:
            label_col = "Q"
            for _, row in df.iterrows():
                value = str(row[label_col]).upper()
                if value != self.positive_label:
                    continue

                wav_name = row[audio_col]
                wav_path = self._resolve_wav_path(audio_dir, wav_name)
                start_time = float(row[start_col])
                end_time = float(row[end_col])

                if self.class_name not in self.class_to_idx:
                    self.class_to_idx[self.class_name] = len(self.class_to_idx)
                class_id = self.class_to_idx[self.class_name]

                self.examples.append(
                    SegmentExample(
                        wav_path=wav_path,
                        start_time=start_time,
                        end_time=end_time,
                        class_id=class_id,
                    )
                )
            return

        # ---------- Case 3: multi-class with CLASS_x columns (original DCASE style) ----------
        class_cols = [c for c in df.columns if c.startswith("CLASS_")]
        if class_cols:
            for _, row in df.iterrows():
                wav_name = row[audio_col]
                wav_path = self._resolve_wav_path(audio_dir, wav_name)
                start_time = float(row[start_col])
                end_time = float(row[end_col])

                for c in class_cols:
                    value = str(row[c]).upper()
                    if value != self.positive_label:
                        continue

                    class_label = c  # e.g. "CLASS_1"
                    if class_label not in self.class_to_idx:
                        self.class_to_idx[class_label] = len(self.class_to_idx)
                    class_id = self.class_to_idx[class_label]

                    self.examples.append(
                        SegmentExample(
                            wav_path=wav_path,
                            start_time=start_time,
                            end_time=end_time,
                            class_id=class_id,
                        )
                    )
            return

        # ---------- Case 4 (fallback): no 'Q' and no 'CLASS_*' ----------
        # Treat ALL rows as positive events for a single class per CSV.
        csv_class_name = ann_path.stem
        if csv_class_name not in self.class_to_idx:
            self.class_to_idx[csv_class_name] = len(self.class_to_idx)
        class_id = self.class_to_idx[csv_class_name]

        for _, row in df.iterrows():
            wav_name = row[audio_col]
            wav_path = self._resolve_wav_path(audio_dir, wav_name)
            start_time = float(row[start_col])
            end_time = float(row[end_col])

            self.examples.append(
                SegmentExample(
                    wav_path=wav_path,
                    start_time=start_time,
                    end_time=end_time,
                    class_id=class_id,
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
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
        k_way: int,
        n_shot: int,
        n_query: int,
        max_frames: int = 512,  # <--- limit time dimension
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.k_way = k_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.max_frames = max_frames

        # map from class_id to list of indices
        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, ex in enumerate(self.base_dataset.examples):
            self.class_to_indices.setdefault(ex.class_id, []).append(idx)

        self.class_ids = list(self.class_to_indices.keys())

    def __len__(self) -> int:
        # arbitrary large number for episode sampling
        return 1000

    def __getitem__(self, idx: int):
        # sample k_way distinct classes
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
            query_idx = chosen[self.n_shot :]

            for i in support_idx:
                x, _ = self.base_dataset[i]
                support_x_list.append(x)
                support_y_list.append(class_id)

            for i in query_idx:
                x, _ = self.base_dataset[i]
                query_x_list.append(x)
                query_y_list.append(class_id)

        # ---------- CROP + PAD TO FIXED MAX FRAMES ----------
        T_max = self.max_frames

        def crop_pad(t: torch.Tensor, T_max: int) -> torch.Tensor:
            # t: (1, n_mels, T)
            T = t.shape[-1]
            if T > T_max:
                # center crop or left crop; here we just take first T_max
                t = t[..., :T_max]
            elif T < T_max:
                diff = T_max - T
                t = F.pad(t, (0, diff))  # pad at the end in time dimension
            return t

        support_x_padded = [crop_pad(t, T_max) for t in support_x_list]
        query_x_padded = [crop_pad(t, T_max) for t in query_x_list]

        support_x = torch.stack(support_x_padded, dim=0)  # (k*n_shot, 1, n_mels, T_max)
        query_x = torch.stack(query_x_padded, dim=0)      # (k*n_query, 1, n_mels, T_max)
        support_y = torch.tensor(support_y_list, dtype=torch.long)
        query_y = torch.tensor(query_y_list, dtype=torch.long)

        return support_x, support_y, query_x, query_y
