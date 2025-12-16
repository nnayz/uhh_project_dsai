from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

from schemas import SegmentExample


class AnnotationService:
    """
    Service for loading and parsing DCASE-style annotation files.
    
    Handles:
      - multi-class CSVs with CLASS_x columns (POS/NEG/UNK)
      - single-class CSVs with a 'Q' column (POS/UNK)
      - fallback: CSVs with only Audiofilename/Starttime/Endtime (all rows are POS)
    """

    def __init__(
        self,
        positive_label: str = "POS",
        class_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the AnnotationService.

        Args:
            positive_label: The label value that indicates a positive example.
            class_name: Optional explicit class name to use for all annotations.
        """
        self.positive_label = positive_label
        self.class_name = class_name
        self.class_to_idx: Dict[str, int] = {}
        self.examples: List[SegmentExample] = []

    def _resolve_wav_path(
        self,
        audio_dir: Path,
        wav_name: Union[str, Path]
    ) -> Path:
        """
        Build a path to the audio file.

        Strategy:
        1. If there is a .wav with the same stem (e.g. 'a1.wav'), use that.
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

    def load_annotations(
        self,
        annotation_paths: List[Union[str, Path]],
    ) -> List[SegmentExample]:
        """
        Load all annotation files and return the list of examples.

        Args:
            annotation_paths: List of paths to annotation CSV files (supports glob patterns).

        Returns:
            List of SegmentExample objects.
        """
        self.examples = []
        self.class_to_idx = {}

        for ann_path in annotation_paths:
            path = Path(ann_path)
            path_str = str(path)
            # Expand glob patterns (e.g., "/data/Training_Set/**/*.csv")
            if "*" in path_str or "?" in path_str:
                # Find the root (non-glob) portion of the path
                parts = path.parts
                root_parts = []
                for part in parts:
                    if "*" in part or "?" in part:
                        break
                    root_parts.append(part)
                root = Path(*root_parts) if root_parts else Path(".")
                # Get the glob pattern relative to root
                glob_pattern = str(path.relative_to(root))
                expanded_paths = sorted(root.glob(glob_pattern))
                for expanded_path in expanded_paths:
                    self._load_annotation_file(expanded_path)
            else:
                self._load_annotation_file(path)

        return self.examples

    def _load_annotation_file(
        self,
        annotation_path: Path,
    ) -> None:
        """
        Load a single annotation file and populate self.examples.

        Args:
            annotation_path: Path to the annotation CSV file.
        """
        if not annotation_path.is_file():
            return

        df = pd.read_csv(annotation_path)

        audio_col = "Audiofilename"
        start_col = "Starttime"
        end_col = "Endtime"

        if audio_col not in df.columns:
            raise ValueError(f"{annotation_path} has no '{audio_col}' column")

        # Directory where WAVs are: same folder as CSV
        audio_dir = annotation_path.parent

        # Case 1: explicit single-class with 'Q' (POS/UNK), no explicit class_name
        if "Q" in df.columns and self.class_name is None:
            self._parse_single_class_q(df, annotation_path, audio_dir, audio_col, start_col, end_col)
            return

        # Case 2: single-class but class_name given explicitly
        if self.class_name is not None and "Q" in df.columns:
            self._parse_explicit_class_name(df, audio_dir, audio_col, start_col, end_col)
            return

        # Case 3: multi-class with CLASS_x columns (original DCASE style)
        class_cols = [c for c in df.columns if c.startswith("CLASS_")]
        if class_cols:
            self._parse_multi_class(df, class_cols, audio_dir, audio_col, start_col, end_col)
            return

        # Case 4 (fallback): no 'Q' and no 'CLASS_*'
        # Treat ALL rows as positive events for a single class per CSV.
        self._parse_fallback(df, annotation_path, audio_dir, audio_col, start_col, end_col)

    def _parse_single_class_q(
        self,
        df: pd.DataFrame,
        annotation_path: Path,
        audio_dir: Path,
        audio_col: str,
        start_col: str,
        end_col: str,
    ) -> None:
        """Parse CSV with 'Q' column, using CSV stem as class name."""
        label_col = "Q"
        # Use CSV stem as class name, e.g., 'BV_file123'
        csv_class_name = annotation_path.stem
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

    def _parse_explicit_class_name(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        audio_col: str,
        start_col: str,
        end_col: str,
    ) -> None:
        """
        
        Parse CSV with 'Q' column, using explicit class_name.

        Args:
            df: The DataFrame to parse.
            audio_dir: The directory containing the audio files.
            audio_col: The column containing the audio file names.
            start_col: The column containing the start time.
            end_col: The column containing the end time.
            
        Returns:
            None
        """
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

    def _parse_multi_class(
        self,
        df: pd.DataFrame,
        class_cols: List[str],
        audio_dir: Path,
        audio_col: str,
        start_col: str,
        end_col: str,
    ) -> None:
        """
        Parse CSV with CLASS_x columns (multi-class DCASE style).

        Args:
            df: The DataFrame to parse.
            class_cols: The list of class columns to parse.
            audio_dir: The directory containing the audio files.
            audio_col: The column containing the audio file names.
            start_col: The column containing the start time.
            end_col: The column containing the end time.

        Returns:
            None
        """
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

    def _parse_fallback(
        self,
        df: pd.DataFrame,
        annotation_path: Path,
        audio_dir: Path,
        audio_col: str,
        start_col: str,
        end_col: str,
    ) -> None:
        """
        Fallback: treat all rows as positive for a single class per CSV.

        Args:
            df: The DataFrame to parse.
            annotation_path: The path to the annotation file.
            audio_dir: The directory containing the audio files.
            audio_col: The column containing the audio file names.
            start_col: The column containing the start time.
            end_col: The column containing the end time.

        Returns:
            None
        """
        csv_class_name = annotation_path.stem
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

    def get_class_to_idx(self) -> Dict[str, int]:
        """Return the mapping from class names to indices."""
        return self.class_to_idx.copy()

    def get_idx_to_class(self) -> Dict[int, str]:
        """Return the mapping from indices to class names."""
        return {v: k for k, v in self.class_to_idx.items()}

    def get_num_classes(self) -> int:
        """Return the number of unique classes."""
        return len(self.class_to_idx)
