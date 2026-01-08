"""
Feature export and validation for training.

This module generates per-audio feature arrays expected by the
sequence-based datamodule (e.g., audio.wav -> audio_logmel.npy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .preprocess import load_audio, waveform_to_logmel, waveform_to_pcen


SUPPORTED_SUFFIXES = {"logmel", "pcen"}


def _expand_annotation_paths(paths: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        path = Path(p)
        if "*" in str(path) or "?" in str(path):
            files.extend(sorted(Path().glob(str(path))))
        else:
            files.append(path)
    return [f for f in files if f.is_file()]


def _wav_path_from_row(csv_path: Path, wav_name: str) -> Path:
    audio_dir = csv_path.parent
    candidate = audio_dir / wav_name
    if candidate.is_file():
        return candidate
    alt = audio_dir / (Path(wav_name).stem + ".wav")
    return alt


def collect_wav_paths(annotation_paths: Sequence[str]) -> List[Path]:
    wavs: List[Path] = []
    for csv_path in _expand_annotation_paths(annotation_paths):
        try:
            df = pd.read_csv(csv_path, usecols=["Audiofilename"])
            names = df["Audiofilename"].dropna().unique().tolist()
        except Exception:
            names = []

        if not names:
            names = [csv_path.with_suffix(".wav").name]

        for name in names:
            wav_path = _wav_path_from_row(csv_path, name)
            wavs.append(wav_path)
    # dedupe while preserving order
    seen = set()
    unique = []
    for w in wavs:
        if w in seen:
            continue
        seen.add(w)
        unique.append(w)
    return unique


def _extract_feature(waveform: np.ndarray, cfg, suffix: str) -> np.ndarray:
    if suffix == "logmel":
        return waveform_to_logmel(waveform, cfg)
    if suffix == "pcen":
        return waveform_to_pcen(waveform, cfg)
    raise ValueError(f"Unsupported feature suffix: {suffix}")


def export_features(
    cfg,
    splits: Iterable[str] = ("train", "val", "test"),
    force: bool = False,
) -> int:
    """Export per-audio feature arrays for training."""
    suffixes = cfg.features.feature_types.split("@")
    unsupported = [s for s in suffixes if s not in SUPPORTED_SUFFIXES]
    if unsupported:
        raise ValueError(
            f"Unsupported feature_types={unsupported}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    split_map = {
        "train": cfg.annotations.train_files,
        "val": cfg.annotations.val_files,
        "test": cfg.annotations.test_files,
    }

    total_written = 0
    for split in splits:
        ann_paths = split_map.get(split, [])
        if not ann_paths:
            continue
        for wav_path in collect_wav_paths(ann_paths):
            if not wav_path.is_file():
                continue
            waveform, _ = load_audio(wav_path, cfg=cfg, mono=True)
            for suffix in suffixes:
                out_path = wav_path.with_suffix(f"_{suffix}.npy")
                if out_path.exists() and not force:
                    continue
                features = _extract_feature(waveform, cfg, suffix)
                np.save(out_path, features)
                total_written += 1
    return total_written


def validate_features(
    cfg,
    splits: Iterable[str] = ("train", "val", "test"),
) -> List[Path]:
    """Return a list of missing feature files for training."""
    suffixes = cfg.features.feature_types.split("@")
    missing: List[Path] = []

    split_map = {
        "train": cfg.annotations.train_files,
        "val": cfg.annotations.val_files,
        "test": cfg.annotations.test_files,
    }

    for split in splits:
        ann_paths = split_map.get(split, [])
        if not ann_paths:
            continue
        for wav_path in collect_wav_paths(ann_paths):
            for suffix in suffixes:
                out_path = wav_path.with_suffix(f"_{suffix}.npy")
                if not out_path.exists():
                    missing.append(out_path)
    return missing
