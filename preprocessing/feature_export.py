"""
Feature export and validation for training.

This module generates per-audio feature arrays expected by the
sequence-based datamodule (e.g., audio.wav -> audio_logmel.npy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import glob
import numpy as np

from .preprocess import load_audio, waveform_to_logmel, waveform_to_pcen


SUPPORTED_SUFFIXES = {"logmel", "pcen"}


def collect_wav_paths_from_dir(root: str) -> List[Path]:
    if not root:
        return []
    pattern = str(Path(root) / "**" / "*.wav")
    files = [Path(p) for p in glob.glob(pattern, recursive=True)]
    # dedupe while preserving order
    seen = set()
    unique = []
    for w in files:
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

    total_written = 0
    for split in splits:
        if split == "train":
            wav_paths = collect_wav_paths_from_dir(cfg.path.train_dir)
        elif split == "val":
            wav_paths = collect_wav_paths_from_dir(cfg.path.eval_dir)
        elif split == "test":
            wav_paths = collect_wav_paths_from_dir(cfg.path.test_dir)
        else:
            wav_paths = []
        for wav_path in wav_paths:
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

    for split in splits:
        if split == "train":
            wav_paths = collect_wav_paths_from_dir(cfg.path.train_dir)
        elif split == "val":
            wav_paths = collect_wav_paths_from_dir(cfg.path.eval_dir)
        elif split == "test":
            wav_paths = collect_wav_paths_from_dir(cfg.path.test_dir)
        else:
            wav_paths = []
        for wav_path in wav_paths:
            for suffix in suffixes:
                out_path = wav_path.with_suffix(f"_{suffix}.npy")
                if not out_path.exists():
                    missing.append(out_path)
    return missing
