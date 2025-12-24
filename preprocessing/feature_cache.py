"""
Feature Extraction and Caching Pipeline for DCASE Few-Shot Bioacoustic.

This module implements the offline feature extraction phase (Phase 1) of baseline v1:
    .wav audio → feature extraction → .npy feature files (cached on disk)

The model never sees raw audio during training - only cached features.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from schemas import SegmentExample
from .ann_service import AnnotationService
from .preprocess import extract_logmel_segment

logger = logging.getLogger(__name__)


@dataclass
class CachedFeature:
    """
    Represents a cached feature file on disk.
    
    Attributes:
        npy_path: Path to the .npy feature file.
        class_id: Integer class identifier.
        class_name: String class name.
        original_wav: Original wav file path (for reference).
        start_time: Start time of segment in original audio.
        end_time: End time of segment in original audio.
        feature_shape: Shape of the feature tensor.
    """
    npy_path: Path
    class_id: int
    class_name: str
    original_wav: Path
    start_time: float
    end_time: float
    feature_shape: Tuple[int, ...]


@dataclass
class FeatureManifest:
    """
    Manifest file containing metadata about cached features.
    
    Stored as JSON alongside the feature files for reproducibility.
    """
    version: str
    config_hash: str
    split: str
    num_samples: int
    num_classes: int
    class_to_idx: Dict[str, int]
    feature_shape: Tuple[int, ...]
    normalization: str
    samples: List[Dict]


def compute_config_hash(cfg: DictConfig) -> str:
    """
    Compute a hash of the feature extraction configuration.
    
    This ensures cached features are invalidated when config changes.
    
    Args:
        cfg: Hydra DictConfig with data and features settings.
        
    Returns:
        A hex string hash of the relevant config values.
    """
    relevant_keys = {
        "sampling_rate": cfg.data.sampling_rate,
        "n_mels": cfg.data.n_mels,
        "frame_length": cfg.data.frame_length,
        "hop_length": cfg.data.hop_length,
        "normalize": cfg.features.normalize,
        "normalize_mode": cfg.features.normalize_mode,
        "min_duration": cfg.annotations.min_duration,
    }
    config_str = json.dumps(relevant_keys, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def get_cache_dir(cfg: DictConfig, split: str) -> Path:
    """
    Get the cache directory for a specific split.
    
    Structure: {cache_dir}/{version}/{config_hash}/{split}/
    
    Args:
        cfg: Hydra DictConfig.
        split: Dataset split ('train', 'val', 'test').
        
    Returns:
        Path to the cache directory.
    """
    cache_root = Path(cfg.features.cache_dir)
    version = cfg.features.version
    config_hash = compute_config_hash(cfg)
    return cache_root / version / config_hash / split


def get_manifest_path(cache_dir: Path) -> Path:
    """Get path to the manifest file for a cache directory."""
    return cache_dir / "manifest.json"


def load_manifest(cache_dir: Path) -> Optional[FeatureManifest]:
    """
    Load the manifest file from a cache directory.
    
    Args:
        cache_dir: Path to the cache directory.
        
    Returns:
        FeatureManifest if exists, None otherwise.
    """
    manifest_path = get_manifest_path(cache_dir)
    if not manifest_path.exists():
        return None
    
    with open(manifest_path, "r") as f:
        data = json.load(f)
    
    return FeatureManifest(
        version=data["version"],
        config_hash=data["config_hash"],
        split=data["split"],
        num_samples=data["num_samples"],
        num_classes=data["num_classes"],
        class_to_idx=data["class_to_idx"],
        feature_shape=tuple(data["feature_shape"]),
        normalization=data["normalization"],
        samples=data["samples"],
    )


def save_manifest(manifest: FeatureManifest, cache_dir: Path) -> None:
    """
    Save the manifest file to a cache directory.
    
    Args:
        manifest: FeatureManifest to save.
        cache_dir: Path to the cache directory.
    """
    manifest_path = get_manifest_path(cache_dir)
    data = {
        "version": manifest.version,
        "config_hash": manifest.config_hash,
        "split": manifest.split,
        "num_samples": manifest.num_samples,
        "num_classes": manifest.num_classes,
        "class_to_idx": manifest.class_to_idx,
        "feature_shape": list(manifest.feature_shape),
        "normalization": manifest.normalization,
        "samples": manifest.samples,
    }
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved manifest to {manifest_path}")


def normalize_features(
    features: np.ndarray,
    mode: str = "per_sample",
    stats: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Normalize feature tensor.
    
    Args:
        features: Feature array of shape (n_mels, time_frames).
        mode: 'per_sample' or 'global'.
        stats: For 'global' mode, dict with 'mean' and 'std' arrays.
        
    Returns:
        Normalized feature array.
    """
    if mode == "per_sample":
        mean = features.mean()
        std = features.std()
        if std > 1e-8:
            features = (features - mean) / std
    elif mode == "global" and stats is not None:
        features = (features - stats["mean"]) / (stats["std"] + 1e-8)
    return features


def extract_and_cache_features(
    cfg: DictConfig,
    split: str,
    annotation_paths: List[Union[str, Path]],
    force_recompute: bool = False,
) -> Tuple[Path, FeatureManifest]:
    """
    Extract features from audio files and cache them as .npy files.
    
    This is the main entry point for Phase 1 (offline feature extraction).
    
    Args:
        cfg: Hydra DictConfig with all settings.
        split: Dataset split name ('train', 'val', 'test').
        annotation_paths: List of paths to annotation CSV files.
        force_recompute: If True, re-extract even if cache exists.
        
    Returns:
        Tuple of (cache_dir, manifest).
    """
    cache_dir = get_cache_dir(cfg, split)
    
    # Check if valid cache exists
    if not force_recompute and cache_dir.exists():
        manifest = load_manifest(cache_dir)
        if manifest is not None:
            expected_hash = compute_config_hash(cfg)
            if manifest.config_hash == expected_hash:
                logger.info(
                    f"Using cached features for {split} from {cache_dir} "
                    f"({manifest.num_samples} samples)"
                )
                return cache_dir, manifest
            else:
                logger.warning(
                    f"Config hash mismatch for {split}. Re-extracting features..."
                )
    
    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting features for {split} to {cache_dir}")
    
    # Load annotations
    annotation_service = AnnotationService(
        positive_label=cfg.annotations.positive_label,
        class_name=cfg.annotations.class_name,
    )
    examples: List[SegmentExample] = annotation_service.load_annotations(
        annotation_paths=[Path(p) for p in annotation_paths]
    )
    class_to_idx = annotation_service.get_class_to_idx()
    idx_to_class = annotation_service.get_idx_to_class()
    
    if not examples:
        raise RuntimeError(
            f"No positive events found in annotations for {split}. "
            f"Check CSV format and positive_label='{cfg.annotations.positive_label}'."
        )
    
    logger.info(f"Found {len(examples)} segments across {len(class_to_idx)} classes")
    
    # Create class subdirectories
    for class_name in class_to_idx.keys():
        (cache_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # Extract and save features
    samples_metadata = []
    feature_shape = None
    
    for idx, example in enumerate(tqdm(examples, desc=f"Extracting {split} features")):
        # Extract log-mel spectrogram
        try:
            logmel = extract_logmel_segment(
                wav_path=example.wav_path,
                start_time=example.start_time,
                end_time=example.end_time,
                cfg=cfg,
            )
        except Exception as e:
            logger.warning(
                f"Failed to extract features from {example.wav_path} "
                f"[{example.start_time:.2f}-{example.end_time:.2f}]: {e}"
            )
            continue
        
        # Normalize if configured
        if cfg.features.normalize:
            logmel = normalize_features(
                logmel,
                mode=cfg.features.normalize_mode,
            )
        
        # Add channel dimension: (n_mels, T) -> (1, n_mels, T)
        feature_tensor = logmel[np.newaxis, ...]
        
        if feature_shape is None:
            feature_shape = (feature_tensor.shape[0], feature_tensor.shape[1])
        
        # Generate unique filename
        class_name = idx_to_class[example.class_id]
        wav_stem = Path(example.wav_path).stem
        feature_filename = f"{wav_stem}_{example.start_time:.3f}_{example.end_time:.3f}.npy"
        npy_path = cache_dir / class_name / feature_filename
        
        # Save feature as .npy
        np.save(npy_path, feature_tensor.astype(np.float32))
        
        # Record metadata
        samples_metadata.append({
            "npy_path": str(npy_path.relative_to(cache_dir)),
            "class_id": example.class_id,
            "class_name": class_name,
            "original_wav": str(example.wav_path),
            "start_time": example.start_time,
            "end_time": example.end_time,
            "shape": list(feature_tensor.shape),
        })
    
    # Create and save manifest
    manifest = FeatureManifest(
        version=cfg.features.version,
        config_hash=compute_config_hash(cfg),
        split=split,
        num_samples=len(samples_metadata),
        num_classes=len(class_to_idx),
        class_to_idx=class_to_idx,
        feature_shape=feature_shape if feature_shape else (1, cfg.data.n_mels),
        normalization=cfg.features.normalize_mode if cfg.features.normalize else "none",
        samples=samples_metadata,
    )
    save_manifest(manifest, cache_dir)
    
    logger.info(
        f"Cached {len(samples_metadata)} features for {split} "
        f"({len(class_to_idx)} classes)"
    )
    
    return cache_dir, manifest


def extract_all_splits(
    cfg: DictConfig,
    force_recompute: bool = False,
) -> Dict[str, Tuple[Path, FeatureManifest]]:
    """
    Extract and cache features for all configured splits.
    
    Args:
        cfg: Hydra DictConfig.
        force_recompute: If True, re-extract even if cache exists.
        
    Returns:
        Dict mapping split names to (cache_dir, manifest) tuples.
    """
    results = {}
    
    splits = [
        ("train", cfg.annotations.train_files),
        ("val", cfg.annotations.val_files),
        ("test", cfg.annotations.test_files),
    ]
    
    for split_name, annotation_paths in splits:
        if annotation_paths:
            cache_dir, manifest = extract_and_cache_features(
                cfg=cfg,
                split=split_name,
                annotation_paths=annotation_paths,
                force_recompute=force_recompute,
            )
            results[split_name] = (cache_dir, manifest)
            logger.info(f"Completed {split_name}: {manifest.num_samples} samples")
        else:
            logger.info(f"Skipping {split_name}: no annotation files configured")
    
    return results


def verify_cache_integrity(cache_dir: Path) -> bool:
    """
    Verify that all files in the manifest exist.
    
    Args:
        cache_dir: Path to the cache directory.
        
    Returns:
        True if all files exist, False otherwise.
    """
    manifest = load_manifest(cache_dir)
    if manifest is None:
        return False
    
    for sample in manifest.samples:
        npy_path = cache_dir / sample["npy_path"]
        if not npy_path.exists():
            logger.warning(f"Missing file: {npy_path}")
            return False
    
    return True


def get_cache_stats(cache_dir: Path) -> Dict:
    """
    Get statistics about cached features.
    
    Args:
        cache_dir: Path to the cache directory.
        
    Returns:
        Dict with cache statistics.
    """
    manifest = load_manifest(cache_dir)
    if manifest is None:
        return {"error": "No manifest found"}
    
    # Calculate total size
    total_size = 0
    for sample in manifest.samples:
        npy_path = cache_dir / sample["npy_path"]
        if npy_path.exists():
            total_size += npy_path.stat().st_size
    
    # Count samples per class
    class_counts = {}
    for sample in manifest.samples:
        class_name = sample["class_name"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return {
        "version": manifest.version,
        "config_hash": manifest.config_hash,
        "split": manifest.split,
        "num_samples": manifest.num_samples,
        "num_classes": manifest.num_classes,
        "total_size_mb": total_size / (1024 * 1024),
        "class_counts": class_counts,
        "feature_shape": manifest.feature_shape,
        "normalization": manifest.normalization,
    }

