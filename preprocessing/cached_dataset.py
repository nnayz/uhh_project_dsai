"""
Cached Feature Dataset for DCASE Few-Shot Bioacoustic.

This module implements Phase 2 (online, repeated) of baseline v1:
    .npy feature files → embedding network → few-shot classification

During training and evaluation:
    - Dataset loaders read .npy files directly
    - Audio libraries are no longer used
    - Feature extraction is skipped entirely
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from omegaconf import DictConfig

from .feature_cache import load_manifest, FeatureManifest, get_cache_dir


class CachedFeatureDataset(Dataset):
    """
    Dataset that loads pre-extracted features from .npy files.
    
    This is the core dataset for Phase 2 - it operates purely on cached features
    and never touches raw audio files.
    
    Attributes:
        cache_dir: Path to the cache directory.
        manifest: FeatureManifest with sample metadata.
        samples: List of sample metadata dicts.
        class_to_idx: Mapping from class names to indices.
    """

    def __init__(
        self,
        cache_dir: Path,
        cfg: Optional[DictConfig] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Initialize the CachedFeatureDataset.
        
        Args:
            cache_dir: Path to the cache directory containing .npy files.
            cfg: Optional Hydra DictConfig (for max_frames if not specified).
            max_frames: Maximum number of time frames (for padding/cropping).
        """
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cfg = cfg
        
        # Load manifest
        self.manifest = load_manifest(self.cache_dir)
        if self.manifest is None:
            raise RuntimeError(
                f"No manifest found in {self.cache_dir}. "
                f"Run feature extraction first."
            )
        
        self.samples = self.manifest.samples
        self.class_to_idx = self.manifest.class_to_idx
        
        # Determine max_frames
        if max_frames is not None:
            self.max_frames = max_frames
        elif cfg is not None and hasattr(cfg, "annotations"):
            self.max_frames = cfg.annotations.max_frames
        else:
            self.max_frames = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load a single feature from cache.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            Tuple of (feature_tensor, class_id).
            Feature tensor has shape (1, n_mels, T) or (1, n_mels, max_frames) if padded.
        """
        sample = self.samples[idx]
        npy_path = self.cache_dir / sample["npy_path"]
        
        # Load feature from .npy file
        feature = np.load(npy_path)
        tensor = torch.from_numpy(feature)
        
        # Ensure channel dimension exists
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # (n_mels, T) -> (1, n_mels, T)
        
        # Pad/crop to fixed length if max_frames is set
        if self.max_frames is not None:
            tensor = self._crop_or_pad(tensor, self.max_frames)
        
        class_id = sample["class_id"]
        return tensor, class_id

    def _crop_or_pad(self, tensor: torch.Tensor, max_frames: int) -> torch.Tensor:
        """Crop or pad tensor to fixed time dimension."""
        T = tensor.shape[-1]
        if T > max_frames:
            tensor = tensor[..., :max_frames]
        elif T < max_frames:
            diff = max_frames - T
            tensor = F.pad(tensor, (0, diff))
        return tensor

    def get_num_classes(self) -> int:
        """Return the number of unique classes."""
        return self.manifest.num_classes

    def get_class_to_idx(self) -> Dict[str, int]:
        """Return the mapping from class names to indices."""
        return self.class_to_idx.copy()

    def get_idx_to_class(self) -> Dict[int, str]:
        """Return the mapping from indices to class names."""
        return {v: k for k, v in self.class_to_idx.items()}


class CachedFewShotEpisodeDataset(Dataset):
    """
    Wraps a CachedFeatureDataset and yields few-shot episodes.
    
    Each __getitem__ returns:
        support_x: (k_way * n_shot, 1, n_mels, T)
        support_y: (k_way * n_shot,)
        query_x:   (k_way * n_query, 1, n_mels, T)
        query_y:   (k_way * n_query,)
    
    This operates purely in feature space - no audio processing.
    """

    def __init__(
        self,
        base_dataset: CachedFeatureDataset,
        cfg: DictConfig,
        num_episodes: Optional[int] = None,
    ) -> None:
        """
        Initialize the CachedFewShotEpisodeDataset.
        
        Args:
            base_dataset: The underlying CachedFeatureDataset.
            cfg: Hydra DictConfig with episodes settings.
            num_episodes: Override for number of episodes per epoch.
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.cfg = cfg
        
        # Episode parameters from config
        self.k_way = cfg.arch.episodes.n_way
        self.n_shot = cfg.arch.episodes.k_shot
        self.n_query = cfg.arch.episodes.n_query
        self.max_frames = cfg.annotations.max_frames
        
        # Number of episodes
        if num_episodes is not None:
            self.num_episodes = num_episodes
        else:
            self.num_episodes = cfg.arch.episodes.episodes_per_epoch
        
        # Build class-to-indices mapping
        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, sample in enumerate(self.base_dataset.samples):
            class_id = sample["class_id"]
            if class_id not in self.class_to_indices:
                self.class_to_indices[class_id] = []
            self.class_to_indices[class_id].append(idx)
        
        self.class_ids = list(self.class_to_indices.keys())
        
        # Validate we have enough classes
        if len(self.class_ids) < self.k_way:
            raise RuntimeError(
                f"Not enough classes ({len(self.class_ids)}) for {self.k_way}-way episodes. "
                f"Available classes: {self.class_ids}"
            )

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a few-shot episode from cached features.
        
        Args:
            idx: Episode index (used for potential seeding).
            
        Returns:
            Tuple of (support_x, support_y, query_x, query_y).
        """
        # Sample k_way distinct classes
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
            
            # Sample with replacement if not enough samples
            if len(indices) < needed:
                chosen = np.random.choice(indices, size=needed, replace=True)
            else:
                chosen = np.random.choice(indices, size=needed, replace=False)
            
            support_idx = chosen[:self.n_shot]
            query_idx = chosen[self.n_shot:]
            
            for i in support_idx:
                x, _ = self.base_dataset[i]
                support_x_list.append(x)
                support_y_list.append(class_id)
            
            for i in query_idx:
                x, _ = self.base_dataset[i]
                query_x_list.append(x)
                query_y_list.append(class_id)
        
        # Stack and pad/crop to fixed max_frames
        def crop_pad(t: torch.Tensor, T_max: int) -> torch.Tensor:
            T = t.shape[-1]
            if T > T_max:
                t = t[..., :T_max]
            elif T < T_max:
                diff = T_max - T
                t = F.pad(t, (0, diff))
            return t
        
        support_x_padded = [crop_pad(t, self.max_frames) for t in support_x_list]
        query_x_padded = [crop_pad(t, self.max_frames) for t in query_x_list]
        
        support_x = torch.stack(support_x_padded, dim=0)
        query_x = torch.stack(query_x_padded, dim=0)
        support_y = torch.tensor(support_y_list, dtype=torch.long)
        query_y = torch.tensor(query_y_list, dtype=torch.long)
        
        return support_x, support_y, query_x, query_y

    def get_num_classes(self) -> int:
        """Return the number of unique classes."""
        return len(self.class_ids)


def create_cached_dataset(
    cfg: DictConfig,
    split: str,
) -> CachedFeatureDataset:
    """
    Factory function to create a CachedFeatureDataset for a split.
    
    Args:
        cfg: Hydra DictConfig.
        split: Dataset split ('train', 'val', 'test').
        
    Returns:
        CachedFeatureDataset instance.
    """
    cache_dir = get_cache_dir(cfg, split)
    
    if not cache_dir.exists():
        raise RuntimeError(
            f"Cache directory not found: {cache_dir}. "
            f"Run feature extraction first with: python main.py extract-features"
        )
    
    return CachedFeatureDataset(
        cache_dir=cache_dir,
        cfg=cfg,
        max_frames=cfg.annotations.max_frames,
    )


def create_cached_episode_dataset(
    cfg: DictConfig,
    split: str,
    num_episodes: Optional[int] = None,
) -> CachedFewShotEpisodeDataset:
    """
    Factory function to create a CachedFewShotEpisodeDataset for a split.
    
    Args:
        cfg: Hydra DictConfig.
        split: Dataset split ('train', 'val', 'test').
        num_episodes: Override for number of episodes.
        
    Returns:
        CachedFewShotEpisodeDataset instance.
    """
    base_dataset = create_cached_dataset(cfg, split)
    return CachedFewShotEpisodeDataset(
        base_dataset=base_dataset,
        cfg=cfg,
        num_episodes=num_episodes,
    )

