from .ann_service import AnnotationService
from .dataset import DCASEEventDataset, FewShotEpisodeDataset
from .preprocess import load_audio, waveform_to_logmel, extract_logmel_segment
from .dataloaders import make_dcase_event_dataset, make_fewshot_dataloaders
from .feature_cache import (
    extract_and_cache_features,
    extract_all_splits,
    get_cache_dir,
    load_manifest,
    verify_cache_integrity,
    get_cache_stats,
)
from .cached_dataset import (
    CachedFeatureDataset,
    CachedFewShotEpisodeDataset,
    create_cached_dataset,
    create_cached_episode_dataset,
)
from .datamodule import DCASEFewShotDataModule, create_datamodule

__all__ = [
    # Annotation service
    "AnnotationService",
    # Legacy datasets (on-the-fly extraction)
    "DCASEEventDataset",
    "FewShotEpisodeDataset",
    # Audio preprocessing
    "load_audio",
    "waveform_to_logmel",
    "extract_logmel_segment",
    # Legacy dataloaders
    "make_dcase_event_dataset",
    "make_fewshot_dataloaders",
    # Feature caching (Phase 1)
    "extract_and_cache_features",
    "extract_all_splits",
    "get_cache_dir",
    "load_manifest",
    "verify_cache_integrity",
    "get_cache_stats",
    # Cached datasets (Phase 2)
    "CachedFeatureDataset",
    "CachedFewShotEpisodeDataset",
    "create_cached_dataset",
    "create_cached_episode_dataset",
    # Lightning DataModule
    "DCASEFewShotDataModule",
    "create_datamodule",
]

