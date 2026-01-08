from .ann_service import AnnotationService
from .dataset import DCASEEventDataset, FewShotEpisodeDataset
from .preprocess import load_audio, waveform_to_logmel, extract_logmel_segment
from .datamodule import DCASEFewShotDataModule, create_datamodule
from .feature_export import export_features, validate_features

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
    # Lightning DataModule
    "DCASEFewShotDataModule",
    "create_datamodule",
    # Task 5 feature helpers
    "export_features",
    "validate_features",
]
