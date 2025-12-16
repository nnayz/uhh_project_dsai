from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import torch

from enum import Enum

class Distance(Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"

@dataclass
class Config:
    """
    Configuration for the v1 Prototypical Network model.
    
    All required paths (TRAIN_ANNOTATION_FILES) must be provided when instantiating.
    """
    # === Data ===
    SAMPLING_RATE: int = 16000  # Sampling rate of the audio
    DATA_DIR: Path = Path("/data/msc-proj/")  # Root directory of the dataset

    N_MELS: int = 64  # Number of mel bins for spectrogram
    FRAME_LENGTH: float = 0.025  # 25ms window length for STFT/mel extraction
    HOP_LENGTH: float = 0.010  # 10ms hop length between frames
    NORMALIZE: bool = True  # Apply mean/variance normalization to the audio

    # === Annotation files ===
    TRAIN_ANNOTATION_FILES: List[Path] = field(default_factory=lambda: [Path("/data/msc-proj/Training_Set/**/*.csv")])
    VAL_ANNOTATION_FILES: List[Path] = field(default_factory=lambda: [Path("/data/msc-proj/Validation_Set_DSAI_2025_2026/**/*.csv")])
    TEST_ANNOTATION_FILES: List[Path] = field(default_factory=lambda: [Path("/data/msc-proj/Evaluation_Set_DSAI_2025_2026/**/*.csv")])
    POSITIVE_LABEL: str = "POS"  # Label value indicating a positive example
    CLASS_NAME: Optional[str] = None  # Explicit class name for all annotations (None to infer from CSV)
    MIN_DURATION: float = 0.5  # Minimum duration in seconds (pads shorter segments)
    MAX_FRAMES: int = 512  # Maximum number of time frames (pads/crops to this)
    BATCH_SIZE: int = 1  # Batch size for dataloaders (usually 1 episode per batch)

    # === Episodes ===
    N_WAY: int = 5  # Number of classes per episode
    K_SHOT: int = 5  # Number of support examples per class
    N_QUERY: int = 10  # Number of query examples per class
    EPISODES_PER_EPOCH: int = 1000  # Number of episodes per epoch
    VAL_EPISODES: int = 100  # Number of validation episodes per eval run
    TEST_EPISODES: int = 100  # Number of test episodes per eval run

    # === Model ===
    ENCODER_TYPE: str = "conv4"  # Type of encoder network
    EMBEDDING_DIM: int = 128  # Dimension of the embedding vector
    CONV_CHANNELS: Tuple[int, ...] = (32, 64, 128, 256)  # Number of channels for each convolutional layer
    DISTANCE: Distance = Distance.EUCLIDEAN  # Distance metric to use

    # === Training ===
    LEARNING_RATE: float = 1e-3  # Learning rate for the optimizer
    WEIGHT_DECAY: float = 1e-4  # L2 Weight decay for the optimizer
    MAX_EPOCHS: int = 10  # Maximum number of epochs to train

    # === Runtime I/O ===
    DEVICE: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    NUM_WORKERS: int = 2  # Number of workers for the dataloader
    LOG_DIR: Path = Path("runs/proto/logs")  # Directory to save the logs
    CKPT_DIR: Path = Path("runs/proto/checkpoints")  # Directory to save the checkpoints
