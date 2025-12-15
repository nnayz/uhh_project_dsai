from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass
class Config:
    # Data
    SAMPLING_RATE: int = 16000 # Sampling rate of the audio
    DATA_DIR: Path = Path("/data/msc-proj/") # Root directory of the dataset

    N_MELS: int = 64 # Number of mel bins for spectrogram
    FRAME_LENGTH: float = 0.025 # 25ms for spectrogram [Window length for STFT/mel extraction]
    HOP_LENGTH: float = 0.010 # 10ms for spectrogram [Hop length for STFT/mel extraction between frames]
    NORMALIZE: bool = True # Whether to normalize the audio [Apply mean/variance normalization to the audio]

    # Episodes
    N_WAY: int = 5 # Number of classes per episode
    K_SHOT: int = 5 # Number of support examples per class

    N_QUERY: int = 10 # Number of query examples per class
    EPISODES_PER_EPOCH: int = 1000 # Number of episodes per epoch
    VAL_EPISODES: int # Number of validation episodes per eval run
    TEST_EPISODES: int # Number of test episodes per eval run
    RANDOM_SEED: int = 42 # Random seed for reproducibility [Set for reproducibility of the random number generator]

    # Model
    ENCODER_TYPE: str = "conv4" # Type of encoder network
    EMBEDDING_DIM: int = 128 # Dimension of the embedding vector
    CONV_CHANNELS: Tuple[int, ...] = (32, 64, 128, 256) # Number of channels for each convolutional layer
    pass # TODO: Add more if needed

    # Training/Optimizer
    LEARNING_RATE: float = 1e-3 # Learning rate for the optimizer
    WEIGHT_DECAY: float = 1e-4 # L2 Weight decay for the optimizer
    OPTIMIZER: str = "adam" # Optimizer to use ["adam", "sgd"]
    SCHEDULER: str = "cosine" # Scheduler to use ["none", "step"]
    MAX_EPOCHS: int = 10 # Maximum number of epochs to train
    GRAD_CLIP: float = 5.0 # Gradient clipping value [0 to disable]

    # Runtime I/o
    DEVICE: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" # Device to use ["cuda", "cpu", "mps"]

    NUM_WORKERS: int = 2 # Number of workers for the dataloader
    LOG_DIR: Path = Path("runs/proto/logs") # Directory to save the logs
    CKPT_DIR: Path = Path("runs/proto/checkpoints") # Directory to save the checkpoints
    SAVE_EVERY: int = 5 # Save the model every N epochs



