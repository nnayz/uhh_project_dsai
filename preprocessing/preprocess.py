from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import librosa

from utils.config import Config


def load_audio(
    path: Union[str, Path],
    config: Config,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as a waveform.

    Args:
        path: Path to .wav file.
        config: Configuration object (required).
        mono: If True, convert to mono.

    Returns:
        waveform: float32 waveform, shape (n_samples,) or (n_channels, n_samples)
        sr: sampling rate (from config.SAMPLING_RATE)

    Raises:
        ValueError: If config is not provided.
    """
    if config is None:
        raise ValueError("config is required and cannot be None")

    path = Path(path)
    waveform, sr = librosa.load(path.as_posix(), sr=config.SAMPLING_RATE, mono=mono)
    return waveform.astype(np.float32), sr


def waveform_to_logmel(
    waveform: np.ndarray,
    config: Config,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Convert waveform to log-mel spectrogram.

    Args:
        waveform: Input waveform array.
        config: Configuration object (required).
        fmin: Minimum frequency for mel filterbank.
        fmax: Maximum frequency for mel filterbank.
        eps: Small constant for numerical stability.

    Returns:
        logmel: np.ndarray, shape (n_mels, n_frames)

    Raises:
        ValueError: If config is not provided.
    """
    if config is None:
        raise ValueError("config is required and cannot be None")

    sr = config.SAMPLING_RATE
    n_fft = int(config.FRAME_LENGTH * sr)
    hop_length = int(config.HOP_LENGTH * sr)
    win_length = n_fft
    n_mels = config.N_MELS

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel + eps)
    return logmel.astype(np.float32)


def extract_logmel_segment(
    wav_path: Union[str, Path],
    start_time: float,
    end_time: float,
    config: Config,
) -> np.ndarray:
    """
    Load a specific time segment from an audio file and convert to log-mel.

    Args:
        wav_path: Path to the .wav file.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        config: Configuration object (required).

    Returns:
        logmel: np.ndarray, shape (n_mels, n_frames)

    Raises:
        ValueError: If config is not provided.
    """
    if config is None:
        raise ValueError("config is required and cannot be None")

    waveform, sr = load_audio(wav_path, config=config, mono=True)

    # Segmenting the audio
    # Valid range and type conversion
    start_sample = max(0, int(start_time * sr))
    end_sample = min(len(waveform), int(end_time * sr))

    segment = waveform[start_sample:end_sample]

    min_duration = config.MIN_DURATION
    if min_duration is not None:
        min_samples = int(min_duration * sr)
        if len(segment) < min_samples:
            pad_width = min_samples - len(segment)
            segment = np.pad(
                segment,
                (0, pad_width),
                mode="constant"
            )

    logmel = waveform_to_logmel(
        waveform=segment,
        config=config,
        fmin=0.0,
    )
    return logmel
