# data/preprocess.py

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import librosa


def load_audio(
    path: str | Path,
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as a waveform.

    Args:
        path: Path to .wav file.
        target_sr: If not None, resample to this sampling rate.
        mono: If True, convert to mono.

    Returns:
        y: float32 waveform, shape (n_samples,) or (n_channels, n_samples)
        sr: sampling rate
    """
    path = Path(path)
    y, sr = librosa.load(path.as_posix(), sr=target_sr, mono=mono)
    return y.astype(np.float32), sr


def waveform_to_logmel(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 64,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Convert waveform to log-mel spectrogram.

    Returns:
        logmel: np.ndarray, shape (n_mels, n_frames)
    """
    mel = librosa.feature.melspectrogram(
        y=y,
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
    wav_path: str | Path,
    start_time: float,
    end_time: float,
    target_sr: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 64,
    min_duration: Optional[float] = None,
) -> np.ndarray:
    """
    Load a specific time segment from an audio file and convert to log-mel.

    Args:
        wav_path: path to the .wav file
        start_time: start time in seconds
        end_time: end time in seconds
        target_sr: sampling rate to resample to
        min_duration: if not None, pad the segment (with zeros) up to this duration in seconds

    Returns:
        logmel: np.ndarray, shape (n_mels, n_frames)
    """
    wav_path = Path(wav_path)
    y, sr = load_audio(wav_path, target_sr=target_sr, mono=True)

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # clip to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(y), end_sample)

    segment = y[start_sample:end_sample]

    if min_duration is not None:
        min_samples = int(min_duration * sr)
        if len(segment) < min_samples:
            pad_width = min_samples - len(segment)
            segment = np.pad(segment, (0, pad_width), mode="constant")

    logmel = waveform_to_logmel(
        segment,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
    )
    return logmel
