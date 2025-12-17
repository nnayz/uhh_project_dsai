from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import librosa

from omegaconf import DictConfig


def load_audio(
    path: Union[str, Path],
    cfg: DictConfig,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as a waveform.

    Args:
        path: Path to .wav file.
        cfg: Hydra DictConfig with data.sampling_rate.
        mono: If True, convert to mono.

    Returns:
        waveform: float32 waveform, shape (n_samples,) or (n_channels, n_samples)
        sr: sampling rate
    """
    path = Path(path)
    sr = cfg.data.sampling_rate
    waveform, sr = librosa.load(path.as_posix(), sr=sr, mono=mono)
    return waveform.astype(np.float32), sr


def waveform_to_logmel(
    waveform: np.ndarray,
    cfg: DictConfig,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Convert waveform to log-mel spectrogram.

    Args:
        waveform: Input waveform array.
        cfg: Hydra DictConfig with data settings.
        fmin: Minimum frequency for mel filterbank.
        fmax: Maximum frequency for mel filterbank.
        eps: Small constant for numerical stability.

    Returns:
        logmel: np.ndarray, shape (n_mels, n_frames)
    """
    sr = cfg.data.sampling_rate
    n_fft = int(cfg.data.frame_length * sr)
    hop_length = int(cfg.data.hop_length * sr)
    win_length = n_fft
    n_mels = cfg.data.n_mels

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
    cfg: DictConfig,
) -> np.ndarray:
    """
    Load a specific time segment from an audio file and convert to log-mel.

    Args:
        wav_path: Path to the .wav file.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        cfg: Hydra DictConfig with data and annotations settings.

    Returns:
        logmel: np.ndarray, shape (n_mels, n_frames)
    """
    waveform, sr = load_audio(wav_path, cfg=cfg, mono=True)

    # Segmenting the audio
    start_sample = max(0, int(start_time * sr))
    end_sample = min(len(waveform), int(end_time * sr))

    segment = waveform[start_sample:end_sample]

    min_duration = cfg.annotations.min_duration
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
        cfg=cfg,
        fmin=0.0,
    )
    return logmel
