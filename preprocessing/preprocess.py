"""
Audio loading and feature extraction for DCASE Few-Shot Bioacoustic.

This module handles:
- Loading audio files (wav format)
- Computing log-mel spectrograms
- PCEN normalization (optional)
- Segment extraction

The feature parameters are configured in cfg.features.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import librosa

from omegaconf import DictConfig


def get_feature_params(cfg: DictConfig) -> dict:
    """
    Extract feature extraction parameters from config.

    Supports both old config format (cfg.data) and new format (cfg.features).

    Args:
        cfg: Hydra DictConfig.

    Returns:
        Dict with sr, n_fft, hop_length, n_mels, fmin, fmax, eps.
    """
    # New config format (features section)
    if hasattr(cfg, "features") and hasattr(cfg.features, "sr"):
        return {
            "sr": cfg.features.sr,
            "n_fft": cfg.features.n_fft,
            "hop_length": cfg.features.hop_mel,
            "n_mels": cfg.features.n_mels,
            "fmin": cfg.features.fmin,
            "fmax": cfg.features.fmax,
            "eps": cfg.features.eps,
        }
    # Legacy config format (data section)
    elif hasattr(cfg, "data"):
        sr = cfg.data.sampling_rate
        return {
            "sr": sr,
            "n_fft": int(getattr(cfg.data, "frame_length", 0.025) * sr),
            "hop_length": int(getattr(cfg.data, "hop_length", 0.010) * sr),
            "n_mels": cfg.data.n_mels,
            "fmin": 0.0,
            "fmax": sr // 2,
            "eps": 1e-10,
        }
    else:
        raise ValueError("Config must have either 'features' or 'data' section")


def load_audio(
    path: Union[str, Path],
    cfg: DictConfig,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as a waveform.

    Args:
        path: Path to .wav file.
        cfg: Hydra DictConfig with feature settings.
        mono: If True, convert to mono.

    Returns:
        waveform: float32 waveform, shape (n_samples,) or (n_channels, n_samples)
        sr: sampling rate
    """
    path = Path(path)
    params = get_feature_params(cfg)
    sr = params["sr"]

    waveform, sr = librosa.load(path.as_posix(), sr=sr, mono=mono)
    return waveform.astype(np.float32), sr


def waveform_to_logmel(
    waveform: np.ndarray,
    cfg: DictConfig,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    eps: Optional[float] = None,
) -> np.ndarray:
    """
    Convert waveform to log-mel spectrogram.

    Args:
        waveform: Input waveform array.
        cfg: Hydra DictConfig with feature settings.
        fmin: Minimum frequency for mel filterbank (overrides config).
        fmax: Maximum frequency for mel filterbank (overrides config).
        eps: Small constant for numerical stability (overrides config).

    Returns:
        logmel: np.ndarray, shape (n_mels, n_frames)
    """
    params = get_feature_params(cfg)

    sr = params["sr"]
    n_fft = params["n_fft"]
    hop_length = params["hop_length"]
    n_mels = params["n_mels"]

    # Use config values if not overridden
    if fmin is None:
        fmin = params["fmin"]
    if fmax is None:
        fmax = params["fmax"]
    if eps is None:
        eps = params["eps"]

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel + eps)
    return logmel.astype(np.float32)


def waveform_to_pcen(
    waveform: np.ndarray,
    cfg: DictConfig,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """
    Convert waveform to PCEN (Per-Channel Energy Normalization) spectrogram.

    PCEN is more robust to background noise than log-mel and is commonly
    used in bioacoustic tasks.

    Args:
        waveform: Input waveform array.
        cfg: Hydra DictConfig with feature settings.
        fmin: Minimum frequency for mel filterbank.
        fmax: Maximum frequency for mel filterbank.

    Returns:
        pcen: np.ndarray, shape (n_mels, n_frames)
    """
    params = get_feature_params(cfg)

    sr = params["sr"]
    n_fft = params["n_fft"]
    hop_length = params["hop_length"]
    n_mels = params["n_mels"]

    if fmin is None:
        fmin = params["fmin"]
    if fmax is None:
        fmax = params["fmax"]

    # Compute mel spectrogram (power)
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )

    # Apply PCEN normalization
    pcen = librosa.pcen(mel, sr=sr, hop_length=hop_length)
    return pcen.astype(np.float32)


def extract_features(
    waveform: np.ndarray,
    cfg: DictConfig,
    feature_type: Optional[str] = None,
) -> np.ndarray:
    """
    Extract features from waveform based on config.

    Args:
        waveform: Input waveform array.
        cfg: Hydra DictConfig with feature settings.
        feature_type: Feature type to extract (overrides config).
            Options: 'logmel', 'pcen'

    Returns:
        features: np.ndarray, shape (n_mels, n_frames)
    """
    if feature_type is None:
        feature_type = getattr(cfg.features, "feature_types", "logmel")

    if feature_type == "logmel":
        return waveform_to_logmel(waveform, cfg)
    elif feature_type == "pcen":
        return waveform_to_pcen(waveform, cfg)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def extract_logmel_segment(
    wav_path: Union[str, Path],
    start_time: float,
    end_time: float,
    cfg: DictConfig,
    feature_type: Optional[str] = None,
) -> np.ndarray:
    """
    Load a specific time segment from an audio file and extract features.

    Args:
        wav_path: Path to the .wav file.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        cfg: Hydra DictConfig with feature and annotation settings.
        feature_type: Feature type to extract (default: from config).

    Returns:
        features: np.ndarray, shape (n_mels, n_frames)
    """
    waveform, sr = load_audio(wav_path, cfg=cfg, mono=True)

    # Segmenting the audio
    start_sample = max(0, int(start_time * sr))
    end_sample = min(len(waveform), int(end_time * sr))

    segment = waveform[start_sample:end_sample]

    # Get minimum duration from config
    min_duration = None
    if hasattr(cfg, "annotations") and hasattr(cfg.annotations, "min_duration"):
        min_duration = cfg.annotations.min_duration
    elif hasattr(cfg, "train_param") and hasattr(cfg.train_param, "seg_len"):
        min_duration = cfg.train_param.seg_len

    # Pad short segments
    if min_duration is not None:
        min_samples = int(min_duration * sr)
        if len(segment) < min_samples:
            pad_width = min_samples - len(segment)
            segment = np.pad(segment, (0, pad_width), mode="constant")

    # Extract features
    if feature_type is None:
        feature_type = getattr(cfg.features, "feature_types", "logmel")

    if feature_type == "logmel":
        features = waveform_to_logmel(waveform=segment, cfg=cfg)
    elif feature_type == "pcen":
        features = waveform_to_pcen(waveform=segment, cfg=cfg)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    return features
