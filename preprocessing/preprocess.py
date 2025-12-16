from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import librosa

from utils.config import Config


def load_audio(
    path: Union[str, Path],
    target_sr: Optional[int] = None,
    mono: bool = True,
    config: Optional[Config] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as a waveform.

    Args:
        path: Path to .wav file.
        target_sr: Target sampling rate. If None, uses config.SAMPLING_RATE.
        mono: If True, convert to mono.
        config: Configuration object. If None, uses default Config().

    Returns:
        waveform: float32 waveform, shape (n_samples,) or (n_channels, n_samples)
        sr: sampling rate
    """
    if config is None:
        config = Config()

    if target_sr is None:
        target_sr = config.SAMPLING_RATE

    path = Path(path)
    waveform, sr = librosa.load(path.as_posix(), sr=target_sr, mono=mono)
    return waveform.astype(np.float32), sr


def waveform_to_logmel(
    waveform: np.ndarray,
    sr: int,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_mels: Optional[int] = None,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    eps: float = 1e-10,
    config: Optional[Config] = None,
) -> np.ndarray:
    """
    Convert waveform to log-mel spectrogram.

    Args:
        waveform: Input waveform array.
        sr: Sampling rate of the waveform.
        n_fft: FFT window size. If None, computed from config.FRAME_LENGTH.
        hop_length: Hop length. If None, computed from config.HOP_LENGTH.
        win_length: Window length. If None, equals n_fft.
        n_mels: Number of mel bins. If None, uses config.N_MELS.
        fmin: Minimum frequency for mel filterbank.
        fmax: Maximum frequency for mel filterbank.
        eps: Small constant for numerical stability.
        config: Configuration object. If None, uses default Config().

    Returns:
        logmel: np.ndarray, shape (n_mels, n_frames)
    """
    if config is None:
        config = Config()

    # Use config values as defaults
    if n_fft is None:
        n_fft = int(config.FRAME_LENGTH * sr)
    if hop_length is None:
        hop_length = int(config.HOP_LENGTH * sr)
    if win_length is None:
        win_length = n_fft
    if n_mels is None:
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
    target_sr: Optional[int] = None,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_mels: Optional[int] = None,
    min_duration: Optional[float] = None,
    config: Optional[Config] = None,
) -> np.ndarray:
    """
    Load a specific time segment from an audio file and convert to log-mel.

    Args:
        wav_path: Path to the .wav file.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        target_sr: Sampling rate to resample to. If None, uses config.SAMPLING_RATE.
        n_fft: FFT window size. If None, computed from config.FRAME_LENGTH.
        hop_length: Hop length. If None, computed from config.HOP_LENGTH.
        win_length: Window length. If None, equals n_fft.
        n_mels: Number of mel bins. If None, uses config.N_MELS.
        min_duration: If not None, pad the segment (with zeros) up to this duration in seconds.
        config: Configuration object. If None, uses default Config().

    Returns:
        logmel: np.ndarray, shape (n_mels, n_frames)
    """
    if config is None:
        config = Config()

    # Use config values as defaults
    if target_sr is None:
        target_sr = config.SAMPLING_RATE

    waveform, sr = load_audio(wav_path, target_sr=target_sr, mono=True, config=config)

    # Compute spectrogram params from config if not provided
    if n_fft is None:
        n_fft = int(config.FRAME_LENGTH * sr)
    if hop_length is None:
        hop_length = int(config.HOP_LENGTH * sr)
    if win_length is None:
        win_length = n_fft
    if n_mels is None:
        n_mels = config.N_MELS

    # Segmenting the audio
    # Valid range and type conversion
    start_sample = max(0, int(start_time * sr))
    end_sample = min(len(waveform), int(end_time * sr))

    segment = waveform[start_sample:end_sample]

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
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=0.0,
        config=config,
    )
    return logmel
