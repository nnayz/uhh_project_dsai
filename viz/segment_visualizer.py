"""
Visualization module for audio segments and their features.

This module provides functions to visualize:
- Audio signals in time domain
- Log mel spectrograms
- PCEN spectrograms
- Comparison between different feature types
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from omegaconf import DictConfig

from preprocessing.ann_service import AnnotationService
from preprocessing.preprocess import (
    load_audio,
    waveform_to_logmel,
    waveform_to_pcen,
)


@dataclass
class SegmentExample:
    """Lightweight container for a labeled audio segment."""

    wav_path: Path
    start_time: float
    end_time: float
    class_id: int


def load_feature_array(
    wav_path: Path, feature_type: str
) -> Optional[np.ndarray]:
    """
    Load pre-extracted feature array if it exists.

    Args:
        wav_path: Path to the audio file.
        feature_type: Feature type ('logmel' or 'pcen').

    Returns:
        Feature array if exists, None otherwise.
    """
    feature_path = wav_path.with_name(f"{wav_path.stem}_{feature_type}.npy")
    if feature_path.exists():
        features = np.load(feature_path)
        # Features are stored as (n_frames, n_mels), transpose if needed
        if features.shape[0] < features.shape[1]:
            features = features.T
        return features
    return None


def extract_segment_from_features(
    features: np.ndarray,
    start_time: float,
    end_time: float,
    sr: int,
    hop_length: int,
) -> np.ndarray:
    """
    Extract a time segment from pre-computed feature array.

    Args:
        features: Full feature array, shape (n_mels, n_frames).
        start_time: Start time in seconds.
        end_time: End time in seconds.
        sr: Sample rate.
        hop_length: Hop length used for feature extraction.

    Returns:
        Segment feature array.
    """
    fps = sr / hop_length
    start_frame = max(0, int(start_time * fps))
    end_frame = min(features.shape[1], int(end_time * fps))
    return features[:, start_frame:end_frame]


def visualize_segment(
    segment: SegmentExample,
    cfg: DictConfig,
    output_path: Optional[Path] = None,
    use_precomputed: bool = True,
    show_plot: bool = True,
) -> None:
    """
    Visualize a single audio segment with time domain, logmel, and PCEN.

    Args:
        segment: SegmentExample containing segment information.
        cfg: Hydra DictConfig with feature settings.
        output_path: Optional path to save the figure.
        use_precomputed: If True, try to use pre-computed feature arrays.
        show_plot: If True, display the plot.
    """
    wav_path = segment.wav_path
    start_time = segment.start_time
    end_time = segment.end_time

    if not wav_path.exists():
        print(f"Warning: Audio file not found: {wav_path}")
        return

    # Load audio segment
    waveform, sr = load_audio(wav_path, cfg=cfg, mono=True)
    start_sample = max(0, int(start_time * sr))
    end_sample = min(len(waveform), int(end_time * sr))
    audio_segment = waveform[start_sample:end_sample]
    time_axis = np.linspace(start_time, end_time, len(audio_segment))

    # Get feature parameters
    params = {
        "sr": cfg.features.sr,
        "n_fft": cfg.features.n_fft,
        "hop_length": cfg.features.hop_mel,
        "n_mels": cfg.features.n_mels,
    }

    # Try to load pre-computed features
    logmel_features = None
    pcen_features = None

    if use_precomputed:
        logmel_full = load_feature_array(wav_path, "logmel")
        pcen_full = load_feature_array(wav_path, "pcen")

        if logmel_full is not None:
            logmel_features = extract_segment_from_features(
                logmel_full,
                start_time,
                end_time,
                params["sr"],
                params["hop_length"],
            )

        if pcen_full is not None:
            pcen_features = extract_segment_from_features(
                pcen_full,
                start_time,
                end_time,
                params["sr"],
                params["hop_length"],
            )

    # Compute features if not available from pre-computed
    if logmel_features is None:
        logmel_features = waveform_to_logmel(audio_segment, cfg)
    if pcen_features is None:
        pcen_features = waveform_to_pcen(audio_segment, cfg)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

    # Plot 1: Time domain waveform
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, audio_segment, linewidth=0.5)
    ax1.set_xlabel("Time (s)", fontsize=10)
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.set_title(
        f"Audio Signal (Time Domain)\n{wav_path.name} [{start_time:.3f}s - {end_time:.3f}s]",
        fontsize=12,
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(start_time, end_time)

    # Plot 2: Log mel spectrogram
    ax2 = fig.add_subplot(gs[1, 0])
    time_frames_logmel = np.linspace(
        start_time, end_time, logmel_features.shape[1]
    )
    im1 = ax2.imshow(
        logmel_features,
        aspect="auto",
        origin="lower",
        extent=[start_time, end_time, 0, params["n_mels"]],
        cmap="viridis",
    )
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylabel("Mel Frequency Bin", fontsize=10)
    ax2.set_title("Log Mel Spectrogram", fontsize=12)
    plt.colorbar(im1, ax=ax2, label="Log Magnitude")

    # Plot 3: PCEN spectrogram
    ax3 = fig.add_subplot(gs[1, 1])
    im2 = ax3.imshow(
        pcen_features,
        aspect="auto",
        origin="lower",
        extent=[start_time, end_time, 0, params["n_mels"]],
        cmap="viridis",
    )
    ax3.set_xlabel("Time (s)", fontsize=10)
    ax3.set_ylabel("Mel Frequency Bin", fontsize=10)
    ax3.set_title("PCEN Spectrogram", fontsize=12)
    plt.colorbar(im2, ax=ax3, label="PCEN Value")

    # Plot 4: Comparison (difference or side-by-side)
    ax4 = fig.add_subplot(gs[2, :])
    # Normalize both for comparison
    logmel_norm = (logmel_features - logmel_features.min()) / (
        logmel_features.max() - logmel_features.min() + 1e-10
    )
    pcen_norm = (pcen_features - pcen_features.min()) / (
        pcen_features.max() - pcen_features.min() + 1e-10
    )
    diff = logmel_norm - pcen_norm
    im3 = ax4.imshow(
        diff,
        aspect="auto",
        origin="lower",
        extent=[start_time, end_time, 0, params["n_mels"]],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    ax4.set_xlabel("Time (s)", fontsize=10)
    ax4.set_ylabel("Mel Frequency Bin", fontsize=10)
    ax4.set_title("Log Mel - PCEN (Normalized Difference)", fontsize=12)
    plt.colorbar(im3, ax=ax4, label="Difference")

    plt.suptitle(
        f"Segment Visualization: Class ID {segment.class_id}\n"
        f"File: {wav_path.name}",
        fontsize=14,
        y=0.995,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_segments_for_class(
    class_name: str,
    cfg: DictConfig,
    split: str = "train",
    max_segments: int = 5,
    output_dir: Optional[Union[str, Path]] = None,
    use_precomputed: bool = True,
    show_plots: bool = False,
) -> List[SegmentExample]:
    """
    Visualize segments for a specific class.

    Args:
        class_name: Name of the class to visualize (can be class name or class ID).
        cfg: Hydra DictConfig with path and feature settings.
        split: Data split to use ('train', 'val', or 'test').
        max_segments: Maximum number of segments to visualize.
        output_dir: Optional directory to save visualizations.
        use_precomputed: If True, try to use pre-computed feature arrays.
        show_plots: If True, display plots interactively.

    Returns:
        List of SegmentExample objects that were visualized.
    """
    # Determine data directory
    if split == "train":
        data_dir = Path(cfg.path.train_dir)
    elif split == "val":
        data_dir = Path(cfg.path.eval_dir)
    elif split == "test":
        data_dir = Path(cfg.path.test_dir)
    else:
        raise ValueError(f"Invalid split: {split}")

    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # Load annotations
    ann_service = AnnotationService()
    csv_pattern = str(data_dir / "**" / "*.csv")
    annotations = ann_service.load_annotations([csv_pattern])

    if not annotations:
        print(f"No annotations found in {data_dir}")
        return []

    # Get class mapping
    class_to_idx = ann_service.get_class_to_idx()
    idx_to_class = ann_service.get_idx_to_class()

    # Find class ID
    class_id = None
    if class_name.isdigit():
        class_id = int(class_name)
    elif class_name in class_to_idx:
        class_id = class_to_idx[class_name]
    else:
        # Try to find by partial match
        for name, idx in class_to_idx.items():
            if class_name.lower() in name.lower():
                class_id = idx
                class_name = name
                break

    if class_id is None:
        print(f"Class '{class_name}' not found.")
        available_classes = sorted(class_to_idx.keys())
        print(f"\nFound {len(available_classes)} available classes:")
        # Show first 20 classes, or all if less than 20
        display_count = min(20, len(available_classes))
        for i, cls in enumerate(available_classes[:display_count], 1):
            print(f"  {i}. {cls}")
        if len(available_classes) > display_count:
            print(f"  ... and {len(available_classes) - display_count} more")
        print(f"\nTip: Use the exact class name from the list above (e.g., 'e1', 'XC100296')")
        return []

    # Filter segments for this class
    class_segments = [seg for seg in annotations if seg.class_id == class_id]

    if not class_segments:
        print(f"No segments found for class '{class_name}' (ID: {class_id})")
        return []

    print(
        f"Found {len(class_segments)} segments for class '{class_name}' (ID: {class_id})"
    )

    # Limit number of segments
    segments_to_viz = class_segments[:max_segments]

    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize each segment
    for i, segment in enumerate(segments_to_viz):
        print(f"\nVisualizing segment {i+1}/{len(segments_to_viz)}")
        print(f"  File: {segment.wav_path.name}")
        print(f"  Time: [{segment.start_time:.3f}s - {segment.end_time:.3f}s]")

        output_path = None
        if output_dir:
            safe_class_name = class_name.replace("/", "_").replace("\\", "_")
            output_path = (
                output_dir
                / f"{safe_class_name}_segment_{i+1:03d}_{segment.wav_path.stem}.png"
            )

        try:
            visualize_segment(
                segment,
                cfg,
                output_path=output_path,
                use_precomputed=use_precomputed,
                show_plot=show_plots,
            )
        except Exception as e:
            print(f"  Error visualizing segment: {e}")
            continue

    return segments_to_viz
