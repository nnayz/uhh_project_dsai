# Audio Signal Processing Pipeline

This document provides a detailed explanation of how audio files are processed in the DCASE Few-Shot Bioacoustic project, including signal transformations, feature extraction, and visual examples.

## Table of Contents

1. [Overview](#overview)
2. [Raw Audio Loading](#1-raw-audio-loading)
3. [Log-Mel Spectrogram Computation](#2-log-mel-spectrogram-computation)
4. [PCEN Normalization](#3-pcen-normalization)
5. [Feature Normalization](#4-feature-normalization)
6. [Segment Extraction](#5-segment-extraction)
7. [Complete Pipeline Diagram](#6-complete-pipeline-diagram)

---

## Overview

The audio processing pipeline converts raw `.wav` files into normalized feature arrays that can be consumed by the neural network. The pipeline operates in two phases:

1. **Offline Feature Export**: Converts entire audio files to feature arrays (`.npy` files)
2. **Online Segment Sampling**: Dynamically extracts segments during training

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE AUDIO PROCESSING FLOW                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   .wav file                                                                     │
│       │                                                                         │
│       ▼                                                                         │
│   ┌─────────────────┐                                                           │
│   │  Load & Resample │  → 22050 Hz, Mono, Normalized to [-1, 1]                 │
│   └────────┬────────┘                                                           │
│            │                                                                    │
│            ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐           │
│   │                    STFT (Short-Time Fourier Transform)          │           │
│   │   • Window: n_fft=1024 samples (~46ms at 22050 Hz)              │           │
│   │   • Hop: 256 samples (~11.6ms)                                  │           │
│   │   • Output: Complex spectrogram (513 frequency bins × T frames) │           │
│   └─────────────────────────────────────────────────────────────────┘           │
│            │                                                                    │
│            ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐           │
│   │                    Mel Filterbank Application                   │           │
│   │   • 128 triangular mel-scale filters                            │           │
│   │   • Frequency range: 50 Hz – 11025 Hz                           │           │
│   │   • Power spectrum (magnitude²)                                 │           │
│   │   • Output: Mel spectrogram (128 bins × T frames)               │           │
│   └─────────────────────────────────────────────────────────────────┘           │
│            │                                                                    │
│            ├──────────────────────┬──────────────────────┐                      │
│            ▼                      ▼                      │                      │
│   ┌─────────────────┐    ┌─────────────────┐             │                      │
│   │   Log-Mel       │    │     PCEN        │             │                      │
│   │   Compression   │    │   Normalization │             │                      │
│   │   log(mel + ε)  │    │   (adaptive)    │             │                      │
│   └────────┬────────┘    └────────┬────────┘             │                      │
│            │                      │                      │                      │
│            ▼                      ▼                      │                      │
│   ┌─────────────────┐    ┌─────────────────┐             │                      │
│   │ audio_logmel.npy│    │ audio_pcen.npy  │             │                      │
│   │ (T × 128)       │    │ (T × 128)       │             │                      │
│   └─────────────────┘    └─────────────────┘             │                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Raw Audio Loading

**Location**: `preprocessing/preprocess.py:load_audio()`

The first step loads the raw audio waveform and prepares it for feature extraction.

### Process

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              RAW AUDIO LOADING                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   Input: BV_file1.wav (any sample rate, mono/stereo)                           │
│                                                                                │
│   Step 1: Resample to 22050 Hz                                                 │
│   ──────────────────────────────────────                                       │
│   Original:    [s₁, s₂, s₃, ..., sₙ] @ 44100 Hz                                │
│                      ↓                                                         │
│   Resampled:   [s₁', s₂', s₃', ..., sₘ'] @ 22050 Hz (m ≈ n/2)                  │
│                                                                                │
│   Step 2: Convert to Mono (if stereo)                                          │
│   ─────────────────────────────────────                                        │
│   Stereo:      Left:  [L₁, L₂, L₃, ...]                                        │
│                Right: [R₁, R₂, R₃, ...]                                        │
│                      ↓                                                         │
│   Mono:        [(L₁+R₁)/2, (L₂+R₂)/2, ...]                                     │
│                                                                                │
│   Step 3: Peak Normalization                                                   │
│   ─────────────────────────────────────                                        │
│   waveform = waveform / max(|waveform|)                                        │
│                                                                                │
│   Before:      [-0.3, 0.8, -0.5, 0.2, ...] (arbitrary amplitude)               │
│                      ↓                                                         │
│   After:       [-0.375, 1.0, -0.625, 0.25, ...] (scaled to [-1, 1])            │
│                                                                                │
│   Output: float32 array, shape (n_samples,)                                    │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Configuration Parameters

| Parameter | Config Path | Default | Description |
|-----------|-------------|---------|-------------|
| `sr` | `features.sr` | 22050 | Target sample rate in Hz |

### Example

For a 10-second audio file at 44100 Hz:
- Input: 441,000 samples
- After resampling to 22050 Hz: 220,500 samples

---

## 2. Log-Mel Spectrogram Computation

**Location**: `preprocessing/preprocess.py:waveform_to_logmel()`

The log-mel spectrogram is the most common audio representation for neural networks. It mimics human auditory perception.

### Process

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        LOG-MEL SPECTROGRAM COMPUTATION                         │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   Step 1: Short-Time Fourier Transform (STFT)                                  │
│   ───────────────────────────────────────────                                  │
│                                                                                │
│   Waveform:  ───────────────────────────────────────────────────────►          │
│              |     |     |     |     |     |     |     |     |                 │
│              ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤                 │
│              │ W₁  │ W₂  │ W₃  │ W₄  │ W₅  │ W₆  │ W₇  │ W₈  │ ...            │
│              └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘                 │
│                                                                                │
│              ←────→ n_fft = 1024 samples (~46.4 ms)                            │
│              ←──→   hop_length = 256 samples (~11.6 ms)                        │
│                                                                                │
│   For each window:                                                             │
│     1. Apply Hann window: w(n) = 0.5 - 0.5·cos(2πn/(N-1))                      │
│     2. Compute FFT: X[k] = Σ x[n]·e^(-j2πkn/N)                                 │
│     3. Keep positive frequencies: 513 bins (0 to Nyquist)                      │
│                                                                                │
│   Output: Complex spectrogram (513 freq bins × T time frames)                  │
│                                                                                │
│   Step 2: Power Spectrum                                                       │
│   ──────────────────────                                                       │
│   Power = |X[k]|² = Re(X[k])² + Im(X[k])²                                      │
│                                                                                │
│   Step 3: Mel Filterbank                                                       │
│   ──────────────────────                                                       │
│                                                                                │
│   Frequency (Hz)                                                               │
│   11025 ─┬────────────────────────────────────────────────────┐                │
│          │                                          ▲ ▲ ▲ ▲ ▲ │ (sparse at    │
│          │                                       ▲ ▲          │  high freq)   │
│          │                                    ▲ ▲             │                │
│          │                                 ▲ ▲                │                │
│          │                              ▲ ▲                   │                │
│          │                           ▲ ▲                      │                │
│          │                        ▲ ▲                         │                │
│          │                     ▲ ▲                            │ 128 filters   │
│          │                  ▲ ▲                               │                │
│          │               ▲ ▲                                  │                │
│          │            ▲ ▲                                     │                │
│          │         ▲ ▲                                        │                │
│          │      ▲ ▲                                           │ (dense at     │
│          │   ▲ ▲                                              │  low freq)    │
│      50 ─┼─▲ ▲───────────────────────────────────────────────┘                │
│          │                                                                     │
│          └────────────────────────────────────────────────────► Mel scale     │
│            0                                                 128              │
│                                                                                │
│   Mel = 2595 · log₁₀(1 + f/700)    (Mel scale formula)                        │
│                                                                                │
│   Each triangular filter:                                                      │
│     - Peaks at center frequency                                                │
│     - Overlaps with neighbors                                                  │
│     - Sums energy in that band                                                 │
│                                                                                │
│   Step 4: Log Compression                                                      │
│   ───────────────────────                                                      │
│   log_mel = log(mel_power + ε)    where ε = 1e-8                               │
│                                                                                │
│   Why log?                                                                     │
│   - Human hearing is logarithmic                                               │
│   - Compresses dynamic range (quiet sounds become visible)                     │
│   - Stabilizes variance for neural network training                            │
│                                                                                │
│   Output: (n_mels × T) = (128 × T) array, then transposed to (T × 128)        │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Visual Example

```
                           LOG-MEL SPECTROGRAM VISUALIZATION
                           
   Frequency                        Time →
   (Mel bin)  ┌──────────────────────────────────────────────────────────────┐
         128 │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
             │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
             │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
         100 │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
             │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
             │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
          75 │░░░░░░░░░░░░░▓▓▓▓░░░░░░░░░░░░░░░░░░░░▓▓▓▓░░░░░░░░░░░░░░░░░░│
             │░░░░░░░░░░░▓▓████▓▓░░░░░░░░░░░░░░░░▓▓████▓▓░░░░░░░░░░░░░░░░│
             │░░░░░░░░░░▓██████▓░░░░░░░░░░░░░░░░░▓██████▓░░░░░░░░░░░░░░░░│
          50 │░░░░░░░░░▓████████▓░░░░░░░░░░░░░░░▓████████▓░░░░░░░░░░░░░░░│
             │░░░░░░░░▓██████████▓░░░░░░░░░░░░░▓██████████▓░░░░░░░░░░░░░░│
             │░░░░░░░░▓██████████▓░░░░░░░░░░░░░▓██████████▓░░░░░░░░░░░░░░│  ← Bird call
          25 │░░░░░░░░░▓████████▓░░░░░░░░░░░░░░░▓████████▓░░░░░░░░░░░░░░░│    events
             │░░░░░░░░░░▓██████▓░░░░░░░░░░░░░░░░░▓██████▓░░░░░░░░░░░░░░░░│
             │▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
           0 │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│ ← Low-freq noise
             └──────────────────────────────────────────────────────────────┘
                  0s        1s        2s        3s        4s        5s
   
   Legend: ░ = low energy, ▒ = medium, ▓ = high, █ = very high
```

### Configuration Parameters

| Parameter | Config Path | Default | Description |
|-----------|-------------|---------|-------------|
| `n_fft` | `features.n_fft` | 1024 | FFT window size in samples |
| `hop_mel` | `features.hop_mel` | 256 | Hop length between windows |
| `n_mels` | `features.n_mels` | 128 | Number of mel frequency bins |
| `fmin` | `features.fmin` | 50 | Minimum frequency (Hz) |
| `fmax` | `features.fmax` | 11025 | Maximum frequency (Hz, Nyquist/2) |
| `eps` | `features.eps` | 1e-8 | Small constant for log stability |

---

## 3. PCEN Normalization

**Location**: `preprocessing/preprocess.py:waveform_to_pcen()`

PCEN (Per-Channel Energy Normalization) is an alternative to log-mel that provides automatic gain control. It's particularly effective for bioacoustic sounds with varying background noise.

### Process

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         PCEN (Per-Channel Energy Normalization)                │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   PCEN adapts to the local energy level, suppressing background noise         │
│   while preserving transient sounds (like animal calls).                       │
│                                                                                │
│   Formula:                                                                     │
│   ────────                                                                     │
│                      E(t,f)                                                    │
│   PCEN(t,f) = ─────────────────────  - δ                                       │
│               (ε + M(t,f))^α                                                   │
│                                                                                │
│   Where:                                                                       │
│   • E(t,f) = mel spectrogram energy at time t, frequency f                     │
│   • M(t,f) = smoothed energy (running average): M = (1-s)·M + s·E              │
│   • s = smoothing coefficient (time constant)                                  │
│   • α = compression exponent (typically 0.98)                                  │
│   • δ = offset (typically 2)                                                   │
│   • ε = small constant for stability                                           │
│                                                                                │
│   Visualization of PCEN Effect:                                                │
│   ─────────────────────────────                                                │
│                                                                                │
│   Log-Mel (traditional):                                                       │
│                                                                                │
│   Energy │         ▲▲▲                    ▲▲▲▲                                 │
│          │        ▲████▲                 ▲████▲         ← Calls visible       │
│          │       ▲██████▲               ▲██████▲                               │
│          │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ← Background     │
│          │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    noise high   │
│          └───────────────────────────────────────────────────► Time           │
│                                                                                │
│   PCEN (normalized):                                                           │
│                                                                                │
│   Energy │         ▲▲▲                    ▲▲▲▲                                 │
│          │        ▲████▲                 ▲████▲         ← Calls preserved     │
│          │       ▲██████▲               ▲██████▲                               │
│          │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ← Background     │
│          │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    suppressed   │
│          └───────────────────────────────────────────────────► Time           │
│                                                                                │
│   Benefits of PCEN:                                                            │
│   • Automatic gain control (adapts to varying background levels)               │
│   • Better for outdoor recordings with wind, rain, etc.                        │
│   • Preserves transient events (animal calls)                                  │
│   • More robust than fixed log compression                                     │
│                                                                                │
│   Output: (n_mels × T) = (128 × T) array, then transposed to (T × 128)        │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### When to Use PCEN vs Log-Mel

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Clean studio recordings | Log-Mel | Simpler, well-understood |
| Field recordings with variable noise | PCEN | Adaptive gain control |
| Multiple recording conditions | PCEN | More robust |
| Baseline comparisons | Log-Mel | Standard in literature |

---

## 4. Feature Normalization

**Location**: `preprocessing/sequence_data/pcen.py:Feature_Extractor`

After loading cached features, they are normalized using global statistics computed across the dataset.

### Process

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           FEATURE NORMALIZATION                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   Step 1: Compute Global Statistics (once, cached)                             │
│   ─────────────────────────────────────────────────                            │
│                                                                                │
│   • Sample up to 1000 feature files                                            │
│   • Flatten all values: [file₁ values, file₂ values, ...]                      │
│   • Compute: μ = mean(all_values), σ = std(all_values)                         │
│                                                                                │
│   Example for logmel:                                                          │
│     mean = -4.2, std = 2.1  (typical log-mel statistics)                       │
│                                                                                │
│   Step 2: Z-Score Normalization (per feature load)                             │
│   ─────────────────────────────────────────────────                            │
│                                                                                │
│   normalized = (feature - mean) / std                                          │
│                                                                                │
│   Before:     [-8.5, -6.2, -3.1, -2.0, ...]  (raw log-mel values)              │
│                      ↓                                                         │
│   After:      [-2.0, -0.95, 0.52, 1.05, ...]  (zero mean, unit variance)       │
│                                                                                │
│   Why normalize?                                                               │
│   • Neural networks train better with standardized inputs                      │
│   • Prevents features with large values from dominating                        │
│   • Faster convergence during training                                         │
│                                                                                │
│   Step 3: Ensure Time-Major Format                                             │
│   ──────────────────────────────────                                           │
│                                                                                │
│   Expected shape: (T frames, 128 mel bins)                                     │
│   If shape is (128, T): transpose to (T, 128)                                  │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Segment Extraction

**Location**: `preprocessing/sequence_data/dynamic_pcen_dataset.py:select_segment()`

During training, fixed-length segments are extracted from the full-audio feature arrays.

### Process

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           SEGMENT EXTRACTION                                   │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   Given: CSV annotation with start/end times for positive events               │
│                                                                                │
│   Time (seconds):     0      1      2      3      4      5      6              │
│                       │      │      │      │      │      │      │              │
│   Full audio:    ─────┴──────┴──────┴──────┴──────┴──────┴──────┴─────►        │
│                                                                                │
│   CSV annotations:           ├──POS──┤         ├──POS──┤                       │
│                              1.2    1.8       3.5    4.1                       │
│                                                                                │
│   Step 1: Add 25ms Padding                                                     │
│   ────────────────────────                                                     │
│   Padded:              ├────POS────┤       ├────POS────┤                       │
│                        1.175      1.825   3.475      4.125                     │
│                                                                                │
│   Step 2: Convert Time to Frames                                               │
│   ──────────────────────────────                                               │
│   fps = sr / hop_mel = 22050 / 256 ≈ 86.13 frames/second                       │
│                                                                                │
│   start_frame = floor(1.175 × 86.13) = 101                                     │
│   end_frame = floor(1.825 × 86.13) = 157                                       │
│   segment_frames = 157 - 101 = 56 frames                                       │
│                                                                                │
│   Step 3: Extract Fixed-Length Segment                                         │
│   ──────────────────────────────────────                                       │
│   Target: seg_len = 0.2s → 0.2 × 86.13 ≈ 17 frames                             │
│                                                                                │
│   Case A: Segment longer than target (56 > 17)                                 │
│   ─────────────────────────────────────────────                                │
│   Original:    │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│       │
│                │◄───────────── 56 frames ─────────────►│                       │
│                                                                                │
│   Random crop: Pick random start within valid range                            │
│                     │◄─── 17 frames ───►│                                      │
│                     └───────────────────┘                                      │
│                           extracted                                            │
│                                                                                │
│   Case B: Segment shorter than target                                          │
│   ─────────────────────────────────────                                        │
│   Original:    │░░░░░░░░│                                                      │
│                │◄─ 8 ──►│                                                      │
│                                                                                │
│   Tile/repeat: │░░░░░░░░│░░░░░░░░│░░│                                          │
│                │◄───── 17 frames ────►│                                        │
│                      (tiles until target length)                               │
│                                                                                │
│   Case C: Empty segment (start >= end)                                         │
│   ─────────────────────────────────────                                        │
│   Return: Zero-filled array of target length                                   │
│                                                                                │
│   Output: (17 frames, 128 mel bins) segment                                    │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Negative Segment Selection

Negative segments (non-event regions) are selected from gaps between positive events:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        NEGATIVE SEGMENT SELECTION                              │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   Audio timeline:    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━►             │
│                      │                                                         │
│   Positive events:      ██████        ████████        ██████                   │
│                         │    │        │      │        │    │                   │
│                         └─P₁─┘        └──P₂──┘        └─P₃─┘                   │
│                                                                                │
│   Negative regions: ████      ████████        ████████      ████               │
│                     │  │      │      │        │      │      │  │               │
│                     │N₀│      │  N₁  │        │  N₂  │      │N₃│               │
│                     └──┘      └──────┘        └──────┘      └──┘               │
│                                                                                │
│   neg_info list:                                                               │
│     [(0, start_P₁), (end_P₁, start_P₂), (end_P₂, start_P₃), ...]              │
│                                                                                │
│   Selection: Random gap, ensuring duration > 0.2s minimum                      │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Complete Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE DATA FLOW DIAGRAM                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ╔═══════════════════════════════════════════════════════════════════════════╗ │
│   ║                    OFFLINE FEATURE EXPORT (run once)                      ║ │
│   ╠═══════════════════════════════════════════════════════════════════════════╣ │
│   ║                                                                           ║ │
│   ║   .wav files                                                              ║ │
│   ║       │                                                                   ║ │
│   ║       ▼                                                                   ║ │
│   ║   ┌─────────────────┐                                                     ║ │
│   ║   │ load_audio()    │  Resample → Mono → Normalize                        ║ │
│   ║   └────────┬────────┘                                                     ║ │
│   ║            │                                                              ║ │
│   ║            ▼                                                              ║ │
│   ║   ┌─────────────────────────────────────────┐                             ║ │
│   ║   │ waveform_to_logmel() / waveform_to_pcen │                             ║ │
│   ║   │   STFT → Mel filterbank → Log/PCEN      │                             ║ │
│   ║   └────────────────┬────────────────────────┘                             ║ │
│   ║                    │                                                      ║ │
│   ║                    ▼                                                      ║ │
│   ║   ┌─────────────────────────────────────────┐                             ║ │
│   ║   │ Save as .npy (T × 128) next to .wav     │                             ║ │
│   ║   │   audio_logmel.npy  or  audio_pcen.npy  │                             ║ │
│   ║   └─────────────────────────────────────────┘                             ║ │
│   ║                                                                           ║ │
│   ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                    │                                            │
│                                    ▼                                            │
│   ╔═══════════════════════════════════════════════════════════════════════════╗ │
│   ║                    TRAINING DATA LOADING (each epoch)                     ║ │
│   ╠═══════════════════════════════════════════════════════════════════════════╣ │
│   ║                                                                           ║ │
│   ║   CSV Annotations          .npy Feature Files                             ║ │
│   ║        │                        │                                         ║ │
│   ║        ▼                        ▼                                         ║ │
│   ║   ┌──────────────┐        ┌──────────────────┐                            ║ │
│   ║   │ Parse labels │        │ Feature_Extractor│                            ║ │
│   ║   │ POS events   │        │ Load + Normalize │                            ║ │
│   ║   └──────┬───────┘        └────────┬─────────┘                            ║ │
│   ║          │                         │                                      ║ │
│   ║          └────────────┬────────────┘                                      ║ │
│   ║                       ▼                                                   ║ │
│   ║         ┌──────────────────────────────┐                                  ║ │
│   ║         │   build_meta()               │                                  ║ │
│   ║         │   • Positive segments (info) │                                  ║ │
│   ║         │   • Negative segments        │                                  ║ │
│   ║         │   • File paths               │                                  ║ │
│   ║         └──────────────┬───────────────┘                                  ║ │
│   ║                        │                                                  ║ │
│   ║                        ▼                                                  ║ │
│   ║         ┌──────────────────────────────┐                                  ║ │
│   ║         │   __getitem__(idx)           │                                  ║ │
│   ║         │   • select_positive()        │                                  ║ │
│   ║         │   • select_negative()        │                                  ║ │
│   ║         │   • select_segment()         │                                  ║ │
│   ║         └──────────────┬───────────────┘                                  ║ │
│   ║                        │                                                  ║ │
│   ║                        ▼                                                  ║ │
│   ║         ┌──────────────────────────────┐                                  ║ │
│   ║         │ Segment: (17 × 128) float32  │                                  ║ │
│   ║         │ Label: class index           │                                  ║ │
│   ║         └──────────────────────────────┘                                  ║ │
│   ║                        │                                                  ║ │
│   ║                        ▼                                                  ║ │
│   ║         ┌──────────────────────────────┐                                  ║ │
│   ║         │   DataLoader + Sampler       │                                  ║ │
│   ║         │   • k_way classes per batch  │                                  ║ │
│   ║         │   • n_shot × 2 per class     │                                  ║ │
│   ║         └──────────────┬───────────────┘                                  ║ │
│   ║                        │                                                  ║ │
│   ║                        ▼                                                  ║ │
│   ║         ┌──────────────────────────────┐                                  ║ │
│   ║         │   Episodic Batch             │                                  ║ │
│   ║         │   Shape: (k×n×2, 17, 128)    │                                  ║ │
│   ║         └──────────────────────────────┘                                  ║ │
│   ║                                                                           ║ │
│   ╚═══════════════════════════════════════════════════════════════════════════╝ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Formulas Summary

| Step | Formula | Purpose |
|------|---------|---------|
| Mel scale | `mel = 2595 × log₁₀(1 + f/700)` | Convert Hz to perceptual mel scale |
| Log compression | `log_mel = log(mel + ε)` | Compress dynamic range |
| PCEN | `PCEN = (E / (ε + M)^α) - δ` | Adaptive normalization |
| Z-score | `x' = (x - μ) / σ` | Standardize features |
| Time to frames | `frame = floor(time × sr / hop)` | Convert seconds to frame index |
| Segment length | `seg_frames = seg_len × sr / hop` | Target segment size in frames |

---

## File Locations

| Component | File Path |
|-----------|-----------|
| Audio loading | `preprocessing/preprocess.py:load_audio()` |
| Log-mel extraction | `preprocessing/preprocess.py:waveform_to_logmel()` |
| PCEN extraction | `preprocessing/preprocess.py:waveform_to_pcen()` |
| Feature export | `preprocessing/feature_export.py:export_features()` |
| Feature normalization | `preprocessing/sequence_data/pcen.py:Feature_Extractor` |
| Segment extraction | `preprocessing/sequence_data/dynamic_pcen_dataset.py:select_segment()` |
