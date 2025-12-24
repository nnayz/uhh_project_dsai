# Feature Extraction, Audio Processing, and Feature Caching
## Baseline v1 (DCASE Few-Shot Bioacoustic)

This document explains how baseline v1 processes raw audio files, extracts features, persists them as `.npy` files, and consumes those cached features during training and evaluation.

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Audio Input and Dataset Layout](#2-audio-input-and-dataset-layout)
3. [Feature Extraction](#3-feature-extraction)
4. [Feature Caching](#4-feature-caching)
5. [Configuration Reference](#5-configuration-reference)
6. [CLI Commands](#6-cli-commands)
7. [Data Flow Summary](#7-data-flow-summary)

## 1. Pipeline Overview

Baseline v1 is split into two distinct phases:

```
PHASE 1 (offline, once)
.wav audio → feature extraction → .npy feature files

PHASE 2 (online, repeated)
.npy feature files → embedding network → few-shot classification
```

The model never sees raw audio during training.

### Why Two Phases?

- Fast training (no audio I/O during training)
- Reproducible experiments (same features every run)
- Easy debugging (inspect cached features)

## 2. Audio Input and Dataset Layout

### Raw Audio Files

- Format: `.wav`
- Audio: Mono or converted to mono
- Sampling rate: 22050 Hz (configurable)

### Dataset Structure

```
/data/msc-proj/
  Training_Set/
    BV/
      BV_file1.wav
      BV_file1.csv
  Validation_Set_DSAI_2025_2026/
    ...
  Evaluation_Set_DSAI_2025_2026/
    ...
```

### Annotation Format

CSV files with columns:
- `Audiofilename`: Name of the audio file
- `Starttime`: Start time of segment (seconds)
- `Endtime`: End time of segment (seconds)
- `Q` or `CLASS_*`: Label (POS/NEG/UNK)

## 3. Feature Extraction

### Log-Mel Spectrogram Computation

For each audio segment:
1. Load audio at target sample rate
2. Apply Short-Time Fourier Transform (STFT)
3. Compute power spectrum
4. Project onto mel filterbanks
5. Convert to dB scale (log)

```python
mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft, ...)
logmel = librosa.power_to_db(mel + eps)
```

### Default Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 22050 Hz |
| n_fft | 1024 |
| hop_length | 256 |
| n_mels | 128 |
| fmin | 50 Hz |
| fmax | 11025 Hz |

### Output Shape

```
(n_mels, time_frames) → (1, n_mels, time_frames)
```

## 4. Feature Caching

### Cache Directory Structure

```
{cache_dir}/{version}/{config_hash}/{split}/
  manifest.json
  {class_name}/
    {wav_stem}_{start}_{end}.npy
```

### Manifest File

Each split has a `manifest.json` containing:
```json
{
    "version": "v1",
    "config_hash": "abc123def456",
    "split": "train",
    "num_samples": 1234,
    "num_classes": 15,
    "class_to_idx": {"BV": 0, "PB": 1},
    "samples": [...]
}
```

### Config Hash

A hash is computed from feature extraction parameters. If any parameter changes, a new cache directory is created to prevent stale features.

## 5. Configuration Reference

### Main Config (`conf/config.yaml`)

```yaml
# Path Configuration
path:
  root_dir: /data/msc-proj
  train_dir: ${path.root_dir}/Training_Set
  eval_dir: ${path.root_dir}/Validation_Set_DSAI_2025_2026
  test_dir: ${path.root_dir}/Evaluation_Set_DSAI_2025_2026

# Feature Extraction
features:
  sr: 22050
  n_fft: 1024
  n_mels: 128
  hop_mel: 256
  fmin: 50
  fmax: 11025
  feature_types: logmel
  embedding_dim: 2048
  drop_rate: 0.1
  non_linearity: leaky_relu
  cache_dir: ${path.root_dir}/features_cache
  use_cache: true

# Training Parameters
train_param:
  seg_len: 0.2
  n_shot: 5
  k_way: 10
  lr_rate: 0.001
  scheduler_gamma: 0.65
  scheduler_step_size: 10
  num_episodes: 2000

# Evaluation Parameters
eval_param:
  seg_len: 0.200
  hop_seg: 0.05
  threshold: 0.9
```

### Architecture Config (`conf/arch/v1.yaml`)

```yaml
name: v1

# Model Architecture
model:
  encoder_type: conv4
  embedding_dim: ${features.embedding_dim}
  conv_channels: [64, 64, 64, 64]
  distance: euclidean
  n_mels: ${features.n_mels}

# Episode Configuration
episodes:
  n_way: ${train_param.k_way}
  k_shot: ${train_param.n_shot}
  episodes_per_epoch: ${train_param.num_episodes}

# Training Configuration
training:
  learning_rate: ${train_param.lr_rate}
  max_epochs: 50
  scheduler: step
  scheduler_gamma: ${train_param.scheduler_gamma}
```

### Callbacks Config (`conf/callbacks/default.yaml`)

```yaml
model_checkpoint:
  _target_: lightning.pytorch.callbacks.ModelCheckpoint
  monitor: "val_acc"
  mode: "max"
  save_top_k: 1

early_stopping:
  _target_: lightning.pytorch.callbacks.EarlyStopping
  monitor: "val_acc"
  patience: 10
```

## 6. CLI Commands

### Feature Extraction (Phase 1)

```bash
# Extract features for all splits
python main.py extract-features

# Extract for specific split
python main.py extract-features --split train

# Force re-extraction
python main.py extract-features --force

# Check cache info
python main.py cache-info

# Verify cache integrity
python main.py verify-cache
```

### Training (Phase 2)

```bash
# Train with cached features
python main.py train-lightning v1

# Train with custom parameters
python main.py train-lightning v1 arch.training.max_epochs=100

# Train without cache (on-the-fly extraction)
python main.py train-lightning v1 --no-cache
```

## 7. Data Flow Summary

| Stage | Input | Output |
|-------|-------|--------|
| Audio loading | `.wav` | waveform |
| Segmentation | waveform | audio chunks |
| STFT + Mel | chunks | mel spectrogram |
| Log scaling | mel | log-mel features |
| Normalization | features | normalized features |
| Tensor shaping | features | CNN-ready tensor |
| Caching | tensor | `.npy` file |
| Training | `.npy` | embeddings |

## Key Mental Model

> Audio processing is an offline preprocessing job.
> Training and evaluation operate only on cached features.

This is the defining characteristic of baseline v1.

## File Reference

| File | Purpose |
|------|---------|
| `preprocessing/preprocess.py` | Audio loading, spectrogram extraction |
| `preprocessing/feature_cache.py` | Feature caching pipeline |
| `preprocessing/cached_dataset.py` | Dataset classes for cached features |
| `preprocessing/datamodule.py` | PyTorch Lightning DataModule |
| `archs/train.py` | Generic trainer |
| `archs/v1/arch.py` | ProtoNet model |
| `archs/v1/lightning_module.py` | Lightning wrapper |
| `conf/config.yaml` | Main configuration |
| `conf/arch/v1.yaml` | Architecture config |
| `conf/callbacks/default.yaml` | Callbacks config |
| `conf/logger/mlflow.yaml` | MLflow logger config |
