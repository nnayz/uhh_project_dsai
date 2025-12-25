# Feature Extraction, Audio Processing, and Feature Caching
## Baseline v1 (DCASE Few-Shot Bioacoustic)

<<<<<<< HEAD
<<<<<<< HEAD
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
=======
This document provides a complete, end-to-end explanation of how baseline v1 processes raw audio files, extracts features, persists them as `.npy` files, and later consumes those cached features during training and evaluation.

---
=======
This document explains how baseline v1 processes raw audio files, extracts features, persists them as `.npy` files, and consumes those cached features during training and evaluation.
>>>>>>> 0fd03c6 (Feature caching support with callbacks)

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

<<<<<<< HEAD
### 2.2 Dataset Structure
>>>>>>> f21206b (feat: feature cachine, reduced train time)
=======
### Dataset Structure
>>>>>>> 0fd03c6 (Feature caching support with callbacks)

```
/data/msc-proj/
  Training_Set/
    BV/
      BV_file1.wav
<<<<<<< HEAD
<<<<<<< HEAD
      BV_file1.csv
=======
      BV_file1.csv       # Annotations
    PB/
      PB_file1.wav
      PB_file1.csv
>>>>>>> f21206b (feat: feature cachine, reduced train time)
=======
      BV_file1.csv
>>>>>>> 0fd03c6 (Feature caching support with callbacks)
  Validation_Set_DSAI_2025_2026/
    ...
  Evaluation_Set_DSAI_2025_2026/
    ...
```

<<<<<<< HEAD
<<<<<<< HEAD
### Annotation Format
=======
### 2.3 Annotation Format
>>>>>>> f21206b (feat: feature cachine, reduced train time)
=======
### Annotation Format
>>>>>>> 0fd03c6 (Feature caching support with callbacks)

CSV files with columns:
- `Audiofilename`: Name of the audio file
- `Starttime`: Start time of segment (seconds)
- `Endtime`: End time of segment (seconds)
- `Q` or `CLASS_*`: Label (POS/NEG/UNK)

<<<<<<< HEAD
<<<<<<< HEAD
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
=======
---
=======
## 3. Feature Extraction
>>>>>>> 0fd03c6 (Feature caching support with callbacks)

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

<<<<<<< HEAD
## 5. Feature Normalization

After time–frequency transformation, features are normalized:

### Per-Sample Normalization (Default)
```python
mean = features.mean()
std = features.std()
features = (features - mean) / std
```

### Configuration
```yaml
features:
  normalize: true
  normalize_mode: per_sample  # or 'global'
```

---

## 6. Tensor Formatting

Before saving or modeling, a channel dimension is added:
>>>>>>> f21206b (feat: feature cachine, reduced train time)
=======
### Output Shape
>>>>>>> 0fd03c6 (Feature caching support with callbacks)

```
(n_mels, time_frames) → (1, n_mels, time_frames)
```

<<<<<<< HEAD
<<<<<<< HEAD
## 4. Feature Caching

### Cache Directory Structure

```
{cache_dir}/{version}/{config_hash}/{split}/
  manifest.json
  {class_name}/
    {wav_stem}_{start}_{end}.npy
```

### Manifest File
=======
This matches CNN input expectations:
```
(batch, channels, height, width) = (B, 1, n_mels, T)
```
=======
## 4. Feature Caching
>>>>>>> 0fd03c6 (Feature caching support with callbacks)

### Cache Directory Structure

```
{cache_dir}/{version}/{config_hash}/{split}/
  manifest.json
  {class_name}/
    {wav_stem}_{start}_{end}.npy
```

<<<<<<< HEAD
### 7.2 Cache Directory Structure

Features are saved in a versioned, hash-organized structure:

```
{cache_dir}/
  {version}/
    {config_hash}/
      train/
        manifest.json           # Metadata about cached features
        BV/
          BV_file1_0.500_1.200.npy
          BV_file1_2.100_3.500.npy
        PB/
          PB_file1_1.000_2.000.npy
      val/
        manifest.json
        ...
      test/
        manifest.json
        ...
```

### 7.3 Config Hash

A hash is computed from feature extraction parameters:
```python
{
    "sampling_rate": 16000,
    "n_mels": 64,
    "frame_length": 0.025,
    "hop_length": 0.010,
    "normalize": true,
    "normalize_mode": "per_sample",
    "min_duration": 0.5
}
```

If any parameter changes, a new cache directory is created.

### 7.4 Manifest File
>>>>>>> f21206b (feat: feature cachine, reduced train time)
=======
### Manifest File
>>>>>>> 0fd03c6 (Feature caching support with callbacks)

Each split has a `manifest.json` containing:
```json
{
    "version": "v1",
    "config_hash": "abc123def456",
    "split": "train",
    "num_samples": 1234,
    "num_classes": 15,
<<<<<<< HEAD
<<<<<<< HEAD
    "class_to_idx": {"BV": 0, "PB": 1},
    "samples": [...]
}
```

### Config Hash

A hash is computed from feature extraction parameters. If any parameter changes, a new cache directory is created to prevent stale features.

## 5. Configuration Reference
=======
    "class_to_idx": {"BV": 0, "PB": 1, ...},
    "feature_shape": [1, 64],
    "normalization": "per_sample",
    "samples": [
        {
            "npy_path": "BV/BV_file1_0.500_1.200.npy",
            "class_id": 0,
            "class_name": "BV",
            "original_wav": "/data/msc-proj/Training_Set/BV/BV_file1.wav",
            "start_time": 0.5,
            "end_time": 1.2,
            "shape": [1, 64, 70]
        },
        ...
    ]
=======
    "class_to_idx": {"BV": 0, "PB": 1},
    "samples": [...]
>>>>>>> 0fd03c6 (Feature caching support with callbacks)
}
```

### Config Hash

A hash is computed from feature extraction parameters. If any parameter changes, a new cache directory is created to prevent stale features.

<<<<<<< HEAD
---

## 8. Codebase Structure

### 8.1 Phase 1: Feature Generation Code

**Location**: `preprocessing/`

```
preprocessing/
  feature_cache.py      # Feature extraction and caching
  preprocess.py         # Audio loading and spectrogram computation
  ann_service.py        # Annotation parsing
  cached_dataset.py     # Dataset classes for cached features
  datamodule.py         # PyTorch Lightning DataModule
```

**Responsibilities**:
- Loading audio
- Segmenting
- Extracting spectrograms
- Normalizing
- Saving `.npy` files

**When to run**: Once per dataset configuration, or whenever feature parameters change.

### 8.2 Phase 2: Training and Evaluation Code

**Location**: `archs/`

```
archs/
  train.py              # Generic Lightning trainer
  v1/
    arch.py             # ProtoNet model
    lightning_module.py # Lightning wrapper
```

**Responsibilities**:
- Loading `.npy` files
- Batching tensors
- Passing features through the embedding network
- Performing few-shot learning

**No audio code exists here.**

---

## 9. Dataset Loading During Training

During training and evaluation:
1. Dataset loaders read `.npy` files directly
2. Audio libraries are **no longer used**
3. Feature extraction is **skipped entirely**

```python
# preprocessing/cached_dataset.py
class CachedFeatureDataset(Dataset):
    def __getitem__(self, idx):
        sample = self.samples[idx]
        npy_path = self.cache_dir / sample["npy_path"]
        feature = np.load(npy_path)
        return torch.from_numpy(feature), sample["class_id"]
```

Data flow becomes:
```
.npy → torch tensor → CNN → embedding
```

This significantly speeds up experiments.

---

## 10. Feature Flow in Few-Shot Episodes

In each episode:

1. **Support and query `.npy` files are selected**
2. **Feature tensors are loaded**
3. **Embeddings are computed** via the encoder
4. **Class prototypes are formed** (mean of support embeddings)
5. **Query samples are classified** via distance metrics

```python
# archs/v1/arch.py
emb_support = self.encoder(support_x)  # (Ns, D)
emb_query = self.encoder(query_x)      # (Nq, D)

prototypes = compute_prototypes(emb_support, support_y)  # (Nc, D)
dists = euclidean_dist(emb_query, prototypes)            # (Nq, Nc)
logits = -dists  # closer = larger logit
```

The few-shot logic operates **purely in feature space**.

---

## 11. Configuration Reference
>>>>>>> f21206b (feat: feature cachine, reduced train time)
=======
## 5. Configuration Reference
>>>>>>> 0fd03c6 (Feature caching support with callbacks)

### Main Config (`conf/config.yaml`)

```yaml
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 0fd03c6 (Feature caching support with callbacks)
# Path Configuration
path:
  root_dir: /data/msc-proj
  train_dir: ${path.root_dir}/Training_Set
  eval_dir: ${path.root_dir}/Validation_Set_DSAI_2025_2026
  test_dir: ${path.root_dir}/Evaluation_Set_DSAI_2025_2026
<<<<<<< HEAD

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
=======
# Feature caching configuration
features:
  cache_dir: ${data.data_dir}/features_cache
  version: v1
  use_cache: true
  force_recompute: false
  format: npy
  normalize: true
  normalize_mode: per_sample

# Audio parameters
data:
  sampling_rate: 16000
  n_mels: 64
  frame_length: 0.025
  hop_length: 0.010
>>>>>>> f21206b (feat: feature cachine, reduced train time)
=======

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
>>>>>>> 0fd03c6 (Feature caching support with callbacks)
```

### Architecture Config (`conf/arch/v1.yaml`)

```yaml
name: v1

<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
# Model Architecture
>>>>>>> 0fd03c6 (Feature caching support with callbacks)
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

<<<<<<< HEAD
## 12. CLI Commands
>>>>>>> f21206b (feat: feature cachine, reduced train time)
=======
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
>>>>>>> 0fd03c6 (Feature caching support with callbacks)

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
<<<<<<< HEAD
<<<<<<< HEAD
# Train with cached features
python main.py train-lightning v1

# Train with custom parameters
python main.py train-lightning v1 arch.training.max_epochs=100
=======
# Train with cached features (default)
python main.py train-lightning v1

# Train with custom parameters
python main.py train-lightning v1 arch.training.learning_rate=0.0005
>>>>>>> f21206b (feat: feature cachine, reduced train time)
=======
# Train with cached features
python main.py train-lightning v1

# Train with custom parameters
python main.py train-lightning v1 arch.training.max_epochs=100
>>>>>>> 0fd03c6 (Feature caching support with callbacks)

# Train without cache (on-the-fly extraction)
python main.py train-lightning v1 --no-cache
```

<<<<<<< HEAD
<<<<<<< HEAD
## 7. Data Flow Summary
=======
---

## 13. Summary of the v1 Data Flow
>>>>>>> f21206b (feat: feature cachine, reduced train time)
=======
## 7. Data Flow Summary
>>>>>>> 0fd03c6 (Feature caching support with callbacks)

| Stage | Input | Output |
|-------|-------|--------|
| Audio loading | `.wav` | waveform |
| Segmentation | waveform | audio chunks |
| STFT + Mel | chunks | mel spectrogram |
| Log scaling | mel | log-mel features |
| Normalization | features | normalized features |
| Tensor shaping | features | CNN-ready tensor |
<<<<<<< HEAD
<<<<<<< HEAD
| Caching | tensor | `.npy` file |
| Training | `.npy` | embeddings |

## Key Mental Model

> Audio processing is an offline preprocessing job.
> Training and evaluation operate only on cached features.

This is the defining characteristic of baseline v1.

=======
| **Caching** | tensor | `.npy` file |
=======
| Caching | tensor | `.npy` file |
>>>>>>> 0fd03c6 (Feature caching support with callbacks)
| Training | `.npy` | embeddings |

## Key Mental Model

> Audio processing is an offline preprocessing job.
> Training and evaluation operate only on cached features.

This is the defining characteristic of baseline v1.

<<<<<<< HEAD
---

## Implications for Your Project

If you adopt baseline v1 faithfully:

1. **Feature extraction should be a separate step** (`python main.py extract-features`)
2. **`.npy` files become your training dataset**
3. **Model code should never touch `.wav` files**
4. **Any change to feature parameters requires re-extraction**

---

>>>>>>> f21206b (feat: feature cachine, reduced train time)
=======
>>>>>>> 0fd03c6 (Feature caching support with callbacks)
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
<<<<<<< HEAD
<<<<<<< HEAD
| `conf/arch/v1.yaml` | Architecture config |
| `conf/callbacks/default.yaml` | Callbacks config |
| `conf/logger/mlflow.yaml` | MLflow logger config |
=======
| `conf/arch/v1.yaml` | Architecture-specific config |

>>>>>>> f21206b (feat: feature cachine, reduced train time)
=======
| `conf/arch/v1.yaml` | Architecture config |
| `conf/callbacks/default.yaml` | Callbacks config |
| `conf/logger/mlflow.yaml` | MLflow logger config |
>>>>>>> 0fd03c6 (Feature caching support with callbacks)
