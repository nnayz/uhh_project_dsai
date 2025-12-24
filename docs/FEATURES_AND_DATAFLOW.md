# Feature Extraction, Audio Processing, and Feature Caching
## Baseline v1 (DCASE Few-Shot Bioacoustic)

This document provides a complete, end-to-end explanation of how baseline v1 processes raw audio files, extracts features, persists them as `.npy` files, and later consumes those cached features during training and evaluation.

---

## Table of Contents

1. [High-Level Pipeline Overview](#1-high-level-pipeline-overview)
2. [Audio Input and Dataset Layout](#2-audio-input-and-dataset-layout)
3. [Audio Loading and Preprocessing](#3-audio-loading-and-preprocessing)
4. [Time–Frequency Feature Extraction](#4-timefrequency-feature-extraction)
5. [Feature Normalization](#5-feature-normalization)
6. [Tensor Formatting](#6-tensor-formatting)
7. [Feature Caching as .npy Files](#7-feature-caching-as-npy-files)
8. [Codebase Structure](#8-codebase-structure)
9. [Dataset Loading During Training](#9-dataset-loading-during-training)
10. [Feature Flow in Few-Shot Episodes](#10-feature-flow-in-few-shot-episodes)
11. [Configuration Reference](#11-configuration-reference)
12. [CLI Commands](#12-cli-commands)
13. [Summary of the v1 Data Flow](#13-summary-of-the-v1-data-flow)

---

## 1. High-Level Pipeline Overview

Baseline v1 is explicitly split into **two distinct phases**:

```
PHASE 1 (offline, once)
.wav audio
  ↓
feature extraction
  ↓
.npy feature files (cached on disk)

PHASE 2 (online, repeated)
.npy feature files
  ↓
embedding network
  ↓
few-shot classification
```

**The model never sees raw audio during training.**

### Why Two Phases?

- **Speed**: Feature extraction is computationally expensive. Computing once saves time.
- **Reproducibility**: Cached features ensure identical inputs across experiments.
- **Flexibility**: Experiments with the model don't require re-processing audio.
- **Debugging**: Cached features can be inspected and verified.

---

## 2. Audio Input and Dataset Layout

### 2.1 Raw Audio Files

- **Input format**: `.wav`
- **Audio**: Mono or converted to mono
- **Sampling rate**: Fixed (default: 16kHz)

### 2.2 Dataset Structure

```
/data/msc-proj/
  Training_Set/
    BV/
      BV_file1.wav
      BV_file1.csv       # Annotations
    PB/
      PB_file1.wav
      PB_file1.csv
  Validation_Set_DSAI_2025_2026/
    ...
  Evaluation_Set_DSAI_2025_2026/
    ...
```

### 2.3 Annotation Format

CSV files with columns:
- `Audiofilename`: Name of the audio file
- `Starttime`: Start time of segment (seconds)
- `Endtime`: End time of segment (seconds)
- `Q` or `CLASS_*`: Label (POS/NEG/UNK)

---

## 3. Audio Loading and Preprocessing

### 3.1 Audio Loading

Each `.wav` file is:
1. Read from disk using `librosa`
2. Resampled to target sampling rate
3. Converted to mono floating-point waveform

```python
# preprocessing/preprocess.py
waveform, sr = librosa.load(path, sr=cfg.data.sampling_rate, mono=True)
```

At this point, data is in the **time domain**:
```
Shape: (audio_samples,)
```

### 3.2 Segmentation

Long recordings are divided into fixed-length segments based on annotations:
```python
start_sample = int(start_time * sr)
end_sample = int(end_time * sr)
segment = waveform[start_sample:end_sample]
```

Segments shorter than `min_duration` are zero-padded.

---

## 4. Time–Frequency Feature Extraction

This is the core transformation stage in baseline v1.

### 4.1 Spectrogram Computation

For each audio segment:
1. **Short-Time Fourier Transform (STFT)**
2. **Power spectrum**
3. **Projection onto mel filterbanks**

```python
# preprocessing/preprocess.py
mel = librosa.feature.melspectrogram(
    y=waveform,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
)
logmel = librosa.power_to_db(mel)
```

**Output shape**:
```
(n_mels, time_frames)
```

### 4.2 Default Parameters

| Parameter | Value | Config Key |
|-----------|-------|------------|
| Sampling rate | 16000 Hz | `data.sampling_rate` |
| Frame length | 0.025s (25ms) | `data.frame_length` |
| Hop length | 0.010s (10ms) | `data.hop_length` |
| n_mels | 64 | `data.n_mels` |
| n_fft | 400 (= frame_length × sr) | Computed |

### 4.3 Log Scaling

Raw mel spectrograms are converted to dB scale:
```python
logmel = librosa.power_to_db(mel + eps)
```

This step:
- Stabilizes magnitude ranges
- Improves robustness to background noise
- Is standard for bioacoustic tasks

---

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

```
(n_mels, time_frames) → (1, n_mels, time_frames)
```

This matches CNN input expectations:
```
(batch, channels, height, width) = (B, 1, n_mels, T)
```

---

## 7. Feature Caching as .npy Files

### 7.1 What Happens After Feature Extraction

**This is the key difference in baseline v1.**

Instead of immediately passing features to the model:
1. Extracted feature tensors are converted to NumPy arrays
2. They are saved to disk as `.npy` files
3. This happens **once, offline**

```
.wav → spectrogram → normalized tensor → .npy file
```

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

Each split has a `manifest.json` containing:
```json
{
    "version": "v1",
    "config_hash": "abc123def456",
    "split": "train",
    "num_samples": 1234,
    "num_classes": 15,
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
}
```

### 7.5 Why .npy Is Used

| Benefit | Description |
|---------|-------------|
| **Fast I/O** | NumPy's binary format is highly optimized |
| **Reproducibility** | Exact numerical values preserved |
| **Zero recomputation** | No feature extraction during training |
| **Easy inspection** | Load with `np.load()` for debugging |

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

### Main Config (`conf/config.yaml`)

```yaml
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
```

### Architecture Config (`conf/arch/v1.yaml`)

```yaml
name: v1

model:
  encoder_type: conv4
  embedding_dim: 2048
  distance: euclidean

episodes:
  n_way: 5
  k_shot: 5
  n_query: 10
  episodes_per_epoch: 1000
  val_episodes: 100
  test_episodes: 100

training:
  learning_rate: 1e-3
  weight_decay: 1e-4
  max_epochs: 10
```

---

## 12. CLI Commands

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
# Train with cached features (default)
python main.py train-lightning v1

# Train with custom parameters
python main.py train-lightning v1 arch.training.learning_rate=0.0005

# Train without cache (on-the-fly extraction)
python main.py train-lightning v1 --no-cache
```

---

## 13. Summary of the v1 Data Flow

| Stage | Input | Output |
|-------|-------|--------|
| Audio loading | `.wav` | waveform |
| Segmentation | waveform | audio chunks |
| STFT + Mel | chunks | mel spectrogram |
| Log scaling | mel | log-mel features |
| Normalization | features | normalized features |
| Tensor shaping | features | CNN-ready tensor |
| **Caching** | tensor | `.npy` file |
| Training | `.npy` | embeddings |

---

## Key Mental Model

> **Audio processing is an offline preprocessing job.**
> **Training and evaluation operate only on cached features.**

This is the defining characteristic of baseline v1.

---

## Implications for Your Project

If you adopt baseline v1 faithfully:

1. **Feature extraction should be a separate step** (`python main.py extract-features`)
2. **`.npy` files become your training dataset**
3. **Model code should never touch `.wav` files**
4. **Any change to feature parameters requires re-extraction**

---

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
| `conf/arch/v1.yaml` | Architecture-specific config |

