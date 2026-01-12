# Features and Data Flow

This document describes the feature extraction pipeline and how features flow through the dataloader and datamodule during training.

## Table of Contents

1. [Overview](#overview)
2. [Feature Export Pipeline](#feature-export-pipeline)
3. [DataModule Architecture](#datamodule-architecture)
4. [Dataset Classes](#dataset-classes)
5. [Batch Sampler](#batch-sampler)
6. [Complete Data Flow Diagram](#complete-data-flow-diagram)

---

## Overview

The data pipeline has two phases:

1. **Offline Feature Export** - Converts `.wav` files to `.npy` feature arrays (run once)
2. **Online Data Loading** - Dynamically samples segments during training

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           HIGH-LEVEL DATA FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Phase 1: Offline (run once)                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                         │   │
│   │   .wav + CSV labels                                                     │   │
│   │        │                                                                │   │
│   │        ▼                                                                │   │
│   │   ┌─────────────────┐                                                   │   │
│   │   │ g5 export-features │  ← preprocessing/feature_export.py            │   │
│   │   └────────┬────────┘                                                   │   │
│   │            │                                                            │   │
│   │            ▼                                                            │   │
│   │   ┌─────────────────┐                                                   │   │
│   │   │ audio_logmel.npy│  ← Full-audio feature array (T × 128)             │   │
│   │   │ audio_pcen.npy  │                                                   │   │
│   │   └─────────────────┘                                                   │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│                                    ▼                                            │
│   Phase 2: Online (each training iteration)                                     │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                         │   │
│   │   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │   │
│   │   │ DCASEFewShot    │────►│ Dataset Classes │────►│ DataLoader      │   │   │
│   │   │ DataModule      │     │ (segment sampling)│    │ + BatchSampler  │   │   │
│   │   └─────────────────┘     └─────────────────┘     └────────┬────────┘   │   │
│   │                                                             │           │   │
│   │                                                             ▼           │   │
│   │                                                   ┌─────────────────┐   │   │
│   │                                                   │ Episodic Batch  │   │   │
│   │                                                   │ (k_way × n_shot │   │   │
│   │                                                   │  × 2 segments)  │   │   │
│   │                                                   └─────────────────┘   │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Feature Export Pipeline

**Entry Point**: `g5 export-features` → `preprocessing/feature_export.py`

### What Gets Written

For each `.wav` file, one or more `.npy` files are created:

```
Training_Set/
└── BirdSpecies_A/
    ├── audio_001.wav           ← Original audio
    ├── audio_001.csv           ← Annotations (POS/NEG labels)
    ├── audio_001_logmel.npy    ← Log-mel spectrogram (T × 128)
    └── audio_001_pcen.npy      ← PCEN features (T × 128)
```

### Feature Array Structure

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE ARRAY STRUCTURE                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   audio_logmel.npy: np.ndarray, shape (T, 128), dtype float32                   │
│                                                                                 │
│   Time (frames)                                                                 │
│       │                                                                         │
│       ▼                                                                         │
│   ┌───────────────────────────────────────────────────────────────┐             │
│   │ [f₀₀, f₀₁, f₀₂, ... f₀₁₂₇]  │  Frame 0                       │             │
│   │ [f₁₀, f₁₁, f₁₂, ... f₁₁₂₇]  │  Frame 1                       │             │
│   │ [f₂₀, f₂₁, f₂₂, ... f₂₁₂₇]  │  Frame 2                       │             │
│   │ ...                          │  ...                           │             │
│   │ [fₜ₀, fₜ₁, fₜ₂, ... fₜ₁₂₇]  │  Frame T-1                     │             │
│   └───────────────────────────────────────────────────────────────┘             │
│                                  │                                              │
│                                  └──── 128 mel frequency bins ────►             │
│                                                                                 │
│   T = audio_duration × sr / hop_mel                                             │
│     = audio_duration × 22050 / 256                                              │
│     ≈ audio_duration × 86.13 frames/second                                      │
│                                                                                 │
│   Example: 60-second audio → ~5168 frames                                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Export Command

```bash
# Export all splits
g5 export-features --split all

# Export specific split
g5 export-features --split train

# Force overwrite existing files
g5 export-features --split all --force

# Validate exports exist
g5 check-features --split all
```

---

## DataModule Architecture

**Location**: `preprocessing/datamodule.py:DCASEFewShotDataModule`

The DataModule is a PyTorch Lightning abstraction that encapsulates all data loading logic.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        DCASEFewShotDataModule                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                            __init__(cfg)                                │   │
│   │                                                                         │   │
│   │   • Stores configuration (paths, features, train_param, eval_param)     │   │
│   │   • Calls init() to set up datasets and loaders                         │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│                                    ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                          prepare_data()                                 │   │
│   │                                                                         │   │
│   │   • Validates that all required .npy files exist                        │   │
│   │   • Raises RuntimeError if features are missing                         │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│                                    ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                              init()                                     │   │
│   │                                                                         │   │
│   │   Creates:                                                              │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │ Training:                                                       │   │   │
│   │   │   • dataset: PrototypeDynamicArrayDataSet                       │   │   │
│   │   │       or PrototypeDynamicArrayDataSetWithEval                   │   │   │
│   │   │   • sampler: IdentityBatchSampler                               │   │   │
│   │   │   • loader: DataLoader(dataset, batch_sampler=sampler)          │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │ Validation:                                                     │   │   │
│   │   │   • dataset: PrototypeDynamicArrayDataSetVal                    │   │   │
│   │   │   • sampler: IdentityBatchSampler                               │   │   │
│   │   │   • loader: DataLoader(dataset, batch_sampler=sampler)          │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │ Testing:                                                        │   │   │
│   │   │   • dataset: PrototypeTestSet                                   │   │   │
│   │   │       or PrototypeAdaSeglenBetterNegTestSetV2                   │   │   │
│   │   │   • loader: DataLoader(dataset, batch_size=1)                   │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   Methods:                                                                      │
│   • train_dataloader() → Returns training DataLoader                            │
│   • val_dataloader() → Returns validation DataLoader                            │
│   • test_dataloader() → Returns test DataLoader                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Dataset Selection Logic

```python
# Training dataset selection
if train_param.use_validation_first_5:
    dataset = PrototypeDynamicArrayDataSetWithEval(...)  # Includes validation samples
else:
    dataset = PrototypeDynamicArrayDataSet(...)          # Training only

# Test dataset selection  
if train_param.adaptive_seg_len:
    test_dataset = PrototypeAdaSeglenBetterNegTestSetV2(...)  # Adaptive segment lengths
else:
    test_dataset = PrototypeTestSet(...)                       # Fixed segment lengths
```

---

## Dataset Classes

### Feature_Extractor (Shared Loading)

**Location**: `preprocessing/sequence_data/pcen.py`

All datasets use this class to load and normalize cached features.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Feature_Extractor                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   __init__(features, audio_path=[...])                                          │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ 1. Scan directories for .wav files                                      │   │
│   │ 2. Parse feature_types (e.g., "logmel" or "logmel@pcen")                │   │
│   │ 3. Call update_mean_std() to compute normalization statistics           │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   update_mean_std()                                                             │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ • Sample up to 1000 .npy files                                          │   │
│   │ • Compute global mean and std for each feature type                     │   │
│   │ • Cache in class-level dict: mean_std[suffix] = [mean, std]             │   │
│   │                                                                         │   │
│   │ Example: mean_std["logmel"] = [-4.2, 2.1]                               │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   extract_feature(audio_path)                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ 1. Load .npy file for each feature type                                 │   │
│   │ 2. Normalize: (features - mean) / std                                   │   │
│   │ 3. Ensure time-major format: (T, 128)                                   │   │
│   │ 4. Concatenate if multiple feature types: (T, 128*N)                    │   │
│   │                                                                         │   │
│   │ Returns: np.ndarray, shape (T, n_features)                              │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Training Dataset

**Location**: `preprocessing/sequence_data/dynamic_pcen_dataset.py`

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     PrototypeDynamicArrayDataSet                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Initialization:                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ 1. Create Feature_Extractor for train_dir and eval_dir                  │   │
│   │ 2. Collect all CSV files from train_dir                                 │   │
│   │ 3. Build metadata (meta dict) from CSVs:                                │   │
│   │    meta[class_name] = {                                                 │   │
│   │        "info": [(start, end), ...],      # Positive segments            │   │
│   │        "neg_info": [(start, end), ...],  # Negative segments            │   │
│   │        "duration": [dur1, dur2, ...],    # Segment durations            │   │
│   │        "file": [path1, path2, ...]       # Audio file paths             │   │
│   │    }                                                                    │   │
│   │ 4. Load feature arrays into self.pcen[audio_path]                       │   │
│   │ 5. Build class-to-index mapping                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   __getitem__(idx) → Returns segment(s) for class at index idx                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                         │   │
│   │   if negative_train_contrast:                                           │   │
│   │       return (pos_segment, neg_segment, pos_label, neg_label, class)    │   │
│   │   else:                                                                 │   │
│   │       return (segment, label, class_name)                               │   │
│   │                                                                         │   │
│   │   Segment shape: (seg_len_frames, n_features) = (17, 128)               │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   select_positive(class_name) → Randomly selects positive segment               │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ 1. Choose random segment from meta[class_name]["info"]                  │   │
│   │ 2. Get (start, end) times                                               │   │
│   │ 3. Call select_segment() to extract fixed-length segment                │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   select_negative(class_name) → Randomly selects negative segment               │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ 1. Choose random segment from meta[class_name]["neg_info"]              │   │
│   │ 2. Ensure duration > 0.2s (minimum length)                              │   │
│   │ 3. Call select_segment() to extract fixed-length segment                │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   select_segment(start, end, features, seg_len) → Fixed-length segment          │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ 1. Convert time to frames                                               │   │
│   │ 2. If segment < seg_len: tile/repeat to fill                            │   │
│   │ 3. If segment > seg_len: random crop                                    │   │
│   │ 4. If segment empty: return zeros                                       │   │
│   │                                                                         │   │
│   │ Returns: np.ndarray, shape (seg_len, n_features)                        │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Validation Dataset

**Location**: `preprocessing/sequence_data/dynamic_pcen_dataset_val.py`

Similar to training dataset but:
- Sources CSVs from `eval_dir` only
- Uses CSV filename as class name
- Provides `eval_class_idxs` for sampler

### Test Dataset

**Location**: `preprocessing/sequence_data/test_loader.py` (or `test_loader_ada_seglen_better_neg_v2.py`)

Returns complete test episodes with support, negative, and query sets:

```
__getitem__(idx) → Returns:
    (X_pos, X_neg, X_query, hop_seg), strt_index_query, audio_path, ...
```

---

## Batch Sampler

**Location**: `preprocessing/sequence_data/identity_sampler.py:IdentityBatchSampler`

The sampler ensures each batch contains the correct episodic structure.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          IdentityBatchSampler                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Parameters:                                                                   │
│   • k_way: Number of classes per episode (e.g., 10)                             │
│   • n_shot: Samples per class (e.g., 5)                                         │
│   • batch_size: n_shot × 2 = 10 (support + query per class)                     │
│   • n_episode: Number of episodes per epoch                                     │
│                                                                                 │
│   Episode Structure:                                                            │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                         │   │
│   │   Batch indices: [c₁, c₁, c₁, c₁, c₁, c₁, c₁, c₁, c₁, c₁,              │   │
│   │                   c₂, c₂, c₂, c₂, c₂, c₂, c₂, c₂, c₂, c₂,              │   │
│   │                   ...                                                   │   │
│   │                   c₁₀, c₁₀, c₁₀, c₁₀, c₁₀, c₁₀, c₁₀, c₁₀, c₁₀, c₁₀]   │   │
│   │                                                                         │   │
│   │   Where c₁...c₁₀ are randomly selected class indices                    │   │
│   │   Each class appears n_shot × 2 = 10 times                              │   │
│   │   Total batch size = k_way × n_shot × 2 = 10 × 5 × 2 = 100              │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   Iteration:                                                                    │
│   • For each episode, randomly select k_way classes                             │
│   • For each class, yield batch_size indices (all pointing to same class)       │
│   • Dataset's __getitem__ handles actual segment sampling                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Batch Structure Example

For 10-way 5-shot with negative contrast:

```
Batch tensor shape: (100, 17, 128)
           or with negatives: (200, 17, 128)

Labels:    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   # Class 0 (10 samples)
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,   # Class 1 → label 2
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4,   # Class 2 → label 4
            ...
            18, 18, 18, 18, 18, 18, 18, 18, 18, 18]  # Class 9 → label 18

With negative contrast, labels are interleaved:
    Even indices (0, 2, 4, ...) = positive class labels
    Odd indices (1, 3, 5, ...) = negative class labels
```

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       COMPLETE TRAINING DATA FLOW                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   1. INITIALIZATION (once per training run)                                     │
│   ═════════════════════════════════════════                                     │
│                                                                                 │
│   archs/train.py                                                                │
│       │                                                                         │
│       ▼                                                                         │
│   DCASEFewShotDataModule(cfg)                                                   │
│       │                                                                         │
│       ├──► prepare_data() ──► validate_features() ──► Check .npy exist         │
│       │                                                                         │
│       └──► init()                                                               │
│             │                                                                   │
│             ├──► PrototypeDynamicArrayDataSet(...)                              │
│             │         │                                                         │
│             │         ├──► Feature_Extractor(...)                               │
│             │         │         │                                               │
│             │         │         └──► update_mean_std() ──► Compute μ, σ         │
│             │         │                                                         │
│             │         ├──► get_all_csv_files() ──► Find all .csv                │
│             │         │                                                         │
│             │         └──► build_meta() ──► Parse CSVs, load features           │
│             │                                                                   │
│             ├──► IdentityBatchSampler(...)                                      │
│             │                                                                   │
│             └──► DataLoader(dataset, batch_sampler=sampler)                     │
│                                                                                 │
│   2. TRAINING LOOP (each iteration)                                             │
│   ══════════════════════════════════                                            │
│                                                                                 │
│   for batch in train_dataloader:                                                │
│       │                                                                         │
│       ▼                                                                         │
│   IdentityBatchSampler.__iter__()                                               │
│       │                                                                         │
│       │   Returns batch of indices: [c₁×10, c₂×10, ..., c₁₀×10]                 │
│       │                                                                         │
│       ▼                                                                         │
│   DataLoader collects samples                                                   │
│       │                                                                         │
│       │   For each index in batch:                                              │
│       │       │                                                                 │
│       │       ▼                                                                 │
│       │   dataset.__getitem__(idx)                                              │
│       │       │                                                                 │
│       │       ├──► class_name = classes[idx]                                    │
│       │       │                                                                 │
│       │       ├──► select_positive(class_name)                                  │
│       │       │         │                                                       │
│       │       │         ├──► Random segment from meta[class]["info"]            │
│       │       │         │                                                       │
│       │       │         └──► select_segment(start, end, features)               │
│       │       │                     │                                           │
│       │       │                     └──► Crop/tile to fixed length              │
│       │       │                                                                 │
│       │       ├──► select_negative(class_name) (if negative_train_contrast)     │
│       │       │         │                                                       │
│       │       │         └──► Same process for negative regions                  │
│       │       │                                                                 │
│       │       └──► Returns: (segment, label, class_name)                        │
│       │                 or: (pos, neg, pos_label, neg_label, class_name)        │
│       │                                                                         │
│       ▼                                                                         │
│   Collated batch tensor                                                         │
│       │                                                                         │
│       │   Shape: (batch_size, seg_len_frames, n_features)                       │
│       │          = (100, 17, 128) without negatives                             │
│       │          = (200, 17, 128) with negatives                                │
│       │                                                                         │
│       ▼                                                                         │
│   Lightning Module training_step(batch)                                         │
│       │                                                                         │
│       ├──► _forward_embed(x) ──► encoder(x) ──► embeddings                      │
│       │                                                                         │
│       └──► prototypical_loss(embeddings, labels) ──► loss, accuracy             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## File Locations Summary

| Component | File Path |
|-----------|-----------|
| Feature Export | `preprocessing/feature_export.py` |
| DataModule | `preprocessing/datamodule.py` |
| Feature Loader | `preprocessing/sequence_data/pcen.py` |
| Training Dataset | `preprocessing/sequence_data/dynamic_pcen_dataset.py` |
| Training Dataset (with Eval) | `preprocessing/sequence_data/dynamic_pcen_dataset_first_5.py` |
| Validation Dataset | `preprocessing/sequence_data/dynamic_pcen_dataset_val.py` |
| Test Dataset (Fixed) | `preprocessing/sequence_data/test_loader.py` |
| Test Dataset (Adaptive) | `preprocessing/sequence_data/test_loader_ada_seglen_better_neg_v2.py` |
| Batch Sampler | `preprocessing/sequence_data/identity_sampler.py` |

---

## CLI Commands

```bash
# Export features (run once)
g5 export-features --split all

# Validate features exist
g5 check-features --split all

# Training will use the DataModule automatically
g5 train v1 --exp-name my_experiment
```

---

## Related Documentation

- [AUDIO_SIGNAL_PROCESSING.md](./AUDIO_SIGNAL_PROCESSING.md) - How audio becomes features
- [CONFIG_REFERENCE.md](./CONFIG_REFERENCE.md) - Configuration parameters
- [PREPROCESSING.md](./PREPROCESSING.md) - CSV parsing and segment selection details
