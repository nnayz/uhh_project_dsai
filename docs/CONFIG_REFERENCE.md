# Configuration Reference

This document provides a complete reference for all configuration parameters used in the DCASE Few-Shot Bioacoustic project. Configuration is managed via Hydra and split across multiple YAML files.

## Table of Contents

1. [Configuration Files Overview](#configuration-files-overview)
2. [Main Configuration (config.yaml)](#main-configuration-configyaml)
3. [Feature Extraction Parameters](#feature-extraction-parameters)
4. [Training Parameters](#training-parameters)
5. [Evaluation Parameters](#evaluation-parameters)
6. [Architecture Configurations](#architecture-configurations)
7. [Trainer Configuration](#trainer-configuration)
8. [Callbacks Configuration](#callbacks-configuration)
9. [Logger Configuration](#logger-configuration)
10. [CLI Override Examples](#cli-override-examples)

---

## Configuration Files Overview

```
conf/
├── config.yaml           # Main configuration (entry point)
├── arch/
│   ├── v1.yaml           # V1 architecture (baseline ResNet)
│   └── v2.yaml           # V2 architecture (ResNet + Attention)
├── callbacks/
│   └── default.yaml      # Training callbacks
├── trainer/
│   └── default.yaml      # PyTorch Lightning trainer settings
├── logger/
│   └── mlflow.yaml       # MLflow logging configuration
└── log_dir/
    └── default.yaml      # Output directory configuration
```

---

## Main Configuration (config.yaml)

**Location**: `conf/config.yaml`

### General Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `print_config` | bool | `true` | Print full configuration at startup |
| `ignore_warnings` | bool | `true` | Suppress Python warnings |
| `train` | bool | `true` | Enable training phase |
| `test` | bool | `true` | Enable testing phase |
| `seed` | int | `1234` | Random seed for reproducibility |
| `disable_cudnn` | bool | `false` | Disable cuDNN (for debugging) |
| `name` | str | `"mlflow_experiments"` | Experiment group name |

### Path Configuration (`path.*`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path.root_dir` | str | `/data/msc-proj` | Root data directory |
| `path.train_dir` | str | `${path.root_dir}/Training_Set` | Training data directory |
| `path.eval_dir` | str | `${path.root_dir}/Validation_Set_DSAI_2025_2026` | Validation data directory |
| `path.test_dir` | str | `${path.root_dir}/Evaluation_Set_DSAI_2025_2026` | Test data directory |
| `path.extra_train_dir` | str | `null` | Optional extra training data |
| `path.mask_dir` | str | `null` | Optional mask directory |

---

## Feature Extraction Parameters

**Location**: `conf/config.yaml` → `features.*`

These parameters control how audio is converted to spectrograms.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION PARAMETER DIAGRAM                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Audio Waveform (sr = 22050 Hz)                                                │
│   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━►            │
│                                                                                 │
│   STFT Parameters:                                                              │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │                                                                        │    │
│   │   n_fft = 1024 samples                                                 │    │
│   │   ├───────────────────────────────────────────────────────────────────┤│    │
│   │   │              Window (~46.4 ms at 22050 Hz)                        ││    │
│   │   └───────────────────────────────────────────────────────────────────┘│    │
│   │                                                                        │    │
│   │   hop_mel = 256 samples (~11.6 ms)                                     │    │
│   │   ├─────────┤ ← Overlap between windows                                │    │
│   │                                                                        │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│   Mel Filterbank:                                                               │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │                                                                        │    │
│   │   Frequency Range: fmin (50 Hz) ──────────────── fmax (11025 Hz)       │    │
│   │                    │                                    │              │    │
│   │                    └──── n_mels = 128 filters ──────────┘              │    │
│   │                                                                        │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Signal Processing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features.sr` | int | `22050` | Sample rate (Hz). Audio is resampled to this rate. |
| `features.n_fft` | int | `1024` | FFT window size in samples. Determines frequency resolution. |
| `features.hop_mel` | int | `256` | Hop length between STFT windows. Determines time resolution. |
| `features.n_mels` | int | `128` | Number of mel frequency bins. More bins = finer frequency detail. |
| `features.fmin` | float | `50` | Minimum frequency (Hz). Filters out low-frequency noise. |
| `features.fmax` | float | `11025` | Maximum frequency (Hz). Typically Nyquist/2 for 22050 Hz. |
| `features.eps` | float | `1e-8` | Small constant for numerical stability in log. |
| `features.feature_types` | str | `"logmel"` | Feature type(s). Options: `logmel`, `pcen`, or `logmel@pcen` for both. |

### Model Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features.embedding_dim` | int | `2048` | Embedding vector dimension from encoder. |
| `features.drop_rate` | float | `0.1` | Dropout rate in encoder. |
| `features.with_bias` | bool | `false` | Use bias in convolutions. |
| `features.non_linearity` | str | `"leaky_relu"` | Activation function. Options: `leaky_relu`, `relu`. |
| `features.time_max_pool_dim` | int | `4` | Time dimension after adaptive pooling. |
| `features.layer_4` | bool | `false` | Enable optional 4th encoder layer. |

### Test-Time Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features.test_seglen_len_lim` | int | `30` | Maximum segment length limit for adaptive test. |
| `features.test_hoplen_fenmu` | int | `3` | Hop length divisor for test sliding window. |

---

## Training Parameters

**Location**: `conf/config.yaml` → `train_param.*`

These control the few-shot learning episode structure and training dynamics.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       EPISODIC TRAINING STRUCTURE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   One Episode (Batch):                                                          │
│                                                                                 │
│   k_way = 10 classes                                                            │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                         │   │
│   │   Class 1:  [S₁][S₂][S₃][S₄][S₅]  [Q₁][Q₂][Q₃][Q₄][Q₅]                 │   │
│   │              └── n_shot=5 ──┘      └── n_shot=5 ──┘                     │   │
│   │              support samples       query samples                        │   │
│   │                                                                         │   │
│   │   Class 2:  [S₁][S₂][S₃][S₄][S₅]  [Q₁][Q₂][Q₃][Q₄][Q₅]                 │   │
│   │   ...                                                                   │   │
│   │   Class 10: [S₁][S₂][S₃][S₄][S₅]  [Q₁][Q₂][Q₃][Q₄][Q₅]                 │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   Total samples per batch = k_way × n_shot × 2 = 10 × 5 × 2 = 100               │
│                                                                                 │
│   With negative_train_contrast=true:                                            │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │   Each sample becomes a (positive, negative) pair:                      │   │
│   │   [P₁, N₁], [P₂, N₂], ...                                               │   │
│   │   Total = 200 segments per batch                                        │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Episode Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_param.k_way` | int | `10` | Number of classes per episode (N-way). |
| `train_param.n_shot` | int | `5` | Number of support/query samples per class (K-shot). |
| `train_param.num_episodes` | int | `2000` | Episodes per epoch. |
| `train_param.seg_len` | float | `0.2` | Segment length in seconds. |

### Training Dynamics

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_param.lr_rate` | float | `0.001` | Initial learning rate. |
| `train_param.scheduler_gamma` | float | `0.65` | LR decay factor per step. |
| `train_param.scheduler_step_size` | int | `10` | Epochs between LR decay. |
| `train_param.device` | str | `"cuda"` | Device for training. |

### Advanced Training Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_param.adaptive_seg_len` | bool | `true` | Use adaptive segment length during testing. |
| `train_param.use_validation_first_5` | bool | `false` | Include first 5 validation examples in training. |
| `train_param.negative_train_contrast` | bool | `true` | Train with negative contrast samples. |
| `train_param.load_weight_from` | str | `null` | Path to pretrained weights. |
| `train_param.negative_seg_search` | bool | `false` | Enable negative segment search. |
| `train_param.merging_segment` | bool | `false` | Merge adjacent segments. |
| `train_param.remove_long_segment` | bool | `false` | Filter out long segments. |
| `train_param.padd_tail` | bool | `false` | Pad segment tails. |

---

## Evaluation Parameters

**Location**: `conf/config.yaml` → `eval_param.*`

These control inference and evaluation behavior.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION SLIDING WINDOW                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Audio file:                                                                   │
│   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━►        │
│                                                                                 │
│   Sliding window (seg_len = 0.2s):                                              │
│                                                                                 │
│   ┌────────┐                                                                    │
│   │  W₁    │                                                                    │
│   └────────┘                                                                    │
│        ┌────────┐                                                               │
│        │  W₂    │     hop_seg = 0.05s (overlap = 0.15s)                        │
│        └────────┘                                                               │
│             ┌────────┐                                                          │
│             │  W₃    │                                                          │
│             └────────┘                                                          │
│                  ┌────────┐                                                     │
│                  │  W₄    │                                                     │
│                  └────────┘                                                     │
│                       ...                                                       │
│                                                                                 │
│   Each window → Encoder → Embedding → Distance to prototypes → Probability     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_param.seg_len` | float | `0.200` | Segment length for evaluation (seconds). |
| `eval_param.hop_seg` | float | `0.05` | Hop length between segments (seconds). |
| `eval_param.samples_neg` | int | `150` | Number of negative samples per iteration. |
| `eval_param.iterations` | int | `3` | Iterations for averaging predictions. |
| `eval_param.query_batch_size` | int | `8` | Batch size for query encoding. |
| `eval_param.negative_set_batch_size` | int | `16` | Batch size for negative set encoding. |
| `eval_param.threshold` | float | `0.9` | Detection probability threshold. |
| `eval_param.negative_estimate` | str | `"freq_mask"` | Method for negative estimation. |

---

## Architecture Configurations

### V1 Architecture (Baseline)

**Location**: `conf/arch/v1.yaml`

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           V1 ARCHITECTURE (ResNet Encoder)                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Input: (batch, 17, 128) spectrogram segment                                   │
│          │                                                                      │
│          ▼                                                                      │
│   ┌─────────────────┐                                                           │
│   │ Reshape to      │ → (batch, 1, 17, 128)                                     │
│   │ (B, C, T, F)    │                                                           │
│   └────────┬────────┘                                                           │
│            │                                                                    │
│            ▼                                                                    │
│   ┌─────────────────┐                                                           │
│   │ Layer 1: 64 ch  │ Conv → BN → ReLU → Conv → BN → ReLU → MaxPool(2)         │
│   └────────┬────────┘                                                           │
│            ▼                                                                    │
│   ┌─────────────────┐                                                           │
│   │ Layer 2: 128 ch │ Conv → BN → ReLU → Conv → BN → ReLU → MaxPool(2)         │
│   └────────┬────────┘                                                           │
│            ▼                                                                    │
│   ┌─────────────────┐                                                           │
│   │ Layer 3: 64 ch  │ Conv → BN → ReLU → Conv → BN → ReLU → MaxPool(2)         │
│   │ + DropBlock     │                                                           │
│   └────────┬────────┘                                                           │
│            ▼                                                                    │
│   ┌─────────────────┐                                                           │
│   │ AdaptiveAvgPool │ → (batch, 64, 4, 8)                                       │
│   └────────┬────────┘                                                           │
│            ▼                                                                    │
│   ┌─────────────────┐                                                           │
│   │ Flatten         │ → (batch, 2048) embedding                                 │
│   └─────────────────┘                                                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arch.model.encoder_type` | str | `"resnet"` | Encoder architecture type. |
| `arch.model.embedding_dim` | int | `2048` | Output embedding dimension. |
| `arch.model.drop_rate` | float | `0.1` | Dropout rate. |
| `arch.model.distance` | str | `"euclidean"` | Distance metric (`euclidean`, `cosine`). |
| `arch.model.n_mels` | int | `128` | Input mel bins. |

### V2 Architecture (Enhanced)

**Location**: `conf/arch/v2.yaml`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arch.model.encoder_type` | str | `"resnet_attention"` | ResNet + Attention encoder. |
| `arch.model.embedding_dim` | int | `1024` | Embedding dimension (memory-efficient). |
| `arch.model.distance` | str | `"learnable"` | Distance metric (`euclidean`, `cosine`, `learnable`). |
| `arch.model.channels` | list | `[32,64,128,256]` | Channel sizes per layer. |
| `arch.model.dropout` | float | `0.1` | Dropout rate. |

#### V2 Augmentation Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arch.augmentation.use_augmentation` | bool | `true` | Enable data augmentation. |
| `arch.augmentation.use_spec_augment` | bool | `true` | Enable SpecAugment. |
| `arch.augmentation.use_noise` | bool | `true` | Add Gaussian noise. |
| `arch.augmentation.time_mask_pct` | float | `0.15` | Max percentage of time to mask. |
| `arch.augmentation.freq_mask_pct` | float | `0.15` | Max percentage of frequency to mask. |

### Episode Configuration (Both Architectures)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arch.episodes.n_way` | int | `${train_param.k_way}` | Classes per episode. |
| `arch.episodes.k_shot` | int | `${train_param.n_shot}` | Samples per class. |
| `arch.episodes.n_query` | int | `5` | Query samples per class. |
| `arch.episodes.episodes_per_epoch` | int | `${train_param.num_episodes}` | Episodes per epoch. |
| `arch.episodes.val_episodes` | int | `200` | Validation episodes. |
| `arch.episodes.test_episodes` | int | `200` | Test episodes. |

### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arch.training.learning_rate` | float | `${train_param.lr_rate}` | Learning rate. |
| `arch.training.weight_decay` | float | `1e-4` | L2 regularization. |
| `arch.training.max_epochs` | int | `200` (V1) / `50` (V2) | Maximum epochs. |
| `arch.training.optimizer` | str | `"adamw"` | Optimizer type. |
| `arch.training.scheduler` | str | `"step"` (V1) / `"cosine"` (V2) | LR scheduler. |
| `arch.training.gradient_clip_val` | float | `null` (V1) / `1.0` (V2) | Gradient clipping. |

---

## Trainer Configuration

**Location**: `conf/trainer/default.yaml`

These are PyTorch Lightning Trainer settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainer.accelerator` | str | `"auto"` | Hardware accelerator. |
| `trainer.devices` | int/str | `"auto"` | Number of devices. |
| `trainer.precision` | str | `"32-true"` | Training precision. |
| `trainer.max_epochs` | int | `${arch.training.max_epochs}` | Maximum epochs. |
| `trainer.check_val_every_n_epoch` | int | `1` | Validation frequency. |
| `trainer.log_every_n_steps` | int | `50` | Logging frequency. |
| `trainer.enable_checkpointing` | bool | `true` | Save checkpoints. |
| `trainer.enable_progress_bar` | bool | `true` | Show progress bar. |

---

## Callbacks Configuration

**Location**: `conf/callbacks/default.yaml`

### ModelCheckpoint

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `callbacks.model_checkpoint.dirpath` | str | `${runtime.ckpt_dir}` | Checkpoint directory. |
| `callbacks.model_checkpoint.filename` | str | `"{arch.name}_{epoch:03d}_{val/fmeasure:.4f}"` | Filename pattern. |
| `callbacks.model_checkpoint.monitor` | str | `"val/fmeasure"` | Metric to monitor. |
| `callbacks.model_checkpoint.mode` | str | `"max"` | Maximize or minimize. |
| `callbacks.model_checkpoint.save_last` | bool | `true` | Save last checkpoint. |
| `callbacks.model_checkpoint.save_top_k` | int | `3` | Keep top K checkpoints. |

### EarlyStopping

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `callbacks.early_stopping.monitor` | str | `"val/fmeasure"` | Metric to monitor. |
| `callbacks.early_stopping.patience` | int | `20` | Epochs before stopping. |
| `callbacks.early_stopping.mode` | str | `"max"` | Maximize or minimize. |

### LearningRateMonitor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `callbacks.lr_monitor.logging_interval` | str | `"epoch"` | Log LR per step or epoch. |

---

## Logger Configuration

**Location**: `conf/logger/mlflow.yaml`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logger.mlflow.experiment_name` | str | `${exp_name}` | MLflow experiment name. |
| `logger.mlflow.tracking_uri` | str | `"file:./mlruns"` | MLflow tracking location. |
| `logger.mlflow.save_dir` | str | `${runtime.log_dir}` | Save directory. |

---

## CLI Override Examples

### Basic Overrides

```bash
# Change learning rate
g5 train v1 --exp-name test arch.training.learning_rate=0.0005

# Reduce episodes for quick test
g5 train v1 --exp-name test train_param.num_episodes=100

# Change segment length
g5 train v1 --exp-name test train_param.seg_len=0.3

# Use PCEN instead of log-mel
g5 train v1 --exp-name test features.feature_types=pcen

# Use both log-mel and PCEN
g5 train v1 --exp-name test features.feature_types=logmel@pcen
```

### Episode Configuration

```bash
# 5-way 3-shot learning
g5 train v1 --exp-name test train_param.k_way=5 train_param.n_shot=3

# More episodes per epoch
g5 train v1 --exp-name test train_param.num_episodes=3000
```

### Training Configuration

```bash
# Longer training
g5 train v1 --exp-name test arch.training.max_epochs=100

# Disable negative contrast
g5 train v1 --exp-name test train_param.negative_train_contrast=false

# Custom scheduler
g5 train v1 --exp-name test arch.training.scheduler=cosine
```

### V2-Specific Options

```bash
# Train V2 with custom augmentation
g5 train v2 --exp-name test \
    arch.augmentation.time_mask_pct=0.2 \
    arch.augmentation.freq_mask_pct=0.2

# Larger V2 model
g5 train v2 --exp-name test \
    arch.model.channels=[64,128,256,512] \
    arch.model.embedding_dim=2048
```

---

## Configuration Interpolation

Hydra supports variable interpolation using `${var}` syntax:

```yaml
# Example interpolations in config.yaml
path:
  root_dir: /data/msc-proj
  train_dir: ${path.root_dir}/Training_Set  # Resolves to /data/msc-proj/Training_Set

train_param:
  sr: ${features.sr}  # Inherits from features.sr (22050)
  seg_len: 0.2

annotations:
  min_duration: ${train_param.seg_len}  # Resolves to 0.2
```

---

## Related Documentation

- [WORKFLOW.md](./WORKFLOW.md) - End-to-end training workflow
- [AUDIO_SIGNAL_PROCESSING.md](./AUDIO_SIGNAL_PROCESSING.md) - Signal processing details
- [CLI_USAGE.md](./CLI_USAGE.md) - Command-line interface guide
